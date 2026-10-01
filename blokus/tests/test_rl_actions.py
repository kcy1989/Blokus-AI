"""Every claim in `rl/actions.py` that matters, checked against the engine.

The oracle here is `engine` itself and the two never share code: the action
index is built from `ORIENTS` lengths, while the legal set comes from
`engine.legal_move_mask`. If the index arithmetic were wrong the round-trip
would still pass, so the tests deliberately compare against the engine's own
move tuples as well.

`random_playout` comes from `bench_engine` because the positions that matter
are the ones real games produce; a synthetic board would mostly produce empty
or full ones.
"""
import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
from bench_engine import random_playout  # noqa: E402
import rl.actions as A  # noqa: E402
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_PASSES = 40


def positions_from_playouts(seeds):
    """Every position of `seeds`, passes included, so the rotation tests see
    the awkward ones too."""
    out = []
    for seed in seeds:
        tape = []
        random_playout(seed, tape=tape, max_passes=MAX_PASSES)
        out.extend(s for s, _mask, _n, _mv in tape)
    return out


# --------------------------------------------------------------- constants

def test_the_action_space_is_91_orientations_times_400_bases():
    assert A.ORIENT_OFFSET[-1] == 91
    assert A.N_ORIENT == 91
    assert A.N_ACTIONS == 91 * 400 == 36400
    assert len(A.ORIENT_OFFSET) == engine.N_PIECES + 1


def test_the_offsets_agree_with_the_action_table():
    """The frozen table enumerates orientations in exactly the order the
    offsets assume, so a shard labelled with an action index means the same
    thing forever."""
    with open(os.path.join(HERE, "action_table.json"), encoding="utf-8") as fh:
        table = json.load(fh)
    actions = table["actions"]
    assert len(actions) == 91
    j = 0
    for piece_idx in range(engine.N_PIECES):
        name = engine.PIECE_ORDER[piece_idx]
        for oi in range(len(engine.ORIENTS[piece_idx])):
            assert actions[j] == [name, oi], (j, actions[j], name, oi)
            assert A.ORIENT_OFFSET[piece_idx] + oi == j
            j += 1
    assert j == 91


# --------------------------------------------------------------- the encoding

def test_index_to_move_inverts_move_to_index_on_every_real_placement():
    seen = set()
    for piece_idx in range(engine.N_PIECES):
        for oi, od in enumerate(engine.ORIENTS[piece_idx]):
            valid = od["valid"]
            while valid:
                low = valid & -valid
                base = low.bit_length() - 1
                valid ^= low
                m = (piece_idx, oi, base)
                idx = A.move_to_index(m)
                assert 0 <= idx < A.N_ACTIONS
                assert idx not in seen, "two placements share one index"
                seen.add(idx)
                assert A.index_to_move(idx) == m
    assert len(seen) == 30433


def test_the_unreal_slots_are_exactly_those_whose_bounding_box_overhangs():
    real = 0
    for piece_idx in range(engine.N_PIECES):
        for oi, od in enumerate(engine.ORIENTS[piece_idx]):
            base = (A.ORIENT_OFFSET[piece_idx] + oi) * 400
            for b in range(400):
                want = bool((od["valid"] >> b) & 1)
                assert A.is_real_index(base + b) == want
                real += want
    assert real == 30433
    assert int(A.REAL_MASK.sum()) == 30433
    assert int(A.REAL_MASK.sum()) == A.REAL_INDICES.size


# --------------------------------------------------------------- legal moves

def test_legal_indices_agrees_with_the_engines_own_move_list():
    for s in positions_from_playouts(range(100)):
        want = sorted(A.move_to_index(m)
                      for m in engine.unpack_move_mask(engine.legal_move_mask(s)))
        got = A.legal_indices(s).tolist()
        if not want:
            assert got == []
        else:
            assert got == want
        assert got == sorted(got), "legal_indices must be sorted"


def test_the_empty_board_offers_exactly_58_moves_to_every_player():
    s = engine.initial_state()
    for owner in range(4):
        assert A.legal_indices(s, owner).size == 58


# --------------------------------------------------------------- rotation

def test_the_rotation_tables_are_permutations_of_the_real_actions():
    real = A.REAL_INDICES
    for k in range(4):
        lut = A.LUT_ROT[k]
        mapped = lut[real]
        assert (mapped >= 0).all(), "k=%d left a real action unmapped" % k
        assert mapped.size == np.unique(mapped).size == real.size
    assert (A.LUT_ROT[0][real] == real).all()
    assert (A.LUT_ROT[0][~A.REAL_MASK] == -1).all()


def test_four_clockwise_steps_come_back_to_the_start():
    """Each table is a quarter turn, so four applications of any of them return
    every real action to itself."""
    real = A.REAL_INDICES
    for k in range(4):
        table = A.LUT_ROT[k]
        back = real
        for _ in range(4):
            back = table[back]
        assert (back == real).all(), "LUT_ROT[%d] does not have order 4" % k
    once = A.LUT_ROT[1][real]
    twice = A.LUT_ROT[1][once]
    three = A.LUT_ROT[1][twice]
    assert (A.LUT_ROT[1][three] == real).all()


def test_the_rotation_tables_agree_with_engine_rotate_move():
    """Spot-check the tables against the engine directly, including pieces
    whose bounding box is not square, which is where a hand-written rotation
    would go wrong."""
    rng = random.Random(4242)
    real = A.REAL_INDICES.tolist()
    for _ in range(4000):
        j = rng.choice(real)
        piece_idx, oi, base = A.index_to_move(j)
        m = (piece_idx, oi, base)
        for k in range(4):
            assert A.LUT_ROT[k][j] == A.move_to_index(m)
            m = engine.rotate_move(m)


def test_rotating_the_position_rotates_its_legal_moves():
    """The equivariance that makes the whole package correct.

    `rotate_state` is the engine's own clockwise step, and it renumbers the
    owners along with the squares. If the legal set moved with the board, then
    the mover's frame and the action indices would disagree and a policy would
    be asked to play a move that does not exist.
    """
    positions = positions_from_playouts(range(100))
    # Cover all four movers, 75 each, so a mismatch that only affects one
    # corner cannot hide.
    by_mover = {}
    for s in positions:
        by_mover.setdefault(s.to_move, []).append(s)
    checked = 0
    for owner in range(4):
        for s in by_mover.get(owner, [])[:75]:
            base = sorted(A.legal_indices(s).tolist())
            rot = s
            for k in range(1, 4):
                rot = engine.rotate_state(rot)
                got = sorted(A.LUT_ROT[k][np.array(base, dtype=np.int32)].tolist()) \
                    if base else []
                assert got == sorted(A.legal_indices(rot).tolist()), (owner, k)
            checked += 1
    assert checked == 300


# --------------------------------------------------------------- the two frames

def test_real_to_view_and_back_are_inverse_on_every_real_action():
    """`view_to_real` inverts `real_to_view`.

    Only that composition is the identity. `real_to_view` is a quarter turn, so
    applying it twice is a half turn and no inverse function undoes it - the
    second assertion would be claiming a rotation does not rotate.
    """
    real = A.REAL_INDICES
    for p in range(4):
        view = A.real_to_view(real, p)
        assert (A.view_to_real(view, p) == real).all(), p
        # The turn really happened: the mapping is not the identity for any
        # seat, or the whole view layer would be doing nothing.
        assert not (view == real).all() or p in (0,), p


def test_real_to_view_uses_the_same_rotation_as_engine_normalize():
    """`engine.normalize` is the authority on which frame the mover sees.

    Checked through a square that cannot be confused: the mover's own corner.
    In the view the mover's stones are channel 0 and they sit at the top-left
    corner, whatever the real corner was.
    """
    f5 = engine.PIECE_IDX["F5"]
    for owner in range(4):
        cx, cy = OWNER_CORNER[owner]
        own = [0, 0, 0, 0]
        own[owner] = 1 << (cx + cy * B)
        hand = [engine.FULL_HAND] * 4
        hand[owner] &= ~(1 << f5)
        st = engine.State(own_bits=tuple(own), hand_bits=tuple(hand),
                          turn_order=CLOCKWISE_OWNERS, to_move=owner,
                          stuck=(False,) * 4)
        view, seat_of = engine.normalize(st)
        assert seat_of[0] == owner
        assert view.to_move == 0
        assert view.own_bits[0] == 1, "me must sit at the top-left corner"
        assert A.seat_of_mover(st) == CLOCKWISE_OWNERS.index(owner)


def test_both_directions_reject_an_unreal_slot():
    real = A.REAL_INDICES
    bad = np.array([-1], dtype=np.int32)
    for p in range(4):
        with pytest.raises(ValueError):
            A.real_to_view(bad, p)
        with pytest.raises(ValueError):
            A.view_to_real(bad, p)
    assert (A.LUT_ROT[1][real] >= 0).all()


# --------------------------------------------------------------- masks

def test_the_view_mask_marks_exactly_the_rotated_legal_moves():
    for s in positions_from_playouts(range(100)):
        mask = A.legal_mask_view(s)
        assert mask.dtype == bool
        assert mask.shape == (A.N_ACTIONS,)
        real_idx = A.legal_indices(s)
        assert int(mask.sum()) == real_idx.size
        if real_idx.size == 0:
            continue
        assert np.flatnonzero(mask).size == real_idx.size
        p = A.seat_of_mover(s)
        assert sorted(A.view_to_real(np.flatnonzero(mask).astype(np.int32),
                                     p).tolist()) == real_idx.tolist()


def test_the_plane_form_round_trips_through_flat_indices():
    for s in positions_from_playouts(range(20)):
        idx = A.legal_indices(s)
        plane = A.indices_to_mask_bits(A.real_to_view(idx, A.seat_of_mover(s)))
        assert plane.shape == (91, B, B)
        assert int(plane.sum()) == idx.size
        back = A.mask_bits_to_indices(plane)
        p = A.seat_of_mover(s)
        assert sorted(A.view_to_real(back, p).tolist()) == sorted(idx.tolist())