"""The features are only correct if they agree with the engine exactly.

Channels 0-6 and the hand vectors are compared bit for bit against
`engine.to_plane` and `engine.hand_vectors` - not against a copy of them. The
four new groups (the opponents' contact rules, the opening squares, the
scalars) have no engine counterpart, so they are checked against arithmetic
derived from `config.OWNER_CORNER` and `engine.need_avoid` instead.

The rotation direction is not assumed either. `test_the_rotation_direction_is_the_one_the_engine_uses`
builds the board rotation from the `(x, y) -> (B - 1 - y, x)` rule and
compares it with `np.rot90`, because getting this backwards produces arrays
that are still self-consistent and still wrong.
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
from bench_engine import random_playout  # noqa: E402
from config import B, CLOCKWISE_OWNERS, N, OWNER_CORNER  # noqa: E402
import rl.features as F  # noqa: E402

MAX_PASSES = 40


def positions(seeds, stride=1):
    out = []
    for seed in seeds:
        tape = []
        random_playout(seed, tape=tape, max_passes=MAX_PASSES)
        out.extend(s for s, _m, _n, _mv in tape[::stride])
    return out


def sample_states(n_games=100, want=300):
    """`want` positions covering all four movers and all three stages."""
    by_mover = {}
    for s in positions(range(n_games), stride=1):
        by_mover.setdefault(s.to_move, []).append(s)
    per = max(1, want // 4)
    out = []
    for owner in range(4):
        out.extend(by_mover.get(owner, [])[:per])
    return out


# --------------------------------------------------------------- constants

def test_the_channel_names_and_count_are_fixed():
    assert F.PLANE_C_V2 == 14
    assert len(F.PLANE_CHANNELS_V2) == 14
    assert F.PLANE_CHANNELS_V2[:7] == engine.PLANE_CHANNELS, \
        "the first seven channels must stay the engine's"
    # 12 per-seat scalars plus `own_move_count`, which is the mover's alone.
    # `plan6.md` H-B0 item 3 added the thirteenth: Hunter's book applies to its
    # first three own moves, and the count is what says so.
    assert F.SCALAR_C_V2 == 13
    assert len(F.SCALAR_NAMES_V2) == 13
    assert F.SCALAR_NAMES_V2[12] == "own_move_count"
    assert len(set(F.PLANE_CHANNELS_V2)) == 14


def test_shapes_and_dtypes():
    s = engine.initial_state()
    planes, scalars, hands = F.featurize(s)
    assert planes.shape == (14, 20, 20)
    assert scalars.shape == (13,)
    assert hands.shape == (4, 21)
    assert planes.dtype == np.float32
    assert scalars.dtype == np.float32
    assert hands.dtype == np.float32
    assert set(np.unique(planes)).issubset({0.0, 1.0})
    assert set(np.unique(hands)).issubset({0.0, 1.0})


# --------------------------------------------------------------- the oracle

def test_the_first_seven_channels_are_the_engines_to_plane():
    checked = 0
    for s in sample_states():
        planes, _scalars, _hands = F.featurize(s)
        assert np.array_equal(planes[:7], engine.to_plane(s)), engine.serialize(s)
        checked += 1
    assert checked >= 300


def test_the_hands_are_the_engines_hand_vectors():
    checked = 0
    for s in sample_states():
        _planes, _scalars, hands = F.featurize(s)
        assert np.array_equal(hands, engine.hand_vectors(s)), engine.serialize(s)
        checked += 1
    assert checked >= 300


def test_featurize_agrees_with_the_normalize_reference():
    """The numpy rotation is a speed choice. `featurize_via_normalize` is the
    definition - it goes through `engine.normalize` - so the two must agree on
    every channel, not just the first seven."""
    checked = 0
    for s in sample_states(want=120):
        p, sc, h = F.featurize(s)
        rp, rsc, rh = F.featurize_via_normalize(s)
        assert np.array_equal(p, rp), engine.serialize(s)
        assert np.array_equal(sc, rsc), engine.serialize(s)
        assert np.array_equal(h, rh), engine.serialize(s)
        checked += 1
    assert checked >= 100


# --------------------------------------------------------------- rotation

def test_the_rotation_direction_is_the_one_the_engine_uses():
    """`np.rot90(a, -1)` is clockwise. Established here rather than assumed."""
    m = np.arange(N, dtype=np.int8).reshape(B, B)
    expected = np.zeros((B, B), dtype=np.int8)
    for y in range(B):
        for x in range(B):
            # engine's clockwise step: (x, y) -> (B - 1 - y, x)
            expected[x, B - 1 - y] = m[y, x]
    assert np.array_equal(F._rot90_clockwise(m, 1), expected)
    assert np.array_equal(F._rot90_clockwise(m, 2), np.rot90(m, k=2))
    assert np.array_equal(F._rot90_clockwise(m, 3), np.rot90(m, k=1))
    assert np.array_equal(F._rot90_clockwise(m, 4), m)
    assert np.array_equal(F._rot90_clockwise(m, 0), m)


def test_the_features_do_not_change_when_the_board_is_rotated():
    """The property rotation augmentation depends on: four games that are
    rotations of each other must featurize identically."""
    checked = 0
    for s in positions(range(100), stride=9)[:120]:
        p, sc, h = F.featurize(s)
        rot = s
        for _ in range(3):
            rot = engine.rotate_state(rot)
            rp, rsc, rh = F.featurize(rot)
            assert np.array_equal(p, rp), engine.serialize(s)
            assert np.array_equal(sc, rsc), engine.serialize(s)
            assert np.array_equal(h, rh), engine.serialize(s)
        checked += 1
    assert checked > 50


def test_seat_of_is_the_same_reading_as_engine_normalize():
    """`featurize` derives the seat order from `CLOCKWISE_OWNERS` instead of
    asking `normalize`. That is a shortcut, so it is checked against the
    function it is standing in for."""
    for s in positions(range(30), stride=11):
        p = CLOCKWISE_OWNERS.index(s.to_move)
        mine = tuple(CLOCKWISE_OWNERS[(p + i) % 4] for i in range(4))
        _view, seat_of = engine.normalize(s)
        assert mine == seat_of
        assert mine[0] == s.to_move


# --------------------------------------------------------------- scalars

def test_the_scalars_are_in_the_declared_order():
    s = engine.initial_state()
    _p, sc, _h = F.featurize(s)
    # Nobody has opened, nobody is stuck, everybody holds 89 cells.
    assert list(sc[0:4]) == [0.0] * 4
    assert list(sc[4:8]) == [1.0] * 4
    assert list(sc[8:12]) == [0.0] * 4
    assert [F.SCALAR_NAMES_V2[i] for i in range(4)] == \
        ["stuck_me", "stuck_next", "stuck_opposite", "stuck_previous"]
    assert [F.SCALAR_NAMES_V2[i] for i in range(4, 8)] == \
        ["remaining_me", "remaining_next", "remaining_opposite",
         "remaining_previous"]
    assert [F.SCALAR_NAMES_V2[i] for i in range(8, 12)] == \
        ["opened_me", "opened_next", "opened_opposite", "opened_previous"]


def test_the_scalar_seat_order_follows_the_channels():
    for s in positions(range(60), stride=7):
        planes, sc, _h = F.featurize(s)
        _view, seat_of = engine.normalize(s)
        for i in range(4):
            o = seat_of[i]
            assert sc[i] == (1.0 if s.stuck[o] else 0.0)
            assert sc[4 + i] == pytest.approx(engine.remaining_cells(s, o) / 89.0)
            assert sc[8 + i] == (1.0 if s.own_bits[o] else 0.0)
            assert planes[i].sum() == float(bin(s.own_bits[o]).count("1"))
            assert planes[10 + i].sum() == (0.0 if s.own_bits[o] else 1.0)


# --------------------------------------------------------------- the new channels

def _handmade(owner, stones, hands=None, stuck=None):
    """A position built from scratch, so a specific situation can be tested."""
    own = [0, 0, 0, 0]
    for o, bits in stones.items():
        own[o] = bits
    hand_bits = [engine.FULL_HAND] * 4
    for o, bits in (hands or {}).items():
        hand_bits[o] = bits
    return engine.State(own_bits=tuple(own), hand_bits=tuple(hand_bits),
                        turn_order=CLOCKWISE_OWNERS, to_move=owner,
                        stuck=stuck or (False, False, False, False))


def test_start_channels_name_the_right_corner_for_each_seat():
    """A position where the mover has opened but the next player has not.

    This is the case the `start_*` channels exist for: `need_avoid` returns
    nothing for a player who has not opened, so nothing else in the array says
    where they are allowed to play.
    """
    f5 = engine.PIECE_IDX["F5"]
    mover = 1                      # top-left corner, so the view has k = 3
    cx, cy = OWNER_CORNER[mover]
    s = _handmade(mover, {mover: 1 << (cx + cy * B)},
                  hands={mover: engine.FULL_HAND & ~(1 << f5)})
    planes, scalars, _h = F.featurize(s)
    corners = F.view_corners(s)
    assert list(scalars[8:12]) == [1.0, 0.0, 0.0, 0.0], \
        "only the mover has opened"
    # Seat 0 has opened, seats 1, 2, 3 have not.
    assert planes[10 + 0].sum() == 0.0
    for i in (1, 2, 3):
        assert planes[10 + i].sum() == 1.0
        x, y = corners[i]
        assert planes[10 + i, y, x] == 1.0
    # The corners come from OWNER_CORNER put through the same clockwise step
    # the planes went through, and with the mover at the top-left the result is
    # the fixed table `_VIEW_CORNER`.
    _view, seat_of = engine.normalize(s)
    k = (4 - CLOCKWISE_OWNERS.index(s.to_move)) % 4
    for i in range(4):
        assert corners[i] == F.rotated_corner(seat_of[i], k)
        assert corners[i] == F._VIEW_CORNER[i]


def test_start_channels_are_zero_once_a_player_has_opened():
    f5 = engine.PIECE_IDX["F5"]
    bits = 0
    for o in range(4):
        x, y = OWNER_CORNER[o]
        bits |= 1 << (x + y * B)
    s = _handmade(1, {o: 1 << (OWNER_CORNER[o][0] + OWNER_CORNER[o][1] * B)
                      for o in range(4)},
                  {1: engine.FULL_HAND & ~(1 << f5)})
    planes, _sc, _h = F.featurize(s)
    for i in range(4):
        assert planes[10 + i].sum() == 0.0, i


def test_the_opponent_need_channels_exclude_their_avoid():
    """The reason `*_need` is `need & ~avoid` and not just `dilate_diag`.

    Two stones of one player side by side make a square that is a diagonal
    neighbour of one of them and an edge neighbour of the other, so it is in
    both sets. `need` is the squares they may touch, `avoid` the ones they may
    not, and the rule is the difference - a square in both is forbidden, so
    the channel has to drop it.
    """
    # (18, 5) and (19, 5), same player. (18, 4) is above the first and
    # diagonal to the second.
    pair = (1 << (18 + 5 * B)) | (1 << (19 + 5 * B))
    s = _handmade(1, {2: pair},
                  hands={1: engine.FULL_HAND & ~(1 << 0)})
    view, seat_of = engine.normalize(s)
    need, avoid = engine.need_avoid(view, seat_of.index(2))
    assert need & avoid, "this position was built to make the sets overlap"
    planes, _sc, _h = F.featurize(s)
    i = seat_of.index(2)
    assert i != 0, "the overlapping player is an opponent here"
    channel = F.need_channel_of_seat(i)
    want = need & ~avoid & engine.empty_of(view.own_bits)
    assert np.array_equal(planes[channel].astype(bool),
                          engine.bits_to_plane(want).astype(bool))
    # The channel is genuinely smaller than a raw diagonal dilation, and the
    # squares it dropped are exactly the ones that were in both sets.
    assert planes[channel].sum() < engine.bits_to_plane(need).sum()
    assert (need & avoid) & engine.empty_of(view.own_bits) != 0


def test_every_player_has_a_need_channel_equal_to_their_own_rule():
    for s in positions(range(40), stride=7):
        planes, _sc, _h = F.featurize(s)
        view, seat_of = engine.normalize(s)
        empt = engine.empty_of(view.own_bits)
        for i in range(4):
            need, avoid = engine.need_avoid(view, i)
            want = need & ~avoid & empt
            assert np.array_equal(
                planes[F.need_channel_of_seat(i)].astype(bool),
                engine.bits_to_plane(want).astype(bool)), i


def test_the_need_channels_are_non_empty_where_contact_is_possible():
    """A sanity floor: after a real opening move there really are squares the
    next player may touch but may not stand on, so a model has something to
    read."""
    found = 0
    for s in positions(range(60), stride=5):
        planes, _sc, _h = F.featurize(s)
        view, seat_of = engine.normalize(s)
        need, avoid = engine.need_avoid(view, 1)
        mine = planes[F.need_channel_of_seat(1)]
        want = need & ~avoid & engine.empty_of(view.own_bits)
        assert np.array_equal(mine.astype(bool),
                              engine.bits_to_plane(want).astype(bool))
        if mine.sum() > 0 and mine.sum() < engine.bits_to_plane(need).sum():
            found += 1
    assert found > 0, ("no position had a next_need channel with the avoid "
                       "squares actually removed from it")


# --------------------------------------------------------------- batching

def test_batch_equals_stacking_the_single_calls():
    batch = positions(range(20), stride=3)[:40]
    planes, scalars, hands = F.featurize_batch(batch)
    assert planes.shape == (len(batch), 14, 20, 20)
    assert scalars.shape == (len(batch), 13)
    assert hands.shape == (len(batch), 4, 21)
    for i, s in enumerate(batch):
        p, sc, h = F.featurize(s)
        assert np.array_equal(planes[i], p)
        assert np.array_equal(scalars[i], sc)
        assert np.array_equal(hands[i], h)


def test_an_empty_batch_is_well_typed():
    planes, scalars, hands = F.featurize_batch([])
    assert planes.shape == (0, 14, 20, 20) and planes.dtype == np.float32
    assert scalars.shape == (0, 13) and scalars.dtype == np.float32
    assert hands.shape == (0, 4, 21) and hands.dtype == np.float32


def test_featurize_does_not_mutate_the_state():
    """`State` is frozen, but the temptation to write one back into a dict is
    strong enough to be worth an assertion."""
    s = positions(range(5), stride=20)[0]
    before = engine.serialize(s)
    F.featurize(s)
    F.featurize_batch([s, s])
    assert engine.serialize(s) == before


def test_the_planes_only_contain_zero_and_one():
    for s in positions(range(20), stride=9):
        planes, _sc, hands = F.featurize(s)
        for c in range(14):
            u = np.unique(planes[c])
            assert set(u.tolist()).issubset({0.0, 1.0}), \
                (F.PLANE_CHANNELS_V2[c], u)
        assert set(np.unique(hands).tolist()).issubset({0.0, 1.0})


def test_the_empty_board_has_an_empty_hand_and_a_full_board():
    planes, scalars, hands = F.featurize(engine.initial_state())
    assert planes[0:4].sum() == 0.0
    assert planes[6].sum() == 400.0
    assert planes[4].sum() == 0.0, "no contact rule before opening"
    assert planes[5].sum() == 0.0
    assert hands.sum() == 4 * 21
    assert list(scalars[4:8]) == [1.0] * 4


def test_a_single_player_who_has_opened_has_a_non_empty_contact_rule():
    s = _handmade(1, {1: 1 << (10 + 10 * B)},
                  {1: engine.FULL_HAND & ~(1 << engine.PIECE_IDX["F5"])})
    planes, _sc, _h = F.featurize(s)
    assert planes[4].sum() > 0, "my_need must be non-empty once I have stones"
    assert planes[5].sum() > 0, "my_avoid must be non-empty once I have stones"
    assert planes[4][planes[5].astype(bool)].sum() == 0.0, \
        "my_need and my_avoid must be disjoint"


def test_rng_is_not_used_so_the_features_are_a_pure_function():
    """No randomness anywhere, so the same state always gives the same array
    whatever the global RNG is doing."""
    s = positions(range(3), stride=7)[0]
    first = F.featurize(s)[0].copy()
    random.seed(1)
    for _ in range(20):
        random.random()
    assert np.array_equal(F.featurize(s)[0], first)

def _own_move_count(state, owner):
    """The scalar's value, computed the way the features compute it."""
    return ((engine.N_PIECES - bin(state.hand_bits[owner]).count("1"))
            / float(engine.N_PIECES))


def test_the_movers_own_move_count_is_the_thirteenth_scalar():
    """`own_move_count` says how many real moves the mover has made.

    Pinned on value, not just on shape: a thirteenth scalar that was always zero
    would pass every count assertion and quietly leave the book step unlearnable,
    which is the only reason this scalar exists (`plan6.md` H-B0 item 3).
    """
    checked = 0
    for s in positions(range(12), stride=1):
        _planes, scalars, _hands = F.featurize(s)
        want = _own_move_count(s, s.to_move)
        assert scalars[12] == pytest.approx(want), \
            ("mover %d: scalar says %r, hand says %r"
             % (s.to_move, scalars[12], want))
        assert 0.0 <= scalars[12] <= 1.0
        checked += 1
    assert checked > 100, "only %d positions checked" % checked


def test_a_pass_leaves_every_players_move_count_untouched():
    """It has to count real moves, which is what the book step is defined against.

    A pass is not a move. If the count moved across one, a blocked seat's book
    step would silently become a different book - the failure `ai/hunter.py`
    already documents for its own counter.
    """
    passes_seen = 0
    for seed in range(8):
        tape = []
        random_playout(seed, tape=tape, max_passes=MAX_PASSES)
        for st, _mask, _n_real, mv in tape:
            if mv is not None:
                continue
            passes_seen += 1
            after = engine.pass_turn(st)
            for owner in range(4):
                assert after.hand_bits[owner] == st.hand_bits[owner], \
                    "a pass changed a hand, so the move count moved with it"
                assert _own_move_count(after, owner) == \
                    _own_move_count(st, owner)
    assert passes_seen > 0, "no pass in 8 games, so the branch is untested"


def test_the_stone_count_alone_cannot_stand_in_for_the_move_count():
    """Why this scalar exists, stated as a test.

    Inside Hunter's book window the mover's own stone count does reveal the step -
    Z5, V5 and W5 are all pentominoes, so 0, 5 and 10 stones are steps 0, 1 and 2
    and nothing else. That coincidence is not a substitute for the count, because
    the book pieces are not representative of the pieces: one I4 covers four
    squares, I3 plus I1 covers the same four squares in two moves, and the two
    are indistinguishable in stone count.

    These two positions have identical board planes and identical first twelve
    scalars - same stones, same cells left in hand - and a different move count.
    The thirteenth scalar is what separates them. (`hands` does differ too, which
    is the other route to the number; the scalar states it outright instead of
    leaving it to be read off a 4x21 mask.)
    """
    mover = 0
    squares = [1 << (x + 5 * B) for x in (2, 3, 4, 5)]
    stones = 0
    for bit in squares:
        stones |= bit
    i1 = engine.PIECE_IDX["I1"]
    i3 = engine.PIECE_IDX["I3"]
    i4 = engine.PIECE_IDX["I4"]

    one_move = _handmade(mover, {mover: stones},
                         hands={mover: engine.FULL_HAND & ~(1 << i4)})
    two_moves = _handmade(mover, {mover: stones},
                          hands={mover: engine.FULL_HAND
                                 & ~((1 << i3) | (1 << i1))})

    p1, sc1, h1 = F.featurize(one_move)
    p2, sc2, h2 = F.featurize(two_moves)

    assert sc1[12] == pytest.approx(1.0 / engine.N_PIECES)
    assert sc2[12] == pytest.approx(2.0 / engine.N_PIECES)
    # the board and the twelve per-seat scalars cannot tell them apart
    assert np.array_equal(p1, p2)
    assert np.array_equal(sc1[:12], sc2[:12])
    # the hands can, in principle: different pieces are absent
    assert not np.array_equal(h1, h2)
