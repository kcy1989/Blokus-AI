"""E1: the second engine must agree with `Game`, move for move.

`engine.py` is a deliberately separate implementation of the Blokus rules, so
the only thing that can make it trustworthy is a real disagreement being able to
happen and being caught. Three layers do that here:

  * the fast bitmask legality test is pinned to `Board.can_place`, which loops
    over every base and every cell and is obviously correct;
  * position state, stuck latches and final scores are compared against `Game`
    itself while a real game is being played;
  * the structural promises (no shared mutable state, mandatory rule checks,
    exact serialisation round trip) are checked on their own, because a
    position that is subtly aliased would pass every correctness test and still
    be unusable for search.
"""
import random

import pytest

import ai
import board as BD
import engine
from config import B, N
from game import Game
from pieces import MASTER

SEEDS = 40


# ------------------------------------------------------------------ helpers

def board_of(state):
    """A `Board` mirroring an engine position, for use as the brute-force oracle."""
    b = BD.Board()
    b.grid = [-1] * N
    for o in range(4):
        m = state.own_bits[o]
        while m:
            low = m & -m
            b.grid[low.bit_length() - 1] = o
            m ^= low
    b.empty_bits = engine.empty_of(state.own_bits)
    b.owner_bits = list(state.own_bits)
    return b


def brute_legal_moves(state, owner):
    """Every legal `(piece_idx, oi, base)`, found by scanning bases and cells."""
    b = board_of(state)
    from config import OWNER_CORNER
    mc = OWNER_CORNER[owner] if not state.own_bits[owner] else None
    reach = None if not state.own_bits[owner] else b.reach(owner)
    out = set()
    hand = state.hand_bits[owner]
    while hand:
        low = hand & -hand
        name = engine.PIECE_ORDER[low.bit_length() - 1]
        for oi, cells in enumerate(MASTER[name]["orientations"]):
            for y in range(B):
                for x in range(B):
                    if b.can_place(x, y, cells, owner, mc, reach):
                        out.add((low.bit_length() - 1, oi, x + y * B))
        hand ^= low
    return out


def random_positions(count, seed=5):
    """Positions reached by random legal play, so they are realistic."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        s = engine.initial_state()
        guard = 0
        while guard < 40 and not engine.is_over(s):
            moves = sorted(engine.legal_moves(s))
            if not moves:
                break
            # Prefer deeper positions so the corner-contact rule gets exercised
            # rather than only the opening special case.
            s = engine.apply_move(s, rng.choice(moves))
            guard += 1
            out.append(s)
    return out


# ------------------------------------------------------------- board algebra

def test_dilate_matches_the_existing_implementation_for_every_square():
    for i in range(N):
        m = 1 << i
        assert engine.dilate(m) == BD.dilate(m)
        assert engine.dilate_diag(m) == BD.dilate_diag(m)


def test_dilate_matches_on_random_masks():
    rng = random.Random(3)
    for _ in range(3000):
        m = 0
        for _ in range(rng.randrange(1, 40)):
            m |= 1 << rng.randrange(N)
        assert engine.dilate(m) == BD.dilate(m)
        assert engine.dilate_diag(m) == BD.dilate_diag(m)


def test_dilate_never_wraps_around_the_board_edge():
    """A square on the edge has three neighbours, not four.

    This is the exact shape of the `COL0` / `COL_LAST` guard bug: shifting an
    x == 0 square "left" lands it on the previous row's last column.
    """
    for i in range(N):
        assert engine.dilate(1 << i).bit_count() == \
            len(BD.neighbors_of()[i])


def test_dilate_diag_excludes_edge_neighbours():
    """Corner contact and edge contact are different sets.

    If `dilate_diag` accidentally included the four edge neighbours, every move
    after the opening would be illegal, because the piece would always be
    adjacent to its own stones.
    """
    for i in range(N):
        assert not (engine.dilate_diag(1 << i) & engine.dilate(1 << i))


# ------------------------------------------------------- legality vs brute force

def test_opening_legality_matches_brute_force_for_all_four_corners():
    s = engine.initial_state()
    for owner in range(4):
        assert engine.legal_moves(s, owner) == brute_legal_moves(s, owner)


def test_the_plus_pentomino_can_never_open():
    """X5's bounding box is 3x3 and centred, so it can never cover a corner.

    Worth pinning because it looks like a bug: it is the rule working. A shape
    whose cells all sit at least one square in from the origin can never put a
    cell on `(0, 0)`, so it is unplayable as an opening piece for every seat.
    """
    s = engine.initial_state()
    xi = engine.PIECE_IDX["X5"]
    for owner in range(4):
        assert not engine._legal_bases_for_orient(s, owner, xi, 0)
    # The cells really are inset from the bounding-box corner.
    cells = MASTER["X5"]["orientations"][0]
    assert (0, 0) not in cells


def test_every_piece_except_the_plus_can_open():
    s = engine.initial_state()
    stuck = [p for p in range(21)
             if not any(engine._legal_bases_for_orient(s, o, p, oi)
                        for oi in range(len(engine.ORIENTS[p]))
                        for o in range(4))]
    assert stuck == [engine.PIECE_IDX["X5"]]


@pytest.mark.parametrize("state_i", range(6))
def test_legality_matches_brute_force_on_random_positions(state_i):
    """The bitmask path against an explicit base-and-cell scan.

    This is the test that makes the fast path trustworthy: `_legal_bases_for_orient`
    works by shifting a shape mask, `can_place` works by asking about each cell
    in turn, and they are different code.
    """
    s = random_positions(1, seed=100 + state_i)[0]
    for owner in range(4):
        assert engine.legal_moves(s, owner) == brute_legal_moves(s, owner), owner


# -------------------------------------------------------- structural promises

def test_legal_moves_agrees_with_the_packed_mask():
    for s in random_positions(3, seed=200):
        for owner in range(4):
            assert engine.legal_moves(s, owner) == \
                engine.unpack_move_mask(engine.legal_move_mask(s, owner))


def test_serialize_round_trips_exactly():
    for s in random_positions(4, seed=300):
        assert engine.deserialize(engine.serialize(s)) == s


def test_serialize_survives_bytes_and_rejects_corruption():
    s = random_positions(1, seed=301)[0]
    blob = engine.serialize(s)
    assert engine.deserialize(blob.encode("ascii")) == s
    with pytest.raises(Exception):
        engine.deserialize(blob[:-4] + 'zzz"')


def test_deserialize_rejects_a_board_with_a_shared_square():
    hands = ",".join([format(engine.FULL_HAND, "x")] * 4)
    bad = ('{"own_bits":["1","1","0","0"],"hand_bits":[%s],'
           '"stuck":[false,false,false,false],"to_move":0,'
           '"turn_order":[0,1,2,3]}' % hands)
    with pytest.raises(ValueError):
        engine.deserialize(bad)


def test_apply_move_does_not_touch_the_state_it_was_given():
    s = engine.initial_state()
    before = (s.own_bits, s.hand_bits, s.stuck, s.to_move)
    moves = sorted(engine.legal_moves(s))
    nxt = engine.apply_move(s, moves[0])
    assert (s.own_bits, s.hand_bits, s.stuck, s.to_move) == before
    assert nxt is not s
    # Two different moves from the same parent are independent of each other.
    again = engine.apply_move(s, moves[10])
    assert again != nxt
    assert again.own_bits != nxt.own_bits
    # ...and neither of them disturbed the parent.
    assert (s.own_bits, s.hand_bits, s.stuck, s.to_move) == before


def test_states_share_nothing_mutable():
    s = engine.initial_state()
    a = engine.apply_move(s, sorted(engine.legal_moves(s))[0])
    b = engine.apply_move(s, sorted(engine.legal_moves(s))[0])
    assert a == b
    assert a.own_bits is not b.own_bits
    assert a.stuck is not b.stuck


def test_initial_state_rejects_a_bad_turn_order():
    for bad in ([0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 4]):
        with pytest.raises(ValueError):
            engine.initial_state(bad)


# ------------------------------------------------------------- rule enforcement

def test_apply_move_rejects_every_kind_of_illegal_move():
    s = engine.initial_state()
    piece_idx, oi, base = sorted(engine.legal_moves(s))[0]

    with pytest.raises(ValueError):
        engine.apply_move(s, (99, oi, base))          # no such piece
    with pytest.raises(ValueError):
        engine.apply_move(s, (-1, oi, base))          # negative piece
    with pytest.raises(ValueError):
        engine.apply_move(s, (piece_idx, 99, base))   # no such orientation
    with pytest.raises(ValueError):
        engine.apply_move(s, (piece_idx, -1, base))

    # A base that is not in this piece's legal set.
    legal = engine._legal_bases_for_orient(s, s.to_move, piece_idx, oi)
    off = next(b for b in range(400) if not legal >> b & 1)
    with pytest.raises(ValueError):
        engine.apply_move(s, (piece_idx, oi, off))

    # A piece this player has already played.
    nxt = engine.apply_move(s, (piece_idx, oi, base))
    with pytest.raises(ValueError):
        engine.apply_move(nxt, (piece_idx, oi, base))


def test_a_piece_the_mover_does_not_hold_is_rejected():
    xi = engine.PIECE_IDX["X5"]
    s = engine.State(own_bits=(0, 0, 0, 0), hand_bits=(engine.FULL_HAND ^ (1 << xi),)
                     + (engine.FULL_HAND,) * 3,
                     turn_order=(0, 1, 2, 3), to_move=0, stuck=(False,) * 4)
    with pytest.raises(ValueError):
        engine.apply_move(s, (xi, 0, 399))


def test_a_base_that_runs_off_the_board_is_never_legal():
    """`valid` is what stands between a 400-bit shift and a row wrap.

    Checked by coordinates rather than by index arithmetic, because a base index
    is `x + y * B` and index 20 is `(0, 1)` rather than "column 20" - the two
    readings only agree for the rows you can actually name.
    """
    for p in range(21):
        for od in engine.ORIENTS[p]:
            w, h = od["w"], od["h"]
            want = 0
            for y in range(B - h + 1):
                for x in range(B - w + 1):
                    want |= 1 << (x + y * B)
            assert od["valid"] == want, engine.PIECE_ORDER[p]
            assert od["valid"].bit_count() == (B - w + 1) * (B - h + 1)


def test_a_stuck_player_cannot_be_made_to_move():
    """`apply_move` with no legal move must raise rather than quietly pass."""
    xi = engine.PIECE_IDX["X5"]
    s = engine.State(own_bits=(0, 0, 0, 0),
                     hand_bits=(1 << xi,) + (engine.FULL_HAND,) * 3,
                     turn_order=(0, 1, 2, 3), to_move=0, stuck=(False,) * 4)
    assert not engine.has_any_legal(s, 0)
    with pytest.raises(ValueError):
        engine.apply_move(s, (xi, 0, 399))


# ------------------------------------------------------------------ scoring

def test_remaining_cells_agrees_with_the_game():
    keys = list(ai.personality_keys())
    g = Game(random.Random(4))
    g.setup_match(random.Random(4).sample(keys, 4))
    g.start()
    s = engine.initial_state(g.turn_order)
    for _ in range(12):
        if g.state != "PLAYING":
            break
        o = g.current_owner()
        name, oi, x, y = _one_move(g, o)
        s = engine.apply_move(s, (engine.PIECE_IDX[name], oi, x + y * B))
        g.act(name, oi, x, y)
    assert engine.result(s) == tuple(g.remaining_cells(o) for o in range(4))


def test_hand_bits_uses_the_action_table_piece_order():
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(here), "action_table.json"),
              encoding="utf-8") as fh:
        table = json.load(fh)
    assert list(engine.PIECE_ORDER) == table["pieces"]
    s = engine.initial_state()
    for i, name in enumerate(engine.PIECE_ORDER):
        assert s.hand_bits[0] >> i & 1, name
        assert engine.PIECE_SIZE[i] == MASTER[name]["size"]


# ---------------------------------------------------- live agreement with Game

def _one_move(g, o):
    mv = ai.choose_move(g.board, g.hands[o].names, o, g.brains[o], g.rng,
                        other_brains=g.brains,
                        must_cover=g.must_cover(o), reach=g.reach(o),
                        other_must_cover={x: g.must_cover(x) for x in range(4)},
                        other_reach={x: g.reach(x) for x in range(4)})
    assert mv is not None
    return mv


def test_engine_tracks_a_live_game_exactly():
    """The whole acceptance check, at a size that runs in CI."""
    keys = list(ai.personality_keys())
    for seed in range(SEEDS):
        g = Game(random.Random(seed))
        g.setup_match(random.Random(seed).sample(keys, 4))
        g.start()
        s = engine.initial_state(g.turn_order)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            o = g.current_owner()
            assert s.to_move == o
            assert s.stuck == tuple(g.stuck), (seed, g.turn_count)
            if not g.has_legal(o):
                assert not engine.has_any_legal(s, o)
                g.act_pass()
                s = engine.pass_turn(s)
                continue
            name, oi, x, y = _one_move(g, o)
            s = engine.apply_move(s, (engine.PIECE_IDX[name], oi, x + y * B))
            g.act(name, oi, x, y)
            assert list(s.own_bits) == g.board.owner_bits, (seed, g.turn_count)
        assert engine.is_over(s) == (g.state == "GAME_OVER")
        assert engine.result(s) == tuple(g.remaining_cells(o) for o in range(4))
