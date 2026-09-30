"""E3: the position as the player to move sees it, as numbers.

The point of normalising is that two games which differ only by a rotation of
the board must produce identical input, otherwise a model has to spend its
capacity learning that the same position can be written down in four different
ways. So the test that matters most is the invariance one, not the shape check.

The plane is the only place in the project that uses numpy, and it is imported
inside the functions that need it. That is deliberate: the rules above must stay
readable and checkable without it, and `test_importing_the_engine_does_not_pull
_in_numpy` pins that.
"""
import random
import sys

import pytest

import engine
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER

np = pytest.importorskip("numpy")


def plan_bits_to_plane(mask):
    """The conversion exactly as the plan proposed it, for comparison."""
    raw = mask.to_bytes(50, "little")
    return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little") \
             .reshape(20, 20)


def random_states(count, seed=41, max_moves=16):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        s = engine.initial_state()
        for _ in range(rng.randrange(1, max_moves)):
            moves = sorted(engine.legal_moves(s))
            if not moves:
                break
            s = engine.apply_move(s, rng.choice(moves))
        out.append(s)
    return out


# ------------------------------------------------------------ bits_to_plane

def test_bits_to_plane_matches_the_conversion_in_the_plan():
    """The plan's snippet was unverified, so it gets checked rather than
    trusted. It is right: bit `y * B + x` is row `y`, column `x`."""
    rng = random.Random(2)
    for _ in range(300):
        m = 0
        for _ in range(rng.randrange(1, 60)):
            m |= 1 << rng.randrange(400)
        assert np.array_equal(engine.bits_to_plane(m), plan_bits_to_plane(m))


def test_bits_to_plane_is_row_major_with_no_transpose():
    """Row is y and column is x, so the array is a picture of the board."""
    for i in (0, 1, 19, 20, 21, 199, 380, 399):
        p = engine.bits_to_plane(1 << i)
        assert p[i // B, i % B] == 1.0
        assert p.sum() == 1.0
    # A full row, to catch an axis swap that a single cell would not show.
    row = sum(1 << (3 * B + x) for x in range(B))
    p = engine.bits_to_plane(row)
    assert p[3].sum() == B
    assert p.sum(axis=1)[3] == B
    assert all(p[y].sum() == 0 for y in range(B) if y != 3)


def test_bits_to_plane_preserves_the_popcount():
    rng = random.Random(4)
    for _ in range(200):
        m = 0
        for _ in range(rng.randrange(1, 80)):
            m |= 1 << rng.randrange(400)
        assert engine.bits_to_plane(m).sum() == float(m.bit_count())


def test_bits_to_plane_agrees_with_the_bits_on_random_cells():
    """The acceptance check the plan asks for, item by item."""
    rng = random.Random(6)
    for _ in range(20):
        m = 0
        for _ in range(rng.randrange(1, 50)):
            m |= 1 << rng.randrange(400)
        p = engine.bits_to_plane(m)
        for i in [rng.randrange(400) for _ in range(100)]:
            assert p[i // B, i % B] == (1.0 if m >> i & 1 else 0.0), i


# ------------------------------------------------------------------ normalize

def test_normalize_puts_the_mover_in_the_top_left_corner():
    for s in random_states(30, seed=7):
        view, seat_of = engine.normalize(s)
        assert view.to_move == 0
        assert seat_of[0] == s.to_move
        # After normalisation the mover's stones sit against the TL corner.
        own = view.own_bits[0]
        if own:
            corners = [i for i in range(400) if own >> i & 1]
            assert min(c[0] for c in [(i % B, i // B) for i in corners]) < 2
            assert min(c[1] for c in [(i % B, i // B) for i in corners]) < 2


def test_normalize_numbers_players_me_next_opposite_previous():
    for s in random_states(20, seed=8):
        _view, seat_of = engine.normalize(s)
        # The rotation has already brought the mover to the top-left corner, so
        # the clockwise cycle from there is always the same four owners.
        assert sorted(seat_of) == [0, 1, 2, 3]
        assert seat_of[0] == s.to_move
        # The player two seats along is the opposite corner, not a neighbour.
        assert seat_of[2] not in (seat_of[1], seat_of[3])
        # ...and "next" really is clockwise from "me" in the original seating.
        assert seat_of[1] == engine.ROT_OWNER[seat_of[0]]


def test_normalize_keeps_the_mover_to_move():
    for s in random_states(20, seed=9):
        view, _ = engine.normalize(s)
        assert view.to_move == 0
        assert view.turn_order[view.turn_pos] == 0


def test_normalize_preserves_the_position_it_started_from():
    for s in random_states(20, seed=10):
        view, seat_of = engine.normalize(s)
        assert sum(engine.result(view)) == sum(engine.result(s))
        assert engine.is_over(view) == engine.is_over(s)
        # Compare against the same number of steps of `rotate_state`, which
        # test_rotation.py already pins against a literal coordinate rotation.
        k = (4 - CLOCKWISE_OWNERS.index(s.to_move)) % 4
        rot = s
        for _ in range(k):
            rot = engine.rotate_state(rot)
        for i, o in enumerate(seat_of):
            assert view.own_bits[i] == rot.own_bits[CLOCKWISE_OWNERS[i]], (i, o)
            assert view.hand_bits[i] == s.hand_bits[o], (i, o)
            assert view.stuck[i] == s.stuck[o], (i, o)
        # Every square is still owned by exactly one player.
        union = (view.own_bits[0] | view.own_bits[1]
                 | view.own_bits[2] | view.own_bits[3])
        assert not (view.own_bits[0] & view.own_bits[1])
        assert not (view.own_bits[1] & view.own_bits[2])
        assert not (view.own_bits[2] & view.own_bits[3])
        assert not (view.own_bits[0] & view.own_bits[2])
        assert union.bit_count() == sum(b.bit_count() for b in view.own_bits)


# ---------------------------------------------------------------- the plane

def test_plane_has_the_declared_shape_and_type():
    for s in random_states(8, seed=11):
        p = engine.to_plane(s)
        assert p.shape == (engine.PLANE_C, B, B)
        assert p.dtype == np.float32
        assert set(np.unique(p)) <= {0.0, 1.0}


def test_plane_channels_reconstruct_the_board():
    for s in random_states(10, seed=12):
        p = engine.to_plane(s)
        stones = p[:4].sum(axis=0)
        empt = p[engine.PLANE_CHANNELS.index("empty")]
        assert np.all(stones + empt == 1.0)          # every square accounted for
        assert np.all(stones <= 1.0)                 # never two owners on one
        view, _ = engine.normalize(s)
        for i in range(4):
            assert p[i].sum() == float(view.own_bits[i].bit_count())


def test_plane_my_need_and_my_avoid_are_the_contact_rule():
    for s in random_states(10, seed=13):
        view, _ = engine.normalize(s)
        p = engine.to_plane(s)
        need, avoid = engine.need_avoid(view, 0)
        empt = engine.empty_of(view.own_bits)
        ch_need = engine.PLANE_CHANNELS.index("my_need")
        ch_avoid = engine.PLANE_CHANNELS.index("my_avoid")
        assert np.array_equal(p[ch_need],
                              engine.bits_to_plane(need & ~avoid & empt))
        assert np.array_equal(p[ch_avoid], engine.bits_to_plane(avoid))


def test_need_and_avoid_can_overlap_and_avoid_wins():
    """A square can touch one of my stones at a corner and another along an
    edge, so `need` and `avoid` are not disjoint - and the rule is that the
    edge wins. This is why the plan's contact set is `need & ~avoid & empty`
    rather than `need`."""
    s = engine.initial_state()
    # My own two stones at (0, 0) and (1, 2): the square (1, 1) touches the
    # first at a corner and the second along an edge.
    own = (1 << 0) | (1 << (1 + 2 * B))
    st = engine.State(own_bits=(own, 0, 0, 0),
                      hand_bits=(engine.FULL_HAND,) * 4,
                      turn_order=(0, 1, 2, 3), to_move=0, stuck=(False,) * 4)
    need, avoid = engine.need_avoid(st, 0)
    assert need >> (1 + 1 * B) & 1        # diagonal contact with (0, 0)
    assert avoid >> (1 + 1 * B) & 1       # ...and edge contact with (1, 2)
    assert need & avoid                   # genuinely in both
    empt = engine.empty_of(st.own_bits)
    assert not (need & ~avoid & empt) & avoid    # the usable set excludes it


def test_opening_positions_have_no_contact_rule_yet():
    """Before the first stone the rule is "cover my corner", not contact."""
    s = engine.initial_state()
    need, avoid = engine.need_avoid(s, 0)
    assert need == 0 and avoid == 0
    p = engine.to_plane(s)
    ch_need = engine.PLANE_CHANNELS.index("my_need")
    ch_avoid = engine.PLANE_CHANNELS.index("my_avoid")
    assert p[ch_need].sum() == 0.0
    assert p[ch_avoid].sum() == 0.0


def test_my_need_is_where_a_move_can_actually_land():
    """The usable contact set is empty squares I may touch but not abut."""
    for s in random_states(10, seed=14):
        view, _ = engine.normalize(s)
        need, avoid = engine.need_avoid(view, 0)
        empt = engine.empty_of(view.own_bits)
        usable = need & ~avoid & empt
        assert not (usable & view.own_bits[0])     # not my own stones
        assert not (usable & avoid)                # and not next to them
        assert not (usable & ~engine.ALL)          # and on the board
        assert usable.bit_count() <= need.bit_count()


# ------------------------------------------------------------- the invariance

def test_a_rotated_position_gives_an_identical_plane():
    """The reason normalisation exists.

    Two positions that are rotations of one another have to become the same
    array, otherwise the model has to learn four equivalent spellings of every
    position it will ever see.
    """
    for s in random_states(40, seed=15):
        plane = engine.to_plane(s)
        for turns in (1, 2, 3):
            rotated = s
            for _ in range(turns):
                rotated = engine.rotate_state(rotated)
            assert np.array_equal(engine.to_plane(rotated), plane), turns


def test_rotation_leaves_the_plane_sum_unchanged():
    """The acceptance check: totals survive the rotation."""
    for s in random_states(30, seed=16):
        for turns in (0, 1, 2, 3):
            r = s
            for _ in range(turns):
                r = engine.rotate_state(r)
            before = engine.to_plane(s).sum(axis=(1, 2))
            after = engine.to_plane(r).sum(axis=(1, 2))
            assert np.array_equal(before, after), turns


# ------------------------------------------------------------ hand vectors

def test_hand_vectors_are_four_players_of_twenty_one():
    for s in random_states(10, seed=17):
        h = engine.hand_vectors(s)
        view, seat_of = engine.normalize(s)
        assert h.shape == (4, engine.N_PIECES)
        assert h.dtype == np.float32
        assert h.sum() == sum(bin(view.hand_bits[p]).count("1") for p in range(4))
        for p, o in enumerate(seat_of):
            for i in range(engine.N_PIECES):
                assert h[p, i] == (1.0 if s.hand_bits[o] >> i & 1 else 0.0)


def test_hand_vectors_only_ever_shrink():
    """A finished game does not mean empty hands.

    A player who cannot move keeps what is left, so the total number of pieces
    in hand falls over a game but need not reach zero. What must hold is that it
    never rises and that it only ever drops by whole pieces.
    """
    s = engine.initial_state()
    prev = float(engine.hand_vectors(s).sum())
    assert prev == 4 * engine.N_PIECES
    guard = 0
    while not engine.is_over(s) and guard < 600:
        moves = sorted(engine.legal_moves(s))
        if not moves:
            s = engine.pass_turn(s)
        else:
            s = engine.apply_move(s, random.Random(guard).choice(moves))
        guard += 1
        now = float(engine.hand_vectors(s).sum())
        assert now <= prev
        prev = now
    assert engine.is_over(s)
    assert engine.hand_vectors(s).shape == (4, engine.N_PIECES)


# ------------------------------------------------------------- dependencies

def test_importing_the_engine_does_not_pull_in_numpy():
    """The rules must stay usable without the training dependency.

    numpy is imported inside the two functions that need it, so a plain
    `import engine` for the rules costs nothing extra.
    """
    for name in ("numpy",):
        assert name not in sys.modules or name == "numpy"
    # Re-importing in a clean subprocess is the honest version of the claim.
    import subprocess
    code = ("import sys, engine; "
            "print('numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout
