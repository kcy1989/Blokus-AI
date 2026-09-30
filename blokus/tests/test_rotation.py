"""E2: the board's rotation symmetry, and the fact that the game has it.

Two claims have to hold, and they are different kinds of claim:

  * **Algebraic.** `rot_orient` and `rot_base` describe one clockwise step
    consistently with a single global coordinate rotation of the whole board,
    and four steps are the identity. This is a property of the tables alone.

  * **Behavioural.** Rotating a position and rotating the legal moves it offers
    give the same answer. If that fails, a search that canonicalises positions
    by rotating them is exploring a different game, and every symmetry-based
    speedup built on it is worthless.

Mirroring is deliberately absent. A mirror reverses the direction of play, and
the clockwise seating is a rule of Blokus rather than a presentation choice, so
mirroring describes a different game.
"""
import random

import pytest

import engine
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER
from pieces import MASTER

GLOBAL_ROT = lambda x, y: (B - 1 - y, x)      # 90 degrees clockwise


# -------------------------------------------------------------- the tables

def test_one_step_is_a_clockwise_rotation_of_the_whole_board():
    """TL must go to TR, not to BL. The other direction is a different game."""
    assert GLOBAL_ROT(0, 0) == (B - 1, 0)
    assert GLOBAL_ROT(B - 1, 0) == (B - 1, B - 1)
    assert GLOBAL_ROT(B - 1, B - 1) == (0, B - 1)
    assert GLOBAL_ROT(0, B - 1) == (0, 0)


@pytest.mark.parametrize("piece_idx", range(21))
def test_four_rotations_are_the_identity(piece_idx):
    for oi in range(len(engine.ORIENTS[piece_idx])):
        back = oi
        for _ in range(4):
            back = engine.rot_orient(piece_idx, back)
        assert back == oi


@pytest.mark.parametrize("piece_idx", range(21))
def test_rot_orient_matches_a_rotation_of_the_piece_itself(piece_idx):
    for oi, od in enumerate(engine.ORIENTS[piece_idx]):
        want = frozenset((od["h"] - 1 - y, x) for x, y in od["cells"])
        got = frozenset(engine.ORIENTS[piece_idx][engine.rot_orient(piece_idx, oi)]
                        ["cells"])
        assert got == want, engine.PIECE_ORDER[piece_idx]


def test_rot_orient_always_stays_inside_the_orientation_set():
    for piece_idx in range(21):
        n = len(engine.ORIENTS[piece_idx])
        for oi in range(n):
            r = engine.rot_orient(piece_idx, oi)
            assert 0 <= r < n
            # Rotating a symmetric piece may legitimately return the same index.
            if r != oi:
                assert MASTER[engine.PIECE_ORDER[piece_idx]]["orientations"][r] \
                    != MASTER[engine.PIECE_ORDER[piece_idx]]["orientations"][oi]


@pytest.mark.parametrize("piece_idx", range(21))
def test_rot_base_matches_a_rotation_of_the_whole_placement(piece_idx):
    """The two tables have to agree with one global rotation of the board.

    Checking `rot_base` and `rot_orient` separately is not enough: each could
    be self-consistent while the pair describes two different rotations.
    """
    rng = random.Random(17)
    checked = 0
    for oi, od in enumerate(engine.ORIENTS[piece_idx]):
        w, h = od["w"], od["h"]
        for _ in range(60):
            x0 = rng.randrange(B - w + 1)
            y0 = rng.randrange(B - h + 1)
            base = x0 + y0 * B
            want = sorted(GLOBAL_ROT(x0 + dx, y0 + dy) for dx, dy in od["cells"])
            nb = engine.rot_base(piece_idx, oi, base)
            no = engine.rot_orient(piece_idx, oi)
            got = sorted((nb % B + dx, nb // B + dy)
                         for dx, dy in engine.ORIENTS[piece_idx][no]["cells"])
            assert want == got, (engine.PIECE_ORDER[piece_idx], oi, base)
            checked += 1
    assert checked > 0


def test_rot_base_always_lands_on_a_valid_base():
    """The rotated base has to be in range exactly when the original was."""
    for piece_idx in range(21):
        for oi, od in enumerate(engine.ORIENTS[piece_idx]):
            valid = od["valid"]
            m = valid
            while m:
                low = m & -m
                base = low.bit_length() - 1
                rb = engine.rot_base(piece_idx, oi, base)
                no = engine.rot_orient(piece_idx, oi)
                assert engine.ORIENTS[piece_idx][no]["valid"] >> rb & 1, \
                    (engine.PIECE_ORDER[piece_idx], oi, base, rb)
                m ^= low


# ------------------------------------------------------------------ seating

def test_rotation_moves_every_seat_one_step_clockwise():
    """`CLOCKWISE_OWNERS` is TL, TR, BR, BL, so a clockwise step must send each
    owner to the next owner in that cycle."""
    for o in CLOCKWISE_OWNERS:
        nxt = CLOCKWISE_OWNERS[(CLOCKWISE_OWNERS.index(o) + 1) % 4]
        assert engine.ROT_OWNER[o] == nxt, o
    # ...which is a 4-cycle, not a swap or a double swap.
    seen, o = [], 0
    for _ in range(4):
        seen.append(o)
        o = engine.ROT_OWNER[o]
    assert sorted(seen) == [0, 1, 2, 3] and o == 0


def test_rotation_sends_each_corner_to_the_next_corner():
    for o, (x, y) in OWNER_CORNER.items():
        nx, ny = GLOBAL_ROT(x, y)
        assert OWNER_CORNER[engine.ROT_OWNER[o]] == (nx, ny)


def test_four_rotations_restore_every_seat():
    for o in range(4):
        back = o
        for _ in range(4):
            back = engine.ROT_OWNER[back]
        assert back == o


# ------------------------------------------------------------ rotating a position

def test_rotate_state_moves_the_squares_and_the_owners():
    s = engine.initial_state()
    mv = sorted(engine.legal_moves(s))[0]
    piece_idx, oi, base = mv
    s = engine.apply_move(s, mv)
    owner = s.turn_order[0]          # whoever opened
    placed = engine.ORIENTS[piece_idx][oi]["m"] << base

    r = engine.rotate_state(s)
    rotated = 0
    m = placed
    while m:
        low = m & -m
        i = low.bit_length() - 1
        dx, dy = GLOBAL_ROT(i % B, i // B)
        rotated |= 1 << (dy * B + dx)
        m ^= low
    new_owner = engine.ROT_OWNER[owner]
    assert r.own_bits[new_owner] == rotated
    for o in range(4):
        if o != new_owner:
            assert r.own_bits[o] == 0
    # The hand has to travel with the stones it belongs to.
    inv = {v: k for k, v in engine.ROT_OWNER.items()}
    for o in range(4):
        assert r.hand_bits[o] == s.hand_bits[inv[o]], o


def test_rotating_a_state_four_times_is_the_identity():
    rng = random.Random(23)
    s = engine.initial_state()
    for _ in range(12):
        moves = sorted(engine.legal_moves(s))
        if not moves:
            break
        s = engine.apply_move(s, rng.choice(moves))
    r = s
    for _ in range(4):
        r = engine.rotate_state(r)
    assert r.own_bits == s.own_bits
    assert r.to_move == s.to_move


def test_rotating_preserves_who_can_move_and_the_score():
    rng = random.Random(29)
    s = engine.initial_state()
    for _ in range(10):
        moves = sorted(engine.legal_moves(s))
        if not moves:
            break
        s = engine.apply_move(s, rng.choice(moves))
    r = engine.rotate_state(s)
    # Scores and legality are indexed by owner, and rotation renumbers the
    # owners, so the comparison has to be made through the permutation rather
    # than position by position.
    inv = {v: k for k, v in engine.ROT_OWNER.items()}
    for o in range(4):
        assert engine.result(r)[o] == engine.result(s)[inv[o]], o
        assert engine.has_any_legal(r, o) == engine.has_any_legal(s, inv[o]), o
        assert r.stuck[o] == s.stuck[inv[o]], o
    assert sum(engine.result(r)) == sum(engine.result(s))
    assert engine.is_over(r) == engine.is_over(s)


# ------------------------------------------------- the symmetry that matters

def test_legal_bases_commute_with_rotation():
    """`rotate(legal(state)) == legal(rotate(state))`, the whole claim of E2.

    A full position is a stronger test than a single piece: it forces the
    corner rule, the corner-contact rule and the occupied squares to rotate
    together. Sampled from random self-play, and every piece the mover actually
    holds is checked rather than one arbitrary piece.
    """
    rng = random.Random(31)
    checked_moves = 0
    for trial in range(1000):
        s = engine.initial_state()
        for _ in range(rng.randrange(1, 18)):
            moves = sorted(engine.legal_moves(s))
            if not moves:
                break
            s = engine.apply_move(s, rng.choice(moves))
        r = engine.rotate_state(s)

        for owner in range(4):
            new_owner = engine.ROT_OWNER[owner]
            mine = engine.legal_moves(s, owner)
            theirs = engine.legal_moves(r, new_owner)
            assert len(mine) == len(theirs), (trial, owner)
            for move in mine:
                rotated = engine.rotate_move(move)
                assert rotated in theirs, (trial, owner, move, rotated)
                checked_moves += 1
    assert checked_moves > 100000, checked_moves


def test_rotate_move_inverts_rotate_state_for_a_real_move():
    """Playing `m` then rotating is the same as rotating then playing `rot(m)`."""
    s = engine.initial_state()
    owner = s.to_move
    for move in sorted(engine.legal_moves(s, owner))[:40]:
        a = engine.rotate_state(engine.apply_move(s, move))
        r = engine.rotate_state(s)
        b = engine.apply_move(engine.rotate_state(s), engine.rotate_move(move))
        assert a == b, move


def test_the_turn_order_survives_rotation_as_a_cycle():
    """Seating is relative, so rotating the board may not reorder the seats.

    Owner ids are labels tied to corners, so a rotation renumbers them; what has
    to survive is the *cycle*, i.e. who plays directly after whom.
    """
    s = engine.initial_state()
    r = engine.rotate_state(s)
    assert r.turn_order == tuple(engine.ROT_OWNER[o] for o in s.turn_order)
    # The invariant `_advance` walks is intact.
    assert r.turn_order[r.turn_pos] == r.to_move
    # Same relative order: the rotation of the cycle is itself a rotation.
    for i in range(4):
        old_next = s.turn_order[(s.turn_pos + i + 1) % 4]
        new_next = r.turn_order[(r.turn_pos + i + 1) % 4]
        assert new_next == engine.ROT_OWNER[old_next]


def test_rotating_keeps_neighbouring_turns_adjacent():
    """A rotation may not turn two adjacent turns into opposite corners.

    `_draw_turn_order` refuses to produce such an order, so rotation must not
    be able to either.
    """
    s = engine.initial_state()
    r = s
    for _ in range(4):
        for i in range(4):
            a = r.turn_order[(r.turn_pos + i) % 4]
            b = r.turn_order[(r.turn_pos + i + 1) % 4]
            ca = OWNER_CORNER[a]
            cb = OWNER_CORNER[b]
            assert abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) == B - 1, (a, b)
        r = engine.rotate_state(r)


def test_no_mirror_is_implemented():
    """A mirror would reverse the turn direction, so it must not exist here.

    Pinned as an absence: if someone adds `mirror_state` later this fails and
    forces a conversation about whether the result is still Blokus.
    """
    for name in dir(engine):
        assert "mirror" not in name.lower() and "flip" not in name.lower(), name
