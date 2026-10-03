"""This layer answers one question: is this move legal, and what is the position afterwards?

It is a second, deliberately independent implementation of the Blokus rules.
It exists so that the position algebra used for training can be checked
against the human-facing `Game`; sharing code with `board.py` would make that
check vacuous, so everything here is re-derived from the rules and then pinned
against `Game` in `tests/test_engine_cross.py`.

Deliberately absent: `pygame`, `ai.*`, `records`, `ui`. A position is pure
integers and must stay that way, so the engine can be driven from a plain
Python process with no display.
"""
import json
import os
from dataclasses import dataclass, replace

from config import B, N, OWNER_CORNER
from pieces import MASTER

# Bumped whenever a change can alter which moves are legal, what the position
# looks like afterwards, or how a state is numbered. Recorded on every line of a
# human game log, so a log can be read back knowing which rules produced it.
ENGINE_VERSION = "engine-1"

# --------------------------------------------------------------------------
# E1: board algebra
#
# A square is bit `y * B + x`. Every set is a Python int, so unions,
# intersections and complements are single machine-word-cheap operations on
# 400-bit values instead of a loop over 400 squares.
# --------------------------------------------------------------------------

ALL = (1 << N) - 1
N_BYTES = N // 8

COL0 = 0          # x == 0
COL_LAST = 0      # x == B - 1
for _y in range(B):
    COL0 |= 1 << (_y * B)
    COL_LAST |= 1 << (_y * B + B - 1)


def dilate(mask):
    """Every square sharing an edge with a set square.

    The column guard masks the *source* cells, never the result: shifting an
    x == 0 square left would otherwise wrap it onto the previous row's last
    column and invent a neighbour that does not exist.
    """
    left = (mask & ~COL0) >> 1
    right = (mask & ~COL_LAST) << 1
    return (left | right | (mask >> B) | (mask << B)) & ALL


def dilate_diag(mask):
    """Every square touching a set square at a *corner* only.

    The four edge neighbours are deliberately absent. Blokus requires a piece
    to share a corner with one of its own stones while sharing no edge, so the
    two dilations are genuinely different sets and folding them together would
    make every move after the opening illegal.
    """
    left = mask & ~COL0
    right = mask & ~COL_LAST
    return ((left >> (B + 1)) | (right >> (B - 1))
            | (left << (B - 1)) | (right << (B + 1))) & ALL


def empty_of(own_bits):
    return ALL & ~(own_bits[0] | own_bits[1] | own_bits[2] | own_bits[3])


# --------------------------------------------------------------------------
# E1: piece geometry
#
# `hand_bits` numbers bits by the `pieces` order in `action_table.json`, so a
# hand is a 21-bit integer and "remove this piece" is one AND-NOT. The order is
# asserted against the table rather than re-sorted, because `action_table.json`
# is frozen: its hash is what any stored position or trained model refers to.
# --------------------------------------------------------------------------

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "action_table.json"), encoding="utf-8") as _fh:
    _TABLE = json.load(_fh)

PIECE_ORDER = tuple(_TABLE["pieces"])
assert sorted(PIECE_ORDER) == sorted(MASTER), \
    "action_table.json and the piece data disagree on the piece set"
PIECE_IDX = {name: i for i, name in enumerate(PIECE_ORDER)}
N_PIECES = len(PIECE_ORDER)
FULL_HAND = (1 << N_PIECES) - 1
PIECE_SIZE = [MASTER[n]["size"] for n in PIECE_ORDER]
HAND_CELLS_TOTAL = sum(PIECE_SIZE)

MAX_ORIENTS = max(len(MASTER[n]["orientations"]) for n in PIECE_ORDER)


def _build_orientations():
    """Precompute, per orientation, the shape mask, its cell offsets and the
    bases where the whole bounding box fits on the board.

    `valid` is the only thing standing between a 400-bit shift and a row wrap,
    so it is computed from the orientation's own extent rather than assumed.
    """
    out = []
    for name in PIECE_ORDER:
        per = []
        for cells in MASTER[name]["orientations"]:
            mx = max(x for x, _ in cells)
            my = max(y for _, y in cells)
            offs = tuple(dy * B + dx for dx, dy in sorted(cells))
            m = 0
            for o in offs:
                m |= 1 << o
            valid = 0
            for y in range(B - my):
                for x in range(B - mx):
                    valid |= 1 << (x + y * B)
            per.append({"m": m, "offs": offs, "valid": valid, "w": mx + 1,
                        "h": my + 1, "cells": tuple(sorted(cells))})
        out.append(tuple(per))
    return tuple(out)


ORIENTS = _build_orientations()

# A move packs into one integer so a whole legal set can be compared with a
# single `==` instead of building thousands of tuples at every step.
_MOVE_STRIDE = N * 8   # 400 bases x up to 8 orientations per piece
MOVE_MASK_BITS = N_PIECES * _MOVE_STRIDE


def _pack(piece_idx, oi, base):
    return 1 << (piece_idx * _MOVE_STRIDE + oi * N + base)


def move_bit(piece_idx, oi, base):
    """The single bit `move` occupies inside a packed legal-move mask.

    Exposed so callers never have to re-derive the packing, which is easy to
    get wrong: the orientation stride is the whole 400-square board, not the
    board's width.
    """
    return 1 << (piece_idx * _MOVE_STRIDE + oi * N + base)


def _unpack(mask):
    out = set()
    for piece_idx in range(N_PIECES):
        m = (mask >> (piece_idx * _MOVE_STRIDE)) & ((1 << _MOVE_STRIDE) - 1)
        if not m:
            continue
        for oi in range(MAX_ORIENTS):
            b400 = (m >> (oi * N)) & ALL
            while b400:
                low = b400 & -b400
                out.add((piece_idx, oi, low.bit_length() - 1))
                b400 ^= low
    return out


# --------------------------------------------------------------------------
# E1: the position
#
# Frozen, and built only out of tuples and ints, so two states can never share
# a mutable object. `apply_move` returns a new state and leaves the old one
# untouched; there is no `unplace` and nothing to undo.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    own_bits: tuple    # 4 x 400-bit ints
    hand_bits: tuple   # 4 x 21-bit ints
    turn_order: tuple  # owner ids, clockwise, starting with whoever moves first
    to_move: int       # owner id
    stuck: tuple       # 4 bools, one-way latches

    @property
    def turn_pos(self):
        return self.turn_order.index(self.to_move)


def initial_state(turn_order=None):
    """An empty board. Everyone holds the full 21-piece hand."""
    if turn_order is None:
        from config import CLOCKWISE_OWNERS
        turn_order = CLOCKWISE_OWNERS
    turn_order = tuple(turn_order)
    if sorted(turn_order) != list(range(4)):
        raise ValueError("turn_order must be a permutation of 0..3, got %r"
                         % (turn_order,))
    return State(own_bits=(0, 0, 0, 0),
                 hand_bits=(FULL_HAND,) * 4,
                 turn_order=turn_order,
                 to_move=turn_order[0],
                 stuck=(False, False, False, False))


def remaining_cells(state, owner):
    """Unplaced squares: the score. Fewer is better."""
    hand = state.hand_bits[owner]
    total = 0
    while hand:
        low = hand & -hand
        total += PIECE_SIZE[low.bit_length() - 1]
        hand ^= low
    return total


def result(state):
    """(remaining cells) for all four owners, indexed by owner id."""
    return tuple(remaining_cells(state, o) for o in range(4))


def is_over(state):
    """The game ends only when *all four* players have no legal move."""
    return all(state.stuck)


# --------------------------------------------------------------------------
# E1: the rules
# --------------------------------------------------------------------------

def _corner_bit(owner):
    x, y = OWNER_CORNER[owner]
    return 1 << (x + y * B)


def _legal_bases_for_orient(state, owner, piece_idx, oi):
    """The bases where this exact piece, in this exact orientation, is legal.

    Two halves, and they are mutually exclusive by construction, mirroring the
    game: a player's opening piece must cover their own corner square; after
    that every piece must cover at least one corner-contact square and no
    edge-adjacent one.
    """
    own = state.own_bits[owner]
    empt = empty_of(state.own_bits)
    od = ORIENTS[piece_idx][oi]

    # Every cell of the piece must land on an empty square, inside the board.
    cand = od["valid"]
    for o in od["offs"]:
        cand &= empt >> o
    if not cand:
        return 0

    if not own:
        # Opening move: the piece has to cover this player's own corner square.
        # `cbit >> o` is already a set of *bases*, so the shape mask must not
        # be ANDed in here: that mixes a base set with a shape, and it fails
        # silently on three of the four corners while still looking right on
        # the top-left one, where the offsets are small enough to cancel out.
        cbit = _corner_bit(owner)
        hit = 0
        for o in od["offs"]:
            hit |= cbit >> o
        return cand & hit

    need = dilate_diag(own) & empt
    avoid = dilate(own) & empt
    touching = 0
    bad = 0
    for o in od["offs"]:
        touching |= need >> o
        bad |= avoid >> o
    return cand & touching & ~bad


def has_any_legal(state, owner):
    hand = state.hand_bits[owner]
    while hand:
        low = hand & -hand
        piece_idx = low.bit_length() - 1
        for oi in range(len(ORIENTS[piece_idx])):
            if _legal_bases_for_orient(state, owner, piece_idx, oi):
                return True
        hand ^= low
    return False


def legal_move_mask(state, owner=None):
    """Every legal move as one integer: bit `piece_idx * 3200 + oi * 400 + base`.

    Same content as `legal_moves`, but comparable with a single `==`, which is
    what the cross-validation needs at every step of every game.
    """
    if owner is None:
        owner = state.to_move
    out = 0
    hand = state.hand_bits[owner]
    while hand:
        low = hand & -hand
        piece_idx = low.bit_length() - 1
        for oi in range(len(ORIENTS[piece_idx])):
            m = _legal_bases_for_orient(state, owner, piece_idx, oi)
            while m:
                lowb = m & -m
                out |= _pack(piece_idx, oi, lowb.bit_length() - 1)
                m ^= lowb
        hand ^= low
    return out


def legal_moves(state, owner=None):
    """Every legal move as `(piece_idx, oi, base)`.

    `piece_idx` indexes `PIECE_ORDER`, the same numbering `hand_bits` uses.
    """
    return frozenset(_unpack(legal_move_mask(state, owner)))


def unpack_move_mask(mask):
    return _unpack(mask)


def _advance(state):
    """Move the turn on, then latch stuck players.

    The latch is a one-way door: a player's contact rule depends only on their
    own stones, which only ever grow, and on the empty squares, which only ever
    shrink. So a player who cannot move now can never move later, and one test
    per seat is enough. The scan stops at the first player who can still move,
    which keeps the latched set identical to the game's.
    """
    to_move = state.turn_order[(state.turn_pos + 1) % 4]
    stuck = list(state.stuck)
    for o in range(4):
        if stuck[o]:
            continue
        if has_any_legal(state, o):
            break
        stuck[o] = True
    return replace(state, to_move=to_move, stuck=tuple(stuck))


def apply_move(state, move):
    """Play a piece and hand the turn on.

    `move` is `(piece_idx, oi, base)`. The legality check is mandatory: an
    illegal move raises rather than being silently ignored, so a search that
    invents a move fails loudly instead of producing a position nobody can
    reach.
    """
    piece_idx, oi, base = move
    owner = state.to_move
    if not 0 <= piece_idx < N_PIECES:
        raise ValueError("no such piece index %r" % (piece_idx,))
    if not 0 <= oi < len(ORIENTS[piece_idx]):
        raise ValueError("piece %s has no orientation %r"
                         % (PIECE_ORDER[piece_idx], oi))
    if not state.hand_bits[owner] >> piece_idx & 1:
        raise ValueError("owner %d does not hold %s"
                         % (owner, PIECE_ORDER[piece_idx]))
    if not _legal_bases_for_orient(state, owner, piece_idx, oi) >> base & 1:
        raise ValueError("illegal move: %s oi=%d base=%d for owner %d"
                         % (PIECE_ORDER[piece_idx], oi, base, owner))

    od = ORIENTS[piece_idx][oi]
    placed = od["m"] << base
    own = list(state.own_bits)
    own[owner] |= placed
    hand = list(state.hand_bits)
    hand[owner] &= ~(1 << piece_idx)
    nxt = State(own_bits=tuple(own), hand_bits=tuple(hand),
                turn_order=state.turn_order, to_move=owner,
                stuck=state.stuck)
    return _advance(nxt)


def pass_turn(state):
    """Spend a turn without playing.

    Passing is not a choice: a stuck player simply never has a move, and the
    rotation still moves on. The turn pointer advances and the board does not
    change, so this is how a hand with nothing playable gets past its owner.
    """
    return _advance(state)


# --------------------------------------------------------------------------
# E1: serialisation
#
# Ints go out as hex strings because JSON has no arbitrary-precision integer.
# --------------------------------------------------------------------------

def serialize(state):
    return json.dumps({
        "own_bits": [format(m, "x") for m in state.own_bits],
        "hand_bits": [format(m, "x") for m in state.hand_bits],
        "turn_order": list(state.turn_order),
        "to_move": state.to_move,
        "stuck": list(state.stuck),
    }, sort_keys=True, separators=(",", ":"))


def deserialize(blob):
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode("ascii")
    d = json.loads(blob)
    own = tuple(int(m, 16) for m in d["own_bits"])
    hand = tuple(int(m, 16) for m in d["hand_bits"])
    if len(own) != 4 or len(hand) != 4:
        raise ValueError("a position has four players")
    for m in own:
        if m & ~ALL:
            raise ValueError("own_bits has bits outside the board")
    for m in hand:
        if m & ~FULL_HAND:
            raise ValueError("hand_bits has bits outside the 21 pieces")
    for i, m in enumerate(own):
        for j in range(i + 1, 4):
            if own[i] & own[j]:
                raise ValueError("square owned by two players at once")
    return State(own_bits=own, hand_bits=hand,
                 turn_order=tuple(d["turn_order"]),
                 to_move=d["to_move"], stuck=tuple(d["stuck"]))


# --------------------------------------------------------------------------
# E2: clockwise rotation
#
# One step is 90 degrees clockwise: (x, y) -> (B-1-y, x), which sends TL to TR.
# Mirroring is deliberately not implemented. A mirror reverses the turn
# direction, and the clockwise seating is a rule rather than a presentation
# detail, so mirroring would describe a different game.
# --------------------------------------------------------------------------

def rot_orient(piece_idx, oi):
    """The orientation index after one clockwise step.

    The piece's own bounding box turns with it, so a w x h cell set becomes
    h x w and the rotation is `(x, y) -> (h-1-y, x)`. Because `pieces.py`
    deduplicates rotations, this can map several orientations to one; four
    applications always come back to the start.
    """
    cells = ORIENTS[piece_idx][oi]["cells"]
    h = ORIENTS[piece_idx][oi]["h"]
    r = frozenset((h - 1 - y, x) for x, y in cells)
    for cand, od in enumerate(ORIENTS[piece_idx]):
        if frozenset(od["cells"]) == r:
            return cand
    raise KeyError("rotation left the orientation set")   # pragma: no cover


def rot_base(piece_idx, oi, base):
    """The base after one clockwise step.

    The board is B wide, so the piece's origin moves to the far side: the new
    x origin is `B - y0 - h` and the new y origin is `x0`. Both stay in range
    exactly when the original placement did.
    """
    h = ORIENTS[piece_idx][oi]["h"]
    x0, y0 = base % B, base // B
    return (B - y0 - h) + x0 * B


# Which owner sits where: TL, TR, BR, BL.
ROT_OWNER = {1: 2, 2: 0, 0: 3, 3: 1}
ROT_CORNER = {OWNER_CORNER[o]: OWNER_CORNER[ROT_OWNER[o]] for o in ROT_OWNER}


def _rot_mask(mask):
    """Rotate a 400-bit set one step clockwise, 20 rows at a time.

    Row by row rather than with a whole-board bit trick, because the four
    rotations only differ in how far each row is shifted and an explicit
    `dest_x` cannot silently wrap.
    """
    row_bits = (1 << B) - 1
    out = 0
    for y in range(B):
        row = (mask >> (y * B)) & row_bits
        if not row:
            continue
        for x in range(B):
            if row >> x & 1:
                dx, dy = B - 1 - y, x
                out |= 1 << (dy * B + dx)
    return out


def rotate_state(state):
    """The position as it looks after one clockwise step of the board.

    Squares move with the board, the pieces' orientations and bases follow, and
    every owner takes the seat of the corner they were sitting in. Because
    `own_bits` and `hand_bits` are both indexed by owner, the hands are
    permuted along with the stones - otherwise a rotated position would credit
    the stones to one player and the pieces in hand to another.

    `turn_order` is permuted for the same reason: owner ids are labels tied to
    corners, so a step of the board renumbers them. Leaving it alone would break
    `turn_order[turn_pos] == to_move`, which is what `_advance` walks. The
    *relative* seating is untouched - a clockwise step sends every owner to the
    next seat in the cycle, so the turn order stays a rotation of the same
    4-cycle and two neighbouring turns are still adjacent corners.
    """
    own = [0, 0, 0, 0]
    hand = [0, 0, 0, 0]
    stuck = [False] * 4
    for o in range(4):
        n = ROT_OWNER[o]
        own[n] = _rot_mask(state.own_bits[o])
        hand[n] = state.hand_bits[o]
        stuck[n] = state.stuck[o]
    return State(own_bits=tuple(own), hand_bits=tuple(hand),
                 turn_order=tuple(ROT_OWNER[o] for o in state.turn_order),
                 to_move=ROT_OWNER[state.to_move], stuck=tuple(stuck))


def rotate_move(move):
    """The same move expressed in the rotated position."""
    piece_idx, oi, base = move
    return (piece_idx, rot_orient(piece_idx, oi), rot_base(piece_idx, oi, base))


# --------------------------------------------------------------------------
# E3: the position as one player sees it
#
# Everything above is a position from the outside: four owner ids that happen
# to be labelled by corner. That is the right shape for the rules and the wrong
# one for a learner, because "my stones" and "the player two seats away" are
# what actually matter and both change meaning from one game to the next.
#
# So the position is first rotated until the player to move sits in the
# top-left corner, then the four players are renumbered to me, the player after
# me, the player opposite, and the player before me. Two games that are
# rotations of each other then produce byte-identical planes, which is what
# makes rotation-based data augmentation possible at all.
#
# numpy appears only in this section; the rest of the engine stays pure Python
# integers so the rules can be read and checked without it.
# --------------------------------------------------------------------------

from config import CLOCKWISE_OWNERS   # noqa: E402


def normalize(state):
    """The same position, rotated and renumbered to the mover's point of view.

    Returns `(view, seat_of)` where `view.to_move == 0` and `seat_of[i]` is the
    *original* owner id now sitting at index `i`:

        seat_of = (me, next, opposite, previous)     clockwise

    The turn order is renumbered to match, so `view` is an ordinary `State` and
    the rest of the engine works on it unchanged.
    """
    p = CLOCKWISE_OWNERS.index(state.to_move)
    k = (4 - p) % 4                 # clockwise steps that bring me to TL
    s = state
    for _ in range(k):
        s = rotate_state(s)

    # Rotation renumbers the owners, so track who ended up where before
    # translating the rotated position back to the original numbering.
    moved = list(range(4))
    for _ in range(k):
        moved = [ROT_OWNER[o] for o in moved]
    back = {new: old for old, new in enumerate(moved)}

    # In the rotated frame the mover sits at TL, and TL heads the clockwise
    # cycle, so "me, next, opposite, previous" is that cycle read from the top.
    # `s` is still in the rotated frame, so it is read with those ids; `seat_of`
    # is the same order translated back for the caller's benefit.
    frame = CLOCKWISE_OWNERS
    seat_of = tuple(back[o] for o in frame)
    assert seat_of[0] == state.to_move
    return State(own_bits=tuple(s.own_bits[o] for o in frame),
                 hand_bits=tuple(s.hand_bits[o] for o in frame),
                 turn_order=tuple(frame.index(o) for o in s.turn_order),
                 to_move=0,
                 stuck=tuple(s.stuck[o] for o in frame)), seat_of


def need_avoid(state, owner):
    """The mover's corner-contact rule as two squares sets.

    `need` are the squares a piece may touch to satisfy the rule, `avoid` are
    the ones it may not. Both are clipped to the empty squares, which is
    cheaper and cannot change an answer, because a piece may only be placed on
    an empty square anyway. On an opening position the rule is the corner, not
    contact, and both sets are empty.
    """
    own = state.own_bits[owner]
    if not own:
        return 0, 0
    empt = empty_of(state.own_bits)
    return dilate_diag(own) & empt, dilate(own) & empt


def bits_to_plane(mask):
    """A 20x20 `float32` array, `1.0` where a bit is set.

    Bit `y * B + x` becomes row `y`, column `x`, so the array is a picture of
    the board with the origin at the top left and no transposition needed.
    """
    import numpy as np
    raw = mask.to_bytes(N_BYTES, "little")
    return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little") \
             .reshape(B, B).astype(np.float32)


# Channel order is fixed, because a training pipeline that renumbers its inputs
# between runs produces a model that has silently learned the wrong thing.
PLANE_CHANNELS = ("me", "next", "opposite", "previous",   # stones on board
                  "my_need", "my_avoid",                 # my contact rule
                  "empty")
PLANE_C = len(PLANE_CHANNELS)


def to_plane(state):
    """The position as `(C, 20, 20) float32`, from the mover's point of view.

    The caller's position is normalised first, so channel 0 is always the
    player to move. `my_need` is the corner-contact rule - the squares a piece
    may touch - and `my_avoid` the squares it may not, which between them are
    what every legal move has to satisfy.
    """
    import numpy as np
    view, _seat_of = normalize(state)
    need, avoid = need_avoid(view, 0)
    empt = empty_of(view.own_bits)
    planes = [bits_to_plane(m) for m in view.own_bits]
    planes.append(bits_to_plane(need & ~avoid & empt))
    planes.append(bits_to_plane(avoid))
    planes.append(bits_to_plane(empt))
    out = np.stack(planes).astype(np.float32)
    assert out.shape == (PLANE_C, B, B), out.shape
    return out


def hand_vectors(state):
    """The four players' remaining pieces, as `(4, 21) float32`.

    From the mover's point of view, so row 0 is the player to move. Kept out of
    the plane on purpose: it is a per-game feature rather than a per-square one,
    and broadcasting it across 400 squares would spend the model's capacity
    re-learning a fact that is nine numbers long.
    """
    import numpy as np
    view, _seat_of = normalize(state)
    out = np.zeros((4, N_PIECES), dtype=np.float32)
    for p in range(4):
        bits = view.hand_bits[p]
        for i in range(N_PIECES):
            if bits >> i & 1:
                out[p, i] = 1.0
    return out
