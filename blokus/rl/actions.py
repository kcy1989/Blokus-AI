"""This layer answers one question: how does a legal move become a single
integer, and how does that integer move between coordinate systems?

Blokus has 30,433 geometrically real placements but a policy wants one flat
axis. The action index is

    index = (ORIENT_OFFSET[piece_idx] + oi) * N + base

which packs orientations densely, one contiguous block of 400 bases each, for
91 orientations and therefore 36,400 slots. Most slots are not real: an
orientation's bounding box does not fit at every base, and a base is only real
for the orientations whose `valid` set contains it. So the index space is a
sparse superset of the real moves, and `LUT_ROT` says which slots those are.

The direction rule is one sentence: the view is obtained by rotating
`(4 - p) % 4` steps clockwise, where `p` is the mover's seat in the clockwise
order. That is what `engine.normalize` does, and matching it exactly is the
whole point - if the two disagreed, features and action masks would be
computed in two different coordinate systems and a model trained on the
features would be scored on the wrong moves.

Rotation is never computed here as a formula. `LUT_ROT` is built by applying
`engine.rotate_move` repeatedly, so there is exactly one rotation rule in the
repository and it is the one the rules are defined with. See
`reports/g_report.md` section G5 for why legal moves are never computed on a
normalised view.
"""
import numpy as np

import engine
from config import B, CLOCKWISE_OWNERS, N

ORIENT_OFFSET = [0]
for _p in range(engine.N_PIECES):
    ORIENT_OFFSET.append(ORIENT_OFFSET[-1] + len(engine.ORIENTS[_p]))
N_ORIENT = ORIENT_OFFSET[-1]           # 91
N_ACTIONS = N_ORIENT * N               # 36400

# The packed-move stride inside `engine`'s 67,200-bit masks. It is *not* the
# action stride: the engine reserves eight orientation slots per piece while
# the action index numbers only the orientations that exist. Mixing the two up
# produces a mask that looks plausible and is wrong, so it is read from the
# engine rather than written out.
_PACK_STRIDE = engine.MAX_ORIENTS * N

_ORIENT_OF_INDEX = np.empty(N_ACTIONS, dtype=np.int32)
_BASE_OF_INDEX = np.empty(N_ACTIONS, dtype=np.int32)
_PIECE_OF_INDEX = np.empty(N_ACTIONS, dtype=np.int16)
_REAL = np.zeros(N_ACTIONS, dtype=bool)
for _p in range(engine.N_PIECES):
    for _oi, _od in enumerate(engine.ORIENTS[_p]):
        _j = ORIENT_OFFSET[_p] + _oi
        _sl = slice(_j * N, (_j + 1) * N)
        _ORIENT_OF_INDEX[_sl] = _j
        _BASE_OF_INDEX[_sl] = np.arange(N, dtype=np.int32)
        _PIECE_OF_INDEX[_sl] = _p
        _REAL[_sl] = [(_od["valid"] >> b) & 1 for b in range(N)]

REAL_MASK = _REAL.copy()
REAL_INDICES = np.flatnonzero(_REAL).astype(np.int32)

# Which orientation index, which piece, and which base does action `j` name?
# Kept as plain numpy arrays because the hot path is `legal_mask_view`, which
# needs the whole 36,400-slot table and not a Python loop.
_ORIENT_PIECE = np.empty(N_ORIENT, dtype=np.int16)
_ORIENT_OI = np.empty(N_ORIENT, dtype=np.int8)
_j = 0
for _p in range(engine.N_PIECES):
    for _oi in range(len(engine.ORIENTS[_p])):
        _ORIENT_PIECE[_j] = _p
        _ORIENT_OI[_j] = _oi
        _j += 1


def move_to_index(move):
    """`(piece_idx, oi, base)` -> the flat action index."""
    piece_idx, oi, base = move
    if not 0 <= piece_idx < engine.N_PIECES:
        raise ValueError("no such piece index %r" % (piece_idx,))
    if not 0 <= oi < len(engine.ORIENTS[piece_idx]):
        raise ValueError("piece %s has no orientation %r"
                         % (engine.PIECE_ORDER[piece_idx], oi))
    if not 0 <= base < N:
        raise ValueError("base %r is not on the board" % (base,))
    return (ORIENT_OFFSET[piece_idx] + oi) * N + base


def index_to_move(idx):
    """The inverse. Unpacks from the tables rather than by division, so it is
    exact for every slot including the unreal ones."""
    idx = int(idx)
    if not 0 <= idx < N_ACTIONS:
        raise ValueError("action index %r is out of range" % (idx,))
    j = _ORIENT_OF_INDEX[idx]
    return int(_PIECE_OF_INDEX[idx]), int(_ORIENT_OI[j]), int(_BASE_OF_INDEX[idx])


def is_real_index(idx):
    """Whether this slot names a placement whose bounding box fits the board."""
    return bool(_REAL[int(idx)])


# --------------------------------------------------------------------------
# legal moves
# --------------------------------------------------------------------------

def _build_packed_to_action():
    """Position in the engine's packed mask -> flat action index.

    The two numberings disagree: the engine reserves eight orientation slots
    per piece (`MAX_ORIENTS`) whether or not the piece has eight
    orientations, while the action index numbers only the ones that exist. So
    packed position `piece * 3200 + oi * 400 + base` maps to
    `(ORIENT_OFFSET[piece] + oi) * 400 + base`, and the gaps are -1.

    Building it once turns decoding a legal set into one numpy pass over the
    mask instead of a Python loop over every set bit, which matters because
    this is the single hottest call in the environment.
    """
    n_packed = engine.N_PIECES * _PACK_STRIDE
    lut = np.full(n_packed, -1, dtype=np.int32)
    for piece_idx in range(engine.N_PIECES):
        for oi in range(len(engine.ORIENTS[piece_idx])):
            dst = piece_idx * _PACK_STRIDE + oi * N
            src = (ORIENT_OFFSET[piece_idx] + oi) * N
            lut[dst:dst + N] = np.arange(src, src + N, dtype=np.int32)
    return lut


_PACKED_TO_ACTION = _build_packed_to_action()
_PACK_BYTES = (engine.N_PIECES * _PACK_STRIDE) // 8


def legal_indices(state, owner=None):
    """Every legal move as sorted action indices, in real coordinates.

    Decoded straight out of the engine's mask - through one byte-unpack and a
    table lookup - rather than through `unpack_move_mask`, which builds a
    Python set of tuples first. That set is the slow half of `legal_moves` and
    it is thrown away immediately, and this call happens once per environment
    step and once per action-mask build.

    Sorted, because `engine.unpack_move_mask` returns a set whose iteration
    order depends on the hash seed: an unsorted array would make a sampled
    action depend on `PYTHONHASHSEED`. The table lookup is ascending and the
    bits come out ascending, so the result is sorted already.
    """
    mask = engine.legal_move_mask(state, owner)
    if mask == 0:
        return np.empty(0, dtype=np.int32)
    raw = mask.to_bytes(_PACK_BYTES, "little")
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")
    idx = _PACKED_TO_ACTION[bits.view(bool)]
    return idx[idx >= 0]


def legal_index_set(state, owner=None):
    """`legal_indices` as a Python set, for tests and for membership checks
    that are not on a hot path."""
    return set(int(i) for i in legal_indices(state, owner))


# --------------------------------------------------------------------------
# rotation between the real board and the mover's view
# --------------------------------------------------------------------------

def _build_lut():
    """`LUT_ROT[k][i]`: action `i` expressed after `k` clockwise steps.

    Built by calling `engine.rotate_move` `k` times on each of the 30,433 real
    moves. Writing the rotation out again here would be a second definition of
    the same transformation, and the two would drift apart the moment a piece
    with a non-square bounding box was involved - the base moves, not just the
    orientation, and that is the part a hand-rolled formula usually forgets.

    Unreal slots map to -1 rather than to themselves, so a caller that forgets
    to mask a policy output gets a -1 rather than a plausible wrong answer.
    """
    lut = []
    for k in range(4):
        table = np.full(N_ACTIONS, -1, dtype=np.int32)
        for j in REAL_INDICES:
            piece_idx, oi, base = index_to_move(int(j))
            for _ in range(k):
                piece_idx, oi, base = engine.rotate_move((piece_idx, oi, base))
            table[j] = move_to_index((piece_idx, oi, base))
        lut.append(table)
    return lut


LUT_ROT = _build_lut()

# The four clockwise rotations of the real board, composed once, so
# `real_to_view` is a single fancy-index rather than k sequential ones.
LUT_VIEW = [LUT_ROT[(4 - p) % 4] for p in range(4)]


def seat_of_mover(state):
    """Where the player to move sits in the clockwise order: 0 is owner 1, the
    top-left seat. The same number `engine.normalize` uses."""
    return CLOCKWISE_OWNERS.index(state.to_move)


def _check_real(idx, what):
    if np.any(idx < 0):
        raise ValueError("%s was given a non-real action (-1); it has no "
                         "meaning in the other frame" % what)


def real_to_view(idx_array, p):
    """Real coordinates -> the mover's frame, by `(4 - p) % 4` clockwise steps.

    This is exactly the rotation `engine.normalize` applies to the position,
    so a feature plane built in the view and an action mask built here agree
    about which square is which. Raises on -1: an unreal slot has no view
    equivalent and silently passing it through would move a different piece.
    """
    idx = np.asarray(idx_array)
    _check_real(idx, "real_to_view")
    return LUT_VIEW[p][idx].astype(np.int32, copy=False)


def view_to_real(idx_array, p):
    """The other direction: `p` clockwise steps, matching `LUT_ROT[p]`."""
    idx = np.asarray(idx_array)
    _check_real(idx, "view_to_real")
    return LUT_ROT[p][idx].astype(np.int32, copy=False)


def legal_mask_view(state):
    """A `N_ACTIONS`-long bool mask in the mover's frame.

    Built by rotating the legal indices and setting bits, rather than rotating
    the whole 36,400-slot array, so the cost is proportional to the number of
    legal moves rather than to the size of the action space.
    """
    out = np.zeros(N_ACTIONS, dtype=bool)
    idx = legal_indices(state)
    if idx.size:
        out[real_to_view(idx, seat_of_mover(state))] = True
    return out


def legal_mask_real(state, owner=None):
    """The same mask without the rotation, for the rare caller that wants it."""
    out = np.zeros(N_ACTIONS, dtype=bool)
    out[legal_indices(state, owner)] = True
    return out


POLICY_SHAPE = (N_ORIENT, B, B)


def mask_bits_to_indices(mask):
    """A `(91, 20, 20)` bool array - what a policy head produces - into the
    flat action indices it marks. Sorted, so a sampled action is reproducible."""
    idx = np.flatnonzero(np.asarray(mask).reshape(-1))
    return idx.astype(np.int32, copy=False)


def indices_to_mask_bits(idx_array, shape=POLICY_SHAPE):
    """The inverse, for turning an action index back into a plane."""
    out = np.zeros(int(np.prod(shape)), dtype=bool)
    idx = np.asarray(idx_array, dtype=np.int64).reshape(-1)
    if idx.size:
        if idx.min() < 0 or idx.max() >= out.size:
            raise ValueError("action index outside the policy head's %d slots"
                             % out.size)
        out[idx] = True
    return out.reshape(shape)