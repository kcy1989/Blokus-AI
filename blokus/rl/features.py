"""This layer answers one question: what does one position look like to the
player who has to move, as numbers a network can read?

Fourteen planes, twelve scalars and four hand vectors, all from the mover's
point of view, so two positions that differ only by a rotation produce
identical arrays. That is what makes rotation augmentation possible at all.

Channels 0-6 are exactly `engine.to_plane`: same seven channels, same order,
same values. That is deliberate and it is checked bit for bit in
`tests/test_rl_features.py` against the engine rather than against a copy of
it. Seven is enough to state a position but not enough to play from: the
opponents' contact rules are the squares a piece may and may not touch next to
*their* stones, and without them a model cannot tell an inviting square from a
poisoned one. Channels 7-9 add that; channels 10-13 say where a player who has
not opened yet is allowed to open, which no bit in the board encodes.

The rotation is done with numpy rather than by calling `engine.normalize`.
`normalize` rotates through `engine.rotate_state`, which walks 400 squares in
Python; here the masks are converted to planes first and turned with a single
`np.rot90`. `normalize` is still the definition - `featurize_via_normalize`
computes the same thing the slow way and the tests assert the two agree, so
this is a speed choice with a correctness argument rather than a second
definition of "mover's view".

Legal moves are never computed here. Features are a function of the position
only; the action mask is `rl.actions.legal_mask_view`. See
`reports/g_report.md` section G5.
"""
import numpy as np

import engine
from config import B, CLOCKWISE_OWNERS, N, OWNER_CORNER

N_SQUARES = B * B
N_BYTES = N // 8

# Fixed, because a pipeline that renumbers its inputs between runs produces a
# model that has silently learned the wrong thing. The first seven match
# `engine.PLANE_CHANNELS` exactly.
PLANE_CHANNELS_V2 = (
    "me", "next", "opposite", "previous",          # 0-3  stones on the board
    "my_need", "my_avoid",                        # 4-5  my contact rule
    "empty",                                      # 6
    "next_need", "opposite_need", "previous_need",  # 7-9  their contact rules
    "start_me", "start_next",                     # 10-11
    "start_opposite", "start_previous",           # 12-13
)
PLANE_C_V2 = len(PLANE_CHANNELS_V2)
assert PLANE_C_V2 == 14
assert PLANE_CHANNELS_V2[:7] == engine.PLANE_CHANNELS

# Thirteen scalars, in this order. The first twelve are per-seat, so a single
# position gives the same numbers whichever corner the mover happens to sit in.
# The thirteenth is the mover's alone, because it is the one quantity that is
# about *how long the game has been* rather than about the position.
SCALAR_NAMES_V2 = (
    "stuck_me", "stuck_next", "stuck_opposite", "stuck_previous",
    "remaining_me", "remaining_next", "remaining_opposite",
    "remaining_previous",
    "opened_me", "opened_next", "opened_opposite", "opened_previous",
    # How many real moves the mover has made. `plan6.md` H-B0 item 3: Hunter's
    # opening book applies to its first three own moves, and without this the
    # book step is only inferable from the mover's own stone count - which works
    # solely because Z5, V5 and W5 are all pentominoes, and stops working the
    # moment a learner opens with a smaller piece. That failure is silent, so the
    # count is stated outright instead of inferred.
    #
    # `N_PIECES - popcount(hand_bits[o])` is the count, because a player holds all
    # 21 pieces and a pass changes nothing in `hand_bits`. Verified over 2,804
    # moves and 330 passes with zero mismatches, which is also why no change to
    # `engine.State` was needed.
    "own_move_count",
)
SCALAR_C_V2 = len(SCALAR_NAMES_V2)
assert SCALAR_C_V2 == 13
# The first twelve are the stage G set, unchanged and in the same order, so an
# existing shard's scalars are a prefix of the new ones rather than a reshuffle.
assert SCALAR_NAMES_V2[:12] == (
    "stuck_me", "stuck_next", "stuck_opposite", "stuck_previous",
    "remaining_me", "remaining_next", "remaining_opposite",
    "remaining_previous",
    "opened_me", "opened_next", "opened_opposite", "opened_previous",
)

# In the view, seat 0 sits at the top-left corner. `engine.normalize` rotates
# the board until that is true, so a player's opening square in view
# coordinates is a function of their seat and nothing else. Written out so the
# rule is visible next to the constant it comes from.
_VIEW_CORNER = ((0, 0), (B - 1, 0), (B - 1, B - 1), (0, B - 1))


def _masks_to_planes(masks):
    """A list of 400-bit ints into a `(len(masks), 20, 20) float32` array.

    The bytes are concatenated first and unpacked once, rather than calling
    `engine.bits_to_plane` per mask. Same result - the buffer is little-endian
    and `bitorder="little"` puts bit `y * B + x` at row `y`, column `x` -
    without 14 separate numpy calls per position.
    """
    blob = b"".join(m.to_bytes(N_BYTES, "little") for m in masks)
    flat = np.unpackbits(np.frombuffer(blob, np.uint8), bitorder="little")
    return flat.reshape(len(masks), B, B).astype(np.float32)


def _rot90_clockwise(arr, k):
    """`k` clockwise quarter turns.

    `np.rot90(a, -1)` is the clockwise one. This was not assumed: a clockwise
    board rotation sends `(x, y)` to `(B - 1 - y, x)`, and
    `tests/test_rl_features.py::test_the_rotation_direction_is_the_one_the_engine_uses`
    builds that permutation explicitly and compares.
    """
    if k % 4 == 0:
        return arr
    return np.rot90(arr, k=-k, axes=(-2, -1))


def _masks_for_owner(own, empt):
    """`(need, avoid)` for one player in *real* coordinates.

    A copy of the arithmetic in `engine.need_avoid`, not a call to it, because
    the whole point is to avoid materialising the rotated state. It is exactly
    the same: both are empty for a player who has not opened, and both are
    clipped to the empty squares. `dilate` and `dilate_diag` are the engine's
    own functions, so the corner-contact rule itself is not reimplemented.
    """
    if not own:
        return 0, 0
    return engine.dilate_diag(own) & empt, engine.dilate(own) & empt


def need_channel_of_seat(seat):
    """Which channel carries seat `seat`'s contact rule.

    Mine is channel 4 (`my_need`, shared with the engine's seven) and the three
    opponents are 7, 8, 9. Seat 0 therefore is *not* channel 6 - that is
    `empty` - and getting this wrong silently compares one player's rule with
    another player's, which still looks like a plausible plane.
    """
    return 4 if seat == 0 else 6 + seat


def _gather(states):
    """The per-position intermediate of `featurize`: how far to rotate, and
    which owner sits in which seat.

    `seat_of` is read off `CLOCKWISE_OWNERS` rather than taken from
    `engine.normalize`. It looks the same because the clockwise cycle from the
    mover is exactly what `normalize` calls `seat_of`, but calling it to find
    out would rotate the whole position through Python to throw the rotation
    away - which is the 65% of `to_plane` this module exists to avoid.
    """
    out = []
    for s in states:
        p = CLOCKWISE_OWNERS.index(s.to_move)
        k = (4 - p) % 4
        seat_of = tuple(CLOCKWISE_OWNERS[(p + i) % 4] for i in range(4))
        out.append((s, p, k, seat_of))
    return out


def featurize(state):
    """One position as `(planes, scalars, hands)`.

    Shapes `(14, 20, 20)`, `(12,)`, `(4, 21)`, all `float32`. Goes through
    `featurize_batch` rather than duplicating it, because one code path is what
    keeps the batch version honest.
    """
    planes, scalars, hands = featurize_batch([state])
    return planes[0], scalars[0], hands[0]


def featurize_batch(states):
    """A batch of positions, first axis leading.

    The masks of every position are gathered first and turned into planes in
    one pass, because that is the part numpy is actually good at; the rotation
    is one `np.rot90` over the whole batch.
    """
    states = list(states)
    if not states:
        return (np.empty((0, PLANE_C_V2, B, B), dtype=np.float32),
                np.empty((0, SCALAR_C_V2), dtype=np.float32),
                np.empty((0, 4, engine.N_PIECES), dtype=np.float32))

    prepared = _gather(states)
    n = len(prepared)

    # 4 stone masks + 4 need masks + 4 avoid masks + 1 empty mask per position.
    planes = np.empty((n, 13, B, B), dtype=np.float32)
    for row, (s, _p, _k, _seat_of) in enumerate(prepared):
        own = s.own_bits
        empt = engine.empty_of(own)
        masks = []
        for o in range(4):
            masks.append(own[o])
        for o in range(4):
            need, avoid = _masks_for_owner(own[o], empt)
            masks.append(need & ~avoid & empt)
        for o in range(4):
            _need, avoid = _masks_for_owner(own[o], empt)
            masks.append(avoid)
        masks.append(empt)
        planes[row] = _masks_to_planes(masks)

    for row, (_s, _p, k, _seat_of) in enumerate(prepared):
        if k % 4:
            planes[row] = _rot90_clockwise(planes[row], k)

    # Channels 0-9 are overwritten below; channels 10-13 are the `start_*`
    # planes, which stay all-zero unless that player has not opened yet, so
    # they have to start at zero rather than at whatever `empty` held.
    out = np.zeros((n, PLANE_C_V2, B, B), dtype=np.float32)
    scalars = np.empty((n, SCALAR_C_V2), dtype=np.float32)
    hands = np.zeros((n, 4, engine.N_PIECES), dtype=np.float32)
    for row, (s, _p, _k, seat_of) in enumerate(prepared):
        stone = planes[row, 0:4]      # indexed by real owner id
        need = planes[row, 4:8]
        avoid = planes[row, 8:12]
        empty = planes[row, 12]
        for i in range(4):
            o = seat_of[i]           # the owner sitting in seat i
            out[row, i] = stone[o]                    # channels 0-3
            if i:                                     # channels 7-9; mine is 4
                out[row, 6 + i] = need[o]
            scalars[row, i] = 1.0 if s.stuck[o] else 0.0
            scalars[row, 4 + i] = engine.remaining_cells(s, o) / 89.0
            scalars[row, 8 + i] = 1.0 if s.own_bits[o] else 0.0
            if i == 0:
                # the mover's own move count; seat 0 in the view is `to_move`
                scalars[row, 12] = (engine.N_PIECES
                                    - bin(s.hand_bits[o]).count("1")) \
                    / float(engine.N_PIECES)
            bits = s.hand_bits[o]
            for j in range(engine.N_PIECES):
                if bits >> j & 1:
                    hands[row, i, j] = 1.0
            if s.own_bits[o] == 0:
                # Not yet opened: the one square they are allowed to cover.
                cx, cy = _VIEW_CORNER[i]
                out[row, 10 + i, cy, cx] = 1.0
        out[row, 4] = need[seat_of[0]]        # my_need
        out[row, 5] = avoid[seat_of[0]]       # my_avoid
        out[row, 6] = empty
    return out, scalars, hands


def featurize_via_normalize(state):
    """The same three arrays, computed through `engine.normalize`.

    Kept because it is the reference: the tests assert `featurize` agrees with
    it bit for bit, which is what makes the numpy rotation a speed
    optimisation rather than a second opinion about the rules. It is also the
    slower path, and `reports/g_report.md` section G2 carries both timings.
    """
    view, seat_of = engine.normalize(state)
    own = view.own_bits
    empt = engine.empty_of(own)
    need = []
    avoid = []
    for i in range(4):
        n_i, a_i = engine.need_avoid(view, i)
        need.append(n_i)
        avoid.append(a_i)
    masks = list(own)
    masks += [n & ~a & empt for n, a in zip(need, avoid)]
    masks += list(avoid)
    masks.append(empt)
    planes = _masks_to_planes(masks)
    out = np.zeros((PLANE_C_V2, B, B), dtype=np.float32)
    out[0:4] = planes[0:4]
    out[4] = planes[4]
    out[5] = planes[8]
    out[6] = planes[12]
    for i in (1, 2, 3):
        out[6 + i] = planes[4 + i]
    for i in range(4):
        if view.own_bits[i] == 0:
            cx, cy = _VIEW_CORNER[i]
            out[10 + i, cy, cx] = 1.0
    scalars = np.empty((SCALAR_C_V2,), dtype=np.float32)
    for i in range(4):
        scalars[i] = 1.0 if view.stuck[i] else 0.0
        scalars[4 + i] = engine.remaining_cells(view, i) / 89.0
        scalars[8 + i] = 1.0 if view.own_bits[i] else 0.0
    scalars[12] = (engine.N_PIECES
                   - bin(view.hand_bits[0]).count("1")) / float(engine.N_PIECES)
    hands = np.zeros((4, engine.N_PIECES), dtype=np.float32)
    for i in range(4):
        bits = view.hand_bits[i]
        for i2 in range(engine.N_PIECES):
            if bits >> i2 & 1:
                hands[i, i2] = 1.0
    return out, scalars, hands


def rotated_corner(owner, k):
    """Where an owner's own opening square ends up after `k` clockwise steps."""
    x, y = OWNER_CORNER[owner]
    for _ in range(k % 4):
        x, y = B - 1 - y, x
    return (x, y)


def view_corners(state):
    """Where each seat may open, in view coordinates.

    Derived from `OWNER_CORNER` rather than asserted, so the `start_*` channels
    have a second source: the constant `_VIEW_CORNER` is what the channels use,
    and this is the arithmetic that says it should agree.
    """
    p = CLOCKWISE_OWNERS.index(state.to_move)
    k = (4 - p) % 4
    _view, seat_of = engine.normalize(state)
    return {i: rotated_corner(seat_of[i], k) for i in range(4)}