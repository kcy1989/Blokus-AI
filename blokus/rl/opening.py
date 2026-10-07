"""This layer answers one question: does the optimizer play better when its
first three moves are dictated by a fixed opening book instead of by its own
objective function?

The book is the standard Blokus corner opening, one variant per player. Player
`o` starts on the corner square `c = OWNER_CORNER[o]` and runs the diagonal `d`
that points at the middle of the board. Its three pieces each have to occupy
two specific squares along that diagonal:

    k = 0   Z5   c,          c + 2d
    k = 1   V5   c + 3d,     c + 5d
    k = 2   W5   c + 6d,     c + 8d

`c` and `d` are *derived* from `OWNER_CORNER`, not written out four times: a
hand-written table would be a second source of truth for where a player's
corner is, and it would be wrong the moment the seating changed. `d` is the
sign of the step from the corner towards the centre, which is a property of the
corner alone.

The candidate set for step `k` is "every legal move of the specified piece that
covers both named squares", so it contains all orientations and mirror images
that fit - the direction along the diagonal is not fixed, only the two squares
that must be covered are. A candidate is therefore legal by construction: it is
filtered out of `legal_indices`, never generated and hoped for.

Two decisions the book does not get to make:

  * An empty candidate set at step `k` **abandons the book for the rest of the
    game**. There is no retry at step `k + 1`. Being blocked on one square is
    not a reason to try a different opening, and a book that kept retrying would
    be a different personality rather than a fixed opening.
  * From the player's fourth real move on, `choose_v2` calls the unmodified v1
    `choose_move`. Book and v1 are the same personality for every move after the
    opening, so any measured difference belongs to the opening alone.

`choose_v2` is a wrapper, not a subclass: `ai/heuristics/optimizer.py` is handed a brain to
call and is never subclassed or edited. `OptimizerBrain` holds no state that
depends on the move count or on history - its `context` is a pure function of
the board, its `restrict` reads piece sizes, `mistake_rate` is 0 and
`uses_lookahead` is false - so letting it observe the book's own stones is
exactly as valid as letting it observe any other stones.

Legal moves are computed on the real coordinates of the `State` and are never
rotated, because `reports/g_report.md` section G5 showed that a normalised view
answers the opening question with the wrong corner.
"""
import random

import ai
import engine
from config import B, OWNER_CORNER
from rl.actions import index_to_move, legal_indices

# (step, piece name, first square offset, second square offset), both offsets
# counted in units of `d` from the player's own corner.
BOOK = (
    (0, "Z5", 0, 2),
    (1, "V5", 3, 5),
    (2, "W5", 6, 8),
)
BOOK_STEPS = len(BOOK)

# A marker bit that separates this module's random-number namespace from the
# per-seat streams `rl/paired.py` hands to the other players. Both namespaces
# are built from the game seed by shifts rather than by addition, so no
# combination of (seat, step) can alias another, and the marker's presence tells
# a book draw from a personality draw by inspection.
_BOOK_MARKER = 0x80


def book_seed(game_seed, owner, step):
    """The generator a book step draws from: a function of these three inputs and
    of nothing else. No global RNG is touched, so the same three inputs always
    pick the same move and two workers on the same seed cannot disagree."""
    return (game_seed << 8) | _BOOK_MARKER | ((owner & 0x3) << 4) | (step & 0xF)


def corner_and_direction(owner):
    """`c = OWNER_CORNER[owner]` and `d`, the unit diagonal step pointing at the
    centre of the board.

    Both come from the corner. A corner sits on an edge, so the sign of the step
    towards the centre is the sign of `centre - corner`, which is the same as
    stepping towards the middle index. It is written that way so the four cases
    cannot drift apart if the board size ever changes.
    """
    cx, cy = OWNER_CORNER[owner]
    dx = 1 if 2 * cx < B - 1 else -1
    dy = 1 if 2 * cy < B - 1 else -1
    return (cx, cy), (dx, dy)


def book_squares(owner, step):
    """The two squares this player's step `step` has to cover."""
    (cx, cy), (dx, dy) = corner_and_direction(owner)
    _k, _name, first, second = BOOK[step]
    return ((cx + first * dx, cy + first * dy),
            (cx + second * dx, cy + second * dy))


def book_piece(step):
    return BOOK[step][1]


def _bit(square):
    x, y = square
    return 1 << (x + y * B)


def book_candidates(state, owner, step):
    """Every legal move of the specified piece covering both named squares.

    Sorted action indices, because a uniform draw over an unsorted list would
    depend on the enumeration order of whatever produced it. Computed on the
    real coordinates of `state`; `owner` is passed explicitly so the same
    function can be asked about a player who is not to move.
    """
    piece_idx = engine.PIECE_IDX[book_piece(step)]
    a, b = book_squares(owner, step)
    need = _bit(a) | _bit(b)
    out = []
    for index in legal_indices(state, owner):
        idx = int(index)
        got, oi, base = index_to_move(idx)
        if got != piece_idx:
            continue
        if (engine.ORIENTS[piece_idx][oi]["m"] << base) & need == need:
            out.append(idx)
    return out


# --------------------------------------------------------------------------
# per-game book state
# --------------------------------------------------------------------------

class BookTracker:
    """How far each player has got through their own opening book.

    One tracker per game, shared by all four seats, so "is this player still
    following the book" is a field rather than something recomputed. `own_moves`
    counts **real** moves: a pass is not a move and must not shift a player's
    book step, or a blocked opening would silently become a different book.
    """

    def __init__(self, game_seed):
        self.game_seed = game_seed
        self.own_moves = [0, 0, 0, 0]
        self.abandoned_at = [None, None, None, None]
        self.used = [0, 0, 0, 0]

    def note_move(self, owner):
        self.own_moves[owner] += 1

    def active(self, owner):
        """Is the book still in play for this player?"""
        return (self.abandoned_at[owner] is None
                and self.own_moves[owner] < BOOK_STEPS)

    def current_step(self, owner):
        return self.own_moves[owner]

    def candidates(self, state, owner):
        """The candidate set, or None when the book does not apply at all.

        None means "ask v1"; an empty list means "the book applied and found
        nothing", which is the signal to abandon. Conflating the two would make a
        player who has run out of book steps keep trying them.
        """
        if not self.active(owner):
            return None
        return book_candidates(state, owner, self.own_moves[owner])

    def abandon(self, owner):
        self.abandoned_at[owner] = self.own_moves[owner]

    def draw(self, candidates, owner):
        """A uniform draw from the candidates, from this player's own stream."""
        step = self.own_moves[owner]
        pick = random.Random(book_seed(self.game_seed, owner,
                                       step)).randrange(len(candidates))
        return candidates[pick]

    def summary(self):
        return {"own_moves": list(self.own_moves),
                "abandoned_at": list(self.abandoned_at),
                "book_moves_used": list(self.used)}


# --------------------------------------------------------------------------
# the v2 choice
# --------------------------------------------------------------------------

def to_act_move(move):
    """`(piece_idx, oi, base)` -> the `(name, oi, x, y)` `Game.act` wants."""
    piece_idx, oi, base = move
    return (engine.PIECE_ORDER[piece_idx], oi, base % B, base // B)


def act_to_engine_move(name, oi, x, y):
    return (engine.PIECE_IDX[name], oi, x + y * B)


def hand_names(state, owner):
    bits = state.hand_bits[owner]
    return [engine.PIECE_ORDER[i] for i in range(engine.N_PIECES)
            if bits >> i & 1]


def must_cover(state, owner):
    return OWNER_CORNER[owner] if state.own_bits[owner] == 0 else None


def reach(state, board, owner):
    return None if state.own_bits[owner] == 0 else board.reach(owner)


def choose_v2(state, board, owner, brain, rng, tracker, *, other_brains=None,
              trace=None):
    """The v2 choice for `owner` on `state`, as an engine move or None.

    `board` is a `Board` showing the same position, because v1's `choose_move`
    takes one. Returns `(move, info)`; `info["source"]` is `"book"` or `"v1"`,
    and on a book move it also carries the step and the size of the candidate
    set the move was drawn from, which is what the report needs to say how much
    choice the book actually offered.

    The book candidate set is filtered out of `legal_indices(state, owner)`, so a
    book move is legal by construction; the membership check below is a cheap
    guard against a mistake in the filter, not a rule of the book.
    """
    candidates = tracker.candidates(state, owner)
    if candidates is not None:
        if not candidates:
            tracker.abandon(owner)
        else:
            index = tracker.draw(candidates, owner)
            move = index_to_move(int(index))
            if index not in legal_indices(state, owner):   # pragma: no cover
                raise RuntimeError("book move %d is not legal for owner %d"
                                   % (index, owner))
            step = tracker.own_moves[owner]
            tracker.used[owner] += 1
            return move, {"source": "book", "step": step,
                          "n_candidates": len(candidates), "index": int(index)}
    mv = ai.choose_move(board, hand_names(state, owner), owner, brain, rng,
                        other_brains=other_brains,
                        must_cover=must_cover(state, owner),
                        reach=reach(state, board, owner),
                        other_must_cover={o: must_cover(state, o)
                                          for o in range(4)},
                        other_reach={o: reach(state, board, o)
                                     for o in range(4)},
                        trace=trace)
    if mv is None:
        return None, {"source": "v1"}
    return act_to_engine_move(mv[0], mv[1], mv[2], mv[3]), {"source": "v1"}


# --------------------------------------------------------------------------
# what the report needs to say
# --------------------------------------------------------------------------

def book_geometry():
    """The derived corner, direction and the six required squares, per player.

    Reported rather than assumed: `plan4.md` says the corners must be derived
    and not written out, and the only way to show that is to print what the
    derivation produced.
    """
    out = {}
    for owner in range(4):
        (cx, cy), (dx, dy) = corner_and_direction(owner)
        out[owner] = {
            "corner": [cx, cy],
            "direction": [dx, dy],
            "steps": [{"step": k, "piece": book_piece(k),
                       "squares": [list(s) for s in book_squares(owner, k)]}
                      for k in range(BOOK_STEPS)],
        }
    return out


def align_to(state, owner):
    """`state` with the turn handed to `owner`.

    `engine.apply_move` plays for `state.to_move`, so enumerating a book for a
    player who is not next needs the turn moved first. The turn pointer is
    rebuilt the way the engine does it, by scanning owners in id order for
    somebody who can still move, rather than by asking for a player the rotation
    would not have offered the turn to. The resulting state is a real reachable
    one, which is the point: a book line explored through a hand-edited turn
    order would not be a line anybody can play.
    """
    current = state
    while current.to_move != owner:
        current = engine.pass_turn(current)
    return current


def all_sequences(state, owner):
    """Every complete three-step book line for `owner`, with the final state.

    Each line is walked with the engine's own `apply_move`, which raises on an
    illegal move, so "all of these lines are legal" is a statement the engine
    enforces rather than one this module asserts about itself. Returns a list of
    `(moves, final_state)` where `moves` is a tuple of action indices.

    Between the player's own moves the turn is passed on the real way, so a
    candidate that the engine would refuse after three passes is not counted as
    a book line.
    """
    lines = []

    def walk(k, moves, current):
        for index in book_candidates(current, owner, k):
            nxt = engine.apply_move(current, index_to_move(int(index)))
            if k + 1 == BOOK_STEPS:
                lines.append((moves + (int(index),), nxt))
            else:
                walk(k + 1, moves + (int(index),), align_to(nxt, owner))

    walk(0, (), align_to(state, owner))
    return lines
