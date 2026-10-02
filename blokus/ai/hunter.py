"""Hunter: the optimizer scored as "my playable cells minus the strongest
opponent's", plus a fixed three-move opening book.

This is the stage H teacher, and it is two things stacked:

  * **The scoring** is `rl/v3.py`'s `V3Brain`, unchanged - same expression, same
    tie-break, same candidate range, same `A_me` term. The single number the
    differential replaces is the one it replaces there too.
  * **The opening book** is the book `rl/opening.py` implements, written a second
    time rather than imported.

Why the book is reimplemented instead of imported
-------------------------------------------------
`rl/` sits *above* `ai/`: `rl/opening.py` imports `ai`, and `rl/actions.py`
imports numpy. Importing either from `ai/` would invert the layering and pull
numpy into a package that has none today. So the book is rebuilt against
`ai/formulas` alone. That leaves two implementations of one opening in the tree,
which is real duplication and is deliberate; `tests/test_hunter.py`
cross-checks this one against `rl/opening.py` on real positions so they cannot
drift apart silently.

The book in view coordinates
----------------------------
`corner_and_direction` reads `config.OWNER_CORNER[owner]`, a table of **real**
corners. `engine.normalize` puts the player to move at the top-left of the
rotated view, and `OWNER_CORNER[0]` is `(19, 19)`, so handing a normalised view
to `book_candidates` computes the book for the wrong corner. Everything here is
in real coordinates for that reason. A caller wanting the view-coordinate
candidate set must map it across; `tests/test_hunter.py` does.

How the book is applied without touching the caller
---------------------------------------------------
`ai.choose_move` is called from `ui.py` and `match.py` directly and neither may be
edited, so the book cannot short-circuit the way `rl/opening.choose_v2` does. It
goes through the two personality hooks that already exist:

  * `restrict` narrows the candidate list to the book's candidates. An empty
    result means the book applies and found nothing, which **abandons** the book
    for the rest of the game and returns the untouched list - the same rule
    `rl/opening.py` uses, and soft either way, so the search can never be left
    with no candidates.
  * `rescore` then returns the one drawn candidate, which `choose_move` picks
    because it is the only one in the shortlist.

Returning one candidate rather than two is deliberate. `choose_move` draws from
the caller's `rng` only when the shortlist holds two or more
(`ai/chooser.py:268`), so a one-element shortlist consumes nothing - matching
`rl/opening.py`, whose book path returns before `choose_move` is ever reached.
Hunter and the version stage H1 tested therefore leave the random stream in the
same place.

Counting the player's own moves
-------------------------------
The book step is "how many real moves this player has made", and `choose_move`
does not pass that in. It is counted here instead. `context` is reached only when
the player has at least one legal move - `ai/chooser.py` returns before it
otherwise - so its call count *is* the player's own real-move count. A seat that
passes never reaches `context`, which is exactly the count a driver would keep,
and a seat that can no longer move never moves again, so the two agree.
`must_cover` cannot serve instead: it is a single bit, `None` only while
`placed[owner] == 0`.

The seed
--------
`game_seed` is an optional constructor argument. A driver that knows the game's
seed passes it and gets a fully reproducible opening.

When it is absent - which is the normal-game case, because `Game` stores only an
`rng` and exposes no seed - the seed is derived from that rng's **state** by
`seed_from_rng`. Two properties matter and both are deliberate:

  * `getstate()` only reads. The game shares one stream across all four seats
    (`ui.py` passes `game.rng` to every `choose_move`), so consuming from it would
    change what the other three personalities do. Deriving rather than drawing
    keeps all four untouched.
  * The result is a pure function of the stream position, so the same game seed
    always produces the same opening. That is not a nicety: C4 makes it an
    invariant of the project (`tests/test_optimised_paths.py` asserts that two
    runs of the same seeds give identical moves), and it holds for the other six
    personalities because they consume no per-game seed at all.

`sha256` of `repr(state)` rather than `hash(state)`. `state` is
`(version, a 625-int tuple, gauss_next)`, and `gauss_next` is `None` until
`random.gauss` is called - `hash(None)` is derived from the object's address in
CPython, so a plain `hash` would not be stable across processes while `repr` and
`sha256` are. `repr` of a tuple of ints is itself deterministic.

The state is read **once**, at construction, and cached in `self.game_seed`.
Re-reading it per move would give a different seed on every move, since the
pipeline draws from the caller's `rng` on most turns.
"""
import hashlib
import random

from . import formulas as F
from .base import HUNTER_KEY
from .optimizer import OPTIMIZER_SCAN, OptimizerBrain
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER


def seed_from_rng(rng):
    """A stable book seed derived from an `rng`'s current state, without
    advancing it."""
    digest = hashlib.sha256(repr(rng.getstate()).encode()).digest()
    return int.from_bytes(digest[:8], "big")

# The weight on the differential term, exactly as in `rl/v3.py`. 1.0 is Hunter;
# 0.0 is the degenerate case that must reproduce v1 move for move.
DEFAULT_DIFF_WEIGHT = 1.0

# The book: (step, piece name, first square offset, second square offset), both
# offsets in units of `d` from the player's own corner. Written out rather than
# imported, because importing `rl/opening.py` is the inversion this module
# exists to avoid; `tests/test_hunter.py` checks the two agree.
BOOK = (
    (0, "Z5", 0, 2),
    (1, "V5", 3, 5),
    (2, "W5", 6, 8),
)
BOOK_STEPS = len(BOOK)

# A marker bit keeping the book's random-number namespace apart from anything else
# derived from the game seed. Same layout and value as `rl/opening.py`, so the
# derivation is identical when a driver injects a seed.
_BOOK_MARKER = 0x80


# --------------------------------------------------------------------------
# the book
# --------------------------------------------------------------------------

def book_seed(game_seed, owner, step):
    """The generator a book step draws from: a function of these three inputs and
    nothing else. Shifts rather than addition, so no `(owner, step)` pair can
    alias another."""
    return (game_seed << 8) | _BOOK_MARKER | ((owner & 0x3) << 4) | (step & 0xF)


def corner_and_direction(owner):
    """`c = OWNER_CORNER[owner]` and `d`, the unit diagonal step pointing at the
    centre of the board.

    A corner sits on an edge, so the sign of the step towards the centre is the
    sign of `centre - corner`. Written that way so the four cases cannot drift
    apart if the board size ever changes.
    """
    cx, cy = OWNER_CORNER[owner]
    dx = 1 if 2 * cx < B - 1 else -1
    dy = 1 if 2 * cy < B - 1 else -1
    return (cx, cy), (dx, dy)


def book_squares(owner, step):
    """The two real-coordinate squares this player's step `step` must cover."""
    (cx, cy), (dx, dy) = corner_and_direction(owner)
    _k, _name, first, second = BOOK[step]
    return ((cx + first * dx, cy + first * dy),
            (cx + second * dx, cy + second * dy))


def book_piece(step):
    return BOOK[step][1]


def bases_covering_all(od, square_bits):
    """Bases at which this orientation covers **every** square in the list.

    One required square at a time: for a square `s`, bit `(s - o)` of `s >> o` is
    set exactly when `base + o` lands on `s`, so OR-ing over the offsets gives the
    bases covering `s`, and AND-ing over the required squares gives the bases
    covering all of them. `od["valid"]` drops the row-wrap cases, the same guard
    `_adjoining_bases` uses for the corner rule.

    Every element of `square_bits` must be a **single** bit. Handing it one mask
    holding several squares would ask for "covers s or covers t" - a different and
    much weaker set. That mistake produced five candidates where there are two
    while this was being written, which is why it is spelled out.
    """
    hit = od["valid"]
    for bit in square_bits:
        one = 0
        for o in od["offs"]:
            one |= bit >> o
        hit &= one
    return hit


def book_candidates(board, owner, step):
    """Every legal `(name, oi, base)` the book offers at `step`, in a fixed order.

    `_legal_bases` supplies the corner-contact rule and the empty-square test; the
    book's own requirement is the extra `bases_covering_all` term. A fixed order
    matters because the draw is uniform over this list and must not depend on
    anything else's enumeration.
    """
    name = book_piece(step)
    a, b = book_squares(owner, step)
    want = (1 << (a[0] + a[1] * B), 1 << (b[0] + b[1] * B))
    reach = F.own_reach(board, owner)
    cbit = 0
    if reach is None:
        cx, cy = OWNER_CORNER[owner]
        cbit = 1 << (cx + cy * B)
    empt = board.empty_bits
    out = []
    for oi, od in F.ODIRS[name].items():
        mask = (F._legal_bases(od, empt, reach, cbit)
                & bases_covering_all(od, want))
        while mask:
            low = mask & -mask
            mask ^= low
            out.append((oi, low.bit_length() - 1))
    return [(name, oi, b) for oi, b in sorted(out)]


def book_draw(game_seed, owner, step, n_candidates):
    """Which candidate index this step plays.

    A **fresh** `random.Random`, so the stream the caller passed in is untouched -
    the property `rl/opening.py` relies on.
    """
    if n_candidates <= 1:
        return 0
    return random.Random(book_seed(game_seed, owner, step)).randrange(n_candidates)


# --------------------------------------------------------------------------
# the differential scoring, unchanged from rl/v3.py
# --------------------------------------------------------------------------

def playable_before(board, owner):
    """`owner`'s playable cells: the null-move value of v1's expression.

    `F.own_reach` already intersects both halves with `empty_bits`, so `need` is
    "empty squares diagonally touching one of their stones" and `avoid` is "empty
    squares edge-sharing one of their stones"; the playable set is need and not
    avoid. A player with no stones has no corner contact to play from on this
    board and scores zero - a documented assumption, not a guarantee: a seat
    reaches its fourth own move only after every other seat has had at least
    three turns, and over 320 real post-book decisions it never arose.
    """
    reach = F.own_reach(board, owner)
    if reach is None:
        return 0
    return reach.need & ~reach.avoid


def after_moves(board):
    """Every opponent's playable-cell mask, keyed by owner.

    `own_reach` rather than `F.board_context`, because only the mask is wanted and
    `board_context` also builds `cells_to_vertices(anchor)` at three and a half
    times the cost.
    """
    return {o: playable_before(board, o) for o in range(4)}


def turn_order_after(owner, turn_order):
    """The three opponents in the order they will be offered the turn after
    `owner`. A parameter rather than a constant so the tie-break is a visible
    input that a test can drive with a known order."""
    i = turn_order.index(owner)
    return [turn_order[(i + k) % len(turn_order)] for k in (1, 2, 3)]


def strongest_opponent(masks, owner, turn_order):
    """The `j*` of the differential term, decided **before** the move.

    Largest playable-cell count among the other three; ties go to the player
    closest to us in the turn order, next player first. When all three counts are
    zero, `max` reaches the first in that same order and the differential term is
    then identically zero for every candidate - `0 & ~placed` is `0` whatever the
    move is - so the degenerate case needs no special branch here.
    """
    order = turn_order_after(owner, turn_order)
    counts = {o: masks[o].bit_count() for o in order}
    best = max(counts.values())
    for o in order:
        if counts[o] == best:
            return o
    return order[0]           # pragma: no cover - max always hits a member


# --------------------------------------------------------------------------
# the personality
# --------------------------------------------------------------------------

class HunterBrain(OptimizerBrain):
    """The optimizer with the opponent-differential term and the opening book.

    A subclass of `OptimizerBrain`, so the urgent rule, the largest-piece
    restriction, the profile and `mistake_rate = 0` are inherited rather than
    copied. `ai/optimizer.py` is neither edited nor shadowed.
    """

    # `ai/registry.py` reads this to decide whether a driver's `game_seed` can be
    # handed to the class, instead of hard-coding a key comparison over there.
    accepts_game_seed = True

    def __init__(self, key, profile, turn_order=CLOCKWISE_OWNERS,
                 diff_weight=DEFAULT_DIFF_WEIGHT, game_seed=None,
                 seed_rng=None):
        OptimizerBrain.__init__(self, key, profile)
        self.turn_order = tuple(turn_order)
        self.diff_weight = float(diff_weight)
        # Resolved once, here, and cached. Three sources in order of preference:
        # an explicit seed, the rng the caller handed over (read, never drawn
        # from), and only then an independent SystemRandom draw - which is the
        # path a test takes when it builds a brain with no game at all, and is
        # the one case that is not reproducible.
        if game_seed is not None:
            self.game_seed = int(game_seed)
        elif seed_rng is not None:
            self.game_seed = seed_from_rng(seed_rng)
        else:
            self.game_seed = random.SystemRandom().randrange(2 ** 31)
        self.game_seed_source = ("argument" if game_seed is not None
                                 else "rng" if seed_rng is not None else "system")
        # Book progress, kept on the brain because `choose_move` is stateless.
        self.book_step = 0
        self.book_abandoned = False
        self.book_moves_used = 0

    # -- book ---------------------------------------------------------------

    def book_active(self):
        return (not self.book_abandoned) and self.book_step < BOOK_STEPS

    def book_candidates_now(self, board, owner):
        """The book's candidates for the current step, or None when the book does
        not apply at all.

        None means "not a book step"; an empty list means the book applied and
        found nothing, which is the abandonment signal. Conflating the two would
        let a player who has run out of steps keep trying them.
        """
        if not self.book_active():
            return None
        return book_candidates(board, owner, self.book_step)

    def summary(self):
        return {"game_seed": self.game_seed, "book_step": self.book_step,
                "book_abandoned": self.book_abandoned,
                "book_moves_used": self.book_moves_used}

    # -- Brain interface ----------------------------------------------------

    def context(self, board, hand_names_, owner, must_cover_, reach_):
        ctx = OptimizerBrain.context(self, board, hand_names_, owner,
                                     must_cover_, reach_)
        masks = after_moves(board)
        jstar = strongest_opponent(masks, owner, self.turn_order)
        cands = self.book_candidates_now(board, owner)
        pick = None
        if cands is not None:
            if not cands:
                # the book applies and found nothing: abandon for the rest of the
                # game rather than retrying the next step
                self.book_abandoned = True
            else:
                index = book_draw(self.game_seed, owner, self.book_step,
                                  len(cands))
                pick = cands[index]
                self.book_moves_used += 1
        ctx["hunter_masks"] = masks
        ctx["hunter_jstar"] = jstar
        ctx["hunter_book_candidates"] = cands
        ctx["hunter_book_pick"] = pick
        # Counted here, after the context is built, because `context` runs only
        # on a turn where this player had a legal move: its call count is that
        # player's own real-move count, so the next call sees the next book step.
        self.book_step += 1
        return ctx

    def opponent_after(self, ctx, placed):
        """`A_j*(after)` for one candidate, without rebuilding a board.

        `F.own_reach` reads `board.owner_bits[owner]`, so my stones never enter
        `j*`'s need or avoid set - they only ever cover squares that were
        playable for them. Hence the pre-move set minus the squares I took.
        """
        return (ctx["hunter_masks"][ctx["hunter_jstar"]] & ~placed).bit_count()

    def restrict(self, cands, ctx):
        """The book filter, then v1's urgent rule and largest piece.

        Soft throughout: a book step that somehow intersects nothing falls through
        to v1 rather than emptying the candidate list.
        """
        cands = self._book_only(cands, ctx)
        return OptimizerBrain.restrict(self, cands, ctx)

    def rescore(self, cands, ctx):
        """On a book step, the one drawn candidate. Otherwise v1's score minus
        `A_j*(after)`, over exactly v1's candidate range.

        `A_me` comes from calling `F.place_state` precisely as v1's `rescore`
        does, so it is bit-identical to v1 by construction rather than by a test
        agreeing with a re-implementation.

        The book pick is only forced when it is *in* `cands`. `restrict` hands
        back the book's candidates intersected with the real, hand-filtered
        candidate list, and falls back to the untouched list when that
        intersection is empty - which is what happens when the book asks for a
        piece the player has already spent. Forcing the pick regardless of that
        decision put a move with a piece the player does not hold at the top of
        the shortlist, and `choose_move` returns the shortlist's head.
        """
        pick = ctx.get("hunter_book_pick")
        if pick is not None and pick in {(c[1], c[2], c[3]) for c in cands}:
            return [(0.0, pick[0], pick[1], pick[2])]
        weight = self.diff_weight
        jmask = ctx["hunter_masks"][ctx["hunter_jstar"]]
        out = []
        for _s, n, oi, b in cands[:OPTIMIZER_SCAN]:
            a_me = F.place_state(ctx["board"], n, oi, b, ctx)[4].bit_count()
            a_j = (jmask & ~(F.ODIRS[n][oi]["m"] << b)).bit_count()
            out.append((a_me - weight * a_j, n, oi, b))
        out.sort(key=lambda t: t[0], reverse=True)
        return out

    def _book_only(self, cands, ctx):
        """`cands` narrowed to the book's set, or unchanged when there is no book.

        `cands` holds `(score, name, oi, base)` tuples, so the book's
        `(name, oi, base)` triples are matched on the last three fields.
        """
        cands_in_book = ctx.get("hunter_book_candidates")
        if not cands_in_book:
            return cands
        wanted = set(cands_in_book)
        sub = [c for c in cands if (c[1], c[2], c[3]) in wanted]
        return sub if sub else cands


def make_brain(profile, game_seed=None, seed_rng=None,
               turn_order=CLOCKWISE_OWNERS, diff_weight=DEFAULT_DIFF_WEIGHT,
               key=HUNTER_KEY):
    """A `HunterBrain` for `profile`.

    The profile is the optimizer's own: Hunter changes the objective function and
    the opening, not the weights, so an optimizer brain and a Hunter given the
    same profile differ in exactly those two things.
    """
    return HunterBrain(key, profile, turn_order=turn_order,
                       diff_weight=diff_weight, game_seed=game_seed,
                       seed_rng=seed_rng)