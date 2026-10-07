"""This layer answers one question: does the optimizer play better when, from its
fourth move on, it maximises *my playable cells after the move* minus *the
strongest opponent's playable cells after the move* instead of just the first
term?

The first term is v1's objective, unchanged: the count of empty squares that
diagonally touch one of my own stones and share no edge with one of them, on the
board after my move. It is `F.place_state(...)[4].bit_count()`
(`ai/heuristics/optimizer.py:91`), the same expression for the same candidate, so v3
replaces exactly one number and leaves the rest of the pipeline alone.

The second term needs a decision the plan pins down rather than leaves open:
"how many options does *that* player have" is v1's expression evaluated at the
**null move**, i.e. the playable cells they hold without having played anything
into this position. That is not a new quantity - `F.place_state` with
`placed == 0` returns `need0 & empt & ~avoid0`, which is exactly the
`legal_before` field `F.board_context` already computes, and
`tests/test_rl_v3.py` pins the identity against the oracle. Choosing the null
move rather than "the best move they could make" is what makes the term
comparable across players and affordable: the latter would mean a nested search
over the opponent's whole candidate set for each of my ~94 candidates.

`A_j*(after)` therefore falls out of a mask:

    A_j*(after) = (playable_before_j & ~placed).bit_count()

My move can only *remove* squares from `j*`'s playable set - `F.own_reach` reads
`board.owner_bits[owner]`, so my stones never enter `j*`'s need or avoid set,
they only ever cover squares that were playable for them. `tests/test_rl_v3.py`
checks that identity on real positions against a board rebuilt through
`Board.place`, so the cheap form is verified rather than trusted.

`j*` is chosen **before** the move, from the pre-move counts of the three
opponents, and is the same `j*` for every candidate of that decision. That is
deliberate and is what the plan requires: re-deciding per candidate would let v3
compare "damage to the current leader" against "damage to whoever just became the
leader", which is two different questions summed together. The consequence is
that a move which hands the lead to a fourth player is scored against the leader
at the time - see the anomaly section of the report.

Everything else is v1: the urgent rule, the largest-piece restriction, the
`OPTIMIZER_SCAN` cap, the shortlist, the corner band, and `mistake_rate = 0`.
The only score that changes is the one `rescore` returns.

`v3_diff_weight` is the weight on the differential term. It exists because
"with the differential term switched off, v3 must be v1" is a claim worth
testing rather than asserting: at 0 the score is `A_me - 0 * A_j*`, numerically
`A_me`, and the same stable sort over the same candidate order must reproduce
v1's pick move for move.

`choose_v3` is a wrapper, not a subclass of anything in `ai/`: `ai/heuristics/optimizer.py`
is never edited. It reuses the opening book from `rl/opening.py` unchanged - the
same seeds, the same candidate filter and the same abandonment rule - so any
measured difference belongs to the scoring change and nothing else.
"""
import ai
from ai import formulas as F
from ai.heuristics.optimizer import OPTIMIZER_SCAN, OptimizerBrain
from config import CLOCKWISE_OWNERS
from rl.actions import index_to_move, legal_indices
from rl.opening import (act_to_engine_move, book_candidates, hand_names,
                        must_cover, reach)

# The weight on `A_j*(after)`. 1.0 is the version under test; 0.0 is the
# degenerate identity case the test file pins against v1.
DEFAULT_DIFF_WEIGHT = 1.0


# --------------------------------------------------------------------------
# the opponent bookkeeping
# --------------------------------------------------------------------------

def playable_before(board, owner):
    """`owner`'s playable cells in this position: the null-move value of v1's
    expression.

    `F.own_reach` already intersects both halves with `empty_bits`, so `need` is
    "empty squares diagonally touching one of their stones" and `avoid` is "empty
    squares edge-sharing one of their stones"; the playable set is need and not
    avoid. Returns 0 for a player who has placed nothing, because `own_reach`
    returns None for them and a player with no stones has no corner contact to
    play from on this board.

    That last case is a documented assumption, not a measurement: over 320 real
    post-book decisions no opponent was ever in it (a player reaches its fourth
    own move only after every other seat has had at least three turns, so a
    player with no stones there has been stuck since its first turn).
    """
    reach = F.own_reach(board, owner)
    if reach is None:
        return 0
    return reach.need & ~reach.avoid


def after_moves(board, weights=None):
    """Every opponent's playable-cell mask, keyed by owner.

    One `own_reach` per opponent, computed once per decision. `own_reach` is used
    rather than `F.board_context` because v3 needs the mask and not the whole
    context: `board_context` also builds `cells_to_vertices(anchor)`, which costs
    8.8 us against `own_reach`'s 2.4 us and would be pure waste here.
    """
    return {o: playable_before(board, o) for o in range(4)}


def turn_order_after(owner, turn_order):
    """The three opponents in the order they will be offered the turn after
    `owner`.

    `turn_order` is a parameter rather than a constant read inside the brain so
    the tie-break rule is a visible input and can be tested with a known order;
    `CLOCKWISE_OWNERS` is the default because that is the order the paired
    experiment plays.
    """
    i = turn_order.index(owner)
    return [turn_order[(i + k) % len(turn_order)] for k in (1, 2, 3)]


def strongest_opponent(masks, owner, turn_order):
    """The `j*` of the differential term, decided before the move.

    The largest playable-cell count among the other three; ties go to the player
    closest to us in the turn order, next player first. When all three counts are
    zero `max` picks the first in that same order and the differential term is
    then identically zero for every candidate, which is the behaviour the plan
    asks for without needing a special case - `0 & ~placed` is `0` whatever `m`
    is.
    """
    order = turn_order_after(owner, turn_order)
    counts = {o: masks[o].bit_count() for o in order}
    best = max(counts.values())
    for o in order:
        if counts[o] == best:
            return o
    return order[0]           # pragma: no cover - max always hits a member


# --------------------------------------------------------------------------
# the brain
# --------------------------------------------------------------------------

class V3Brain(OptimizerBrain):
    """v1 with the differential term added to the score `rescore` returns.

    A subclass of `OptimizerBrain` rather than of `Brain`, so the urgent rule,
    the largest-piece restriction and the personality's key, profile and
    `mistake_rate = 0` are inherited rather than copied. `ai/heuristics/optimizer.py` itself
    is not edited or shadowed: this class adds `context` and replaces `rescore`,
    and `restrict` is untouched.
    """

    def __init__(self, key, profile, turn_order=CLOCKWISE_OWNERS,
                 diff_weight=DEFAULT_DIFF_WEIGHT):
        OptimizerBrain.__init__(self, key, profile)
        # Held on the instance rather than read from a module global at scoring
        # time, so a test can build a weight-0 brain without mutating anything
        # another test can see.
        self.turn_order = tuple(turn_order)
        self.v3_diff_weight = float(diff_weight)

    def context(self, board, hand_names_, owner, must_cover_, reach_):
        ctx = OptimizerBrain.context(self, board, hand_names_, owner,
                                     must_cover_, reach_)
        masks = after_moves(board)
        jstar = strongest_opponent(masks, owner, self.turn_order)
        ctx["v3_masks"] = masks
        ctx["v3_jstar"] = jstar
        return ctx

    def opponent_after(self, ctx, placed):
        """`A_j*(after)` for one candidate, without rebuilding a board.

        `playable_before` is v1's expression at the null move, and the only thing
        my move changes for `j*` is which squares are no longer empty, so the
        post-move playable set is the pre-move set minus the squares I covered.
        """
        return (ctx["v3_masks"][ctx["v3_jstar"]] & ~placed).bit_count()

    def rescore(self, cands, ctx):
        """v1's score for every candidate, minus `A_j*(after)`.

        The candidate range is v1's: `restrict` has already run, and the
        `OPTIMIZER_SCAN` cap is applied to the same prefix of the same order. The
        `A_me` term is computed by calling `F.place_state` exactly as v1's
        `rescore` does, so it is bit-identical to v1 by construction rather than
        by a test agreeing with a re-implementation.
        """
        weight = self.v3_diff_weight
        jmask = ctx["v3_masks"][ctx["v3_jstar"]]
        out = []
        for _s, n, oi, b in cands[:OPTIMIZER_SCAN]:
            a_me = F.place_state(ctx["board"], n, oi, b, ctx)[4].bit_count()
            a_j = (jmask & ~(F.ODIRS[n][oi]["m"] << b)).bit_count()
            out.append((a_me - weight * a_j, n, oi, b))
        out.sort(key=lambda t: t[0], reverse=True)
        return out


def make_brain(profile, turn_order=CLOCKWISE_OWNERS,
               diff_weight=DEFAULT_DIFF_WEIGHT):
    """A v3 brain for `profile`, the tested personality staying `optimizer`.

    The profile is v1's own: v3 changes the objective function, not the weights,
    so a v1 brain and a v3 brain given the same profile differ in exactly one
    thing. That is what makes "weight 0 is v1" and "v3 vs v2 differ only from
    move four on" both checkable.
    """
    return V3Brain(OptimizerBrain.key, profile, turn_order=turn_order,
                   diff_weight=diff_weight)


# --------------------------------------------------------------------------
# the v3 choice
# --------------------------------------------------------------------------

def book_state(state, owner, tracker):
    """The book candidate set for `owner`, or None when the book no longer
    applies. Returns an empty list when the book applies and found nothing, which
    is the abandonment signal."""
    if not tracker.active(owner):
        return None
    return book_candidates(state, owner, tracker.current_step(owner))


def choose_v3(state, board, owner, brain, rng, tracker, *, other_brains=None,
              trace=None):
    """The v3 choice for `owner` on `state`, as an engine move or None.

    Identical to `rl.opening.choose_v2` for the first three moves: the same
    `BookTracker`, the same per-step seed derivation, the same candidate filter
    and the same rule that an empty set abandons the book for the rest of the
    game rather than retrying the next step. From the fourth real move on the
    difference is only in which brain is asked, so the two modules cannot drift
    apart on the opening.

    Returns `(move, info)` with `info["source"]` in `{"book", "v3"}`, plus the
    step and candidate count on a book move.
    """
    candidates = book_state(state, owner, tracker)
    if candidates is not None:
        if not candidates:
            tracker.abandon(owner)
        else:
            index = tracker.draw(candidates, owner)
            move = index_to_move(int(index))
            if index not in legal_indices(state, owner):   # pragma: no cover
                raise RuntimeError("book move %d is not legal for owner %d"
                                   % (index, owner))
            step = tracker.current_step(owner)
            tracker.used[owner] += 1
            return move, {"source": "book", "step": step,
                          "n_candidates": len(candidates),
                          "index": int(index)}
    mv = ai.choose_move(board, hand_names(state, owner), owner, brain, rng,
                        other_brains=other_brains,
                        must_cover=must_cover(state, owner),
                        reach=reach(state, board, owner),
                        other_must_cover={o: must_cover(state, o)
                                          for o in range(4)},
                        other_reach={o: reach(state, board, o) for o in range(4)},
                        trace=trace)
    if mv is None:
        return None, {"source": "v3"}
    return act_to_engine_move(mv[0], mv[1], mv[2], mv[3]), {"source": "v3"}


# --------------------------------------------------------------------------
# what the report needs to say
# --------------------------------------------------------------------------

def differential_is_constant(state, board, owner, brain, seed=0):
    """Over one decision: how many distinct values `A_j*(after)` takes across the
    candidates v1 would score.

    `plan5.md` H1-1 test 5 asks for the fraction of positions where this varies,
    because a term that never varies is a term that cannot change a ranking and
    would make the whole experiment uninformative without failing loudly. The
    value itself is only a report input; the decision never reads it.
    """
    from ai.chooser import _candidates
    names = hand_names(state, owner)
    open_a, block, defend = F.board_feats(board.grid)
    cands = _candidates(board, names, owner, brain.profile,
                        must_cover(state, owner), reach(state, board, owner),
                        open_a, block, defend, board.corner_regions,
                        board.borders, board.empty_bits)
    cands.sort(key=lambda t: t[0], reverse=True)
    ctx = brain.context(board, names, owner, must_cover(state, owner),
                        reach(state, board, owner))
    cands = brain.restrict(cands, ctx)
    jmask = ctx["v3_masks"][ctx["v3_jstar"]]
    scanned = cands[:OPTIMIZER_SCAN]
    seen = set()
    for _s, n, oi, b in scanned:
        seen.add((jmask & ~(F.ODIRS[n][oi]["m"] << b)).bit_count())
    return {"n_candidates": len(scanned), "distinct": len(seen),
            "varies": len(seen) > 1, "jstar": ctx["v3_jstar"],
            "jstar_count": jmask.bit_count()}