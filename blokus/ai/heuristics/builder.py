"""Builder: estimates how many cells of the remaining hand can never be
played again.

The main term of the objective function is `formulas.pack_lost` -- FFD-pack
the remaining hand into the live-region bins of the post-move board, and
count the cells that will not fit with a negative weight. Since scoring
rewards "fewer leftovers", and leftover cells have exactly one source (you
still hold a piece, but the board has no legal placement for it), this
quantity is on the same scale as the score.

The size preference (`BUILDER_W_SIZE`) and the total placeable space
(`BUILDER_W_TOTAL`) are only secondary terms: the former makes it look like
it also favours 5 cells in ordinary positions, the latter makes it pick the
roomier of two placements that are equally "nothing gets dropped". Both are
deliberately too small to ever outweigh the main term -- hence no stages and
no urgent rule: those are natural consequences of the main term, and adding
them only turns the thing into two sets of rules fighting each other.

Likewise no mistake rolls and no opponent prediction, for the same reasons as
`intruder`.
"""
from .. import formulas as F
from ..base import BUILDER_KEY, Brain
from pieces import MASTER

# The builder's cap for stage 2. It has no stage filter, and what it scores is
# the **full** candidate list (thousands in practice), so this cap really does
# block things: first screen with the cheap "placeable cell count", then run
# the closure and packing on the top BUILDER_SCAN.
BUILDER_SCAN = 800
# The builder's weights. `BUILDER_W_LOST` is measured in cells (0-89), while
# the other two top out at only +/-2 and 4, so "dropping 5 cells" always
# outweighs any difference in size or total space -- which matches the
# scoring: leftovers are the score. The size preference only makes it look
# like it also favours 5 cells in ordinary positions.
BUILDER_W_LOST = 1.0
BUILDER_W_SIZE = 0.35
BUILDER_W_TOTAL = 0.05


class BuilderBrain(Brain):
    """Builder: estimates how many cells of the remaining hand can never be
    played again.

    The main term of the objective function is `pack_lost` -- FFD-pack the
    remaining hand into the live-region bins of the post-move board, and
    count the cells that will not fit with a negative weight. Since scoring
    rewards "fewer leftovers", and leftover cells have exactly one source (you
    still hold a piece, but the board has no legal placement for it), this
    quantity is on the same scale as the score.

    The size preference (`BUILDER_W_SIZE`) and the total placeable space
    (`BUILDER_W_TOTAL`) are only secondary terms: the former makes it look
    like it also favours 5 cells in ordinary positions, the latter makes it
    pick the roomier of two placements that are equally "nothing gets
    dropped". Both are deliberately too small to ever outweigh the main term
    -- hence no stages and no urgent rule: those are natural consequences of
    the main term, and adding them only turns the thing into two sets of
    rules fighting each other.
    """

    key = BUILDER_KEY
    uses_lookahead = False

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        self.mistake_rate = 0.0

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = F.board_context(board, hand_names, owner, must_cover)
        # The remaining hand sizes after each piece class is played out,
        # largest first. Computed once for the whole hand (21 entries) so that
        # every candidate does not run a list.remove. Piece names in the hand
        # never repeat, so using the name as the key is safe.
        sizes = sorted((MASTER[n]["size"] for n in hand_names), reverse=True)
        ctx["sizes_after"] = {name: tuple(sizes[:i] + sizes[i + 1:])
                              for i, name in enumerate(hand_names)}
        return ctx

    def restrict(self, cands, ctx):
        """No stages: see the class docstring, `lost` already absorbs them."""
        return cands

    def rescore(self, cands, ctx):
        """Two-stage. The candidates get no stage filtering, so the first
        stage is a necessary layer, not just a speedup.

        The first stage (all of them, cheap) only computes the placeable
        cell count from `place_state` -- that is the optimizer's already
        validated, correctly directed metric, used as a coarse screen. Only
        the second stage runs the closure, the live-region bins and the
        packing on the top `BUILDER_SCAN`, for the full score.
        """
        board = ctx["board"]
        wide = [(F.place_state(board, n, oi, b, ctx)[4].bit_count(), n, oi, b)
                for _s, n, oi, b in cands]
        wide.sort(key=lambda t: t[0], reverse=True)
        scored = [self._score(n, oi, b, ctx) for _total, n, oi, b
                  in wide[:BUILDER_SCAN]]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _score(self, name, oi, base, ctx):
        board = ctx["board"]
        _own, empt_after, need, avoid, legal = F.place_state(board, name, oi, base, ctx)
        total = legal.bit_count()
        bins = F.fill_components(F.fillable_closure(empt_after, avoid, legal))
        lost = F.pack_lost(ctx["sizes_after"][name], bins)
        od = F.ODIRS[name][oi]
        s = (-BUILDER_W_LOST * lost + BUILDER_W_SIZE * (od["size"] - 3)
             + BUILDER_W_TOTAL * min(total, F.SQ_CAP))
        return (s, name, oi, base)
