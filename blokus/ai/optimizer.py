"""Optimizer: plays the largest piece first, then picks the placement that
keeps the most placeable space.

The objective function measures exactly one thing -- **how many placeable
cells you still have after the move**. Scoring rewards "fewer leftovers", so
the number of cells you can still play is the score, and playing out one
5-cell piece is worth 5 cells; that is why the big pieces go first, while
among pieces of the same size the one leaving more placeable space wins.

The urgent rule is its safety valve: when a piece is down to its last single
placement, play it first. Once that spot is blocked off, the piece stays in
hand forever -- under this scoring that amounts to handing over several cells
for nothing.

Likewise no mistake rolls and no opponent prediction, for the same reasons as
`intruder`.
"""
from . import formulas as F
from .base import OPTIMIZER_KEY, Brain
from pieces import MASTER

# The optimizer's urgency threshold: when a piece is down to this many
# placements, play it out first.
URGENT_PLACES = 1
# Safety cap for stage 2 (the expensive evaluation).
OPTIMIZER_SCAN = 1200


class OptimizerBrain(Brain):
    """Efficiency personality: plays the largest piece first, then picks the
    placement that keeps the most placeable space.

    The objective function measures exactly one thing -- **how many
    placeable cells you still have after the move**. Scoring rewards "fewer
    leftovers", so the number of cells you can still play is the score, and
    playing out one 5-cell piece is worth 5 cells; that is why the big pieces
    go first, while among pieces of the same size the one leaving more
    placeable space wins.

    The urgent rule is its safety valve: when a piece is down to its last
    single placement, play it first. Once that spot is blocked off, the piece
    stays in hand forever -- under this scoring that amounts to handing over
    several cells for nothing.
    """

    key = OPTIMIZER_KEY
    uses_lookahead = False

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        self.mistake_rate = 0.0

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = F.board_context(board, hand_names, owner, must_cover)
        # On the very first move, "down to a single placement" is an artifact
        # of the corner rule rather than a signal that the board has
        # tightened up -- early on, the number of placements is naturally
        # small (O4 has only one placement at all). So the urgent rule does
        # not apply to the opening, otherwise the optimizer would spend its
        # first move on O4 / I1 instead of a 5-cell piece.
        if must_cover is not None:
            ctx["urgent"] = []
        else:
            counts = F.placement_counts(board, hand_names, owner, must_cover)
            # When several pieces are down to one placement, save the big one
            # first: if you cannot save them all anyway, leaving 5 fewer cells
            # on the board hurts less.
            ctx["urgent"] = sorted((n for n, k in counts.items()
                                    if 0 < k <= URGENT_PLACES),
                                   key=lambda n: -MASTER[n]["size"])
        return ctx

    def restrict(self, cands, ctx):
        """The urgent rule comes first, then the largest playable piece.

        Both are soft restrictions: if nothing survives the filter we fall back
        to the original candidate list, so this never lets `choose_move`
        return None.
        """
        if ctx["urgent"]:
            keep = set(ctx["urgent"])
            sub = [c for c in cands if c[1] in keep]
            if sub:
                return sub
        by_size = {}
        for c in cands:
            by_size.setdefault(F.ODIRS[c[1]][c[2]]["size"], []).append(c)
        return by_size[max(by_size)] if by_size else cands

    def rescore(self, cands, ctx):
        scored = [(F.place_state(ctx["board"], n, oi, b, ctx)[4].bit_count(),
                   n, oi, b)
                  for _s, n, oi, b in cands[:OPTIMIZER_SCAN]]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored
