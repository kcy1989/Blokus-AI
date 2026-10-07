"""Intruder: uses "crossing" to burrow into the opponent's territory.

Crossing is the win condition of Blokus -- once one diagonal of a 2x2 square
is all yours and the other one has an opponent on it, you have won the game.
The Intruder breaks that goal down into three stages:

  Rule 1  if a crossing piece is still in hand, use only crossing pieces
  Rule 2  otherwise use long-arm pieces (L5 / N5 / I5)
  Rule 3  with none of those left, use a piece that can take a "key cell"
  Rule 4  this move itself completes a crossing -> a large bonus, plus a
          requirement that there is still somewhere to play after crossing
  Rule 5  normal positions: 5 cells first, relaxed to 4 and 3 cells at the
          strategically interesting spots

Every stage is a **soft** limit: if the designated piece class cannot be
placed at all, fall back to the next stage. Made into a hard filter instead,
the candidate list would be emptied, `choose_move` would return None, and the
player would be wrongly judged as having no move and would auto-pass.

This personality does not roll for mistakes (`mistake_rate = 0`): a strategic
bonus stops meaning anything the moment randomness buries it, and only then
are properties like "use only crossing pieces for the first 3 moves" even
checkable. It also does no opponent prediction, because prediction runs on
the weighted score, whose magnitude is completely different from the
objective function used here.
"""
from .. import formulas as F
from ..base import INTRUDER_KEY, Brain

# Safety cap for stage 2 (the expensive evaluation). Once the stage limits
# and the corner-diagonal rule are in play, the candidate list is usually
# down to a few hundred (measured at no more than 400), so this cap almost
# never blocks anything; its only job is to give "simulate a placement for
# every candidate" a definite ceiling on cost.
INTRUDER_SCAN = 1200


class IntruderBrain(Brain):
    """Rule-based personality: picks pieces by extension and placeable space,
    no weighted sum."""

    key = INTRUDER_KEY
    uses_lookahead = False

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        # Rule-based personalities do not roll for mistakes: a strategic bonus
        # stops meaning anything the moment randomness buries it, and only
        # then are properties like "use only crossing pieces for the first
        # 3 moves" even checkable.
        self.mistake_rate = 0.0

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = F.board_context(board, hand_names, owner, must_cover)
        ctx.update(F.crossing_context(board, hand_names, owner))
        return ctx

    def restrict(self, cands, ctx):
        """Stages for rules 1/2/5: crossing pieces -> long arms -> general.

        Every stage is soft: if not a single cell of that piece class can be
        played, fall through to the next one, and the last fallback is the
        full candidate list. Made into a hard filter, the candidate list
        would be emptied, `choose_move` would return None, and the player
        would be wrongly judged as having no move and would auto-pass.
        """
        for cls in (F.LEAPERS, F.STRETCHERS):
            sub = [c for c in cands if c[1] in cls]
            if sub:
                return sub
        return cands

    def rescore(self, cands, ctx):
        scored = []
        for _s, name, oi, base in cands[:INTRUDER_SCAN]:
            scored.append(self._score(name, oi, base, ctx))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _score(self, name, oi, base, ctx):
        od = F.ODIRS[name][oi]
        (own_after, empt_after, need, avoid, legal, fresh,
         anchors) = F.place_geometry(ctx["board"], name, oi, base, ctx)

        ext = F.extension(fresh, anchors)
        usable = F.cells_to_vertices(fresh).bit_count()
        squares = legal.bit_count()

        # Rule 4: this move itself completes the crossing. `pre` holds the
        # crossing squares as they were before the placement, used to discount
        # the square that "was already crossed and merely got touched by this
        # move".
        crossed = False
        for o, m in ctx["opps"]:
            if F.cross_anchors(own_after, m) & (od["touch"] << base) \
                    & ~ctx["pre"][o]:
                crossed = True
                break

        # Rule 3: grab key cells while no crossing has happened yet
        strategic = crossed
        if not crossed and ctx["uncrossed"]:
            strategic = self._sets_up(ctx, od, own_after, empt_after,
                                      need, avoid) or crossed

        s = F.size_bonus(od["size"], strategic)
        s += (F.crossing_bonus(squares, usable) if crossed
              else F.W_SQUARES * min(squares, F.SQ_CAP))
        s += F.W_VERTICES * min(usable, F.SQ_CAP)
        s += F.W_EXT * min(ext, F.EXT_CAP)
        return (s, name, oi, base)

    def _sets_up(self, ctx, od, own_after, empt_after, need, avoid):
        """After this placement, is there a legal move that completes a
        crossing (the "key cell" of rule 3).

        Only checks this piece in hand plus the crossing pieces, and only for
        opponents that have not been crossed yet -- that is the most expensive
        part of this rule, so the remaining opponents are skipped outright.
        """
        reach = F.Reach(need, avoid)
        for o, m in ctx["uncrossed"]:
            key = F.key_cells(own_after, m, empt_after)
            if not key:
                continue
            kb = 0
            for off in od["offs"]:
                kb |= key >> off
            if kb and F._adjoining_bases(od, reach, 0) & kb \
                    & F._free_bases(od, empt_after):
                return True
            sp = ctx["selfpair"][o]
            for n, oi, lod in ctx["leaps"]:
                lkb = 0
                for off in lod["offs"]:
                    lkb |= key >> off
                hit = lkb | sp[(n, oi)]
                if not hit:
                    continue
                if F._adjoining_bases(lod, reach, 0) & hit \
                        & F._free_bases(lod, empt_after):
                    return True
        return False
