"""Shared base types and weight profiles for the personalities.

`Profile` is the six weights of a weighted personality plus a mistake rate;
`Brain` is the "how to read a position" interface. Candidate enumeration,
opponent lookahead and the draw all live in `chooser.choose_move`; a
personality itself does only three things:

    context   quantities in a position that are independent of the candidate,
              computed once up front
    restrict  stage (soft) filter
    rescore   re-rank the scores
"""
from dataclasses import dataclass

# The full personality table: three weighted ones plus four rule-based ones.
INTRUDER_KEY = "intruder"
OPTIMIZER_KEY = "optimizer"
BUILDER_KEY = "builder"
HUNTER_KEY = "hunter"
RULE_KEYS = (INTRUDER_KEY, OPTIMIZER_KEY, BUILDER_KEY, HUNTER_KEY)


@dataclass(frozen=True)
class Profile:
    """The weight profile of a weighted personality.

    The six `w_*` are the linear terms of candidate scoring: `w_corner`
    corner-connected region, `w_center` closeness to the centre, `w_block`
    stones against an opponent, `w_defend` opponent stones against ours,
    `w_open` number of adjacent empty squares, `w_large` piece size.
    `mistake_rate` is the chance of deliberately picking a suboptimal move
    from the shortlist on each turn.
    """
    w_corner: float
    w_center: float
    w_block: float
    w_defend: float
    w_open: float
    w_large: float
    mistake_rate: float


def make_profile(spec, rng):
    """Multiplying a set of base weights by a random 0.7-1.3 perturbation.

    The perturbation makes the same personality differ slightly from game to
    game, otherwise the leaderboard would be a replay of one fixed set of
    matchups.
    """
    wc, wc2, wb, wd, wo, wl, mr = spec
    u = rng.uniform
    return Profile(
        wc * u(0.7, 1.3), wc2 * u(0.7, 1.3), wb * u(0.7, 1.3), wd * u(0.7, 1.3),
        wo * u(0.7, 1.3), wl * u(0.7, 1.3), mr * u(0.7, 1.3),
    )


class Brain:
    """The evaluation layer.

    Candidate enumeration, opponent lookahead and the draw all stay in
    `choose_move`; a personality only handles "how to read a position":
    `context` collects the shared up-front quantities for a position,
    `restrict` does the stage (soft) filter, and `rescore` re-ranks.
    """

    key = "chess"
    # The opponent lookahead uses **weighted** scoring, whose magnitude is
    # completely different from the rule-based objective function. Rule-based
    # personalities do not use it: that term would simply override the rule
    # ordering, i.e. wasted work.
    uses_lookahead = True

    def __init__(self, key, profile):
        self.key = key
        self.profile = profile
        self.mistake_rate = profile.mistake_rate

    def context(self, board, hand_names, owner, must_cover, reach):
        return None

    def restrict(self, cands, ctx):
        return cands

    def rescore(self, cands, ctx):
        return cands


class WeightedBrain(Brain):
    """Shared base for wolf/chess/fox: a weighted sum, with `restrict` and
    `rescore` both the identity.

    The real differences all live in the six `Profile` weights, so all three
    have exactly the same scoring path.
    """
