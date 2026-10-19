"""Cover the wall-clock cap arm of the opponent lookahead.

`ai/chooser.choose_move` runs the lookahead only when the mover has a
lookahead brain, there is an opponent to model, and the cap has not been
reached:

    other_brains and opps and brain.uses_lookahead
        and (not F.USE_WALL_BUDGET or perf_counter() - t0 < F.WALL_BUDGET)

Every other test that exercises the lookahead runs with the cap *on* but a
loose budget, so the `or ... < WALL_BUDGET` arm is always satisfied and the
skip branch (cap reached -> `adj = top`) is never taken. `test_trace`'s
`test_lookahead_used_only_for_weighted_personalities` only ever observes the
rule-based `uses_lookahead = False` case. This file takes the missing branch.

**What is proven: the branch is reachable, not how it behaves under real
load.** `WALL_BUDGET = 0` makes `elapsed < 0` false because `perf_counter` is
monotonic, so the cap counts as reached without waiting for a real 0.9s - the
test never touches the clock. Whether the lookahead actually helps or hurts
when a machine is genuinely slow is a different question, answered by
`test_optimised_paths`' load test, not here.
"""
import random

import ai
from ai import formulas as F
from game import Game


def _setup(seed=1):
    game = Game(random.Random(seed))
    game.setup_match(("wolf", "chess", "fox", "hunter"))
    game.start()
    return game


def _weighted_owner(game):
    for owner in range(4):
        if game.brains[owner].uses_lookahead:
            return owner
    raise AssertionError("no lookahead brain seated")


def _lookahead_used(game, owner, monkeypatch, use_budget, wall):
    monkeypatch.setattr(F, "USE_WALL_BUDGET", use_budget)
    monkeypatch.setattr(F, "WALL_BUDGET", wall)
    trace = []
    ai.choose_move(
        game.board, game.hands[owner].names, owner, game.brains[owner],
        game.rng, other_brains=game.brain_map(),
        must_cover=game.must_cover(owner), reach=game.reach(owner),
        other_must_cover={x: game.must_cover(x) for x in range(4)},
        other_reach={x: game.reach(x) for x in range(4)}, trace=trace)
    assert trace, "expected one decision record"
    return trace[-1]["lookahead_used"]


def test_the_cap_reached_skips_the_lookahead(monkeypatch):
    """Cap on and already reached -> `adj = top`, no lookahead.

    Paired with the same call under a loose cap, so the assertion is not
    vacuous: the only difference between the two is the budget.
    """
    game = _setup()
    owner = _weighted_owner(game)
    assert _lookahead_used(game, owner, monkeypatch, True, 1e9) is True
    assert _lookahead_used(game, owner, monkeypatch, True, 0.0) is False


def test_the_cap_off_runs_the_lookahead_regardless_of_budget(monkeypatch):
    """`not F.USE_WALL_BUDGET` is the other way the arm is satisfied: with the
    switch off, even a zero budget runs the lookahead."""
    game = _setup()
    owner = _weighted_owner(game)
    assert _lookahead_used(game, owner, monkeypatch, False, 0.0) is True
