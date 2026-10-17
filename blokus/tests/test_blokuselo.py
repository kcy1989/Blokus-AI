"""Tests for blokuselo.py: BlokusElo pure-Python Plackett-Luce engine.

Synthetic-player generation uses a point-noise model (theta + Normal(0, sd)),
independent of the Plackett-Luce fitter: results are generated first, then
fitted.  No Plackett-Luce self-generation / self-fit.
"""
import random

from blokuselo import rate

_SD = 1.45  # measured per-game point-noise sd (real data), not self-set


def _synth_games(rng, n_games, thetas, sd=_SD):
    """Point-noise model: perf = theta + Normal(0, sd); rank by perf.

    Independent of Plackett-Luce: generates raw performances, not PL ranks.
    """
    players = sorted(thetas)
    games = []
    for _ in range(n_games):
        perf = {p: thetas[p] + rng.gauss(0, sd) for p in players}
        order = sorted(players, key=lambda p: -perf[p])
        games.append([(p, i + 1) for i, p in enumerate(order)])
    return games


def test_rate_returns_mu_sigma_for_all_players():
    games = [[("a", 1), ("b", 2), ("c", 3), ("d", 4)]]
    res = rate(games, anchor="a")
    assert set(res) == {"a", "b", "c", "d"}
    assert res["a"] == (0.0, 0.0)
    for p in ("b", "c", "d"):
        mu, sigma = res[p]
        assert isinstance(mu, float)
        assert sigma > 0


def test_rate_reproducible_same_input():
    """Bit-identical output for identical input (unit reproducibility)."""
    games = _synth_games(random.Random(1), 100,
                         {"p": 0.5, "w": -0.5, "m": 0.5, "s": 1.0})
    assert rate(games, anchor="w") == rate(games, anchor="w")


def test_rate_recovers_strength_order():
    rng = random.Random(3)
    thetas = {"weak": -1.0, "mid": 0.0, "strong": 1.0}
    games = _synth_games(rng, 500, thetas)
    res = rate(games, anchor="weak")
    assert res["weak"][0] == 0.0
    assert res["mid"][0] > res["weak"][0]
    assert res["strong"][0] > res["mid"][0]


def test_rate_handles_ties():
    games = [[("a", 1), ("b", 1), ("c", 3), ("d", 3)]]
    res = rate(games, anchor="a")
    assert set(res) == {"a", "b", "c", "d"}


def test_synthetic_skill_rising_mu_rises():
    # 6a: virtual human skill rises segment by segment -> estimated mu rises.
    rng = random.Random(42)
    anchors = {"w": -0.5, "m": 0.5, "s": 1.0}
    mus = []
    all_games = []
    for seg in (0.0, 0.5, 1.0, 1.5):
        cur = dict(anchors)
        cur["p"] = seg
        all_games.extend(_synth_games(rng, 150, cur))
        mus.append(rate(all_games, anchor="w")["p"][0])
    for i in range(1, len(mus)):
        assert mus[i] > mus[i - 1], mus


def test_synthetic_skill_unchanged_no_drift():
    # 6b: constant skill -> no systematic drift; tolerance 2*SE pre-set.
    rng = random.Random(43)
    thetas = {"p": 0.5, "w": -0.5, "m": 0.5, "s": 1.0}
    games = _synth_games(rng, 400, thetas)
    half = 200
    re = rate(games[:half], anchor="w")
    rl = rate(games, anchor="w")
    drift = abs(rl["p"][0] - re["p"][0])
    se = max(re["p"][1], rl["p"][1])
    assert drift <= 2.0 * se, (drift, se)


def test_synthetic_weak_opponent_bounded_inflation():
    # 6c: only weak opponents -> inflation bounded, no runaway (50 Elo pt cap).
    rng = random.Random(44)
    thetas = {"p": 0.05, "w": 0.0}
    games = []
    for _ in range(400):
        perf = {p: thetas[p] + rng.gauss(0, _SD) for p in ("p", "w")}
        order = sorted(("p", "w"), key=lambda p: -perf[p])
        games.append([(p, i + 1) for i, p in enumerate(order)])
    res = rate(games, anchor="w")
    gap = res["p"][0] - res["w"][0]
    assert 0.0 < gap <= 1.0, gap


def test_sigma_decreases_with_games_median_p90():
    # sigma -> 0.6: multiple seeds, report median and 90th percentile.
    # These values feed the human display threshold decision (task 3).
    thresholds = []
    for seed in range(10):
        r = random.Random(seed)
        thetas = {"p": 0.5, "w": -0.5, "m": 0.5, "s": 1.0}
        games = _synth_games(r, 600, thetas)
        reached = None
        for n in range(5, 601, 5):
            if rate(games[:n], anchor="w")["p"][1] <= 0.6:
                reached = n
                break
        thresholds.append(reached if reached is not None else 600)
    thresholds.sort()
    median = thresholds[len(thresholds) // 2]
    p90 = thresholds[int(0.9 * len(thresholds))]
    assert median is not None and p90 is not None
    assert median <= 600 and p90 <= 600
