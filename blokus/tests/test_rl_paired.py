"""What the paired experiment claims about itself, checked against a run of it.

Eight checks, in the order the plan lists them. The first two are the ones the
experiment's validity rests on: if a pair is not reproducible, or if the two
games of a pair do not face the same three opponents in the same seats, then
every number downstream is a comparison of two different experiments.

The whole file must finish inside a minute, which is far less than playing the
2,000 + 1,000 pairs of the real run needs. So it plays a handful of pairs once,
at module level, and shares them between the checks that only need to look at a
result. What that cannot show - that 2,000 pairs behave like 4 - is shown by the
real run's own reproducibility, which is re-checked there rather than here.

The last test is the H0-0 correction from the throughput formula. It lives in a
stage H0 file because `plan4.md` allows only two new test files and neither is
named after `rl/smoke_gpu.py`; `tests/test_rl_isolation.py` covers that module
and is not to be modified. It needs no GPU, because the function under test is
fed a fixed list of timings.
"""
import json
import os
import sys

import ai.formulas  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl.paired as P  # noqa: E402
from rl.env import average_places  # noqa: E402

_CACHE = {}


def cached_pairs(n=3, group="A"):
    """A few real pairs, played once and shared by the checks that read them."""
    key = (group, n)
    if key not in _CACHE:
        configs = P.configurations_a() if group == "A" else P.configurations_b()
        assignment = P.assign(n, configs)
        _CACHE[key] = [
            P.play_pair(i, group, P.SEED_BASE_A + i, assignment[i][0],
                        assignment[i][1])
            for i in range(n)
        ]
    return _CACHE[key]


# --------------------------------------------------------------------------
# 1. a pair is a pure function of its seed and configuration
# --------------------------------------------------------------------------

def test_the_same_pair_replayed_gives_identical_fields():
    seed = P.SEED_BASE_A
    tested, controllers = P.configurations_a()[7]
    first = P.play_pair(7, "A", seed, tested, controllers)
    second = P.play_pair(7, "A", seed, tested, controllers)
    assert first == second
    # compared field by field as well, so a failure says which field moved
    for key in first:
        assert first[key] == second[key], key
    # and as serialised, because that is how they are written down
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_pair_record_carries_every_field_the_plan_lists():
    row = cached_pairs(1)[0]
    for key in ("pair_id", "group", "game_seed", "tested_seat", "controllers",
                "opponents", "rank_v1", "rank_v2", "remaining_v1",
                "remaining_v2", "book_abandoned_v1_at", "book_abandoned_v2_at"):
        assert key in row, key
    assert len(row["opponents"]) == 3
    assert row["d"] == row["rank_v2"] - row["rank_v1"]
    assert row["d_remaining"] == row["remaining_v2"] - row["remaining_v1"]


# --------------------------------------------------------------------------
# 2. the two games of a pair face the same opponents
# --------------------------------------------------------------------------

def test_both_games_of_a_pair_face_the_same_opponents_in_the_same_seats():
    """Re-derived from the game's own summary, not read out of the record.

    A record that says the opponents were the same proves nothing; this plays
    both games again and compares the controllers each one was actually given.
    """
    for group, base in (("A", P.SEED_BASE_A), ("B", P.SEED_BASE_B)):
        configs = P.configurations_a() if group == "A" else P.configurations_b()
        for i in range(2):
            tested, controllers = configs[i]
            v1 = P.play_game(base + i, controllers, tested, use_book=False)
            v2 = P.play_game(base + i, controllers, tested, use_book=True)
            assert v1["controllers"] == v2["controllers"] == list(controllers)
            assert v1["tested_seat"] == v2["tested_seat"] == tested
            # the two games really are different games, not a copy
            assert v1["ply"] or v2["ply"]


def test_opponent_streams_do_not_mention_whether_the_tested_seat_is_v2():
    """The randomness requirement, stated as an equation.

    A seat's two streams are `seat_seed(game_seed, seat, salt)`, which has no
    argument for the version under test, so the two games of a pair hand the
    opponents bit-identical generators. This checks the property the plan states
    by recomputing the seeds and by confirming that a brain built from the
    profile stream is identical in both games.
    """
    import random
    import ai
    for seat in range(4):
        for salt in (P.PROFILE_SALT, P.CHOICE_SALT):
            value = P.seat_seed(P.SEED_BASE_A, seat, salt)
            assert P.seat_seed(P.SEED_BASE_A, seat, salt) == value
            assert P.seat_seed(P.SEED_BASE_A + 1, seat, salt) != value
    a, _ = P._brain_and_rng(P.SEED_BASE_A, 2, "wolf")
    b, _ = P._brain_and_rng(P.SEED_BASE_A, 2, "wolf")
    assert a.profile == b.profile, "the weight perturbation is not reproducible"
    assert a.mistake_rate == b.mistake_rate
    # and the streams of two different seats do not collide
    streams = {P.seat_seed(P.SEED_BASE_A, s, salt)
               for s in range(4)
               for salt in (P.PROFILE_SALT, P.CHOICE_SALT)}
    assert len(streams) == 8


def test_seat_streams_are_injective_over_seed_seat_and_salt():
    seen = {}
    for game_seed in (P.SEED_BASE_A, P.SEED_BASE_B, P.SEED_BASE_A + 1):
        for seat in range(4):
            for salt in (0, 1):
                key = P.seat_seed(game_seed, seat, salt)
                assert key not in seen, (key, seen.get(key))
                seen[key] = (game_seed, seat, salt)


# --------------------------------------------------------------------------
# 3. the configurations are spread evenly
# --------------------------------------------------------------------------

def test_group_a_has_the_240_configurations_the_plan_asks_for():
    configs = P.configurations_a()
    assert len(configs) == 240 == 10 * 4 * 6
    assert len(set(configs)) == 240
    trios = set()
    for tested, ctrl in configs:
        assert ctrl[tested] == P.TESTED_KEY
        others = [ctrl[s] for s in range(4) if s != tested]
        assert len(set(others)) == 3, "an opponent appears twice in one game"
        trios.add(tuple(sorted(others)))
    assert len(trios) == 10, "C(5,3) opponent trios"
    assert all(trio == tuple(sorted(P.POOL_A[i] for i in idx))
               for trio in trios
               for idx in [(P.POOL_A.index(trio[0]),
                            P.POOL_A.index(trio[1]),
                            P.POOL_A.index(trio[2]))])


def test_group_b_has_the_24_configurations_the_plan_asks_for():
    configs = P.configurations_b()
    assert len(configs) == 24 == 4 * 6
    assert len(set(configs)) == 24
    for tested, ctrl in configs:
        others = [ctrl[s] for s in range(4) if s != tested]
        assert sorted(others) == sorted(P.GROUP_B_OPPONENTS)
        assert ctrl[tested] == P.TESTED_KEY
    # the third opponent is v1, and is never the version under test
    assert P.GROUP_B_OPPONENTS[2] == "optimizer"


def test_no_group_a_configuration_puts_the_tested_version_in_the_pool():
    for tested, ctrl in P.configurations_a():
        for s in range(4):
            if s != tested:
                assert ctrl[s] in P.POOL_A


@pytest.mark.parametrize("group,pairs,expected", [("A", 2_000, 240),
                                                  ("A", 2_001, 240),
                                                  ("B", 1_000, 24),
                                                  ("B", 1_001, 24),
                                                  ("A", 17, 240)])
def test_configuration_counts_differ_by_at_most_one(group, pairs, expected):
    configs = P.configurations_a() if group == "A" else P.configurations_b()
    assignment = P.assign(pairs, configs)
    counts = {}
    for entry in assignment:
        counts[entry] = counts.get(entry, 0) + 1
    assert len(counts) == min(pairs, expected)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert sum(counts.values()) == pairs


def test_assignment_cycles_rather_than_shuffling():
    configs = P.configurations_a()
    assignment = P.assign(500, configs)
    assert assignment == [configs[i % len(configs)] for i in range(500)]
    assert assignment[:240] == assignment[240:480] == configs


# --------------------------------------------------------------------------
# 4. seed ranges
# --------------------------------------------------------------------------

def test_the_planned_seed_ranges_are_accepted():
    for base, count in ((P.SEED_BASE_A, P.PAIRS_A), (P.SEED_BASE_B, P.PAIRS_B),
                        (P.SEED_BASE_A_EXPAND, P.PAIRS_A_EXPAND),
                        (P.SEED_BASE_B_EXPAND, P.PAIRS_B_EXPAND)):
        seeds = [base + i for i in range(count)]
        assert P.check_seed_ranges(seeds) == []


@pytest.mark.parametrize("lo,hi", [
    (0, 199),                          # bench_engine.py
    (20_240_101, 20_336_965),          # tools/benchmark.py
    (1_000_000, 1_001_999),            # stage G training
    (2_000_000, 2_000_199),            # stage G validation
])
def test_a_range_that_overlaps_a_published_number_is_refused(lo, hi):
    with pytest.raises(ValueError):
        P.check_seed_ranges(list(range(lo, hi + 1)))


def test_a_range_below_the_plans_floor_is_refused():
    with pytest.raises(ValueError):
        P.check_seed_ranges([2_999_999] + [P.SEED_BASE_A + i
                                            for i in range(10)])
    with pytest.raises(ValueError):
        P.check_seed_ranges([0, 5])


def test_a_range_that_merely_touches_a_boundary_is_still_accepted():
    """Membership, not span: a seed just past a used range is fine.

    `1_002_000` is the first seed above the stage G training block and
    `2_000_200` the first above its validation block. A span check called the
    two of them an overlap, which they are not - so the check was rewritten to
    test membership and this is the case that pins it.
    """
    assert P.check_seed_ranges([3_000_000, 3_000_001]) == []


def test_the_first_seed_above_each_used_range_is_not_rejected_for_that_range():
    """Each boundary seed is inside no used range, even though its neighbours are.

    They are all below the plan's 3,000,000 floor, so they are checked against
    the four used blocks directly rather than through `check_seed_ranges`.
    """
    import rl.paired as M
    for boundary, used in (
            (M.BENCH_SEED_MAX + 1, range(0, M.BENCH_SEED_MAX + 1)),
            (M.BENCHMARK_SEED_BASE
             + M.BENCHMARK_SEED_STRIDE * M.BENCHMARK_MAX_GAMES + 1,
             range(M.BENCHMARK_SEED_BASE,
                   M.BENCHMARK_SEED_BASE
                   + M.BENCHMARK_SEED_STRIDE * M.BENCHMARK_MAX_GAMES + 1)),
            (M.G_TRAIN_BASE + M.G_TRAIN_GAMES,
             range(M.G_TRAIN_BASE, M.G_TRAIN_BASE + M.G_TRAIN_GAMES)),
            (M.G_VALID_BASE + M.G_VALID_GAMES,
             range(M.G_VALID_BASE, M.G_VALID_BASE + M.G_VALID_GAMES))):
        assert boundary not in used
        assert boundary - 1 in used


# --------------------------------------------------------------------------
# 5. one process and four processes agree
# --------------------------------------------------------------------------

def test_one_worker_and_four_workers_produce_identical_results():
    one, _ = P.run_group("A", 4, P.SEED_BASE_A, workers=1)
    four, _ = P.run_group("A", 4, P.SEED_BASE_A, workers=4)
    assert one == four
    for a, b in zip(one, four):
        for key in a:
            assert a[key] == b[key], key
    assert [r["pair_id"] for r in one] == [0, 1, 2, 3]


def test_the_group_b_run_also_agrees_across_worker_counts():
    one, _ = P.run_group("B", 2, P.SEED_BASE_B, workers=1)
    three, _ = P.run_group("B", 2, P.SEED_BASE_B, workers=3)
    assert one == three


# --------------------------------------------------------------------------
# 6. the ranking is the project's ranking
# --------------------------------------------------------------------------

def test_places_match_the_ranking_oracle():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    import benchmark
    for remaining in ([0, 1, 2, 3], [5, 5, 9, 9], [10, 10, 10, 10],
                      [3, 3, 3, 3], [1, 2, 3, 20], [7, 8, 9, 10], [0, 0, 0, 1]):
        assert average_places(remaining) == benchmark.average_ranks(remaining)
    # a tie really does share a place, which is what the pairing relies on
    assert average_places([5, 5, 9, 9]) == [1.5, 1.5, 3.5, 3.5]


def test_the_places_in_a_pair_come_from_that_rule():
    for row in cached_pairs(2):
        for group_row in (row,):
            assert 1.0 <= group_row["rank_v1"] <= 4.0
            assert 1.0 <= group_row["rank_v2"] <= 4.0
    # and recomputing them from the recorded remaining counts agrees
    seed = P.SEED_BASE_A
    tested, controllers = P.configurations_a()[0]
    for use_book in (False, True):
        summary = P.play_game(seed, controllers, tested, use_book)
        assert summary["tested_place"] == average_places(
            summary["remaining"])[tested]


# --------------------------------------------------------------------------
# 7. the wall-clock budget is put back
# --------------------------------------------------------------------------

def test_the_budget_is_off_during_a_pair_and_restored_after_it():
    before = ai.formulas.USE_WALL_BUDGET
    cached_pairs(1)
    assert ai.formulas.USE_WALL_BUDGET is before


def test_a_full_group_run_leaves_the_budget_alone():
    before = ai.formulas.USE_WALL_BUDGET
    P.run_group("A", 1, P.SEED_BASE_A, workers=1)
    assert ai.formulas.USE_WALL_BUDGET is before


# --------------------------------------------------------------------------
# 8. the engine and the Game agree at the end of every game
# --------------------------------------------------------------------------

def test_the_v1_side_book_column_is_actually_probed():
    """The v1 column of the record must be a measurement, not a blank.

    v1 never consults the book, so "would the book have been abandoned on this
    line" has to be asked of the position directly. The first version of
    `play_game` read it off the tracker, which v1 never updates - the field was
    structurally `None` and the two columns did not mean the same thing. Here the
    v1 game is checked against the diagnostic's own answer for the same game.
    """
    from rl.paired import _diagnose_game, diagnose_v1_opening
    tested, controllers = P.configurations_a()[0]
    v1 = P.play_game(P.SEED_BASE_A, controllers, tested, use_book=False)
    opened, abandoned = _diagnose_game(P.SEED_BASE_A, controllers, tested)
    assert v1["book_abandoned_at"] == abandoned
    assert abandoned is not None, (
        "on v1's own line the book should be unavailable; if this is None the "
        "probe is not running")
    assert v1["book_moves"] == 0, "v1 plays no book move"
    # and the v2 side really does use all three
    v2 = P.play_game(P.SEED_BASE_A, controllers, tested, use_book=True)
    assert v2["book_moves"] == 3
    assert v2["book_abandoned_at"] is None
    del diagnose_v1_opening


def test_stuck_flags_and_remaining_agree_at_the_end_of_every_game():
    """Checked by playing, not by reading the record.

    `play_game` raises if the two implementations of the turn ever disagree, so
    reaching the end of a game at all is the evidence; the remaining-cell
    comparison below is the extra assertion, because a game can finish with the
    turn order right and the scoreboard wrong.
    """
    for group, base in (("A", P.SEED_BASE_A), ("B", P.SEED_BASE_B)):
        configs = P.configurations_a() if group == "A" else P.configurations_b()
        for i in range(2):
            tested, controllers = configs[i]
            for use_book in (False, True):
                summary = P.play_game(base + i, controllers, tested, use_book)
                assert sum(summary["remaining"]) > 0
                assert len(summary["places"]) == 4
                assert sum(summary["places"]) == 10.0, (
                    "competition ranking must add to 1+2+3+4")


# --------------------------------------------------------------------------
# the decision rule, exercised on synthetic intervals
# --------------------------------------------------------------------------

def _fake_stats(a_low, a_high, b_low, b_high, a_mean=0.0, b_mean=0.0):
    return {"primary": {"group_A_d": {"n": 100, "mean": a_mean,
                                      "ci_low": a_low, "ci_high": a_high},
                       "group_B_d": {"n": 100, "mean": b_mean,
                                      "ci_low": b_low, "ci_high": b_high}}}


@pytest.mark.parametrize("a,b,verdict", [
    ((-0.10, -0.01), (-0.05, 0.02), "adopt v2"),
    ((-0.10, -0.01), (0.01, 0.20), "keep v1"),
    ((-0.10, 0.03), (-0.05, 0.02), "keep v1"),
    ((0.02, 0.20), (-0.05, 0.02), "keep v1"),
    ((-0.10, 0.00), (0.00, 0.20), "keep v1"),
    ((-0.10, -0.01), (-0.05, 0.00), "adopt v2"),
])
def test_the_decision_follows_the_rule_that_was_fixed_in_advance(a, b, verdict):
    got = P.decide(_fake_stats(a[0], a[1], b[0], b[1]))
    assert got["verdict"] == verdict, got
    # the reported endpoints are the ones the verdict was read off
    assert got["A_ci_high"] == a[1]
    assert got["B_ci_low"] == b[0]


def test_an_interval_touching_zero_is_not_below_zero():
    """A high of exactly 0 does not satisfy "upper bound < 0"."""
    got = P.decide(_fake_stats(-0.1, 0.0, -0.1, 0.0))
    assert got["verdict"] == "keep v1"
    assert P.expansion_needed(_fake_stats(-0.1, 0.0, -0.1, 0.0)) is True


def test_expansion_is_needed_exactly_when_the_primary_interval_has_zero():
    assert P.expansion_needed(_fake_stats(-0.1, -0.01, -0.1, 0.0)) is False
    assert P.expansion_needed(_fake_stats(0.01, 0.1, -0.1, 0.0)) is False
    assert P.expansion_needed(_fake_stats(-0.1, 0.05, -0.1, 0.0)) is True


def test_the_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    import random
    rng = random.Random(7)
    values = [rng.gauss(-0.05, 0.5) for _ in range(200)]
    a = P.bootstrap_ci(values, iters=2_000)
    b = P.bootstrap_ci(values, iters=2_000)
    assert a == b
    assert a["ci_low"] < a["mean"] < a["ci_high"]
    assert a["n"] == 200 and a["iters"] == 2_000
    # a constant sample has a degenerate interval, which is the sanity check
    flat = P.bootstrap_ci([0.25] * 50, iters=500)
    assert flat["mean"] == flat["ci_low"] == flat["ci_high"] == 0.25
    # The width shrinks as the sample grows, which is the point of an interval.
    # (The number of resamples only refines the percentile grid; it does not
    # make the interval narrower, which is a common misreading.)
    small = P.bootstrap_ci(values[:20], iters=5_000)
    large = P.bootstrap_ci(values, iters=5_000)
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"]
                                                   - small["ci_low"])


def test_summarise_reports_everything_the_plan_lists_as_secondary():
    rows = cached_pairs(3)
    stats = P.summarise(rows, iters=500)
    sec = stats["secondary"]
    for key in ("group_A_d_remaining", "group_B_d_remaining", "group_A_d_sd",
                "rank_correlation", "share_d_zero", "group_A_d_by_tested_seat",
                "group_B_d_by_tested_seat", "group_A_d_by_opponent",
                "book_abandoned_v2_share", "book_abandoned_v2_at_counts",
                "book_abandoned_v1_at_counts", "book_moves_v2_mean",
                "d_when_book_abandoned", "d_when_book_kept"):
        assert key in sec, key
    assert set(stats["primary"]) == {"group_A_d", "group_B_d"}
    assert stats["counts"]["games"] == 2 * len(rows)
    # the descriptive numbers are in their own block, so the decision cannot
    # reach them
    assert set(P.decide(stats)) <= {"verdict", "reason", "A_ci_high",
                                    "B_ci_low", "A_ci_low", "B_ci_high",
                                    "A_mean", "B_mean"}


# --------------------------------------------------------------------------
# the opening diagnostic
# --------------------------------------------------------------------------

def test_the_diagnostic_uses_real_games_and_is_reproducible():
    from rl.paired import diagnose_v1_opening
    a = diagnose_v1_opening(2)
    assert a == diagnose_v1_opening(2)
    assert a["games"] == 2
    assert sum(a["steps_observed"]) == 6, "two games, three first moves each"
    for rec in a["records"]:
        assert len(rec["opened"]) == 3
        assert [r["step"] for r in rec["opened"]] == [0, 1, 2]
        for row in rec["opened"]:
            assert row["size"] in (1, 2, 3, 4, 5)


def test_the_module_imports_nothing_heavy():
    import ast
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rl", "paired.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"torch", "pygame"}), imported


# --------------------------------------------------------------------------
# H0-0: the throughput formula, fed a fixed list of timings
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch,times", [
    (64, [0.0035304] * 10),
    (256, [0.0044157] * 10),
    (1024, [0.0275162] * 10),
    (256, [0.0216507] * 50),
    (7, [0.001, 0.002, 0.003, 0.010, 0.0005]),
    (1, [0.5]),
])
def test_throughput_equals_batch_over_the_mean(batch, times):
    """`positions_per_second` must be `batch / (mean_ms / 1000)`.

    The stage G report had `bs / sum(times)`, which is the same quantity
    divided by the iteration count, so the table said 363.1 next to a mean of
    3.5256 ms - 50 times too slow, and internally inconsistent. No GPU is
    involved: `timing_record` is pure, and the timings are numbers.
    """
    from rl.smoke_gpu import timing_record
    record = timing_record(batch, times, warmup=10)
    want = batch / (record["mean_ms"] / 1000.0)
    assert record["positions_per_second"] == pytest.approx(want, rel=1e-6)
    assert record["mean_ms"] == pytest.approx(sum(times) / len(times) * 1000.0,
                                              rel=1e-12)
    assert record["batch"] == batch
    assert record["iterations"] == len(times)
    assert record["min_ms"] <= record["median_ms"] <= record["max_ms"]


def test_throughput_is_not_the_batch_over_the_total():
    """The old formula, pinned so it cannot come back unnoticed.

    `sum(times)` over 50 iterations of 4 ms is 0.2 s, and 256 / 0.2 is 1,280 -
    fifty times too small, and the number the stage G report printed for a batch
    of 64.
    """
    from rl.smoke_gpu import timing_record
    times = [0.004] * 50
    record = timing_record(256, times, warmup=10)
    old = 256 / sum(times)
    assert record["positions_per_second"] == pytest.approx(old * 50, rel=1e-9)
    assert record["mean_ms"] == pytest.approx(4.0, rel=1e-12)
    assert record["positions_per_second"] == pytest.approx(64_000.0, rel=1e-9)
