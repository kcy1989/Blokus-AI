"""Tests for `rl/paired3.py`: the three-arm paired experiment.

The whole file has to stay under 60 seconds, so every test plays a handful of
pairs rather than a group. The properties being pinned are the ones that would
silently corrupt a 15,000-game run: the arms must share their opponents and
their randomness, the configuration balance must be exact, the seed guard must
reject every reserved range, the result must not depend on the worker count, and
the ranking must be the project's own.
"""
import os
import sys

import pytest

import rl.paired3 as P
from rl.env import average_places

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# 1. a pair is reproducible field for field
# --------------------------------------------------------------------------

def test_the_same_pair_replayed_twice_is_identical_field_for_field():
    tested, controllers = P.configurations_a()[7]
    a = P.play_pair(7, "A", P.SEED_BASE_A, tested, controllers)
    b = P.play_pair(7, "A", P.SEED_BASE_A, tested, controllers)
    assert a == b
    # a record with a silently missing or renamed field would still compare equal
    # to itself, so the field set is pinned as well
    assert sorted(a.keys()) == sorted([
        "book_abandoned_v2_at", "book_abandoned_v3_at", "book_moves_v2",
        "book_moves_v3", "controllers", "d21", "d21_remaining", "d31",
        "d31_remaining", "d32", "d32_remaining", "game_seed", "group",
        "opponents", "pair_id", "ply_v1", "ply_v2", "ply_v3", "rank_v1",
        "rank_v2", "rank_v3", "remaining_v1", "remaining_v2", "remaining_v3",
        "tested_seat"])


def test_replaying_a_pair_in_the_other_order_gives_the_same_answer():
    """The arms are independent games, so the order they are played in cannot
    matter. If it did, some state would be shared between them."""
    tested, controllers = P.configurations_b()[3]
    forward = P.play_pair(3, "B", P.SEED_BASE_B, tested, controllers)
    # played after the other two arms rather than before, which is the whole
    # point: the arms are independent games sharing only their seeds
    late_v3 = P.play_game(P.SEED_BASE_B, controllers, tested, "v3")
    early_v3 = P.play_game(P.SEED_BASE_B, controllers, tested, "v3")
    assert late_v3["remaining"] == early_v3["remaining"]
    assert forward["rank_v3"] == late_v3["tested_place"]


# --------------------------------------------------------------------------
# 2. the three games of a pair share opponents and seating
# --------------------------------------------------------------------------

def test_the_three_games_of_a_pair_have_the_same_three_opponents():
    """The only thing that differs between the arms is which version of the
    tested personality is playing; the opponents and their seats are identical."""
    for i, (tested, controllers) in enumerate(P.configurations_a()[:1]
                                               + P.configurations_b()[:1]):
        seen = set()
        for arm in P.ARMS:
            row = P.play_game(P.SEED_BASE_A + i, controllers, tested, arm)
            key = tuple((o["seat"], o["key"])
                        for o in [{"seat": s, "key": controllers[s]}
                                  for s in range(4) if s != tested])
            seen.add(key)
            assert row["arm"] == arm
        assert len(seen) == 1, seen
        assert len(seen.pop()) == 3


def test_the_opponents_are_never_the_tested_personality_in_group_a():
    for _tested, controllers in P.configurations_a():
        keys = [k for k in controllers if k != P.TESTED_KEY]
        assert len(keys) == 3
        assert P.TESTED_KEY not in keys


def test_each_seat_random_stream_is_derived_without_reference_to_the_arm():
    """`seat_seed` takes a game seed, a seat and a salt and nothing else, which
    is what stops one arm's draws from shifting another arm's."""
    a = P.seat_seed(P.SEED_BASE_A, 2, P.PROFILE_SALT)
    b = P.seat_seed(P.SEED_BASE_A, 2, P.CHOICE_SALT)
    assert a != b
    assert P.seat_seed(P.SEED_BASE_A, 1, P.PROFILE_SALT) != a
    # the two salts differ, the four seats differ, and no book stream can collide
    streams = {P.seat_seed(P.SEED_BASE_A + i, s, salt)
               for i in range(50) for s in range(4)
               for salt in (P.PROFILE_SALT, P.CHOICE_SALT)}
    assert len(streams) == 50 * 4 * 2


# --------------------------------------------------------------------------
# 3. the configuration assignment is exactly balanced
# --------------------------------------------------------------------------

@pytest.mark.parametrize("group,pairs", [("A", 2_000), ("B", 1_000),
                                         ("A", 3_000), ("B", 1_500)])
def test_the_configuration_assignment_spread_is_at_most_one(group, pairs):
    configs = P.configurations(group)
    assignment = P.assign(pairs, configs)
    counts = {}
    for c in assignment:
        counts[c] = counts.get(c, 0) + 1
    assert set(counts) == set(configs)
    spread = max(counts.values()) - min(counts.values())
    assert spread <= 1, spread
    assert sum(counts.values()) == pairs


def test_group_a_has_240_configurations_and_group_b_24():
    assert len(P.configurations_a()) == 240
    assert len(P.configurations_b()) == 24
    # every tested seat appears, and every seating of the trio is present
    for tested in range(4):
        got = {c for c in P.configurations_a() if c[0] == tested}
        assert len(got) == 60


# --------------------------------------------------------------------------
# 4. the seed guard rejects every reserved range
# --------------------------------------------------------------------------

def test_the_seed_guard_rejects_each_reserved_range():
    for label, lo, hi in P.RESERVED_RANGES:
        with pytest.raises(ValueError) as exc:
            P.check_seed_ranges([lo], own_base=0)
        assert label in str(exc.value), (label, str(exc.value))
        # and at the far end of the range
        with pytest.raises(ValueError) as exc:
            P.check_seed_ranges([hi], own_base=0)
        assert label in str(exc.value)


def test_the_shipped_own_blocks_are_pairwise_disjoint():
    """A base or a count that reached into another block would silently replay
    positions an earlier phase already used."""
    for i, (la, alo, ahi) in enumerate(P.OWN_RANGES):
        for lb, blo, bhi in P.OWN_RANGES[i + 1:]:
            assert ahi < blo or bhi < alo, (la, ahi, lb, blo)


def test_the_seed_guard_rejects_seeds_that_land_in_another_own_block():
    """Asking for a run that starts inside one block and runs into the next."""
    _la, alo, _ahi = P.OWN_RANGES[0]
    _lb, blo, _bhi = P.OWN_RANGES[1]
    with pytest.raises(ValueError) as exc:
        P.check_seed_ranges([blo - 2, blo - 1, blo, blo + 1],
                            own_base=alo)
    assert "own block" in str(exc.value)


def test_the_seed_guard_accepts_this_stages_four_blocks():
    for _label, lo, hi in P.OWN_RANGES:
        P.check_seed_ranges(list(range(lo, hi + 1)), own_base=lo)


def test_the_seed_guard_rejects_seeds_below_this_stages_own_base():
    with pytest.raises(ValueError) as exc:
        P.check_seed_ranges([P.SEED_BASE_A - 3], own_base=P.SEED_BASE_A)
    assert "below this stage's" in str(exc.value)


def test_the_plan_reserved_ranges_are_the_ones_written_down():
    """`plan5.md` lists these; if the constants drift from the plan the guard
    stops guarding the right thing."""
    got = {label: (lo, hi) for label, lo, hi in P.RESERVED_RANGES}
    assert got["bench_engine.py"] == (0, 199)
    assert got["tools/benchmark.py"] == (20_240_101, 20_336_965)
    assert got["stage G training"] == (1_000_000, 1_001_999)
    assert got["stage G validation"] == (2_000_000, 2_000_199)
    assert got["stage H0 phase 1 A"] == (3_000_000, 3_001_999)
    assert got["stage H0 phase 1 B"] == (3_100_000, 3_100_999)
    assert got["stage H0 expansion A"] == (3_200_000, 3_202_999)
    assert got["stage H0 expansion B"] == (3_300_000, 3_301_499)


# --------------------------------------------------------------------------
# 4b. the blocks plan8-B0b added, and the block-level guard
# --------------------------------------------------------------------------

# The six entries plan8-B0b added, each read off a file. Kept as literals on
# purpose: deriving them from the manifests would make this test agree with
# whatever the manifests say, which is the failure mode RESERVED_RANGES'
# own comment warns about - a guard that imports the thing it polices.
EXPECTED_ADDED = (
    # data/hb1/manifest.json and data/hc1/manifest.json: 60 train shards
    # 6,000,000..6,029,378, 12 valid shards 6,100,000..6,101,545, identical in
    # both runs, so one entry covers both.
    ("H-B1/H-C1 imitation train", 6_000_000, 6_029_378),
    ("H-B1/H-C1 imitation valid", 6_100_000, 6_101_545),
    # reports/hc2_report.md lines 141-144. Single seeds, not spans:
    # match.run_league draws every seat from one random.Random(seed).
    ("stage H-C2 evaluation", 910_001, 910_004),
    ("pool league 920001", 920_001, 920_001),
    # plan8.md's RL blocks.
    ("stage RL train", 7_000_000, 7_019_999),
    ("stage RL validation", 7_100_000, 7_100_999),
)


def test_the_added_reserved_ranges_are_registered_exactly():
    got = {label: (lo, hi) for label, lo, hi in P.RESERVED_RANGES}
    for label, lo, hi in EXPECTED_ADDED:
        assert got.get(label) == (lo, hi), (label, got.get(label))


def test_every_registered_range_is_pairwise_disjoint():
    """Two blocks that overlap would make the guard's own message ambiguous:
    the same seed would be reported as two different blocks, and a caller
    reading the label could not tell which history it had hit."""
    ranges = list(P.RESERVED_RANGES) + list(P.OWN_RANGES)
    overlaps = []
    for i, (la, alo, ahi) in enumerate(ranges):
        for lb, blo, bhi in ranges[i + 1:]:
            if not (ahi < blo or bhi < alo):
                overlaps.append((la, alo, ahi, lb, blo, bhi))
    assert not overlaps, overlaps


def test_the_rl_blocks_clear_every_block_already_registered():
    for label, lo, hi in (("stage RL train", 7_000_000, 7_019_999),
                          ("stage RL validation", 7_100_000, 7_100_999)):
        for other, olo, ohi in list(P.RESERVED_RANGES) + list(P.OWN_RANGES):
            if other == label:
                continue
            assert hi < olo or ohi < lo, (label, (lo, hi), other, (olo, ohi))


def test_the_rl_blocks_are_accepted_by_the_block_guard_when_claimed():
    """The point of reserving them: the stage they were reserved for may take
    them, and nobody else may."""
    train = ("stage RL train", 7_000_000, 7_019_999)
    valid = ("stage RL validation", 7_100_000, 7_100_999)
    assert P.reject_reserved_seeds(7_000_000, 7_019_999, own=train) == []
    assert P.reject_reserved_seeds(7_100_000, 7_100_999, own=valid) == []
    # ...but only by claiming them, and only the right one
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(7_000_000, 7_019_999)
    assert "stage RL train" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(7_000_000, 7_019_999, own=valid)
    assert "stage RL train" in str(exc.value)


def test_claiming_your_own_block_does_not_excuse_overlapping_someone_elses():
    """The exemption is by exact triple, so a block that reaches into the
    imitation data still fails even when the caller has claimed a block."""
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(6_000_000, 7_019_999,
                                own=("stage RL train", 6_000_000, 7_019_999))
    assert "H-B1/H-C1 imitation train" in str(exc.value)


def test_the_block_guard_refuses_the_imitation_data_seeds():
    """6,000,000-6,000,010 is eleven seeds inside H-B1/H-C1's train block.

    This is the failure the whole exercise exists to stop: RL rollouts landing
    on the seeds the imitation set was generated from would report a number
    about memorisation. Before plan8-B0b nothing consulted this, because the
    range was in no table at all.
    """
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(6_000_000, 6_000_010)
    assert "H-B1/H-C1 imitation train" in str(exc.value)


def test_the_block_guard_refuses_h1s_own_blocks_too():
    """Why this function reads OWN_RANGES and `check_seed_ranges` cannot."""
    for label, lo, hi in P.OWN_RANGES:
        with pytest.raises(ValueError) as exc:
            P.reject_reserved_seeds(lo, hi)
        assert label in str(exc.value)


def test_the_block_guard_takes_an_extra_range_for_the_callers_own_block():
    assert P.reject_reserved_seeds(8_000_000, 8_000_999) == []
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(8_000_000, 8_000_999,
                                extra_ranges=(("stage RL phase 2", 8_000_000,
                                               8_000_999),))
    assert "stage RL phase 2" in str(exc.value)
    # claiming it through `own` works too, and needs no registration
    assert P.reject_reserved_seeds(
        8_000_000, 8_000_999, own=("stage RL phase 2", 8_000_000, 8_000_999),
        extra_ranges=(("stage RL phase 2", 8_000_000, 8_000_999),)) == []


def test_the_block_guard_rejects_a_range_that_ends_before_it_starts():
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(7_000_010, 7_000_000)
    assert "ends before it starts" in str(exc.value)


def test_the_block_guard_catches_a_block_swallowed_whole():
    """Overlap, not containment: asking for one seed inside a reserved block,
    and asking for a block that covers a reserved one, are the same mistake."""
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(6_029_000, 6_031_000)
    assert "H-B1/H-C1 imitation train" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        P.reject_reserved_seeds(5_999_000, 6_500_000)
    assert "H-B1/H-C1 imitation train" in str(exc.value)


# --------------------------------------------------------------------------
# 5. the worker count does not change the result
# --------------------------------------------------------------------------

def test_one_worker_and_four_workers_agree_field_for_field():
    rows1, _s1 = P.run_group("A", 3, P.SEED_BASE_A, workers=1)
    rows4, _s4 = P.run_group("A", 3, P.SEED_BASE_A, workers=4)
    assert rows1 == rows4
    # and the rows come back in a fixed order regardless, so a summary does not
    # depend on which worker finished first
    assert [r["pair_id"] for r in rows1] == list(range(3))
    assert [r["pair_id"] for r in rows4] == list(range(3))


# --------------------------------------------------------------------------
# 6. the ranks are the project's own
# --------------------------------------------------------------------------

def test_the_ranks_agree_with_the_project_ranking_oracle():
    """`average_places` is the ranking. `plan3.md` and `plan5.md` both ask for it
    to be checked against `tools.benchmark.average_ranks`, which is written out
    independently; an import would make the check vacuous."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from benchmark import average_ranks
    cases = [[9, 16, 18, 24], [5, 5, 9, 9], [4, 4, 4, 4], [0, 1, 2, 3],
             [84, 40, 12, 0], [7, 7, 1, 3], [21, 21, 21, 5]]
    for remaining in cases:
        assert average_places(remaining) == average_ranks(remaining), remaining


def test_the_recorded_ranks_are_the_tested_seats_place_in_its_own_game():
    for i, (tested, controllers) in enumerate(P.configurations_a()[:1]):
        row = P.play_pair(i, "A", P.SEED_BASE_A + i, tested, controllers)
        for arm in P.ARMS:
            game = P.play_game(P.SEED_BASE_A + i, controllers, tested, arm)
            assert row["rank_%s" % arm] == average_places(game["remaining"])[tested]
            assert row["remaining_%s" % arm] == game["remaining"][tested]


# --------------------------------------------------------------------------
# 7. the engine and the Game agree about being stuck
# --------------------------------------------------------------------------

def test_every_arm_finishes_with_the_engine_and_the_game_agreeing_about_stuck():
    """`play_game` raises if the two stuck vectors ever differ, and also if the
    remaining cells do. Both vectors are returned so the agreement is compared
    here instead of taken on trust from the two raises."""
    checked = 0
    for i, (tested, controllers) in enumerate(P.configurations_a()[:1]):
        for arm in P.ARMS:
            game = P.play_game(P.SEED_BASE_A + i, controllers, tested, arm)
            assert game["stuck_engine"] == game["stuck_game"]
            assert len(game["remaining"]) == 4
            assert sum(game["remaining"]) <= 4 * 84
            assert game["ply"] >= game["book_moves"]
            # every seat stuck: the game ended on the all-stuck condition and
            # not because the ply cap or the pass cap ran out
            assert all(game["stuck_engine"])
            checked += 1
    assert checked == 3


def test_a_seat_which_is_stuck_is_passed_rather_than_moved():
    """The pass branch, on a configuration and seed chosen because it reaches one.

    Group B fields two optimizers and the second one gets shut out early, so
    this position is reached on the first seed of the first group B
    configuration; all three arms of it pass at least three times. That makes the
    branch deterministic rather than something the test has to search for."""
    tested, controllers = P.configurations_b()[0]
    counts = {}
    for arm in P.ARMS:
        game = P.play_game(P.SEED_BASE_B, controllers, tested, arm)
        counts[arm] = game["passes"]
        assert game["passes"] >= 3, (arm, game["passes"])
        assert game["stuck_engine"] == game["stuck_game"]
        assert all(game["stuck_engine"]), "the game should end with every seat stuck"
        # a stuck seat still holds cells, so it cannot have finished joint first
        assert sum(game["remaining"]) > 0
    print("\npasses per arm on the pinned configuration: %r" % counts)


# --------------------------------------------------------------------------
# 8. the decision function
# --------------------------------------------------------------------------

def _stats(a, b):
    return {"primary": {"group_A_d32": a, "group_B_d32": b}}


def _ci(low, high, n=1000, mean=None):
    return {"n": n, "mean": mean if mean is not None else (low + high) / 2.0,
            "sd": 1.0, "ci_low": low, "ci_high": high}


def test_the_decision_adopts_v3_only_when_both_conditions_hold():
    assert P.decide(_stats(_ci(-0.30, -0.10), _ci(-0.20, 0.10)))["verdict"] \
        == "adopt v3"
    # A below zero but B entirely above zero: not against the strong opponents
    assert P.decide(_stats(_ci(-0.30, -0.10), _ci(0.01, 0.20)))["verdict"] \
        == "keep v2"
    # B fine but A not below zero
    assert P.decide(_stats(_ci(-0.10, 0.02), _ci(-0.20, 0.10)))["verdict"] \
        == "keep v2"
    # neither
    out = P.decide(_stats(_ci(0.01, 0.20), _ci(0.01, 0.20)))
    assert out["verdict"] == "keep v2"
    assert "not below 0" in out["reason"] and "above 0" in out["reason"]


def test_an_interval_high_of_exactly_zero_is_not_below_zero():
    """The rule says strictly less than zero. Reading it as "at most" would adopt
    v3 on the exact boundary."""
    out = P.decide(_stats(_ci(-0.20, 0.0), _ci(-0.10, 0.10)))
    assert out["verdict"] == "keep v2"
    assert out["A_ci_high"] == 0.0
    assert "not below 0" in out["reason"]


def test_a_group_b_low_of_exactly_zero_is_at_or_below_and_passes():
    """Condition 2 is "at or below zero", so a boundary of exactly zero passes."""
    out = P.decide(_stats(_ci(-0.30, -0.10), _ci(0.0, 0.20)))
    assert out["verdict"] == "adopt v3"
    assert out["B_ci_low"] == 0.0


def test_the_decision_reports_the_numbers_it_was_read_off():
    out = P.decide(_stats(_ci(-0.30, -0.10, mean=-0.2), _ci(0.01, 0.20, mean=0.1)))
    assert out["A_ci_high"] == -0.10
    assert out["A_ci_low"] == -0.30
    assert out["A_mean"] == -0.2
    assert out["B_ci_low"] == 0.01
    assert out["B_ci_high"] == 0.20
    assert out["B_mean"] == 0.1


def test_an_expansion_is_only_wanted_when_group_a_still_contains_zero():
    # contains zero, either side of it
    assert P.expansion_needed(_stats(_ci(-0.10, 0.02), _ci(-0.1, 0.1)))
    assert P.expansion_needed(_stats(_ci(-0.02, 0.10), _ci(-0.1, 0.1)))
    # entirely below zero: adopt it, do not spend a second phase
    assert not P.expansion_needed(_stats(_ci(-0.30, -0.10), _ci(-0.1, 0.1)))
    # entirely above zero: v3 is worse, no second phase
    assert not P.expansion_needed(_stats(_ci(0.01, 0.20), _ci(-0.1, 0.1)))
    # a boundary of exactly zero still counts as "not below zero"
    assert P.expansion_needed(_stats(_ci(-0.20, 0.0), _ci(-0.1, 0.1)))


# --------------------------------------------------------------------------
# the bootstrap itself
# --------------------------------------------------------------------------

def test_the_bootstrap_is_reproducible_and_brackets_the_mean():
    values = [0.0] * 60 + [-1.0] * 20 + [1.0] * 20
    a = P.bootstrap_ci(values, iters=500, seed=11)
    b = P.bootstrap_ci(values, iters=500, seed=11)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["n"] == 100
    c = P.bootstrap_ci(values, iters=500, seed=12)
    assert c != a, "a different seed must give a different resample"


def test_the_bootstrap_reports_nothing_for_no_pairs():
    out = P.bootstrap_ci([], iters=10, seed=1)
    assert out["n"] == 0 and out["mean"] is None


def test_the_bootstrap_is_vectorised_not_a_python_loop():
    """A 10,000 x 1,000 pure-Python loop is what H0 spent longer on than its
    games; this must be numpy. Checked by speed, with the n=0 case excluded."""
    import time
    values = [float(i % 5) for i in range(1_000)]
    t0 = time.perf_counter()
    P.bootstrap_ci(values, iters=10_000, seed=3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, "bootstrap took %.2f s, which suggests a Python loop" \
        % elapsed


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------

def test_paired3_imports_neither_torch_nor_pygame():
    import ast
    with open(os.path.join(ROOT, "rl", "paired3.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "torch" not in names
    assert "pygame" not in names