"""A dataset is only usable if it is reproducible and if its rows are true.

Those are the two things tested here. Reproducibility is checked by running the
same seeds twice and comparing every field byte for byte - a dataset that
reproduces only in its aggregate would still let two runs disagree about which
moves were chosen, which is the thing that matters.

Truth is checked by rebuilding each sample's position from the stored columns
alone and asking the engine whether the teacher's action was legal there. The
reconstruction goes through `own_bits`, `hand_bits`, `stuck` and `to_move`, so a
column that is wrong in a way that still parses cannot pass.

Fast by construction: whole games are cached per session and the expensive
checks run over a few seeds rather than a few hundred, because a game costs a
third of a second and this file has to stay under a minute.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai.formulas  # noqa: E402
import engine  # noqa: E402
from config import B, CLOCKWISE_OWNERS, N  # noqa: E402
import rl.actions as A  # noqa: E402
import rl.collect as C  # noqa: E402

# The plan asks for 20 games here; that is what these constants mean.
SPOT_CHECK_GAMES = 20
TRAIN_BASE = C.TRAIN_SEED_BASE
VALID_BASE = C.VALID_SEED_BASE

_CACHE = {}


def games(seeds):
    rows = []
    summary = []
    for seed in seeds:
        if seed not in _CACHE:
            prefix_len, controllers = C.draw_setup(seed)
            _CACHE[seed] = C.collect_game(seed, prefix_len, controllers)
        r, g = _CACHE[seed]
        rows.extend(r)
        summary.append(g)
    return rows, summary


def train_seeds(n=SPOT_CHECK_GAMES):
    return list(range(TRAIN_BASE, TRAIN_BASE + n))


def valid_seeds(n=5):
    return list(range(VALID_BASE, VALID_BASE + n))


def same(a, b):
    """Field equality that treats NaN as equal to NaN.

    `np.array_equal` reports False for two identical arrays of NaN unless told
    otherwise, and `shortlist_score` pads its unused slots with NaN on purpose,
    so the padding would read as a difference on every row.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.dtype.kind == "f":
            return np.array_equal(a, b, equal_nan=True)
        return np.array_equal(a, b)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def rebuild(row):
    """The `engine.State` a row describes, from the stored columns only."""
    own = tuple(int.from_bytes(bytes(row["own_bits"][o]), "little")
                for o in range(4))
    hand = tuple(int(row["hand_bits"][o]) for o in range(4))
    stuck = tuple(bool(row["stuck"][o]) for o in range(4))
    return engine.State(own_bits=own, hand_bits=hand,
                        turn_order=CLOCKWISE_OWNERS,
                        to_move=int(row["to_move"]), stuck=stuck)


# --------------------------------------------------------------- reproducibility

def test_the_same_seed_produces_the_same_samples_twice():
    seeds = train_seeds(8)
    a_rows, a_games = games(seeds)
    _CACHE.clear()
    b_rows, b_games = games(seeds)
    assert len(a_rows) == len(b_rows) > 0
    for x, y in zip(a_rows, b_rows):
        for name in C.FIELD_ORDER:
            assert same(x[name], y[name]), (name, int(x["game_id"]))
    assert [g["controllers"] for g in a_games] == \
           [g["controllers"] for g in b_games]


def test_rows_come_back_sorted_by_game_and_ply():
    rows, _summary = games(train_seeds(8))
    keys = [(int(r["game_id"]), int(r["ply"])) for r in rows]
    assert keys == sorted(keys)
    # And ply never goes backwards inside a game.
    for gid in {int(r["game_id"]) for r in rows}:
        plies = [int(r["ply"]) for r in rows if int(r["game_id"]) == gid]
        assert plies == sorted(plies)
        assert len(set(plies)) == len(plies), "one position produced two samples"


# --------------------------------------------------------------- legality

def test_every_recorded_action_is_legal_in_its_own_position():
    """The spot check the plan asks for: 20 games, position rebuilt from the
    columns, action checked against the engine."""
    rows, _summary = games(train_seeds())
    assert len({int(r["game_id"]) for r in rows}) == SPOT_CHECK_GAMES
    for r in rows:
        state = rebuild(r)
        mask = engine.legal_move_mask(state)
        assert mask != 0, "a sample was recorded at a pass decision point"
        move = A.index_to_move(int(r["action"]))
        assert engine.move_bit(*move) & mask, (int(r["game_id"]), int(r["ply"]))
        assert int(r["n_legal"]) == mask.bit_count()


def test_the_reconstruction_is_exact_not_merely_legal():
    """Comparing the stored columns against the engine's own serialisation of
    the replayed position, which is a stricter check than legality."""
    rows, _summary = games(train_seeds(6))
    for r in rows:
        state = rebuild(r)
        assert state.to_move == int(r["to_move"])
        assert all(state.hand_bits[o] == int(r["hand_bits"][o])
                   for o in range(4))
        assert all(state.stuck[o] == bool(r["stuck"][o]) for o in range(4))
        # The stored bytes have to reconstruct the exact integers, and the number of
        # set bits has to match the number of stones played so far. Note the
        # byte *sum* is not the bit count - one byte can be 0x37, five set bits
        # and a value of 55.
        for o in range(4):
            raw = bytes(r["own_bits"][o])
            assert int.from_bytes(raw, "little") == int(state.own_bits[o]), o
            assert len(raw) == 50
        placed = sum(bin(int(state.own_bits[o])).count("1") for o in range(4))
        # Every cell in a hand is a cell not yet on the board, and a whole
        # game starts with 4 x 89 of them, so the two must agree.
        in_hand = sum(engine.remaining_cells(state, o) for o in range(4))
        assert placed + in_hand == 4 * 89, (placed, in_hand)
        assert placed >= int(r["ply"]), "fewer squares than pieces played"


def test_pass_decision_points_never_appear():
    for seed in train_seeds(10):
        rows, _summary = games([seed])
        for r in rows:
            assert engine.legal_move_mask(rebuild(r)) != 0


def test_a_game_whose_stuck_flags_diverge_would_be_rejected():
    """The lockstep check is real, not decorative: a deliberately wrong stuck
    flag has to be detectable from the stored columns."""
    rows, _summary = games(train_seeds(2))
    r = rows[0]
    broken = dict(r)
    broken["stuck"] = np.array([not bool(v) for v in r["stuck"]], dtype=bool)
    state = rebuild(broken)
    assert state.stuck != tuple(bool(v) for v in r["stuck"])


# --------------------------------------------------------------- labels

def test_every_sample_comes_from_an_optimizer_seat():
    rows, summary = games(train_seeds())
    controllers = {g["game_id"]: g["controllers"] for g in summary}
    for r in rows:
        assert controllers[int(r["game_id"])][int(r["to_move"])] == C.TEACHER


def test_the_teacher_always_takes_the_shortlists_first_entry():
    """The plan flags this as an anomaly if it is not so. It is, because the
    teacher is deterministic and stage F' measured its mistake rate at zero."""
    rows, _summary = games(train_seeds(10))
    first = [int(r["shortlist_idx"][0]) for r in rows]
    assert all(i >= 0 for i in first), "an empty shortlist"
    assert all(int(r["action"]) == f for r, f in zip(rows, first))
    # And the shortlist is a superset of the chosen move.
    for r in rows:
        idx = r["shortlist_idx"]
        assert (idx == int(r["action"])).any()


def test_the_shortlist_padding_is_minus_one_and_nan():
    rows, _summary = games(train_seeds(6))
    for r in rows:
        idx = r["shortlist_idx"]
        score = r["shortlist_score"]
        assert idx.shape == (C.SHORTLIST_PAD,)
        assert score.shape == (C.SHORTLIST_PAD,)
        filled = idx >= 0
        # Padding is contiguous at the end: a gap would mean the shortlist was
        # written out of order.
        assert list(filled) == sorted(filled, reverse=True)
        assert np.isnan(score[~filled]).all()
        assert np.isfinite(score[filled]).all()
        # Every listed action is a real placement.
        for i in idx[filled]:
            assert A.is_real_index(int(i))


# --------------------------------------------------------------- splits

def test_training_and_validation_games_are_disjoint():
    t_rows, _ = games(train_seeds(10))
    v_rows, _ = games(valid_seeds(5))
    t_ids = {int(r["game_id"]) for r in t_rows}
    v_ids = {int(r["game_id"]) for r in v_rows}
    assert t_ids and v_ids
    assert not (t_ids & v_ids)


def test_the_seed_ranges_avoid_the_already_used_ones():
    trains = train_seeds(10)
    valids = valid_seeds(5)
    C.check_seed_ranges(trains, valids)          # must not raise
    # bench_engine.py used 0..199, and tools/benchmark.py drew from 20240101.
    for seed in trains + valids:
        assert seed > C.BENCH_SEED_MAX
        assert not (C.BENCHMARK_SEED_BASE <= seed
                    <= C.BENCHMARK_SEED_BASE
                    + C.BENCHMARK_SEED_STRIDE * C.BENCHMARK_MAX_GAMES)
    # And the check rejects a range that reuses one.
    with pytest.raises(ValueError):
        C.check_seed_ranges([0, 1], [VALID_BASE])
    with pytest.raises(ValueError):
        C.check_seed_ranges([C.BENCHMARK_SEED_BASE + 1009], [VALID_BASE])
    with pytest.raises(ValueError):
        C.check_seed_ranges(train_seeds(3), train_seeds(2))


# --------------------------------------------------------------- setup draws

def test_every_game_has_at_least_one_teacher_seat():
    for seed in range(TRAIN_BASE, TRAIN_BASE + 200):
        prefix_len, controllers = C.draw_setup(seed)
        assert C.TEACHER in controllers
        assert prefix_len in C.PREFIX_CHOICES
        assert len(controllers) == 4


def test_the_controller_weights_are_the_ones_the_plan_asks_for():
    """All seven personalities, the teacher at half and the rest splitting it.

    Hunter is the teacher (`plan6.md` stage H) and appears exactly once: a weight
    table is a per-seat draw, so a second entry would put two Hunters in a game.
    """
    weights = dict(C.CONTROLLER_WEIGHTS)
    assert C.TEACHER == "hunter"
    assert weights["hunter"] == 0.5
    for key in ("wolf", "chess", "fox", "intruder", "optimizer", "builder"):
        assert weights[key] == pytest.approx(0.5 / 6), key
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == set(C.CONTROLLER_KEYS)
    # one entry per personality: no duplicates in the draw table
    assert len(C.CONTROLLER_KEYS) == len(set(C.CONTROLLER_KEYS)) == 7
    assert len(C.CONTROLLER_WEIGHTS) == 7


def test_the_drawn_seat_share_is_close_to_the_design():
    """200 games x 4 seats = 800 draws; the teacher is drawn half the time."""
    counts = {}
    for seed in range(TRAIN_BASE, TRAIN_BASE + 200):
        _prefix, controllers = C.draw_setup(seed)
        for key in controllers:
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    assert total == 800
    for key, weight in C.CONTROLLER_WEIGHTS:
        share = counts[key] / total
        assert abs(share - weight) < 0.05, (key, share, weight, counts)


def test_the_prefix_length_distribution_is_the_design():
    counts = {}
    for seed in range(TRAIN_BASE, TRAIN_BASE + 400):
        prefix_len, _controllers = C.draw_setup(seed)
        counts[prefix_len] = counts.get(prefix_len, 0) + 1
    assert set(counts) == set(C.PREFIX_CHOICES)
    assert sum(counts.values()) == 400
    assert all(v > 0 for v in counts.values())


def test_no_teacher_sample_comes_from_an_opening_prefix():
    """B1: the prefix is off for the teacher's own seat.

    The old version of this test asserted the opposite - that `in_prefix` samples
    exist and that they sit inside the prefix. That was right while the prefix
    applied to every seat, and it measured 8.312% of samples. It cannot hold now:
    a prefix ply asks the teacher for a move and then discards it, which advances
    Hunter's book counter without the player having moved, so the teacher's first
    three moves stop being book moves (measured: 79.9% / 49.8% / 21.0% of book
    steps 0 / 1 / 2 displaced). `plan6.md` decision 7 requires those three to
    survive as supervised labels.

    What the prefix still does is move the board: the other three seats keep
    playing random opening moves, so the positions the teacher is later asked
    about are reached differently. Only the *labelled* prefix samples are gone.
    """
    rows, _summary = games(train_seeds(10))
    by_game = {g["game_id"]: g["prefix_len"] for g in _summary}
    assert rows, "no samples at all"
    inside_prefix_len = 0
    for r in rows:
        assert not bool(r["in_prefix"]), \
            "a teacher sample was marked in_prefix; the prefix must be off for it"
        if int(r["ply"]) < by_game[int(r["game_id"])]:
            inside_prefix_len += 1
    # the prefix still covers plies: it just does not relabel the teacher's move
    assert inside_prefix_len > 0, \
        "no sample sits inside its game's prefix length, so the prefix is inert"


# --------------------------------------------------------------- the budget switch

def test_the_wall_budget_is_restored_after_a_run():
    before = ai.formulas.USE_WALL_BUDGET
    games(train_seeds(2))
    assert ai.formulas.USE_WALL_BUDGET == before


def test_the_wall_budget_is_restored_even_when_the_body_raises():
    """A collection that dies half way through must not leave the engine with
    its clock switched off for the rest of the process."""
    before = ai.formulas.USE_WALL_BUDGET
    assert before is True, "the default should be on, or this proves nothing"
    with pytest.raises(RuntimeError):
        with C._Budget(False):
            assert ai.formulas.USE_WALL_BUDGET is False
            raise RuntimeError("boom")
    assert ai.formulas.USE_WALL_BUDGET == before


def test_collection_succeeds_with_the_budget_on_and_restores_it():
    """The whole path, with the default left in place. If a future change
    forgot the context manager, this is where it would show."""
    before = ai.formulas.USE_WALL_BUDGET
    prefix_len, controllers = C.draw_setup(TRAIN_BASE + 500)
    rows, summary = C.collect_game(TRAIN_BASE + 500, prefix_len, controllers)
    assert rows and summary["ply"] > 0
    assert ai.formulas.USE_WALL_BUDGET == before


# --------------------------------------------------------------- shards

def test_a_shard_round_trips_and_matches_the_field_spec(tmp_path):
    rows, _summary = games(train_seeds(6))
    path = str(tmp_path / "shard.npz")
    written = C.write_shard(path, rows)
    assert set(written) == set(C.FIELD_ORDER)
    back = C.load_shard(path)
    assert set(back) == set(C.FIELD_ORDER)
    for name in C.FIELD_ORDER:
        assert back[name].dtype == np.dtype(C.FIELD_SPEC[name][0]), name
        assert back[name].shape == (len(rows),) + C.FIELD_SPEC[name][1], name
        want = C.stack(rows, C.FIELD_ORDER.index(name))
        assert same(back[name], want), name


def test_a_shard_name_carries_the_seed_range():
    assert C.shard_name("imit_train", 1_000_000, 1_000_099) \
        == "imit_train_1000000_1000099.npz"


def test_the_summary_counts_what_the_rows_contain(tmp_path):
    rows, summary = games(train_seeds(6))
    stats = C.summarise(rows, summary)
    assert stats["samples"] == len(rows)
    assert stats["games"] == len(summary)
    assert 0.0 <= stats["in_prefix_share"] <= 1.0
    # The teacher is the shortlist's head: true when it scores the shortlist
    # itself, and also when the book narrows it to the single drawn candidate,
    # because then the head *is* the teacher's pick by construction.
    assert stats["teacher_equals_shortlist_first_share"] == 1.0
    assert 0.0 <= stats["teacher_mean_utility"] <= 1.0
    assert stats["n_legal"]["max"] <= 30433
    assert stats["n_legal"]["min"] >= 1
    assert stats["shortlist_len"]["max"] <= C.SHORTLIST_PAD


def test_the_manifest_records_what_the_data_is(tmp_path):
    out = str(tmp_path)
    trains = train_seeds(6)
    valids = valid_seeds(3)
    manifest, results = C.run(trains, valids, out_dir=out, workers=1,
                              prefix="probe")
    assert manifest["total_games"] == len(trains) + len(valids)
    assert manifest["total_samples"] > 0
    assert manifest["teacher"] == C.TEACHER == "hunter"
    assert manifest["action_table_hash"] == "91252a354b090d43"
    assert manifest["code"]["commit"]
    assert os.path.exists(os.path.join(out, "manifest.json"))
    for shard in manifest["shards"]:
        assert shard["samples"] <= C.MAX_SAMPLES_PER_SHARD
        assert os.path.exists(os.path.join(out, shard["path"]))
    # Seeds do not overlap between the two groups.
    train_lo_hi = [(s["seed_lo"], s["seed_hi"]) for s in manifest["shards"]
                   if s["prefix"].endswith("train")]
    valid_lo_hi = [(s["seed_lo"], s["seed_hi"]) for s in manifest["shards"]
                   if s["prefix"].endswith("valid")]
    for t in train_lo_hi:
        for v in valid_lo_hi:
            assert t[1] < v[0] or v[1] < t[0], (t, v)


def test_one_worker_and_four_workers_produce_the_same_samples(tmp_path):
    """The parallelism claim: no RNG is shared, so the division of seeds does
    not change the rows."""
    seeds = train_seeds(8)
    one_dir = str(tmp_path / "one")
    four_dir = str(tmp_path / "four")
    C.run(seeds, [], out_dir=one_dir, workers=1, prefix="probe")
    C.run(seeds, [], out_dir=four_dir, workers=4, prefix="probe")

    def read_all(directory):
        """Every sample keyed by (game, ply), compared field by field with
        `same` so the NaN padding does not read as a difference."""
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.loads(fh.read())
        rows = {}
        for shard in manifest["shards"]:
            data = C.load_shard(os.path.join(directory, shard["path"]))
            for i in range(data["action"].size):
                key = (int(data["game_id"][i]), int(data["ply"][i]))
                rows[key] = {name: data[name][i].copy()
                             for name in C.FIELD_ORDER}
        return rows

    a = read_all(one_dir)
    b = read_all(four_dir)
    assert set(a) == set(b)
    for key in a:
        for name in C.FIELD_ORDER:
            assert same(a[key][name], b[key][name]), (key, name)
