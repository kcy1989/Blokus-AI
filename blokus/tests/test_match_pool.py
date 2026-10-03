"""`match.py --pool`: which options a league run draws from.

The pool is a **set**, not a multiset. Naming `hunter` alongside the
`no_imitation` preset does not give `hunter` a second chance, because it is
already in that preset; every distinct option in the expanded pool is worth
1/n. And the expansion order is fixed by `seats.automated_options()` rather
than by the order the items were typed in, so two `--pool` arguments with the
same contents play exactly the same games from the same seed.

The last test is the one that matters most for regression: without `--pool`, the
league must still play the same games it always did, byte for byte. The
expected values in it were captured from the code before this feature existed.
"""
import collections
import json
import math
import os
import subprocess
import sys

import pytest

import match
import seats as seats_mod
from match import (POOL_PRESETS, expand_pool, league_options, main, play_match,
                   run_league, summarise)

AUTOMATED = seats_mod.automated_options()
PERSONALITIES = [k for k in AUTOMATED if not k.startswith("step_")]
STEPS = [k for k in AUTOMATED if k.startswith("step_")]


def flat(rows):
    """Every seat row of every game."""
    return [r for game in rows for r in game]


def seats_of(rows):
    """`(option, colour, remaining)` only, for comparing against old output."""
    return [(k, c, r) for k, c, r, _rk, _p in flat(rows)]


# One 40-game run, shared. Two of the tests below need a few hundred seats to
# tell a fair 1/7 from a doubled 2/8 - 123 seats is about the least that can -
# and playing two such runs instead of one doubles the CPU this file puts on a
# parallel test run, which is what a neighbouring wall-clock assertion notices.
_SHARE = {}


def share_run(games=40, seed=777, pool="no_imitation"):
    key = (games, seed, pool)
    if key not in _SHARE:
        rows = run_league(games, seed=seed, options=expand_pool(pool))
        cnt = collections.Counter(k for k, _c, _r, _rk, _p in flat(rows))
        _SHARE[key] = cnt
    return _SHARE[key]


# ------------------------------------------------------------------ errors

def test_an_unknown_name_is_refused_and_lists_what_is_legal():
    with pytest.raises(ValueError) as exc:
        expand_pool("wizard")
    msg = str(exc.value)
    assert "wizard" in msg
    for name in ("all", "no_imitation", "imitation_only"):
        assert name in msg
    for name in AUTOMATED:
        assert name in msg


def test_an_unknown_name_among_valid_ones_still_fails():
    """A typo must not quietly drop an option and change what is measured."""
    with pytest.raises(ValueError) as exc:
        expand_pool("no_imitation,step_20000")
    assert "step_20000" in str(exc.value)


def test_an_empty_pool_is_refused():
    for text in ("", "   ", ",", " , , "):
        with pytest.raises(ValueError):
            expand_pool(text)


def test_the_cli_reports_a_bad_pool_as_a_usage_error(tmp_path):
    out = str(tmp_path / "o.json")
    with pytest.raises(SystemExit) as exc:
        main(["--games", "1", "--pool", "wizard", "--dry", "--out", out])
    assert exc.value.code == 2
    assert not os.path.exists(out)


# ----------------------------------------------------------------- presets

def test_the_three_presets_have_the_ruled_sizes():
    assert len(POOL_PRESETS["all"]) == 11
    assert len(POOL_PRESETS["no_imitation"]) == 7
    assert len(POOL_PRESETS["imitation_only"]) == 4
    assert POOL_PRESETS["no_imitation"] == tuple(PERSONALITIES)
    assert POOL_PRESETS["imitation_only"] == tuple(STEPS)
    assert len(expand_pool("all")) == 11
    assert len(expand_pool("no_imitation")) == 7
    assert len(expand_pool("imitation_only")) == 4


def test_presets_expand_to_the_expected_names():
    assert expand_pool("all") == list(AUTOMATED)
    assert expand_pool("no_imitation") == PERSONALITIES
    assert expand_pool("imitation_only") == STEPS


# -------------------------------------------------------------- mixed lists

def test_a_preset_plus_one_name_is_a_union():
    """`no_imitation,step_2000` is the seven personalities plus one checkpoint."""
    pool = expand_pool("no_imitation,step_2000")
    assert len(pool) == 8
    assert pool == [k for k in AUTOMATED
                    if k in PERSONALITIES or k == "step_2000"]


def test_the_ruled_mixed_examples():
    assert len(expand_pool("no_imitation,step_2000")) == 8
    assert len(expand_pool("no_imitation,hunter")) == 7
    assert len(expand_pool("no_imitation,step_2000,step_38000")) == 9
    assert len(expand_pool("step_2000,step_2000")) == 1


def test_an_overlapping_name_adds_no_weight():
    """`hunter` is already in `no_imitation`, so naming it changes nothing -
    not the list, and not the draw."""
    assert expand_pool("no_imitation,hunter") == expand_pool("no_imitation")
    assert "hunter" in expand_pool("no_imitation,hunter")
    assert seats_of(run_league(1, seed=3,
                               options=expand_pool("no_imitation,hunter"))) \
        == seats_of(run_league(1, seed=3,
                               options=expand_pool("no_imitation")))


def test_a_repeated_item_collapses_to_one():
    assert expand_pool("step_2000,step_2000") == ["step_2000"]
    assert len(expand_pool("step_2000,step_2000")) == 1


# ------------------------------------------------------------------- order

def test_the_expanded_order_does_not_depend_on_the_input_order():
    pairs = [("step_2000,no_imitation", "no_imitation,step_2000"),
             ("all", "no_imitation,imitation_only"),
             ("hunter,wolf", "wolf,hunter"),
             ("step_38000,imitation_only,no_imitation", "all")]
    for a, b in pairs:
        assert expand_pool(a) == expand_pool(b), (a, b)


def test_the_expanded_order_is_the_fixed_option_order():
    for text in ("all", "step_38000,step_2000", "hunter,wolf",
                 "no_imitation,imitation_only"):
        pool = expand_pool(text)
        assert pool == [k for k in AUTOMATED if k in set(pool)], text


def test_the_same_pool_played_in_two_orders_gives_the_same_games():
    """Not just the same list - the same games. The draw is over the list, so a
    different order would shuffle every seat."""
    a = seats_of(run_league(2, seed=5, options=expand_pool("no_imitation,step_2000")))
    b = seats_of(run_league(2, seed=5, options=expand_pool("step_2000,no_imitation")))
    assert a == b


def test_whitespace_and_quoting_are_tolerated():
    want = expand_pool("no_imitation,step_2000")
    for text in ("no_imitation,step_2000",
                 "no_imitation, step_2000",
                 "  no_imitation ,  step_2000  ",
                 '"no_imitation,step_2000"',
                 "'no_imitation,step_2000'"):
        assert expand_pool(text) == want, text


# -------------------------------------------------------------- what is in

def test_no_option_outside_the_pool_ever_appears():
    for text, allowed in (("no_imitation", set(PERSONALITIES)),
                          ("imitation_only", set(STEPS)),
                          ("step_2000,step_22000", {"step_2000",
                                                     "step_22000"}),
                          ("no_imitation,step_2000",
                           set(PERSONALITIES) | {"step_2000"})):
        rows = run_league(2, seed=11, options=expand_pool(text))
        assert {k for k, _c, _r, _rk, _p in flat(rows)} <= allowed, text
    # specifically: no step_* in the personality pool
    rows = run_league(2, seed=11, options=expand_pool("no_imitation"))
    assert not any(k.startswith("step_") for k, *_ in flat(rows))


def test_every_option_in_a_pool_gets_its_share():
    """A uniform draw from seven options over 160 seats.

    A fair share is 1/7 = 0.143 with a standard error of 0.028, so the band is
    a little over three sigma - and it is also the band that tells a fair 1/7
    apart from the 0.25 that a doubled weight would produce.
    """
    cnt = share_run()
    assert set(cnt) == set(PERSONALITIES)
    total = sum(cnt.values())
    assert total == 40 * 4
    for k in PERSONALITIES:
        assert 0.07 < cnt[k] / total < 0.22, (k, cnt[k] / total)


def test_an_overlapping_name_is_drawn_at_one_seventh_not_two_eighths():
    """The statistical half of "a repeated item adds no weight"."""
    cnt = share_run()
    total = sum(cnt.values())
    share = cnt["hunter"] / total
    assert 0.07 < share < 0.22, share
    assert share < 2.0 / 8.0, share


def test_repeat_drawing_is_still_allowed():
    rows = run_league(20, seed=5, options=expand_pool("no_imitation"))
    repeated = sum(1 for game in rows if len({k for k, *_ in game}) < 4)
    assert repeated > 5, repeated


# ------------------------------------------------------------ reproducibility

def test_the_same_seed_replays():
    for text in ("all", "no_imitation", "no_imitation,step_2000"):
        a = seats_of(run_league(2, seed=31, options=expand_pool(text)))
        b = seats_of(run_league(2, seed=31, options=expand_pool(text)))
        assert a == b, text


def test_different_seeds_differ():
    a = seats_of(run_league(2, seed=1, options=expand_pool("all")))
    b = seats_of(run_league(2, seed=2, options=expand_pool("all")))
    assert a != b


def test_no_pool_at_all_means_all():
    assert expand_pool(None) == list(league_options())
    assert expand_pool(None) == list(AUTOMATED)


# The values below were produced by `run_league(2, seed=...)` before `--pool`
# existed, with the options list being `league_options()` and each row being
# `(option, colour, remaining)`. They are the regression guard for the claim
# that a default run is unchanged: same length and same order in the expanded
# pool means `rng.choice` consumes the stream identically, so the seats, the
# colours and the scores all come out the same.
GOLDEN = {
    20260928: [('step_38000', 'red', 5), ('step_10000', 'yellow', 8),
               ('optimizer', 'blue', 17), ('wolf', 'green', 31),
               ('step_22000', 'yellow', 4), ('hunter', 'red', 26),
               ('hunter', 'green', 12), ('wolf', 'blue', 21)],
    7: [('builder', 'green', 14), ('fox', 'red', 22), ('hunter', 'yellow', 12),
        ('step_38000', 'blue', 0), ('builder', 'yellow', 4), ('wolf', 'blue', 14),
        ('step_2000', 'red', 10), ('builder', 'green', 26)],
    99: [('hunter', 'yellow', 8), ('hunter', 'red', 18), ('intruder', 'blue', 13),
         ('step_22000', 'green', 7), ('hunter', 'red', 5), ('chess', 'blue', 29),
         ('chess', 'yellow', 18), ('chess', 'green', 25)],
}


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_a_default_run_still_plays_the_games_it_always_did(seed):
    assert seats_of(run_league(2, seed=seed)) == GOLDEN[seed]


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_pool_all_is_identical_to_no_pool_at_all(seed):
    """Same list, same order, same length - so the same random draws."""
    assert expand_pool("all") == expand_pool(None)
    assert seats_of(run_league(2, seed=seed, options=expand_pool("all"))) \
        == GOLDEN[seed]
    assert seats_of(run_league(2, seed=seed)) == \
        seats_of(run_league(2, seed=seed, options=expand_pool("all")))


def test_all_equals_the_two_presets_that_partition_it():
    rows = seats_of(run_league(1, seed=20260928, options=expand_pool("all")))
    same = seats_of(run_league(1, seed=20260928,
                               options=expand_pool("no_imitation,imitation_only")))
    assert rows == same


# ------------------------------------------------------------------- output

def test_each_seat_row_carries_its_own_rank_and_points():
    rows = run_league(1, seed=20260928)
    assert len(rows) == 1 and len(rows[0]) == 4
    for row in rows[0]:
        assert len(row) == 5
        key, colour, remaining, rank, points = row
        assert key in AUTOMATED
        assert colour in ("blue", "green", "red", "yellow")
        assert isinstance(rank, int) and 1 <= rank <= 4
        assert isinstance(points, int) and 1 <= points <= 4
    assert sorted(r for _k, _c, _rem, r, _p in rows[0]) == [1, 2, 3, 4]
    assert sorted(p for _k, _c, _rem, _r, p in rows[0]) == [1, 2, 3, 4]


def test_a_repeated_option_keeps_two_distinct_places():
    """The fix that made same-option seats score separately: with a one-option
    pool all four seats share a key, and each still carries its own rank."""
    rows = run_league(1, seed=3, options=expand_pool("step_2000"))
    assert {k for k, *_ in rows[0]} == {"step_2000"}
    assert sorted(r for _k, _c, _rem, r, _p in rows[0]) == [1, 2, 3, 4]
    # a tie would share a place, so distinct ranks means distinct scores
    assert len({rem for _k, _c, rem, _r, _p in rows[0]}) == 4


def test_summarise_counts_every_seat_not_every_key():
    rows = [[("a", "blue", 0, 1, 4), ("a", "green", 30, 4, 1),
             ("b", "red", 10, 2, 3), ("c", "yellow", 20, 3, 2)]]
    got = {r["option"]: r for r in summarise(rows[0])}
    assert got["a"]["appearances"] == 2
    assert got["a"]["avg_points"] == 2.5
    assert got["a"]["avg_remaining"] == 15.0
    assert got["b"]["appearances"] == 1


def test_summarise_reports_the_sample_standard_error():
    rows = [[("a", "blue", rem, 1, p) for rem, p in
             ((0, 4), (10, 4), (20, 4), (30, 4))]]
    row = summarise(rows[0])[0]
    n = 4
    var = sum((p - 4.0) ** 2 for p in (4, 4, 4, 4)) / (n - 1)
    assert var == 0.0
    assert row["points_stderr"] == 0.0
    # a spread case
    rows = [[("a", "blue", 0, 1, p) for p in (4, 1)]]
    row = summarise(rows[0])[0]
    assert row["appearances"] == 2
    mean = 2.5
    var = ((4 - mean) ** 2 + (1 - mean) ** 2) / 1
    assert row["points_stderr"] == pytest.approx(math.sqrt(var) / math.sqrt(2))


def test_summarise_leaves_the_stderr_at_zero_for_one_appearance():
    row = summarise([("a", "blue", 5, 3, 2)])[0]
    assert row["appearances"] == 1
    assert row["points_stderr"] == 0.0


# -------------------------------------------------------------- the CLI

def test_the_cli_writes_its_own_file_and_not_the_leaderboard(tmp_path, monkeypatch):
    """The batch output must not land in records.json."""
    rec = tmp_path / "records.json"
    monkeypatch.setattr(match, "RECORDS_DEFAULT", str(rec), raising=False)
    out = tmp_path / "batch.json"
    rc = main(["--games", "3", "--seed", "99", "--pool", "no_imitation",
               "--dry", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pool"] == PERSONALITIES
    assert data["pool_raw"] == "no_imitation"
    assert data["pool_size"] == 7
    assert data["seed"] == 99
    assert data["games"] == 3
    assert len(data["games_detail"]) == 3
    assert all(len(g) == 4 for g in data["games_detail"])
    assert all(len(row) == 5 for g in data["games_detail"] for row in g)
    assert sum(data["appearances"].values()) == 12
    # --dry, so the leaderboard was never opened for writing
    assert not rec.exists()


def test_the_default_output_path_is_under_data_and_not_records(tmp_path, monkeypatch):
    monkeypatch.setattr(match, "DEFAULT_OUT_DIR", str(tmp_path / "m"), raising=False)
    path = match.default_out_path(7, 3, "no_imitation,step_2000")
    assert os.path.dirname(path) == str(tmp_path / "m")
    assert path.endswith(".json")
    assert "records" not in os.path.basename(path)
    assert os.path.isdir(os.path.dirname(path))


def test_the_default_output_path_is_the_same_for_the_same_run(tmp_path, monkeypatch):
    monkeypatch.setattr(match, "DEFAULT_OUT_DIR", str(tmp_path / "m"), raising=False)
    a = match.default_out_path(7, 3, "no_imitation,step_2000")
    b = match.default_out_path(7, 3, "no_imitation,step_2000")
    assert a == b


def test_the_steps_flag_is_the_imitation_preset(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    main(["--games", "2", "--seed", "5", "--steps", "--dry", "--out", str(a)])
    main(["--games", "2", "--seed", "5", "--pool", "imitation_only",
          "--dry", "--out", str(b)])
    ja, jb = json.loads(a.read_text(encoding="utf-8")), \
        json.loads(b.read_text(encoding="utf-8"))
    assert ja["games_detail"] == jb["games_detail"]
    assert ja["pool"] == STEPS


def test_the_help_says_how_the_union_weights():
    out = subprocess.run([sys.executable, "match.py", "--help"],
                         capture_output=True, text=True).stdout
    assert "--pool" in out
    assert "no_imitation" in out and "imitation_only" in out and "all" in out
    # the weighting rule and the de-duplication are both documented
    assert "1/n" in out
    assert "去重" in out


def test_nothing_is_written_into_the_gitignored_data_directory_by_the_tests(tmp_path):
    """The pool tests themselves leave no files in the repository."""
    assert not (tmp_path / "records.json").exists()