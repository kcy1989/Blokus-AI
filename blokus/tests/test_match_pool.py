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
PERSONALITIES = [k for k in AUTOMATED if not seats_mod.is_imitation_key(k)]
STEPS = [k for k in AUTOMATED if seats_mod.is_imitation_key(k)]


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


def test_the_retired_step_names_are_refused_rather_than_remapped():
    """H-B2's option names must not resolve to anything.

    The files are still in `data/hb2` and `load_brain` would still open them, so
    the only thing standing between a stale `--pool` string and a network trained
    on a `stuck` column that disagreed with the rules is this refusal. A name
    that quietly mapped to a current checkpoint would be worse than either
    option: the run would succeed and the number would mean something else.
    """
    for old in ("step_2000", "step_10000", "step_22000", "step_38000"):
        with pytest.raises(ValueError) as exc:
            expand_pool(old)
        assert old in str(exc.value)
        with pytest.raises(ValueError):
            expand_pool("no_imitation," + old)
        # and it is not a legal seat kind either
        with pytest.raises(ValueError):
            seats_mod.kind_of(old)


def test_an_imitation_shaped_name_with_no_file_behind_it_is_refused():
    """`hc_9999` has the right prefix and no checkpoint.

    `is_imitation_key` is a set membership rather than a prefix test for exactly
    this: a prefix test would classify it as a playable seat and fail later, at
    the moment the file is opened.
    """
    assert not seats_mod.is_imitation_key("hc_9999")
    with pytest.raises(ValueError):
        seats_mod.kind_of("hc_9999")
    with pytest.raises(ValueError):
        expand_pool("hc_9999")


def test_the_cli_reports_a_bad_pool_as_a_usage_error(tmp_path):
    out = str(tmp_path / "o.json")
    with pytest.raises(SystemExit) as exc:
        main(["--games", "1", "--pool", "wizard", "--dry", "--out", out])
    assert exc.value.code == 2
    assert not os.path.exists(out)


# ----------------------------------------------------------------- presets

def test_the_three_presets_have_the_ruled_sizes():
    assert len(POOL_PRESETS["all"]) == 10
    assert len(POOL_PRESETS["no_imitation"]) == 7
    assert len(POOL_PRESETS["imitation_only"]) == 3
    assert POOL_PRESETS["no_imitation"] == tuple(PERSONALITIES)
    assert POOL_PRESETS["imitation_only"] == tuple(STEPS)
    assert len(expand_pool("all")) == 10
    assert len(expand_pool("no_imitation")) == 7
    assert len(expand_pool("imitation_only")) == 3


def test_presets_expand_to_the_expected_names():
    assert expand_pool("all") == list(AUTOMATED)
    assert expand_pool("no_imitation") == PERSONALITIES
    assert expand_pool("imitation_only") == STEPS


# -------------------------------------------------------------- mixed lists

def test_a_preset_plus_one_name_is_a_union():
    """`no_imitation,hc_2000` is the seven personalities plus one checkpoint."""
    pool = expand_pool("no_imitation,hc_2000")
    assert len(pool) == 8
    assert pool == [k for k in AUTOMATED
                    if k in PERSONALITIES or k == "hc_2000"]


def test_the_ruled_mixed_examples():
    assert len(expand_pool("no_imitation,hc_2000")) == 8
    assert len(expand_pool("no_imitation,hunter")) == 7
    assert len(expand_pool("no_imitation,hc_2000,hc_10000")) == 9
    assert len(expand_pool("hc_2000,hc_2000")) == 1


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
    assert expand_pool("hc_2000,hc_2000") == ["hc_2000"]
    assert len(expand_pool("hc_2000,hc_2000")) == 1


# ------------------------------------------------------------------- order

def test_the_expanded_order_does_not_depend_on_the_input_order():
    pairs = [("hc_2000,no_imitation", "no_imitation,hc_2000"),
             ("all", "no_imitation,imitation_only"),
             ("hunter,wolf", "wolf,hunter"),
             ("hc_10000,imitation_only,no_imitation", "all")]
    for a, b in pairs:
        assert expand_pool(a) == expand_pool(b), (a, b)


def test_the_expanded_order_is_the_fixed_option_order():
    for text in ("all", "hc_10000,hc_2000", "hunter,wolf",
                 "no_imitation,imitation_only"):
        pool = expand_pool(text)
        assert pool == [k for k in AUTOMATED if k in set(pool)], text


def test_the_same_pool_played_in_two_orders_gives_the_same_games():
    """Not just the same list - the same games. The draw is over the list, so a
    different order would shuffle every seat."""
    a = seats_of(run_league(2, seed=5, options=expand_pool("no_imitation,hc_2000")))
    b = seats_of(run_league(2, seed=5, options=expand_pool("hc_2000,no_imitation")))
    assert a == b


def test_whitespace_and_quoting_are_tolerated():
    want = expand_pool("no_imitation,hc_2000")
    for text in ("no_imitation,hc_2000",
                 "no_imitation, hc_2000",
                 "  no_imitation ,  hc_2000  ",
                 '"no_imitation,hc_2000"',
                 "'no_imitation,hc_2000'"):
        assert expand_pool(text) == want, text


# -------------------------------------------------------------- what is in

def test_no_option_outside_the_pool_ever_appears():
    for text, allowed in (("no_imitation", set(PERSONALITIES)),
                          ("imitation_only", set(STEPS)),
                          ("hc_1000,hc_10000", {"hc_1000",
                                                 "hc_10000"}),
                          ("no_imitation,hc_2000",
                           set(PERSONALITIES) | {"hc_2000"})):
        rows = run_league(2, seed=11, options=expand_pool(text))
        assert {k for k, _c, _r, _rk, _p in flat(rows)} <= allowed, text
    # specifically: no imitation seat in the personality pool
    rows = run_league(2, seed=11, options=expand_pool("no_imitation"))
    assert not any(seats_mod.is_imitation_key(k) for k, *_ in flat(rows))


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
    for text in ("all", "no_imitation", "no_imitation,hc_2000"):
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


# The values below were captured from `run_league(2, seed=...)` before `--pool`
# existed, with the options list being `league_options()` and each row being
# `(option, colour, remaining)`. They are the regression guard for the claim
# that a default run is unchanged: same length and same order in the expanded
# pool means `rng.choice` consumes the stream identically, so the seats, the
# colours and the scores all come out the same.
#
# **Recaptured when the imitation seats were replaced.** The pool went from
# eleven options to ten - the seven personalities plus `hc_1000`, `hc_2000` and
# `hc_10000` in place of the four H-B2 steps - so `rng.choice` draws different
# indices and every seat in these games changes. What the test still guards is
# the property that survives the rename: a default run and an explicit `--pool
# all` play exactly the same games, and that pair keeps doing so as the pool
# changes. The absolute numbers below are pinned to the current pool, so a
# future pool change must recapture them - which is the intended cost of
# catching a silent change to what a league seat can be.
GOLDEN = {
    20260928: [('hc_2000', 'yellow', 17), ('optimizer', 'red', 4),
               ('wolf', 'green', 35), ('intruder', 'blue', 9),
               ('builder', 'red', 16), ('hc_10000', 'green', 21),
               ('chess', 'yellow', 13), ('fox', 'blue', 28)],
    7: [('builder', 'green', 0), ('fox', 'yellow', 22), ('hunter', 'red', 19),
        ('wolf', 'blue', 38), ('hunter', 'yellow', 15), ('intruder', 'red', 8),
        ('fox', 'blue', 27), ('chess', 'green', 30)],
    99: [('hunter', 'yellow', 12), ('hunter', 'red', 32), ('intruder', 'blue', 15),
         ('hc_10000', 'green', 16), ('fox', 'yellow', 38), ('builder', 'blue', 0),
         ('hc_1000', 'red', 13), ('wolf', 'green', 13)],
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
    rows = run_league(1, seed=3, options=expand_pool("hc_2000"))
    assert {k for k, *_ in rows[0]} == {"hc_2000"}
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
    path = match.default_out_path(7, 3, "no_imitation,hc_2000")
    assert os.path.dirname(path) == str(tmp_path / "m")
    assert path.endswith(".json")
    assert "records" not in os.path.basename(path)
    assert os.path.isdir(os.path.dirname(path))


def test_the_default_output_path_is_the_same_for_the_same_run(tmp_path, monkeypatch):
    monkeypatch.setattr(match, "DEFAULT_OUT_DIR", str(tmp_path / "m"), raising=False)
    a = match.default_out_path(7, 3, "no_imitation,hc_2000")
    b = match.default_out_path(7, 3, "no_imitation,hc_2000")
    assert a == b


def test_the_steps_flag_is_now_an_unknown_argument(tmp_path):
    """`--steps` was a second spelling of `--pool imitation_only` and it had to
    go, so a script still passing it fails loudly instead of quietly running the
    full ten-option pool."""
    with pytest.raises(SystemExit) as exc:
        main(["--games", "1", "--steps", "--dry",
              "--out", str(tmp_path / "o.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "o.json").exists()


def test_the_steps_flag_says_it_is_unrecognised():
    err = subprocess.run([sys.executable, "match.py", "--steps", "--dry"],
                         capture_output=True, text=True)
    assert err.returncode == 2
    assert "unrecognized arguments: --steps" in err.stderr


def test_the_imitation_preset_on_its_own(tmp_path):
    """What `--steps` used to do, now said the one way."""
    out = tmp_path / "imitation.json"
    assert main(["--games", "2", "--seed", "5", "--pool", "imitation_only",
                 "--dry", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pool"] == STEPS
    assert data["pool_size"] == 3
    assert all(seats_mod.is_imitation_key(k) for k in data["pool"])
    assert set(data["appearances"]) == set(STEPS)
    assert sum(data["appearances"].values()) == 8
    # and nothing outside the three ever appears
    for game in data["games_detail"]:
        for row in game:
            assert row[0] in STEPS, row
    # the same seed replays it
    again = tmp_path / "again.json"
    main(["--games", "2", "--seed", "5", "--pool", "imitation_only",
          "--dry", "--out", str(again)])
    assert json.loads(again.read_text(encoding="utf-8"))["games_detail"] == \
        data["games_detail"]


def test_the_help_says_how_the_union_weights():
    out = subprocess.run([sys.executable, "match.py", "--help"],
                         capture_output=True, text=True).stdout
    assert "--pool" in out
    assert "no_imitation" in out and "imitation_only" in out and "all" in out
    # the weighting rule and the de-duplication are both documented
    assert "1/n" in out
    assert "去重" in out


SEED_RECORD = ('{"wolf": {"games": 9, "total_points": 20.0, '
               '"total_remaining": 300.0}}')


def _seeded_leaderboard(tmp_path, monkeypatch):
    """A records.json of known content inside tmp_path, with **every** way of
    constructing `Records` redirected to it.

    Two things make the obvious version of this not work. `RECORDS_PATH` is
    bound as a default argument when `Records.__init__` is defined, so patching
    the module constant does not move the file. And patching only the name
    `match` uses misses any code that reaches `records.Records` directly - which
    is exactly the shape of bug this test exists to catch, as the mutation check
    showed. So the class's own initialiser is redirected, and every route lands
    in tmp_path. The class itself is untouched, so the real write path still
    runs.
    """
    import records as records_mod
    rec = tmp_path / "records.json"
    rec.write_text(SEED_RECORD, encoding="utf-8")
    real_init = records_mod.Records.__init__

    def redirected(self, path=None, *_a, **_k):
        real_init(self, str(rec))

    monkeypatch.setattr(records_mod.Records, "__init__", redirected)
    # And the constant itself, so code that opens the path directly lands in
    # tmp_path too. The second mutation check is what asked for this: a batch
    # path that rewrote the leaderboard byte-for-byte with a bare `open` slipped
    # past a sandbox that only redirected the class.
    monkeypatch.setattr(records_mod, "RECORDS_PATH", str(rec))
    return rec


def test_a_batch_run_does_not_touch_the_leaderboard(tmp_path, monkeypatch):
    """The batch output goes to its own file and to nothing else.

    This is the test that replaced one which asserted a `tmp_path` was empty,
    which is true by construction and verified nothing.
    """
    rec = _seeded_leaderboard(tmp_path, monkeypatch)
    before_bytes = rec.read_bytes()
    before_mtime = rec.stat().st_mtime_ns

    out = tmp_path / "batch.json"
    assert main(["--games", "2", "--seed", "7", "--pool", "no_imitation",
                 "--dry", "--out", str(out)]) == 0

    # the batch write path really ran, so this is not a test of a branch that
    # was never reached
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pool"] == PERSONALITIES and data["games"] == 2
    assert len(data["games_detail"]) == 2

    # and the leaderboard is untouched, byte for byte and to the nanosecond
    assert rec.read_bytes() == before_bytes
    assert rec.stat().st_mtime_ns == before_mtime


def test_the_same_run_without_dry_does_write_the_leaderboard(tmp_path, monkeypatch):
    """The guard above only means something if something would have written.

    Without this the first test would also pass if `Records` were never
    constructed at all, and it would be the second invalid test in a row.
    """
    rec = _seeded_leaderboard(tmp_path, monkeypatch)
    before_bytes = rec.read_bytes()
    out = tmp_path / "batch.json"
    assert main(["--games", "2", "--seed", "7", "--pool", "no_imitation",
                 "--out", str(out)]) == 0
    assert out.exists()
    # the league rows landed in the leaderboard, and the old entry survived
    assert rec.read_bytes() != before_bytes
    data = json.loads(out.read_text(encoding="utf-8"))
    written = json.loads(rec.read_text(encoding="utf-8"))
    # `wolf` was already in the file and is in this pool, so it is folded into
    # rather than replaced - and by exactly what the batch output says it did.
    seeded = json.loads(SEED_RECORD)["wolf"]
    assert written["wolf"]["games"] == seeded["games"] + data["appearances"]["wolf"]
    assert written["wolf"]["total_points"] > seeded["total_points"]
    assert written["wolf"]["total_remaining"] > seeded["total_remaining"]
    # only the options that were actually drawn have an entry; with 8 seats out
    # of 7 options some go unmentioned, which is the point of counting seats
    drawn = {k for k, n in data["appearances"].items() if n}
    assert set(written) == drawn | {"wolf"}
    # every option this run drew, bar the seeded one, is a fresh entry holding
    # exactly its own appearance count
    for key in drawn - {"wolf"}:
        assert written[key]["games"] == data["appearances"][key], key


def test_the_batch_output_goes_under_data_by_default(tmp_path, monkeypatch):
    """Where it lands when `--out` is not given: `data/match/`, never the
    leaderboard's file name."""
    monkeypatch.setattr(match, "DEFAULT_OUT_DIR",
                        str(tmp_path / "m"), raising=False)
    path = match.default_out_path(1, 1, "all")
    assert os.path.dirname(path) == str(tmp_path / "m")
    assert os.path.basename(path).endswith(".json")
    assert "records" not in os.path.basename(path)
