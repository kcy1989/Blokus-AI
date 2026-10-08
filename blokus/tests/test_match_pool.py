"""`match.py --pool`: which options a league run draws from.

The pool is a **set**, not a multiset. Naming `hunter` alongside the
`no_imitation` preset does not give `hunter` a second chance, because it is
already in that preset; every distinct option in the expanded pool is worth
1/n. And the expansion order is `ai.registry.pool_order` - by key, ascending -
rather than the order the items were typed in or the order the roster happens
to be registered in, so two `--pool` arguments with the same contents play
exactly the same games from the same seed.

The `GOLDEN` block near the bottom is the one that matters most for regression:
it pins what three seeds actually play, against a pool spelled out in the test
rather than read from the registry. Its values were recaptured when the pool
switched to key order (plan9a stage 3); the old values and the commit they came
from are in README.md under 「GOLDEN 重採」.
"""
import collections
import gzip
import json
import math
import os
import subprocess
import sys

import pytest

import ai.registry as R
import match
import seats as seats_mod
from match import (POOL_PRESETS, expand_pool, league_options, main, play_match,
                   run_league, summarise)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUTOMATED = seats_mod.automated_options()
# Both exclusions, not one. `not is_imitation_key` alone would sweep the trained
# policy into PERSONALALITIES, and every preset-size assertion below would then
# be quietly comparing the wrong sets.
PERSONALITIES = [k for k in AUTOMATED
                 if not seats_mod.is_imitation_key(k)
                 and not seats_mod.is_rl_key(k)]
STEPS = [k for k in AUTOMATED if seats_mod.is_imitation_key(k)]
TRAINED = [k for k in AUTOMATED if seats_mod.is_rl_key(k)]


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

def test_the_four_presets_match_the_registry_and_partition_all():
    """Stage 8: the sizes follow the roster, so the shape has to carry the test.

    Every `== 14` that used to live here is now a comparison against
    `ai/registry.json`, which means registering a fifteenth seat does not
    touch this file. What still has to hold at *any* roster size - and what
    takes the place of those numbers - is the structure: `match` and the
    registry read the same membership, the three sub-pools are pairwise
    disjoint (the plan names `imitation_only & rl_only == []`), together they
    are exactly `all`, and every pool is in key order.

    The literal count survives elsewhere: `GOLDEN_POOL` below is a
    fourteen-element list compared against `expand_pool("all")`, so a roster
    edit still fails loudly, in a test that says which key moved.
    """
    for name, keys in POOL_PRESETS.items():
        assert list(keys) == list(R.pool(name)), name
        assert list(expand_pool(name)) == list(keys), name
        assert list(keys) == sorted(keys), name
    sets = {n: set(k) for n, k in POOL_PRESETS.items()}
    assert sets["no_imitation"] & sets["imitation_only"] == set()
    assert sets["no_imitation"] & sets["rl_only"] == set()
    assert sets["imitation_only"] & sets["rl_only"] == set()
    assert (sets["no_imitation"] | sets["imitation_only"]
            | sets["rl_only"]) == sets["all"]
    assert sets["all"] == set(expand_pool(None)) == set(league_options())
    # and the three sub-pools are still the three kinds of contestant
    assert sets["no_imitation"] == set(PERSONALITIES)
    assert sets["imitation_only"] == set(STEPS)
    assert sets["rl_only"] == set(TRAINED)


def test_no_imitation_leaves_out_the_trained_policy_too():
    """The name means "no network", so a network that is not an imitation
    checkpoint still has to be excluded - otherwise the preset quietly measures
    the trained policy too and its seven is an eight."""
    assert "rl_h1000_20k" not in POOL_PRESETS["no_imitation"]
    assert "rl_h1000_20k" not in expand_pool("no_imitation")
    assert "rl_h1000_20k" in POOL_PRESETS["all"]
    # and imitation_only keeps meaning imitation
    assert "rl_h1000_20k" not in POOL_PRESETS["imitation_only"]
    # the pre-rename spelling resolves to the same option and so is excluded too
    assert "rl_1000_20k" not in expand_pool("no_imitation")


def test_presets_expand_to_the_expected_names():
    assert expand_pool("all") == list(AUTOMATED)
    assert expand_pool("no_imitation") == PERSONALITIES
    assert expand_pool("imitation_only") == STEPS
    assert expand_pool("rl_only") == TRAINED


# -------------------------------------------------------------- mixed lists

def test_a_preset_plus_one_name_is_a_union():
    """`no_imitation,hc_2000` is every personality plus one checkpoint.

    Said as membership rather than as a length: the length was 8 only while
    the roster had seven personalities, and a union does not need to know
    either number to be right.
    """
    pool = expand_pool("no_imitation,hc_2000")
    assert pool == [k for k in AUTOMATED
                    if k in PERSONALITIES or k == "hc_2000"]
    assert set(pool) == set(PERSONALITIES) | {"hc_2000"}
    assert len(pool) == len(set(pool))


def test_the_ruled_mixed_examples():
    """The same three unions, each spelled out as a set.

    The lengths these used to assert (8, 7, 9, 1) were roster sizes wearing a
    different hat; membership is what the examples were ever about, and it
    holds for any roster.
    """
    assert set(expand_pool("no_imitation,hc_2000")) == \
        set(PERSONALITIES) | {"hc_2000"}
    assert expand_pool("no_imitation,hunter") == PERSONALITIES
    assert set(expand_pool("no_imitation,hc_2000,hc_10000")) == \
        set(PERSONALITIES) | {"hc_2000", "hc_10000"}
    assert expand_pool("hc_2000,hc_2000") == ["hc_2000"]


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
             ("all", "no_imitation,imitation_only,rl_only"),
             ("hunter,wolf", "wolf,hunter"),
             ("hc_10000,imitation_only,no_imitation,rl_only", "all")]
    for a, b in pairs:
        assert expand_pool(a) == expand_pool(b), (a, b)


def test_the_expanded_order_is_the_fixed_option_order():
    for text in ("all", "hc_10000,hc_2000", "hunter,wolf",
                 "no_imitation,imitation_only,rl_only"):
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
                          ("hc_1000,hc_10000", {"rl_h1000_0k",
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
# **Recaptured three times.** First when the imitation seats were replaced - the
# pool went from eleven options to ten, the seven personalities plus `hc_1000`,
# `hc_2000` and `hc_10000` in place of the four H-B2 steps - and again when the
# trained policy `rl_h1000_20k` was registered, taking it to eleven. Either way
# `rng.choice` draws different indices and every seat in these games changes.
#
# **And touched once, without recapturing.** plan9a stage 1 renamed `hc_1000` to
# `rl_h1000_0k`. Under registration order the key sat at the same index of the
# same eleven-long list, so the draws were unchanged and only the name column
# moved - which is what `test_seat_rename` checked.
#
# **Recaptured a third time by plan9a stage 3**, which orders every pool by key.
# `rng.choice` draws an index, so a different order is a different game from the
# same seed and all three seeds changed. Old values, new values and the commits
# are in README.md under 「GOLDEN 重採」; the point of writing them down is that
# this was an ordering change and nothing else - the eleven keys are the same
# eleven.
#
# **Recaptured a fourth time by plan9a stage 6**, which registered the three
# 1000-step students. This one *is* a composition change: the pool is fourteen
# keys, so the index every draw reads has moved for every key that sorts after
# `rl_b1000_0k`, and all three seeds changed again. `hc_2000` and `hc_10000`
# stay in the pool (the user's ruling), which is why the pool is fourteen and
# not the twelve the plan originally wrote.
#
# **The pool is spelled out rather than read.** `GOLDEN_POOL` below is a literal
# list: GOLDEN is evidence, and evidence that derives itself from the registry
# follows the registry instead of failing when the roster changes.
# `test_the_pinned_pool_is_still_the_registry_s_pool` is the one place the two
# are allowed to meet, and it is what turns a roster edit into a loud failure
# here rather than a silent drift.
GOLDEN_POOL = ["builder", "chess", "fox", "hc_10000", "hc_2000", "hunter",
               "intruder", "optimizer", "rl_b1000_0k", "rl_h1000_0k",
               "rl_h1000_20k", "rl_i1000_0k", "rl_o1000_0k", "wolf"]

GOLDEN = {
    20260928: [('rl_h1000_20k', 'green', 0), ('rl_b1000_0k', 'red', 18),
               ('hc_2000', 'yellow', 12), ('rl_i1000_0k', 'blue', 9),
               ('hunter', 'green', 14), ('rl_h1000_0k', 'blue', 8),
               ('hc_10000', 'yellow', 10), ('rl_i1000_0k', 'red', 12)],
    7: [('hunter', 'green', 25), ('fox', 'red', 26), ('intruder', 'yellow', 5),
        ('rl_h1000_20k', 'blue', 5), ('rl_h1000_20k', 'blue', 11),
        ('wolf', 'green', 25), ('optimizer', 'red', 21),
        ('hc_2000', 'yellow', 4)],
    99: [('intruder', 'yellow', 19), ('intruder', 'red', 9),
         ('hc_10000', 'blue', 20), ('rl_h1000_0k', 'green', 16),
         ('builder', 'green', 30), ('intruder', 'red', 10),
         ('hunter', 'blue', 12), ('fox', 'yellow', 21)],
}


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_a_pinned_pool_still_plays_the_games_pinned_here(seed):
    """The pool is spelled out, not derived - see the comment on `GOLDEN_POOL`.

    What a default run does is still covered, just not by this test: it is the
    composition of `test_no_pool_at_all_means_all` (no pool means
    `league_options()`), `test_the_pinned_pool_is_still_the_registry_s_pool`
    (the registry's pool is this list) and this one. Three separate claims that
    each fail loudly, rather than one claim that quietly follows the roster.
    """
    assert seats_of(run_league(2, seed=seed, options=GOLDEN_POOL)) == GOLDEN[seed]


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_pool_all_is_identical_to_no_pool_at_all(seed):
    """Same list, same order, same length - so the same random draws."""
    assert expand_pool("all") == expand_pool(None)
    assert seats_of(run_league(2, seed=seed, options=expand_pool("all"))) \
        == GOLDEN[seed]
    assert seats_of(run_league(2, seed=seed)) == \
        seats_of(run_league(2, seed=seed, options=expand_pool("all")))


def test_the_pinned_pool_is_still_the_registry_s_pool():
    """The one place GOLDEN and the roster meet.

    Written as its own test rather than inside the GOLDEN ones so that a roster
    edit fails here - with a message about the roster - instead of failing three
    seeds of seating data with no hint which of the two moved.
    """
    assert GOLDEN_POOL == expand_pool("all") == expand_pool(None)
    assert GOLDEN_POOL == list(league_options())
    assert GOLDEN_POOL == sorted(GOLDEN_POOL)


def test_all_equals_the_three_presets_that_partition_it():
    rows = seats_of(run_league(1, seed=20260928, options=expand_pool("all")))
    same = seats_of(run_league(1, seed=20260928,
                               options=expand_pool("no_imitation,imitation_only,"
                                                   "rl_only")))
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
    assert data["pool_size"] == len(data["pool"])
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
    full eleven-option pool."""
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
    assert data["pool_size"] == len(data["pool"])
    assert all(seats_mod.is_imitation_key(k) for k in data["pool"])
    # Who actually *sat* is a draw, not a promise: eight seats cannot be
    # assumed to cover every key in the pool once the pool grows, so this is
    # containment rather than equality. What has to hold either way is that
    # nothing outside the preset appeared.
    assert set(data["appearances"]) <= set(STEPS)
    assert sum(data["appearances"].values()) == 8
    # and nothing outside the six ever appears
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
    assert "--pool-order" in out
    assert "no_imitation" in out and "imitation_only" in out and "all" in out
    # the weighting rule and the de-duplication are both documented
    assert "1/n" in out
    assert "去重" in out


# --------------------------------------------------------------- literal


def test_literal_keeps_the_order_the_duplicates_and_the_spelling():
    """Everything sorted mode takes away, given back - on purpose."""
    given = "hunter,wolf,hunter,optimizer"
    assert expand_pool(given, order="literal") == \
        ["hunter", "wolf", "hunter", "optimizer"]
    # the sorted default does the opposite on all three counts
    assert expand_pool(given) == ["hunter", "optimizer", "wolf"]
    # and a retired spelling is a spelling here, not something to improve
    assert expand_pool("rl_1000_20k", order="literal") == ["rl_1000_20k"]
    assert expand_pool("rl_1000_20k") == ["rl_h1000_20k"]


def test_literal_is_refused_beside_a_preset():
    """A preset's order *is* the registry's - the one thing literal assumes away.

    Checked for a bare preset and for one buried in a name list, because the
    second is the easy one to let through: the names on either side are legal,
    so only the preset itself can be the tell.
    """
    for text in ("no_imitation", "wolf,no_imitation", "all,wolf"):
        with pytest.raises(ValueError) as exc:
            expand_pool(text, order="literal")
        msg = str(exc.value)
        assert "literal" in msg and "preset" in msg, text


def test_literal_is_refused_without_an_explicit_pool():
    """`literal` has no default, because a default is a derived order."""
    with pytest.raises(ValueError) as exc:
        expand_pool(None, order="literal")
    assert "explicit" in str(exc.value)
    for text in ("", " , "):
        with pytest.raises(ValueError):
            expand_pool(text, order="literal")


def test_literal_still_refuses_a_name_that_is_not_a_league_option():
    """No sort, no dedup, no `enabled` filter - but a typo is still a typo, and
    a human seat is not something a league can draw for."""
    for text in ("wizard", "hc_9999"):
        with pytest.raises(ValueError) as exc:
            expand_pool(text, order="literal")
        assert text in str(exc.value)
    with pytest.raises(ValueError) as exc:
        expand_pool("human", order="literal")
    assert "human" in str(exc.value)


def test_the_cli_turns_a_literal_violation_into_a_usage_error(tmp_path):
    out = str(tmp_path / "o.json")
    for argv in (["--pool-order", "literal"],
                 ["--pool-order", "literal", "--pool", "no_imitation"],
                 ["--pool-order", "literal", "--pool", "wolf,all"]):
        with pytest.raises(SystemExit) as exc:
            main(argv + ["--games", "1", "--seed", "1", "--dry", "--out", out])
        assert exc.value.code == 2, argv
        assert not os.path.exists(out)


# ------------------------------------------- replaying a batch from before


def test_literal_replays_a_batch_committed_before_the_sort(tmp_path):
    """Stage 3 check (a): an old batch still plays, game for game.

    `eval/plan9/argmax-hc_1000-games.json.gz` was written while the pool was
    ordered by the registry; after the switch to key order the same command
    draws different opponents from the same seed. Handing the recorded pool back
    through `--pool-order literal` - no sort, no dedup, and the subject kept in
    the spelling the batch used - is what makes the games come out identical.

    Forty rather than the batch's two thousand: enough to reach game 0, which is
    where sorted order first diverges, and cheap enough for the suite. The
    committed file remains the authority for the other 1960; this guards the
    mechanism, and it fails on the first differing seat rather than on a digest.
    """
    path = os.path.join(_ROOT, "eval", "plan9", "argmax-hc_1000-games.json.gz")
    if not os.path.exists(path):
        pytest.skip("the 0.5b batch is not in this checkout")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)

    n = 40
    out = tmp_path / "replay.json"
    rc = main(["--games", str(n), "--seed", str(payload["seed"]),
               "--pool", ",".join(payload["pool"]),
               "--pool-order", "literal",
               "--paired-rng", "--subject", payload["subject"],
               "--mode", payload["mode"], "--dry", "--out", str(out)])
    assert rc == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["pool_order"] == "literal"
    assert got["pool"] == payload["pool"]          # same list, same order
    assert got["subject"] == payload["subject"]    # the batch's spelling, kept
    assert got["games_detail"] == payload["games_detail"][:n]


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
