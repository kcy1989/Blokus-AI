"""The two spellings of the 0k checkpoint are both out of the roster now.

Plan9a stage 1 renamed the 0k checkpoint's option key `hc_1000` ->
`rl_h1000_0k` and kept the old spelling as a permanent alias, because two
committed evidence batches were recorded as `--subject hc_1000` and
`records.json` still had a row under it. Plan9 task 4 then retired the seat
itself - with `hc_2000`, `hc_10000` and the three students - so *neither*
spelling names a contestant any more.

What that costs, and what this file is now:
  * both names are absent from `R.keys()`, from `automated_options()` and from
    every pool, and `kind_of` / `expand_pool` refuse them - the check that used
    to be "the alias is not a second seat" is now "neither spelling is a seat";
  * no alias was kept for either, so `canonical_key` hands both back byte for
    byte: a leaderboard row written before the retirement still reads, and a
    *new* result filed under the old spelling grows its own row rather than
    being folded into a seat that no longer exists. That is a change from stage
    9's normalise-on-write, and it is what `records.json`'s own contract says -
    it is keyed by the name the game was played under;
  * the alias machinery itself is still exercised, by the one alias that
    survived: `rl_1000_20k` -> `rl_h1000_20k`.

The rest of the old file - both spellings loading the same weights, choosing
the same move, the write-through landing under the new key - tested a seat that
exists. Those assertions are gone with the seat; the bytes they checked are
still pinned where the files are, in `tests/test_seats.py` and in the committed
evidence's own md5 fields.
"""
import pytest

import ai.registry as REG
import match as M
import records as R
import seats as S

OLD = "hc_1000"
NEW = "rl_h1000_0k"
RETIRED = (OLD, NEW, "hc_2000", "hc_10000", "rl_o1000_0k", "rl_b1000_0k",
           "rl_i1000_0k")


# --------------------------------------------------------------------------
# neither spelling is a seat
# --------------------------------------------------------------------------

def test_neither_spelling_is_registered_or_selectable():
    for key in RETIRED:
        assert key not in REG.keys()
        assert key not in S.automated_options()
        assert key not in S.seat_options(include_humans=False)
        for pool in REG.POOL_NAMES:
            assert key not in REG.pool(pool), (key, pool)
        assert key not in M.POOL_PRESETS["all"]


def test_neither_spelling_is_an_alias_of_anything():
    """No alias was kept, so both come back byte for byte - and are refused.

    `canonical_key` is a pure table lookup; the point of pinning it here is that
    a *retired* name must not be quietly re-pointed at a surviving seat, which
    is the one way the pool could grow a contestant nobody registered.
    """
    assert REG.aliases() == {"rl_1000_20k": "rl_h1000_20k"}
    assert S.ALIASES == REG.aliases()
    assert S.RL_ALIASES == {"rl_1000_20k": "rl_h1000_20k"}
    for key in RETIRED:
        assert S.canonical_key(key) == key
        assert REG.resolve(key) == key
        with pytest.raises(ValueError):
            S.kind_of(key)
        with pytest.raises(ValueError):
            M.expand_pool(key)


def test_the_one_alias_that_survived_still_resolves():
    """The machinery is still load-bearing, so it is still tested."""
    assert S.canonical_key("rl_1000_20k") == "rl_h1000_20k"
    assert M.expand_pool("rl_1000_20k") == ["rl_h1000_20k"]
    assert S.kind_of("rl_1000_20k") == S.KIND_RL
    assert M.validate_subject("rl_1000_20k", ["wolf"], True) == "rl_h1000_20k"


def test_canonical_key_leaves_what_it_does_not_recognise_alone():
    """The alias table rewrites the names it knows and nothing else.

    `records.py` runs every key it writes through `canonical_key` (plan9a
    stage 9), so anything outside the alias table - the human seats, the
    player key, a key typed by hand - has to come back byte for byte. A
    resolver that returned `None`, dropped the key or guessed a seat would
    take the leaderboard with it, and this is what says so in one place.
    """
    for key in (OLD, NEW, "player", "human", "human_log", "random_ai",
                "made_up_key", ""):
        assert S.canonical_key(key) == key


# --------------------------------------------------------------------------
# history keeps working
# --------------------------------------------------------------------------

def test_a_records_row_under_a_retired_key_still_reads(tmp_path):
    """`records.json` is keyed by the name a game was played under.

    Neither row is rewritten by the retirement - dropping or merging one would
    silently lose the games already counted under it, and both rows name
    weights that still exist on disk.
    """
    path = tmp_path / "records.json"
    path.write_text('{"hc_1000": {"games": 3, "total_points": 9.0,'
                    ' "total_remaining": 30.0},'
                    ' "rl_h1000_0k": {"games": 2, "total_points": 5.0,'
                    ' "total_remaining": 15.0}}', encoding="utf-8")
    board = R.Records(str(path))
    assert set(board.entries) == {OLD, NEW}
    assert board.entries[OLD]["games"] == 3
    assert board.entries[NEW]["games"] == 2
    # and the provenance fields of a pre-rename file read as unknown, not as an
    # error and not as an inferred value
    from records import META_FIELDS
    for field in META_FIELDS:
        assert board.meta(OLD)[field] is None


def test_a_write_under_a_retired_key_stays_under_that_key(tmp_path):
    """Stage 9 normalised an alias write onto the canonical row. There is no
    canonical row any more, so the name is kept.

    The alternative - folding a new result into a seat that no longer exists,
    or dropping it - would be a silent loss either way. `canonical_key` is the
    single place that decides, and it is a pure table lookup, so this test is
    really an assertion about that table: a retired name must not reappear in
    it under any circumstances.
    """
    board = R.Records(str(tmp_path / "records.json"))
    rows = board.record([(OLD, 10.0), ("wolf", 20.0),
                         ("fox", 30.0), ("builder", 40.0)])
    assert board.entries[OLD]["games"] == 1
    assert board.entries[OLD]["total_points"] == 4.0       # fewest remaining
    assert board.entries[OLD]["total_remaining"] == 10.0
    assert NEW not in board.entries
    assert rows == [(OLD, 1, 4), ("wolf", 2, 3), ("fox", 3, 2),
                    ("builder", 4, 1)]
    again = R.Records(str(tmp_path / "records.json"))
    assert set(again.entries) == {OLD, "wolf", "fox", "builder"}
