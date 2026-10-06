"""The provenance fields plan9 step 1.5 added to `records.json`.

A leaderboard row used to be three numbers, and it is still three numbers: the
scoring reads `games`, `total_points` and `total_remaining` and nothing else.
What step 1.5 added sits beside them and answers a different question - not
"how did this contestant score" but "what is this contestant".

The tests here are mostly about what happens to files that predate the change,
because that is where a schema extension actually breaks:

  * a file with none of the new keys must still open, and must not be
    *rewritten* to invent values for them;
  * a file with some of them - two contestants annotated, two not - must open
    too, since that is what a repository accumulates as models get registered;
  * `None` has to mean "not known". Inferring a `teacher` from the name of a
    checkpoint would produce a file that looks complete and claims something
    about how a model was trained that may be false, and once written there is
    nothing in the file to tell the two apart.

The arithmetic is pinned too: adding and reading provenance must not move a
single average.
"""
import json
import os

import pytest

from records import META_FIELDS, Records


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# files that predate the change
# --------------------------------------------------------------------------

def test_a_file_with_none_of_the_new_fields_still_opens(tmp_path):
    """The case that matters: every existing `records.json` looks like this."""
    p = tmp_path / "records.json"
    _write(p, {"rl_h1000_20k": {"games": 2, "total_points": 8.0,
                                "total_remaining": 6.0},
               "hc_2000": {"games": 1, "total_points": 3.0,
                           "total_remaining": 7.0}})
    r = Records(str(p))
    assert set(r.entries) == {"rl_h1000_20k", "hc_2000"}
    assert r.entries["rl_h1000_20k"]["games"] == 2
    assert r.entries["rl_h1000_20k"]["total_points"] == 8.0
    # and the absent fields read as not known, not as an error or a default
    for field in META_FIELDS:
        assert r.meta("rl_h1000_20k")[field] is None


def test_loading_an_old_file_does_not_rewrite_it_with_invented_values(tmp_path):
    """Opening is not a licence to fill things in.

    `None` is the honest answer for a field nobody wrote down. A rewrite that
    materialised defaults would make the file indistinguishable from one where
    the values were actually recorded - and `commit` in particular would then
    claim a provenance nobody established.
    """
    p = tmp_path / "records.json"
    _write(p, {"wolf": {"games": 1, "total_points": 1.0,
                        "total_remaining": 25.0}})
    before = _read(p)
    r = Records(str(p))
    r.rows()                       # the read path
    assert _read(p) == before, "reading a leaderboard must not write to it"


def test_a_partly_annotated_file_opens(tmp_path):
    """What a repository accumulates: some contestants annotated, some not."""
    p = tmp_path / "records.json"
    _write(p, {
        "rl_h1000_20k": {"games": 1, "total_points": 4.0,
                         "total_remaining": 3.0,
                         "history": "單打 20k", "train_seed": 7000000,
                         "teacher": "hunter", "pool_contains_teacher": True,
                         "commit": "0119360"},
        "hc_2000": {"games": 1, "total_points": 3.0, "total_remaining": 7.0},
    })
    r = Records(str(p))
    assert r.meta("rl_h1000_20k")["history"] == "單打 20k"
    assert r.meta("rl_h1000_20k")["teacher"] == "hunter"
    assert r.meta("rl_h1000_20k")["pool_contains_teacher"] is True
    # the other row keeps every field unknown rather than inheriting its neighbour's
    for field in META_FIELDS:
        assert r.meta("hc_2000")[field] is None
    assert r.meta("hc_2000") != r.meta("rl_h1000_20k")


def test_a_record_written_now_carries_every_field(tmp_path):
    """So the file does not end up with two shapes in it."""
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("wolf", 10), ("wolf", 30), ("fox", 20), ("human", 0)])
    stored = _read(p)["wolf"]
    assert set(stored) == {"games", "total_points", "total_remaining",
                           *META_FIELDS}
    for field in META_FIELDS:
        assert stored[field] is None


# --------------------------------------------------------------------------
# writing provenance
# --------------------------------------------------------------------------

def test_set_meta_stores_and_round_trips(tmp_path):
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("rl_h1000_20k", 3), ("wolf", 9), ("fox", 12), ("human", 0)])
    r.set_meta("rl_h1000_20k", history="單打 20k", train_seed=7000000,
               teacher="hunter", pool_contains_teacher=True,
               commit="0119360")
    again = Records(str(p))
    assert again.meta("rl_h1000_20k") == {
        "history": "單打 20k", "train_seed": 7000000, "teacher": "hunter",
        "pool_contains_teacher": True, "commit": "0119360"}


def test_set_meta_may_register_a_contestant_before_it_has_played(tmp_path):
    """Provenance can be known before there is a result to attach it to."""
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.set_meta("rl_o1000_20k", history="模仿 1000 步", teacher="optimizer")
    assert r.entries["rl_o1000_20k"]["games"] == 0
    # and it must not take the leaderboard down with a division by zero
    assert r.rows() == []
    r.record([("rl_o1000_20k", 5), ("wolf", 9), ("fox", 12), ("human", 0)])
    rows = dict((k, g) for k, g, _ap, _ar in r.rows())
    assert rows["rl_o1000_20k"] == 1


def test_an_unknown_field_name_is_refused(tmp_path):
    """A typo in `train_seed` would otherwise store an extra key silently."""
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("wolf", 10), ("fox", 20), ("human", 0), ("chess", 30)])
    with pytest.raises(ValueError) as e:
        r.set_meta("wolf", train_sed=7)
    assert "train_sed" in str(e.value)
    assert set(_read(p)["wolf"]) == {"games", "total_points",
                                     "total_remaining", *META_FIELDS}


# --------------------------------------------------------------------------
# provenance must not touch the scoring
# --------------------------------------------------------------------------

def test_setting_provenance_moves_no_average(tmp_path):
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("rl_h1000_20k", 3), ("wolf", 9), ("fox", 12), ("human", 0)])
    before = r.rows()
    r.set_meta("rl_h1000_20k", history="單打 20k", commit="0119360")
    assert r.rows() == before
    assert Records(str(p)).rows() == before


def test_recording_a_game_does_not_disturb_provenance(tmp_path):
    """The two directions are separate, and both directions have to hold."""
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("rl_h1000_20k", 3), ("wolf", 9), ("fox", 12), ("human", 0)])
    r.set_meta("rl_h1000_20k", history="單打 20k", teacher="hunter",
               pool_contains_teacher=True)
    r.record([("rl_h1000_20k", 4), ("wolf", 11), ("fox", 8), ("human", 0)])
    again = Records(str(p))
    assert again.meta("rl_h1000_20k")["history"] == "單打 20k"
    assert again.meta("rl_h1000_20k")["pool_contains_teacher"] is True
    assert again.entries["rl_h1000_20k"]["games"] == 2


def test_the_arena_of_a_field_that_is_not_a_number(tmp_path):
    """`pool_contains_teacher` is a bool and `train_seed` is an int; neither is
    scored, and neither is allowed to be coerced into one."""
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("rl_h1000_20k", 3), ("wolf", 9), ("fox", 12), ("human", 0)])
    r.set_meta("rl_h1000_20k", pool_contains_teacher=False, train_seed=0)
    again = Records(str(p))
    assert again.meta("rl_h1000_20k")["pool_contains_teacher"] is False
    assert again.meta("rl_h1000_20k")["train_seed"] == 0
    assert all(isinstance(ap, float) for _k, _g, ap, _ar in again.rows())


def test_meta_of_an_unknown_contestant_is_all_none(tmp_path):
    r = Records(str(tmp_path / "records.json"))
    assert r.meta("nobody") == {f: None for f in META_FIELDS}


def test_meta_of_everyone_is_keyed_by_contestant(tmp_path):
    p = tmp_path / "records.json"
    r = Records(str(p))
    r.record([("wolf", 9), ("fox", 12), ("human", 0), ("chess", 30)])
    r.set_meta("fox", teacher="fox")
    everything = r.meta()
    assert set(everything) == {"wolf", "fox", "human", "chess"}
    assert everything["fox"]["teacher"] == "fox"
    assert everything["wolf"]["teacher"] is None