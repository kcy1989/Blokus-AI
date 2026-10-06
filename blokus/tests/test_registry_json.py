"""`ai/registry.json` against the code tables it is supposed to describe.

plan9a stage 2 moved *which contestants exist, in which order, and under which
names* into data. Two things stayed in code, on purpose:

  * how a move is scored - `WEIGHTED_SPECS` / `RULE_BRAIN_CLASSES`, because a
    JSON file cannot hold a `Brain`;
  * which file a network seat means - `IMITATION_STEPS` / `RL_SEATS`, because a
    trained policy's identity *is* its file and `data/` is still the training
    factory.

So the same roster is written down twice, and the only acceptable relation
between the two copies is agreement. That is what every test below checks: set
*and* order, key by key and field by field. Editing one side alone is a
failure, not a silent divergence.

None of these assertions pin a composition. Each one reads one side from the
JSON and the other from the code, so a stage-6 pool change that moves both
together still passes and one that moves only one does not.
"""
import json
import os
import subprocess
import sys

import pytest

import ai.registry as R
import match as M
import seats as S
from ai.registry import RULE_BRAIN_CLASSES, WEIGHTED_SPECS

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# --------------------------------------------------------------------------
# the roster agrees with the code, set and order
# --------------------------------------------------------------------------

def test_the_json_lists_the_personalities_in_the_class_tables_order():
    """Seven keys, in `WEIGHTED_SPECS` order followed by `RULE_BRAIN_CLASSES`.

    The upper layer's order is the UI order and the draw order of
    `personality_keys()`. A JSON row that merely contains the same seven, in a
    different arrangement, would pass a set comparison and reseat every
    same-seed game - which is the bug this test exists to catch.
    """
    from_code = (tuple(key for key, _cls, _spec in WEIGHTED_SPECS)
                 + tuple(RULE_BRAIN_CLASSES))
    from_json = tuple(e["key"] for e in R.ENTRIES
                      if e["family"] == "personality")
    assert from_json == from_code
    assert from_code == R.personality_keys()


def test_the_json_lists_the_checkpoints_in_step_order():
    """`IMITATION_STEPS` decides both which files and which keys, in order."""
    from_code = tuple(S.IMITATION_KEYS[step] for step in S.IMITATION_STEPS)
    from_json = tuple(e["key"] for e in R.ENTRIES
                      if e["family"] == "imitation")
    assert from_json == from_code


def test_the_json_lists_the_trained_policies_in_table_order():
    from_json = tuple(e["key"] for e in R.ENTRIES if e["family"] == "rl")
    assert from_json == S.rl_keys()
    assert from_json == tuple(key for key, _path in S.RL_SEATS)


def test_the_json_has_exactly_the_keys_the_seat_model_has():
    """No extra contestant, no missing one, no reshuffled one.

    `automated_options()` is what `rng.choice` draws an index into, so the
    order half of this is load-bearing: it is the same tuple the league, the
    seat menu and `RANDOM_AI_KEY` all walk.
    """
    assert R.keys() == S.automated_options()
    assert set(R.keys()) == set(S.seat_options(include_humans=False))


def test_every_json_key_is_a_seat_the_engine_will_accept():
    for key in R.keys():
        assert S.kind_of(key) in (S.KIND_AI, S.KIND_IMITATION, S.KIND_RL)
    # ...and nothing else claims to be one
    for key in ("hc_9999", "rl_9999", "nope"):
        assert key not in R.keys()


# --------------------------------------------------------------------------
# the field-by-field cross-checks
# --------------------------------------------------------------------------

def test_kind_and_family_agree_with_kind_of():
    expected = {"personality": S.KIND_AI,
                "imitation": S.KIND_IMITATION,
                "rl": S.KIND_RL}
    for e in R.ENTRIES:
        assert expected[e["family"]] == S.kind_of(e["key"]), e["key"]


def test_network_entries_name_the_files_the_seats_load():
    """The JSON records a path; `seats` still decides the path.

    Until plan9a stage 4 these are deliberately two sources, so the only safe
    state is that they say the same thing. A checkpoint moved by editing one
    side alone fails here rather than at first use.
    """
    rl_paths = dict(S.RL_SEATS)
    for e in R.ENTRIES:
        if e["family"] == "rl":
            assert e["checkpoint"] == rl_paths[e["key"]]
        elif e["family"] == "imitation":
            step = S.imitation_step(e["key"])
            assert e["checkpoint"] == os.path.join(
                S.IMITATION_CHECKPOINT_DIR, "step_%06d.pt" % step)
        else:
            assert "checkpoint" not in e


def test_heuristic_entries_name_the_module_the_brain_class_lives_in():
    classes = {key: cls for key, cls, _spec in WEIGHTED_SPECS}
    classes.update(RULE_BRAIN_CLASSES)
    for e in R.ENTRIES:
        if e["family"] != "personality":
            assert "module" not in e
            continue
        assert classes[e["key"]].__module__ == e["module"], e["key"]


def test_the_alias_table_is_the_one_seats_reads():
    """One alias table in the repository, seen from both sides."""
    assert S.ALIASES == R.aliases()
    assert S.RL_ALIASES == {"rl_1000_20k": "rl_h1000_20k"}
    for old, new in S.ALIASES.items():
        assert R.resolve(old) == S.canonical_key(old) == new
        assert new in R.keys()
        # an alias is never a second seat
        assert old not in R.keys()
        assert old not in R.automated_options()


def test_the_four_pools_in_the_json_are_the_four_presets_in_match():
    """`match` derives its presets from `kind_of`; the JSON lists them.

    Stage 2 keeps `POOL_PRESETS` derived and the JSON descriptive, so the two
    are compared rather than one being made to wrap the other. Stage 6 is where
    the presets start reading the registry - and this is what will notice if
    that changes an answer.
    """
    assert R.POOL_NAMES == tuple(M.POOL_PRESETS)
    for name in R.POOL_NAMES:
        assert R.pool(name) == M.POOL_PRESETS[name], name
        assert list(M.expand_pool(name)) == list(R.pool(name))
    assert list(M.expand_pool(None)) == list(R.automated_options())


def test_the_label_and_description_keys_exist_in_config():
    """Structural, so it also runs at import - see the module docstring.

    The rendering itself is checked from the UI side, in `test_ui_smoke`: the
    JSON knows the `config.I` keys, and only `ui.seat_label` / `ui.seat_desc`
    know how to fill them in.
    """
    from config import I
    for e in R.ENTRIES:
        assert e["label"] in I, e["key"]
        assert e["desc_key"] in I, e["key"]
        if e["family"] == "personality":
            assert e["label"] == e["key"]
            assert e["desc_key"] == e["key"] + "_desc"
        else:
            assert e["label"] in ("imitation_fmt", "rl_fmt")
            assert e["desc_key"] in ("imitation_desc_fmt", "rl_desc_fmt")


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------

def test_anchors_names_one_seat_and_it_is_the_registered_spelling():
    """The KL anchor is a seat, and never the retired spelling.

    Two spellings for one anchor would give `data/rl1/eval.jsonl` a second
    column for a single model, and the whole point of that file is that its
    rows are comparable to each other.
    """
    assert R.anchors() == ("rl_h1000_0k",)
    assert "hc_1000" not in R.anchors()
    for key in R.anchors():
        assert key in R.keys()
        assert R.resolve(key) == key


# --------------------------------------------------------------------------
# files: checked here and by --check, never at import
# --------------------------------------------------------------------------

def test_check_files_finds_nothing_wrong_with_the_registered_weights():
    assert R.check_files() == []


def test_check_files_reports_a_missing_or_rewritten_checkpoint(tmp_path,
                                                               monkeypatch):
    """The check has to be able to fail, or it is decoration."""
    broken = dict(R.ENTRIES[0])
    broken.update({"key": "ghost", "kind": "network", "family": "rl",
                   "checkpoint": str(tmp_path / "nope.pt"), "source": "x",
                   "sha256": "0" * 64})
    monkeypatch.setattr(R, "ENTRIES", R.ENTRIES + (broken,))
    problems = R.check_files()
    assert len(problems) == 1
    assert "ghost" in problems[0] and "does not exist" in problems[0]

    # a real file with the wrong digest is caught too
    real = str(tmp_path / "weights.pt")
    with open(real, "wb") as f:
        f.write(b"not the weights")
    broken["checkpoint"] = real
    problems = R.check_files()
    assert len(problems) == 1
    assert "sha256" in problems[0]

    # and `verify_sha256=False` is the "does it even exist" mode
    assert R.check_files(verify_sha256=False) == []


def test_importing_the_seat_model_never_hashes_a_checkpoint():
    """Structure at import, bytes only on demand.

    The subprocess is the only way to observe an import: patching
    `hashlib.sha256` from inside an already-imported module would prove
    nothing. If this fails, something started calling `check_files()` from
    module scope and every `import seats` in the game began hashing
    twenty megabytes.
    """
    code = ("import hashlib\n"
            "def boom(*a, **k):\n"
            "    raise AssertionError('hashed at import time')\n"
            "hashlib.sha256 = boom\n"
            "import seats\n"
            "print(len(seats.automated_options()))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    assert r.stdout.strip() == b"11"


def test_the_check_command_reports_the_pools_and_exits_zero(capsys):
    assert R._main(["--check"]) == 0
    out = capsys.readouterr().out
    for name in R.POOL_NAMES:
        assert name in out
    assert R._main([]) == 2


# --------------------------------------------------------------------------
# the import-time validation actually fires
# --------------------------------------------------------------------------

def _entry(**over):
    """A minimal entry the loader would accept, then mutated per test."""
    e = {"key": "wolf", "aliases": [], "kind": "heuristic",
         "family": "personality", "module": "ai.wolf",
         "pools": ["all", "no_imitation"], "enabled": True,
         "selectable": True, "label": "wolf", "desc_key": "wolf_desc",
         "note": ""}
    e.update(over)
    return e


def test_a_sane_entry_validates():
    R._validate([_entry()])


@pytest.mark.parametrize("mutate, needle", [
    ({"key": ""}, "`key` must be a non-empty string"),
    ({"kind": "weights"}, "kind"),
    ({"family": "network"}, "family"),
    ({"kind": "network"}, "do not go together"),
    ({"pools": ["all", "weekly"]}, "not one of"),
    ({"desc_key": "no_such_key"}, "config.I"),
    ({"label": "no_such_key"}, "config.I"),
    ({"enabled": "yes"}, "must be true or false"),
    ({"wat": 1}, "unknown field"),
    ({"kind": "network", "family": "rl", "module": "ai.wolf",
      "checkpoint": "p", "source": "s"}, "sha256"),
])
def test_a_structurally_broken_entry_is_refused(mutate, needle):
    with pytest.raises(ValueError) as exc:
        R._validate([_entry(**mutate)])
    assert needle in str(exc.value)


def test_a_duplicate_key_is_refused():
    with pytest.raises(ValueError) as exc:
        R._validate([_entry(), _entry()])
    assert "duplicate key" in str(exc.value)


def test_an_alias_that_collides_with_a_key_is_refused():
    with pytest.raises(ValueError) as exc:
        R._validate([_entry(), _entry(key="chess", aliases=["wolf"])])
    assert "collides with a registered key" in str(exc.value)


def test_one_alias_claimed_by_two_seats_is_refused():
    with pytest.raises(ValueError) as exc:
        R._validate([_entry(aliases=["old"]),
                     _entry(key="chess", aliases=["old"])])
    assert "claimed by both" in str(exc.value)


def test_an_anchor_that_is_not_registered_is_refused():
    with pytest.raises(ValueError) as exc:
        R._validate_roster(["chess"])
    assert "RL_KL_ANCHOR" in str(exc.value)
    R._validate_roster(["chess", "rl_h1000_0k"])       # the real roster shape


def test_validation_reports_every_problem_at_once():
    """An editor who made two mistakes should see both in one import."""
    with pytest.raises(ValueError) as exc:
        R._validate([_entry(pools=["weekly"], desc_key="nope"),
                     _entry(key="")])
    msg = str(exc.value)
    assert "not one of" in msg and "config.I" in msg and "`key`" in msg


# --------------------------------------------------------------------------
# the file itself
# --------------------------------------------------------------------------

def test_the_json_is_utf8_and_ordered_the_way_the_pool_is():
    path = os.path.join(os.path.dirname(R.__file__), "registry.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert [e["key"] for e in raw["entries"]] == list(S.automated_options())
    assert raw["entries"][0]["key"] == "wolf"
    assert raw["entries"][-1]["key"] == "rl_h1000_20k"


def test_the_registry_file_is_committed():
    """A roster that is not in the repository is not a roster."""
    rel = os.path.join("ai", "registry.json")
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                       cwd=_ROOT, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip("git could not be consulted: %s"
                    % r.stderr.decode().strip())
    assert r.returncode == 0, (
        "ai/registry.json is not tracked; a fresh clone would have no roster")
