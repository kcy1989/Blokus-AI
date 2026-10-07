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
import hashlib
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

    `automated_options()` is what `rng.choice` draws an index into, so the order
    half of this is load-bearing: it is the same tuple the league, the seat menu
    and `RANDOM_AI_KEY` all walk. Since stage 3 that order is `pool_order` - by
    key - so what the test pins is "the registered keys, sorted", and the JSON's
    own array order is free to be whatever the file says.
    """
    assert tuple(sorted(R.keys())) == S.automated_options()
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
    """Since stage 4 the JSON *is* the path, and `seats` reads it.

    Two separate claims. The first is one-authority: `rl_checkpoint` and
    `imitation_checkpoint` hand back exactly what the JSON says, so there is no
    second table for a path to drift in. The second keeps the training record
    honest - `source` still says where the weights were trained, and it has to
    agree with `RL_SEATS` and with H-C2's own directory, because those two are
    the only surviving evidence that a published copy is the right file.

    The imitation filename is pinned too: `build_brain` reopens the path as
    `<dirname>/step_%06d.pt` built from `imitation_step`, so a checkpoint whose
    basename broke that convention would load a different file than the JSON
    names and nothing else would notice.
    """
    rl_sources = dict(S.RL_SEATS)
    seen = 0
    for e in R.ENTRIES:
        if e["kind"] != "network":
            assert "checkpoint" not in e, e["key"]
            continue
        seen += 1
        if e["family"] == "rl":
            assert S.rl_checkpoint(e["key"]) == e["checkpoint"], e["key"]
            assert e["source"] == rl_sources[e["key"]], e["key"]
            continue
        assert S.imitation_checkpoint(e["key"]) == e["checkpoint"], e["key"]
        step = S.imitation_step(e["key"])
        assert e["source"] == os.path.join(
            S.IMITATION_CHECKPOINT_DIR, "step_%06d.pt" % step), e["key"]
        assert os.path.basename(e["checkpoint"]) == "step_%06d.pt" % step, \
            e["key"]
    assert seen == 4


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


def test_reordering_the_json_rows_changes_no_pool_and_no_game(monkeypatch):
    """Stage 3 check (b): registration order stopped being observable.

    The array order used to be the index-to-key mapping every same-seed game
    depended on, so reordering the file reseated every committed batch. Now
    every pool leaves `ai.registry` through `pool_order`, and the only thing a
    reorder can change is the file.

    `ENTRIES` is the live global, so reversing it is a faithful stand-in for
    reordering the JSON and restarting: `automated_options()` and `pool()` read
    it on every call. `match._ORDER` is deliberately not covered - it is built
    once at import and would have to be rebuilt to see a change, which is a
    property of `match`, not of the registry.
    """
    entries = R.ENTRIES
    raw_before = tuple(e["key"] for e in entries)
    pool_before = list(R.pool("no_imitation"))
    options_before = list(S.automated_options())
    assert raw_before != options_before, \
        "the array order must differ from key order for this to mean anything"
    games_before = M.run_league(3, seed=1701, options=pool_before)

    monkeypatch.setattr(R, "ENTRIES", tuple(reversed(entries)))
    raw_after = tuple(e["key"] for e in R.ENTRIES)
    assert raw_after == tuple(reversed(raw_before)), "the reorder must land"
    assert raw_after != raw_before

    # what a draw reads - both the whole pool and one named preset
    assert list(S.automated_options()) == options_before
    assert list(R.pool("no_imitation")) == pool_before
    assert list(R.pool("all")) == options_before

    # and the games, not just the lists
    games_after = M.run_league(3, seed=1701, options=pool_before)
    assert games_after == games_before


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
# files: existence is a suite concern, bytes are --check's
# --------------------------------------------------------------------------

def test_every_registered_checkpoint_hashes_as_recorded():
    """Existence *and* digest, back in the suite as of plan9a stage 4.

    The digest assertion was pulled at stage 2 because the files lived under
    `data/` - the training factory, which a retrain rewrites - so a routine
    retrain would have turned the suite red for no reason. Stage 4 published the
    two published weights into `ai/checkpoints/`, where nothing but a deliberate
    re-publication can touch them, and `check_files()` runs in full again.

    `hc_2000` and `hc_10000` are the remaining case, and hashing them on purpose:
    they are still under `data/` and still in the registry, so a retrain that
    overwrote one *should* fail here - it would mean the pool's numbers moved
    underneath the evidence that cites them.
    """
    assert R.check_files() == []


def test_ai_checkpoints_holds_nothing_unpublished():
    """Stage 4's other half: the repository proper holds only registered weights.

    The mirror of `check_files` - that one asks "does every registered seat have
    a file", this one asks "does every file belong to a registered seat". Without
    it, a stray copy left in `ai/checkpoints/` ships forever: it weighs 5.6 MB,
    it looks official because of where it sits, and nothing names it.

    This is also why stage 4 published two weights and not five. The three
    `rl_*1000_0k` students are not in the roster until stage 6, so moving them
    here first would put exactly this file on disk with nothing to say what it
    is.
    """
    root = os.path.join(os.path.dirname(R.__file__), "checkpoints")
    if not os.path.isdir(root):
        pytest.skip("ai/checkpoints/ does not exist in this checkout")
    published = sorted(
        os.path.relpath(os.path.join(dirpath, name), root)
        for dirpath, _dirs, names in os.walk(root)
        for name in names)
    registered = sorted(
        os.path.relpath(e["checkpoint"], root)
        for e in R.ENTRIES
        if e["kind"] == "network"
        and e["checkpoint"].startswith("ai/checkpoints/"))
    assert published == registered


def test_check_files_reports_a_missing_checkpoint(tmp_path, monkeypatch):
    """The check has to be able to fail, or it is decoration.

    `ENTRIES` is replaced outright rather than extended: leaving the real
    entries in place would make this test hash `data/` again, which is the very
    coupling the test above avoids.
    """
    ghost = {"key": "ghost", "kind": "network", "family": "rl",
             "checkpoint": str(tmp_path / "nope.pt"), "source": "x",
             "sha256": "0" * 64}
    monkeypatch.setattr(R, "ENTRIES", (ghost,))
    problems = R.check_files(verify_sha256=False)
    assert len(problems) == 1
    assert "ghost" in problems[0] and "does not exist" in problems[0]


def test_check_files_detects_a_rewritten_checkpoint(tmp_path, monkeypatch):
    """The sha256 mode, exercised against a scratch file and nothing else."""
    weights = tmp_path / "weights.pt"
    blob = b"the weights"
    weights.write_bytes(blob)
    entry = {"key": "scratch", "kind": "network", "family": "rl",
             "checkpoint": str(weights), "source": "x", "sha256": "0" * 64}
    monkeypatch.setattr(R, "ENTRIES", (entry,))

    problems = R.check_files()
    assert len(problems) == 1 and "sha256" in problems[0]
    assert "scratch" in problems[0]

    entry["sha256"] = hashlib.sha256(blob).hexdigest()
    assert R.check_files() == []
    # `verify_sha256=False` is the "does it even exist" mode
    assert R.check_files(verify_sha256=False) == []


def test_importing_never_hashes_a_checkpoint_and_never_warns():
    """Structure at import, bytes only on demand - and no runpy noise.

    Two facts, one subprocess each. The first is why `check_files` is a
    function and not a module-level call: patching `hashlib.sha256` from
    inside an already-imported module would prove nothing, so the import has
    to be observed from outside. The second is why the entry point is
    `python -m ai` rather than `python -m ai.registry`: the package re-exports
    `ai.registry`, so running the submodule executes it twice and runpy warns.
    """
    hash_code = ("import hashlib\n"
                 "def boom(*a, **k):\n"
                 "    raise AssertionError('hashed at import time')\n"
                 "hashlib.sha256 = boom\n"
                 "import seats\n"
                 "print(len(seats.automated_options()))\n")
    r = subprocess.run([sys.executable, "-c", hash_code], cwd=_ROOT,
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    assert r.stdout.strip() == b"11"
    assert b"RuntimeWarning" not in r.stderr

    r = subprocess.run([sys.executable, "-c", "import ai, seats"],
                       cwd=_ROOT, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    assert b"RuntimeWarning" not in r.stderr, r.stderr.decode()


def test_the_package_entry_point_runs_without_the_duplicate_module_warning():
    """`python -m ai --check` is the command; `python -m ai.registry` warns.

    The exit code is deliberately not asserted: `--check` verifies digests,
    and a digest mismatch against `data/` is a stage-4 concern, not this
    suite's. What is under test here is the wiring - `ai/__main__.py` reaches
    `registry.main`, all four pools are printed, and runpy has nothing to
    complain about.
    """
    r = subprocess.run([sys.executable, "-m", "ai", "--check"],
                       cwd=_ROOT, capture_output=True)
    assert r.returncode != 2, r.stderr.decode()      # not a usage error
    assert b"RuntimeWarning" not in r.stderr, r.stderr.decode()
    out = r.stdout.decode()
    for name in R.POOL_NAMES:
        assert name in out, name


def test_the_check_command_reports_the_pools_and_exits_zero(capsys,
                                                            monkeypatch):
    """The digest path is stubbed for the same reason as above: `data/` may be
    rewritten by a retrain, and this test asserts the CLI, not the weights.
    The digest itself is covered by
    `test_check_files_detects_a_rewritten_checkpoint`."""
    monkeypatch.setattr(R, "check_files", lambda verify_sha256=True: [])
    assert R.main(["--check"]) == 0
    out = capsys.readouterr().out
    for name in R.POOL_NAMES:
        assert name in out
    assert R.main([]) == 2
    assert "python -m ai --check" in capsys.readouterr().out


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


def test_enabled_and_selectable_must_agree_and_the_message_names_the_key():
    """Risk 4 of stage 2: one flag flipped alone is a silent split-brain.

    `enabled` and `selectable` document two different questions, but today
    they gate two views of the same eleven keys - `pool()` reads one,
    `automated_options()` the other. Flipping only `selectable` would take a
    key out of the seat menu while leaving it in every league, with nothing
    failing anywhere. The message has to name the key, because a bare
    "flags disagree" in a file with eleven rows is not actionable.
    """
    for enabled, selectable in ((True, False), (False, True)):
        with pytest.raises(ValueError) as exc:
            R._validate([_entry(enabled=enabled, selectable=selectable)])
        msg = str(exc.value)
        assert "wolf" in msg, msg
        assert "enabled=%s and selectable=%s disagree" % (enabled, selectable) \
            in msg, msg

    # agreeing, either way, is fine
    R._validate([_entry(enabled=True, selectable=True)])
    R._validate([_entry(enabled=False, selectable=False)])

    # and the disagreement is caught alongside any other complaint
    with pytest.raises(ValueError) as exc:
        R._validate([_entry(pools=["weekly"], selectable=False)])
    assert "not one of" in str(exc.value) and "wolf" in str(exc.value)


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

def test_the_json_rows_are_unchanged_and_the_pool_is_them_sorted():
    """Two orders, both pinned, on purpose.

    The array keeps the registration order stage 3 deliberately did **not**
    touch - proof that the re-sort of `GOLDEN` came from a rule and not from an
    edit to the roster - while the pool is that same set sorted by key, which is
    what every draw now reads.
    """
    path = os.path.join(os.path.dirname(R.__file__), "registry.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    keys = [e["key"] for e in raw["entries"]]
    assert keys == ["wolf", "chess", "fox", "intruder", "optimizer", "builder",
                    "hunter", "rl_h1000_0k", "hc_2000", "hc_10000",
                    "rl_h1000_20k"]
    assert sorted(keys) == list(S.automated_options())


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
