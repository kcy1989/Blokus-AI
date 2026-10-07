"""`rl_1000_20k` and `rl_h1000_20k` are the same policy.

plan9 step 1 renamed the trained-policy seat from `rl_1000_20k` to
`rl_h1000_20k` and kept the old name as an alias. A rename is only free if the two
names are genuinely interchangeable, which is the property these tests pin - and
three ways it could quietly not hold:

  * the alias might not be recognised, so an old command line stops working;
  * it might be recognised as a *separate* contestant, so the pool grows to twelve
    and a league draws the same weights twice as often;
  * it might resolve to the right name but the wrong file, or the right file with
    different weights, which is a rename that quietly changed the model.

The third is why the checkpoint's own `engine_version`, `feature_version` and
`action_table_hash` are compared, and why two brains are asked for a move on the
same position rather than merely being asked to load. Loading the same path twice
proves nothing about whether the name you passed reached it.
"""
import hashlib
import json
import os
import random

import pytest

import match as M
import seats as S

NEW = "rl_h1000_20k"
OLD = "rl_1000_20k"


def _blob():
    import torch
    return torch.load(S.rl_checkpoint(NEW), map_location="cpu",
                      weights_only=False)


# --------------------------------------------------------------------------
# the two names reach the same file
# --------------------------------------------------------------------------

def test_both_names_resolve_to_one_file():
    """One file, one path, two spellings.

    The path moved to `ai/checkpoints/` at plan9a stage 4, because that is the
    file the pool reads and it belongs with the code; `ai/registry.json`'s
    `source` still records `data/rl1/step_000040.pt` as where it came from.
    """
    assert S.rl_checkpoint(OLD) == S.rl_checkpoint(NEW) == \
        "ai/checkpoints/rl_h1000_20k/step_000040.pt"
    assert os.path.exists(S.rl_checkpoint(NEW))


def test_the_alias_is_one_hop_and_resolves_to_the_registered_name():
    assert S.RL_ALIASES == {OLD: NEW}
    assert S.canonical_key(OLD) == NEW
    assert S.canonical_key(NEW) == NEW
    assert S.canonical_key("hunter") == "hunter"


def test_both_names_are_recognised_as_a_trained_policy():
    assert S.is_rl_key(OLD) and S.is_rl_key(NEW)
    assert S.kind_of(OLD) == S.kind_of(NEW) == S.KIND_RL
    assert S.rl_keys() == (NEW,), "the table names the policy once"


# --------------------------------------------------------------------------
# the alias is not a second contestant
# --------------------------------------------------------------------------

def test_the_alias_does_not_enter_the_pool():
    """The failure this guards is a pool of twelve, not a name that stopped working.

    A second row in `RL_SEATS` would have been the obvious way to "support both
    names" and it is wrong twice over: the pool grows by one, so every option's
    draw probability falls, and one model is represented twice, so a league says
    it played it twice as often as the pool actually implies.
    """
    options = S.automated_options()
    assert len(options) == 11
    assert NEW in options
    assert OLD not in options
    assert len(M.POOL_PRESETS["all"]) == 11
    assert M.POOL_PRESETS["rl_only"] == (NEW,)
    assert len(M.POOL_PRESETS["no_imitation"]) == 7


def test_pool_parsing_accepts_the_alias_and_expands_to_the_new_name():
    assert M.expand_pool(OLD) == [NEW]
    assert M.expand_pool("no_imitation,%s" % OLD) == list(
        M.expand_pool("no_imitation,%s" % NEW))
    # and it is not a silent acceptance of anything unknown
    with pytest.raises(ValueError):
        M.expand_pool("rl_9999")


def test_a_subject_named_by_its_old_name_is_recorded_under_the_new_one():
    """The alias must not reach the evidence.

    `subject` lands in the batch file and in the filename, and those outlive the
    rename. An old name written into a committed batch would need explaining to
    every later reader.
    """
    assert M.validate_subject(OLD, ("hunter",), True) == NEW
    assert M.validate_subject(NEW, ("hunter",), True) == NEW
    assert M.validate_subject(None, ("hunter",), False) is None


# --------------------------------------------------------------------------
# the two names are the same weights
# --------------------------------------------------------------------------

def test_both_names_return_the_same_model():
    """Fingerprints, weights, and a move - not just "it loaded".

    `engine_version`, `feature_version` and `action_table_hash` are the three
    things that decide whether two checkpoints are the same policy at all; a
    mismatch in any of them means the alias points at a different model wearing
    the same name.
    """
    new = S.build_brain(NEW, random.Random(0), mode="argmax")
    old = S.build_brain(OLD, random.Random(0), mode="argmax")
    assert new.step == old.step == 40
    assert new.key == NEW and old.key == OLD, "each keeps the name it was asked for"

    blob = _blob()
    assert blob["engine_version"] == "engine-1"
    assert blob["feature_version"] == "features-v2-14plane-13scalar"
    assert blob["action_table_hash"] == "91252a354b090d43"

    # identical parameters, tensor for tensor
    new_sd, old_sd = new.net.state_dict(), old.net.state_dict()
    assert set(new_sd) == set(old_sd)
    for name in new_sd:
        assert torch_equal(new_sd[name], old_sd[name]), name


def test_both_names_choose_the_same_move_on_the_same_position():
    """The end-to-end version: same board, same answer.

    The weights being equal implies this, so it is not a second proof - it is the
    check that nothing *around* the weights differs, which is where an alias can
    leak in: a different mode, a different temperature, a different key reaching
    the feature builder. Driven through `choose`, which is the path a real game
    takes, and in argmax so the answer is deterministic.
    """
    from game import Game

    def move_of(key):
        brain = S.build_brain(key, random.Random(0), mode="argmax")
        g = Game(random.Random(3))
        g.setup_seats(["hunter", "wolf", "fox", "builder"],
                      ["blue", "green", "yellow", "red"],
                      random.Random(3), mode="argmax")
        brain.attach(g)
        return brain.choose(g)

    assert move_of(NEW) is not None, "an opening position has legal moves"
    assert move_of(NEW) == move_of(OLD)


def test_the_rename_did_not_change_a_single_game():
    """The old spelling never reaches a game; the new one always does.

    The designated seat is pinned rather than hoped for. This test used to lean
    on a two-game sample landing on the renamed key by chance, which was a fact
    about one seed and one pool order - plan9a stage 3 reordered every pool by
    key, the renamed key sorts somewhere else, and the sample stopped landing.
    Pinning it as the subject says the same thing without depending on a draw:
    the name in the rows is the new one, in every game, and the retired one
    appears nowhere.
    """
    rows = M.run_league(games=2, seed=20260928,
                        options=tuple(S.automated_options()), mode="argmax",
                        paired_rng=True, subject=NEW)
    flat = [(k, c, r) for game in rows for k, c, r, _rk, _p in game]
    assert OLD not in {k for k, _c, _r in flat}
    assert NEW in {k for k, _c, _r in flat}
    assert all(any(k == NEW for k, *_rest in game) for game in rows)


def test_records_json_names_only_the_new_key():
    """The rename step renamed it; an alias does not merge leaderboard rows.

    `records.json` is keyed by the name a game was played under and knows nothing
    about aliases, so a result recorded through the old name would open a second
    row for one policy. Renaming the existing row is what keeps the history in
    one place.
    """
    with open(S.__file__.replace("seats.py", "records.json"),
              encoding="utf-8") as fh:
        entries = json.load(fh)
    assert OLD not in entries
    assert NEW in entries


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)