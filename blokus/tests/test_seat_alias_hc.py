"""`hc_1000` and `rl_h1000_0k` are one seat, spelled two ways.

plan9a stage 1 renamed the 0k checkpoint's option key. The old name counted
steps of an unnamed run; the new one says what the checkpoint *is* - the 0k
starting point of the `rl_h1000` chain, distilled from `hunter` and trained for
1,000 imitation steps. The file never moved: `data/hc2/step_001000.pt`.

Five ways a rename like this quietly breaks, each with its own test below:

  * the retired spelling stops resolving - and two committed evidence batches
    were recorded as `--subject hc_1000`, with a `records.json` row under it;
  * the retired spelling becomes a *second* seat, so the pool grows and a league
    draws the same weights twice as often;
  * the seat moves within the option list - when the rename landed, the pool
    was still in registration order, the key sat at the same index and only the
    name column moved, which is what `GOLDEN` recorded. plan9a stage 3 then
    ordered every pool by key, and a rename changes where the key *sorts*, so
    this is no longer a thing a rename can avoid; what is still checkable is
    that the set of seats did not change;
  * the two spellings reach different weights, which would be a rename that
    quietly changed the model;
  * a leaderboard row under the old key stops being readable.
"""
import random

import pytest

import ai.registry as REG
import match as M
import records as R
import seats as S

OLD = "hc_1000"
NEW = "rl_h1000_0k"


# --------------------------------------------------------------------------
# the old spelling still resolves
# --------------------------------------------------------------------------

def test_the_old_spelling_still_names_step_1000():
    assert S.canonical_key(OLD) == NEW
    assert S.canonical_key(NEW) == NEW
    assert S.imitation_key(1000) == NEW
    assert S.imitation_step(OLD) == S.imitation_step(NEW) == 1000
    assert S.is_imitation_key(OLD) and S.is_imitation_key(NEW)
    assert S.kind_of(OLD) == S.kind_of(NEW) == S.KIND_IMITATION
    # a prefix of the right shape and no file behind it is still refused
    assert not S.is_imitation_key("hc_9999")
    with pytest.raises(ValueError):
        S.imitation_step("wolf")


def test_a_pool_and_a_subject_named_by_the_old_spelling_expand_to_the_new_one():
    """The old spelling must never reach the evidence.

    `expand_pool` and `validate_subject` are the two doors a name enters the
    measurement through, and both write their answer into a batch file that
    outlives the rename.
    """
    assert M.expand_pool(OLD) == [NEW]
    assert M.expand_pool("no_imitation,%s" % OLD) == \
        M.expand_pool("no_imitation,%s" % NEW)
    assert M.validate_subject(OLD, M.POOL_PRESETS["no_imitation"], True) == NEW
    assert M.validate_subject(NEW, M.POOL_PRESETS["no_imitation"], True) == NEW
    assert M.validate_subject(None, M.POOL_PRESETS["no_imitation"], False) is None


# --------------------------------------------------------------------------
# the alias is not a second seat
# --------------------------------------------------------------------------

def test_the_alias_does_not_enter_the_pool():
    options = S.automated_options()
    # Stage 8: sizes come from the registry, so the claim these were making -
    # "the rename added no seat and no preset" - is now two lists agreeing
    # rather than two numbers matching fourteen.
    assert list(options) == list(REG.pool("all"))
    assert len(set(options)) == len(options)
    assert NEW in options
    assert OLD not in options
    for name, keys in M.POOL_PRESETS.items():
        assert list(keys) == list(REG.pool(name)), name
    assert M.POOL_PRESETS["rl_only"] == ("rl_h1000_20k",)


def test_the_rename_left_every_other_seat_where_it_was():
    """The pool is the same fourteen keys, one of them spelled differently.

    What is checkable is the *set*: nothing added, nothing removed, nothing
    substituted beyond that one string. The order is `ai.registry.pool_order`
    since plan9a stage 3, so pinning a tuple here would pin key order - and key
    order deliberately *does* move when a key is renamed, because the sort key
    is the name. That is a consequence worth stating rather than hiding: the
    seat list a player sees reorders on a rename, and the games change with it.
    """
    options = S.automated_options()
    # The set pinned below is the count now: it names every seat, so a roster
    # change fails there, against a list a person wrote.
    assert len(options) == len(set(options))
    assert NEW in options
    assert OLD not in options
    assert options == tuple(sorted(options))
    assert set(options) == {
        "wolf", "chess", "fox", "intruder", "optimizer", "builder", "hunter",
        NEW, "hc_2000", "hc_10000", "rl_h1000_20k", "rl_b1000_0k",
        "rl_i1000_0k", "rl_o1000_0k"}


# --------------------------------------------------------------------------
# both spellings are the same weights
# --------------------------------------------------------------------------

def test_both_spellings_load_the_same_checkpoint():
    old = S.build_brain(OLD, random.Random(0), mode="argmax")
    new = S.build_brain(NEW, random.Random(0), mode="argmax")
    assert old.step == new.step == 1000
    # each keeps the spelling it was asked for, so `brain.key == owner_key` still
    # holds for a game set up with either name
    assert old.key == OLD and new.key == NEW

    old_sd, new_sd = old.net.state_dict(), new.net.state_dict()
    assert set(old_sd) == set(new_sd)
    for name in old_sd:
        assert _equal(old_sd[name], new_sd[name]), name


def test_both_spellings_choose_the_same_move_on_the_same_position():
    """End to end: same board, same answer, driven through `choose`.

    Weights being equal implies this, so what it adds is the check that nothing
    *around* the weights differs - a different checkpoint directory reached by
    the old prefix scheme, or a different key reaching the feature builder.
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


# --------------------------------------------------------------------------
# history keeps working
# --------------------------------------------------------------------------

def test_a_records_row_under_the_old_key_still_reads(tmp_path):
    """`records.json` is keyed by the name a game was played under.

    The row for `hc_1000` exists and is not rewritten by this rename - a rename
    that merged or dropped it would silently lose the games already counted
    under it.
    """
    path = tmp_path / "records.json"
    path.write_text('{"hc_1000": {"games": 3, "total_points": 9.0,'
                    ' "total_remaining": 30.0}}', encoding="utf-8")
    board = R.Records(str(path))
    assert set(board.entries) == {OLD}
    assert board.entries[OLD]["games"] == 3
    assert board.entries[OLD]["total_points"] == 9.0
    # and the provenance fields of a pre-rename file read as unknown, not as an
    # error and not as an inferred value
    from records import META_FIELDS
    for field in META_FIELDS:
        assert board.meta(OLD)[field] is None


def _equal(a, b):
    import torch
    return torch.equal(a, b)
