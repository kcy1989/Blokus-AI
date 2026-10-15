"""The training loop: does a run resume within float range, and do the
guardrails stop it cleanly?

The property that makes a 40-round run worth starting is that round 7 means the
same thing whether the process survived to reach it or was killed and resumed.
So resume is tested for the same parameters within a 1e-4 float tolerance, not
for "roughly the same loss". The tolerance is not an easing of the standard but
an honest one: the pipeline is not bit-reproducible across processes - measured
2026-10-13, the games come back bitwise identical while the float path diverges
from round 1, reaching max|d| ~ 2.6e-5 in the weights after two Adam steps
(3 failures in 60 runs under `torch.equal`). A resume that drifts *beyond* that
noise would quietly turn one 20,000-episode run into several shorter ones with
overlapping seed blocks, which is the failure this whole seed-block machinery
exists to prevent.

The guardrail test matters for the same reason: a run that stops should stop
*with the round that tripped it saved*, so the numbers are recoverable.

Everything writes under `tmp_path`. No test creates `data/rl1`, and the
evaluation test checks that a round's evaluation leaves `records.json` and every
`data/h*` directory byte-identical - which is the property the B0c guard watches
for the rest of the project.
"""
import copy
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import rl.paired3 as paired3
import rl.ppo as ppo_mod
import rl.rl_train as T
import rl.rollout as rollout_mod
import seats
from rl.policy import load_policy

STEP = seats.IMITATION_STEPS[0]
# Plan9 task 4 retired `rl_h1000_0k` from the roster and moved the weights back
# under H-C2's own directory, so `IMITATION_CHECKPOINT_DIR` names a file a fresh
# clone carries again. The registry has no row for it any more, which is why this
# is a path rather than `seats.imitation_checkpoint`.
CHECKPOINT_DIR = seats.IMITATION_CHECKPOINT_DIR
# The 0k seat under the name it used to be registered as. Nothing in the roster
# answers to it now; `bind_retired_0k` below is what makes it a seat again, the
# same way `match.py --adhoc` does.
LEARNER_KEY = "rl_h1000_0k"
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "step_%06d.pt" % STEP)
SMALL = ppo_mod.Config(epochs=1, minibatch=32, kl_coef=0.0)

needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="no H-C2 checkpoint at %s" % CHECKPOINT)


@pytest.fixture(autouse=True)
def bind_retired_0k():
    """Seat the retired 0k checkpoint for the duration of one test.

    `rl_h1000_0k` left the roster in plan9 task 4, and `--adhoc` is the
    supported way to bring it back for a run - so this makes the same call
    `match.py --adhoc KEY=PATH` makes rather than reaching into `kind_of`.

    Function-scoped, not module-scoped: `seats.ADHOC` is process-global, so a
    module-scoped binding stays alive while xdist (`--dist load`) dispatches
    other modules' tests to the same worker - which made
    `test_seat_alias_hc.py::test_neither_spelling_is_an_alias_of_anything`
    flaky (~1 failure per 4 full-suite runs under `-n 8`). The undone here
    still happens at test end either way; the scope is what stops the leak.
    """
    if not os.path.exists(CHECKPOINT):
        yield
        return
    seats.register_adhoc(LEARNER_KEY, CHECKPOINT)
    yield
    seats.ADHOC.pop(LEARNER_KEY, None)


@pytest.fixture(autouse=True)
def one_thread():
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


def small_cfg(tmp_path, **over):
    base = T.TrainConfig(games=20, per_round=10, rounds=2, n_procs=2,
                         rollout_device="cpu", update_device="cpu",
                         eval_every=0, eval_games=4)
    return base._replace(**over)


def quiet(*_a, **_k):
    return None


def read_log(out_dir):
    path = os.path.join(out_dir, "rounds.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


# --------------------------------------------------------------- 8a a run

@needs_checkpoint
def test_two_rounds_of_ten_games_run_and_produce_a_loadable_checkpoint(tmp_path):
    """The whole path, once: loop, save, and the saved file usable as a *seat*.

    Loading through `rl.policy.load_policy` rather than through this module's own
    reader is the point - a checkpoint nothing else can open is not a checkpoint.
    """
    out = tmp_path / "rl1"
    result = T.train(small_cfg(str(out)), str(out), ppo_cfg=SMALL,
                     init_path=CHECKPOINT, log=quiet, nice=0)
    assert result.rounds_done == 2
    assert result.stopped_by == "", result.stopped_by

    rows = read_log(str(out))
    assert len(rows) == 2
    for row in rows:
        assert row["games"] == 10
        assert np.isfinite(row["learner_reward_mean"])
        # the counts are string-keyed so the row survives JSON unchanged
        assert set(row["learner_rank_counts"]) <= {"1", "2", "3", "4"}
        assert sum(row["learner_rank_counts"].values()) == row["games"]
        assert row["learner_rank_mean"] == pytest.approx(
            sum(int(k) * v for k, v in row["learner_rank_counts"].items())
            / row["games"], abs=1e-6)
        assert set(row["stats"]) >= {"policy_loss", "value_loss", "entropy",
                                     "clip_fraction", "explained_variance",
                                     # plan9 3c: EV and the value-return
                                     # correlation are logged apart
                                     "value_return_corr"}
        # 平均餘格 beside 平均名次 (plan9 3c); remaining is cells, 0..21-ish
        assert row["learner_remaining_mean"] is not None
        assert row["learner_remaining_mean"] >= 0.0
        assert all(not g["fired"] for g in row["guardrails"])

    step_path = os.path.join(str(out), "step_000002.pt")
    assert os.path.exists(step_path)
    assert os.path.exists(os.path.join(str(out), "latest.pt"))

    net, meta = load_policy(step_path)
    assert meta["in_ch"] == 27

    import torch as _t
    start, _m = load_policy(CHECKPOINT)
    sd_new = net.state_dict()
    sd_old = start.state_dict()
    shared = [k for k in sd_old if not k.startswith("rl_value.")]
    assert shared, "no policy parameters compared"
    assert any(not _t.equal(sd_new[k], sd_old[k]) for k in shared), \
        "the policy head is identical to hc_1000, so nothing was trained"
    assert all(_t.equal(sd_new[k], sd_old[k]) for k in ("value.weight",
                                                         "value.bias")), \
        "the checkpoint's dead value head should be untouched"


@needs_checkpoint
def test_every_play_batch_call_names_the_learner(tmp_path, monkeypatch):
    """The CLI must not lean on the retired 0k fallback.

    A worker reloads `current.pt` itself and `load_policy` does not restore
    `net.key` from the blob's `name`, so with no `learner_key` `seat_for` has
    no candidate and falls back to `rl_h1000_0k` - which left the roster in
    plan9 task 4. Every round then dies with "no seat key to stand the
    learner in under". The suite hides that behind `bind_retired_0k`, so a
    green run proves nothing about the CLI; this pins the call instead.

    The key is checked **after** that binding is removed: the point is that
    the fix does not depend on the retired seat still being a seat.
    """
    calls = []
    real = rollout_mod.play_batch

    def spy(*args, **kwargs):
        calls.append(kwargs.get("learner_key"))
        return real(*args, **kwargs)

    monkeypatch.setattr(rollout_mod, "play_batch", spy)
    out = tmp_path / "named"
    cfg = small_cfg(str(out), rounds=1, eval_every=1, eval_games=4)
    result = T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT,
                     log=quiet, nice=0)
    assert result.rounds_done == 1

    # one rollout call plus the round's evaluation of current and anchor
    assert len(calls) == 3, calls
    saved = seats.ADHOC.pop("rl_h1000_0k", None)
    try:
        for key in calls:
            assert key is not None, "play_batch was called without learner_key"
            assert seats.kind_of(key) == "rl", key
    finally:
        if saved is not None:
            seats.ADHOC["rl_h1000_0k"] = saved


@needs_checkpoint
def test_a_custom_pool_reaches_both_ends_of_the_training_loop(tmp_path,
                                                              monkeypatch):
    """plan10 §2 end to end, and the reason it is two ends.

    `make_specs` picks the opponents and `play_batch` checks that every one of
    them is in the pool it was handed. Handing them *different* pools is the
    failure that would actually hurt: `play_episode` would refuse a name the
    spec legitimately drew, three seats into a round, after the seed block had
    already been claimed. So the assertion is that both saw the same value,
    read back from a real round rather than from the call.
    """
    seen = []
    real_make, real_play = rollout_mod.make_specs, rollout_mod.play_batch

    def spy_make(*args, **kwargs):
        out = real_make(*args, **kwargs)
        seen.append(("make", kwargs.get("opponent_pool"), out[0]))
        return out

    def spy_play(specs, *args, **kwargs):
        seen.append(("play", kwargs.get("opponent_pool"), specs))
        return real_play(specs, *args, **kwargs)

    monkeypatch.setattr(rollout_mod, "make_specs", spy_make)
    monkeypatch.setattr(rollout_mod, "play_batch", spy_play)

    def one_round(pool):
        del seen[:]
        out = tmp_path / ("custom" if pool else "default")
        cfg = small_cfg(str(out), rounds=1, opponent_pool=pool)
        result = T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT,
                         log=quiet, nice=0)
        assert result.rounds_done == 1
        return list(seen)

    pool = ("chess", "fox", "wolf")
    made = one_round(pool)
    assert [(w, p) for w, p, _ in made] == [("make", pool), ("play", pool)]
    for _where, _p, specs in made:
        assert specs, "a round with no specs"
        for spec in specs:
            assert set(spec.opponent_names) <= set(pool), spec

    # and the default stays None at both ends - the seven are the fallback,
    # not a pool someone rebuilt out of the same seven keys
    default = one_round(None)
    assert [(w, p) for w, p, _ in default] == [("make", None), ("play", None)]
    for _where, _p, specs in default:
        for spec in specs:
            assert set(spec.opponent_names) <= set(rollout_mod.PERSONALITY_POOL)


def test_the_seed_claim_follows_the_registered_block_for_this_base():
    """A run must exempt its own reserved row, not the h-version one.

    Only RL's own train blocks are candidates: an imitation base has to fall
    back to a claim the guard will then refuse, not find a way to exempt itself.
    """
    assert T._claimed_train_block(rollout_mod.RL_SEED_BASE) \
        is rollout_mod.RL_TRAIN_BLOCK
    assert T._claimed_train_block(rollout_mod.RL_STUDENTS_SEED_BASE) \
        is rollout_mod.RL_STUDENTS_BLOCK
    assert T._claimed_train_block(6_000_000) is rollout_mod.RL_TRAIN_BLOCK


def test_the_default_base_claim_is_the_pre_v2_one():
    """The registrations must not move a default-base run by a single seed.

    `_claimed_train_block` gained an anchor match ahead of the containment
    scan. A base that is not an anchor has to stay bit-identical, so the
    assertions are literals rather than the constants they name - a constant
    that drifted would carry the expectation with it.

    The two anchors (7,460,000 for the h rerun, 7,150,000 for the i rerun) are
    the deliberate exceptions and are pinned by their own tests; both are
    outside every registered row, so nothing that claimed by containment
    before still does.
    """
    assert T._claimed_train_block(7_000_000) \
        == ("stage RL train", 7_000_000, 7_019_999)
    assert T._claimed_train_block(7_200_000) \
        == ("stage RL students", 7_200_000, 7_219_999)
    # a base next to an anchor, and one nowhere near any row
    assert T._claimed_train_block(7_150_001) \
        == ("stage RL train", 7_000_000, 7_019_999)
    assert T._claimed_train_block(6_500_000) \
        == ("stage RL train", 7_000_000, 7_019_999)

    # and the round-1 draw itself, opponents included
    specs, man = rollout_mod.make_specs(
        7_000_000, 3, rng_seed=7_000_000,
        own=T._claimed_train_block(7_000_000))
    assert man["reserved_range_checked"] \
        == ["stage RL train", 7_000_000, 7_019_999]
    assert [(s.seed, s.learner_seat, s.opponent_names) for s in specs] == [
        (7_000_000, 3, ("optimizer", "intruder", "fox")),
        (7_000_001, 3, ("chess", "hunter", "intruder")),
        (7_000_002, 0, ("optimizer", "fox", "builder")),
    ]


def test_the_v2_anchor_claims_the_row_its_rounds_81_to_100_draw():
    """The whole point of the anchor match: the base is outside the row.

    `rl_train` derives a round's seeds as `seed_base + (round - 1) * per_round`,
    so a 7,460,000 base resuming at round 81 draws 7,500,000..7,509,999 - a
    range that a containment scan on the base would never find, and that the
    fallback `RL_TRAIN_BLOCK` would not exempt.
    """
    base = rollout_mod.RL_V2_H_SEED_BASE
    assert T._claimed_train_block(base) is rollout_mod.RL_V2_H_BLOCK

    lo = base + (81 - 1) * 500
    hi = base + (100 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_V2_H_BLOCK[1:]

    # the claim the run will actually make, and the row it is exempted against
    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_V2_H_BLOCK)
    assert [s.seed for s in specs][0] == 7_500_000
    assert [s.seed for s in specs][-1] == 7_509_999

    # every other base keeps falling through, so an anchor cannot be a way to
    # exempt an arbitrary range
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)


def test_the_i50k_anchor_claims_rounds_41_to_100_and_leaves_the_students_alone():
    """The i rerun's anchor, and the regression it must not cause.

    The base is checked before containment, so putting 7,200,000 in the anchor
    table would have shadowed `RL_STUDENTS_BLOCK` and broken every future
    student run. It is not in the table, and this asserts that directly rather
    than by absence: the students' base still resolves to the students' row,
    and the students' seeds still claim it.
    """
    base = rollout_mod.RL_I50K_V2_SEED_BASE
    assert T._claimed_train_block(base) is rollout_mod.RL_I50K_V2_BLOCK

    lo = base + (41 - 1) * 500
    hi = base + (100 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_I50K_V2_BLOCK[1:]

    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_I50K_V2_BLOCK)
    assert [s.seed for s in specs][0] == 7_170_000
    assert [s.seed for s in specs][-1] == 7_199_999

    # the regression: containment still owns the students' base
    assert T._claimed_train_block(rollout_mod.RL_STUDENTS_SEED_BASE) \
        is rollout_mod.RL_STUDENTS_BLOCK
    _, st = rollout_mod.make_specs(
        rollout_mod.RL_STUDENTS_SEED_BASE, 10,
        rng_seed=rollout_mod.RL_STUDENTS_SEED_BASE,
        own=T._claimed_train_block(rollout_mod.RL_STUDENTS_SEED_BASE))
    assert st["reserved_range_checked"] == list(rollout_mod.RL_STUDENTS_BLOCK)

    # an anchor is an exact key, not a prefix of one
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    assert T._claimed_train_block(base - 1) is rollout_mod.RL_TRAIN_BLOCK
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)


def test_the_anchor_table_is_exactly_the_seven_registered_runs():
    """The whole table, as literals, plus the property that keeps it safe.

    Anchors are matched *before* containment, so the table is the one place a
    typo can redirect a run's seed claim or - worse - shadow a registered row
    and break every run that claims that row by containment. Pinned as
    literals so an eighth entry, a moved base or a changed row fails here.

    The property is the half that changed with the v3 run: no anchor may sit
    inside any *other* registered row. The first three entries are arithmetic
    bases outside their own rows, because those runs resumed above round 1
    and round 1 never drew them. The v3, h-v2, i-v3 and h-v4 bases are the
    deliberate exception - a cold start at round 1, so their bases are drawn
    and sit at the floor of the rows they claim. Inside its own row an anchor
    claims nothing but itself; inside any other row it would win the
    pre-containment match and steal that row's claim. `7,200,000` is the base
    that would still break the rule, and it is not in the table.
    """
    assert rollout_mod.TRAIN_ANCHOR_BLOCKS == (
        (7_460_000, ("stage RL v2 h train", 7_500_000, 7_509_999)),
        (7_150_000, ("stage RL i50k v2 train", 7_170_000, 7_199_999)),
        (7_530_000, ("stage RL o50k v2 train", 7_550_000, 7_579_999)),
        (7_620_000, ("stage RL o50k v3 train", 7_620_000, 7_649_999)),
        (7_680_000, ("stage RL h50k v2 train", 7_680_000, 7_709_999)),
        (7_730_000, ("stage RL i50k v3 train", 7_730_000, 7_759_999)),
        (7_780_000, ("stage RL h50k v4 train", 7_780_000, 7_799_999)),
    )
    rows = [(label, lo, hi) for label, lo, hi in paired3.RESERVED_RANGES]
    for base, block in rollout_mod.TRAIN_ANCHOR_BLOCKS:
        assert T._claimed_train_block(base) is block
        for label, lo, hi in rows:
            if (label, lo, hi) == block:
                continue          # its own row: that claim is the point
            assert not (lo <= base <= hi), (base, label)
    # the base that would shadow the students is not one of them
    assert rollout_mod.RL_STUDENTS_SEED_BASE not in \
        [b for b, _blk in rollout_mod.TRAIN_ANCHOR_BLOCKS]
    # and the students still resolve to the students, by containment
    assert T._claimed_train_block(rollout_mod.RL_STUDENTS_SEED_BASE) \
        is rollout_mod.RL_STUDENTS_BLOCK


def test_the_o50k_anchor_claims_rounds_41_to_100():
    """The o rerun's anchor: arithmetic base outside its row, exact triple."""
    base = rollout_mod.RL_O50K_V2_SEED_BASE
    assert T._claimed_train_block(base) is rollout_mod.RL_O50K_V2_BLOCK

    lo = base + (41 - 1) * 500
    hi = base + (100 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_O50K_V2_BLOCK[1:]

    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_O50K_V2_BLOCK)
    assert [s.seed for s in specs][0] == 7_550_000
    assert [s.seed for s in specs][-1] == 7_579_999

    # an anchor is an exact key, and the two reruns beside it still hold
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    assert T._claimed_train_block(rollout_mod.RL_V2_H_SEED_BASE) \
        is rollout_mod.RL_V2_H_BLOCK
    assert T._claimed_train_block(rollout_mod.RL_I50K_V2_SEED_BASE) \
        is rollout_mod.RL_I50K_V2_BLOCK
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)


def test_the_o50k_v3_anchor_claims_rounds_1_to_60():
    """The cold-start shape: round 1 draws the base itself, so the base sits
    at the floor of the row it claims - the only anchor that is allowed to
    be inside a registered row, because the row it is inside is its own."""
    base = rollout_mod.RL_O50K_V3_SEED_BASE
    assert base == 7_620_000
    assert T._claimed_train_block(base) is rollout_mod.RL_O50K_V3_BLOCK

    lo = base + (1 - 1) * 500
    hi = base + (60 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_O50K_V3_BLOCK[1:]

    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_O50K_V3_BLOCK)
    assert [s.seed for s in specs][0] == 7_620_000
    assert [s.seed for s in specs][-1] == 7_649_999

    # an anchor is an exact key: a neighbouring base is not this row's claim
    # and would fall through to the default block
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    # and without the claim the registered row refuses the run
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)


def test_the_h50k_v2_anchor_claims_rounds_1_to_60():
    """The h learner's cold start: same shape as the v3 anchor above - round 1
    draws the base itself, so the base sits at the floor of the row it claims.
    The pool this run trains under is the caller's ten seats, which keeps the
    learner's own frozen start; that is a property of the pool, not of the seed
    row, so it is asserted in the run's evidence rather than here."""
    base = rollout_mod.RL_H50K_V2_SEED_BASE
    assert base == 7_680_000
    assert T._claimed_train_block(base) is rollout_mod.RL_H50K_V2_BLOCK

    lo = base + (1 - 1) * 500
    hi = base + (60 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_H50K_V2_BLOCK[1:]

    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_H50K_V2_BLOCK)
    assert [s.seed for s in specs][0] == 7_680_000
    assert [s.seed for s in specs][-1] == 7_709_999

    # an anchor is an exact key: a neighbouring base is not this row's claim
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    # and without the claim the registered row refuses the run
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)

    # the acceptance block is claimed the ordinary way: an eval base never
    # enters `TRAIN_ANCHOR_BLOCKS`, because no training round draws it
    eval_specs, eval_man = rollout_mod.make_specs(
        rollout_mod.RL_H50K_V2_EVAL_SEED_BASE, 1_000,
        rng_seed=rollout_mod.RL_H50K_V2_EVAL_SEED_BASE,
        own=rollout_mod.RL_H50K_V2_EVAL_BLOCK)
    assert eval_man["reserved_range_checked"] == list(
        rollout_mod.RL_H50K_V2_EVAL_BLOCK)
    assert [s.seed for s in eval_specs][-1] == 7_715_999
    with pytest.raises(ValueError):
        rollout_mod.make_specs(rollout_mod.RL_H50K_V2_EVAL_SEED_BASE, 1_000,
                               rng_seed=rollout_mod.RL_H50K_V2_EVAL_SEED_BASE)


def test_the_i50k_v3_anchor_claims_rounds_1_to_60():
    """The i learner's cold start: the same shape as the v3 and h-v2 anchors -
    round 1 draws the base itself, so the base sits at the floor of the row it
    claims. The pool this run trains under is the caller's ten seats, which
    keeps the learner's own start out; that is a property of the pool, not of
    the seed row, so it is asserted in the run's evidence rather than here."""
    base = rollout_mod.RL_I50K_V3_SEED_BASE
    assert base == 7_730_000
    assert T._claimed_train_block(base) is rollout_mod.RL_I50K_V3_BLOCK

    lo = base + (1 - 1) * 500
    hi = base + (60 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_I50K_V3_BLOCK[1:]

    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_I50K_V3_BLOCK)
    assert [s.seed for s in specs][0] == 7_730_000
    assert [s.seed for s in specs][-1] == 7_759_999

    # an anchor is an exact key: a neighbouring base is not this row's claim
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    # and without the claim the registered row refuses the run
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)

    # one evaluation row, and the decree runs argmax only: the block is claimed
    # the ordinary way and nothing reserves a second look
    eval_specs, eval_man = rollout_mod.make_specs(
        rollout_mod.RL_I50K_V3_EVAL_SEED_BASE, 1_000,
        rng_seed=rollout_mod.RL_I50K_V3_EVAL_SEED_BASE,
        own=rollout_mod.RL_I50K_V3_EVAL_BLOCK)
    assert eval_man["reserved_range_checked"] == list(
        rollout_mod.RL_I50K_V3_EVAL_BLOCK)
    assert [s.seed for s in eval_specs][-1] == 7_765_999
    with pytest.raises(ValueError):
        rollout_mod.make_specs(rollout_mod.RL_I50K_V3_EVAL_SEED_BASE, 1_000,
                               rng_seed=rollout_mod.RL_I50K_V3_EVAL_SEED_BASE)


def test_the_h50k_v4_anchor_claims_rounds_1_to_40():
    """The h learner's fourth run: the same cold-start shape as the rows above -
    round 1 draws the base itself, so the base sits at the floor of the row it
    claims. 40 rounds, not 60, because the start is the 60-step
    `rl_h1000_30k` snapshot and 40 more rounds of 500 games is the 20,000 that
    takes the line to 50,000. The start being a registered seat rather than a
    training checkpoint is what the run's evidence records, not this test."""
    base = rollout_mod.RL_H50K_V4_SEED_BASE
    assert base == 7_780_000
    assert T._claimed_train_block(base) is rollout_mod.RL_H50K_V4_BLOCK

    lo = base + (1 - 1) * 500
    hi = base + (40 - 1) * 500 + 499
    assert (lo, hi) == rollout_mod.RL_H50K_V4_BLOCK[1:]

    specs, man = rollout_mod.make_specs(
        lo, hi - lo + 1, rng_seed=base, own=T._claimed_train_block(base))
    assert man["reserved_range_checked"] == list(rollout_mod.RL_H50K_V4_BLOCK)
    assert [s.seed for s in specs][0] == 7_780_000
    assert [s.seed for s in specs][-1] == 7_799_999

    # an anchor is an exact key: a neighbouring base is not this row's claim
    assert T._claimed_train_block(base + 1) is rollout_mod.RL_TRAIN_BLOCK
    # and without the claim the registered row refuses the run
    with pytest.raises(ValueError):
        rollout_mod.make_specs(lo, hi - lo + 1, rng_seed=base)

    # the acceptance block is claimed the ordinary way: an eval base never
    # enters `TRAIN_ANCHOR_BLOCKS`, because no training round draws it
    eval_specs, eval_man = rollout_mod.make_specs(
        rollout_mod.RL_H50K_V4_EVAL_SEED_BASE, 1_000,
        rng_seed=rollout_mod.RL_H50K_V4_EVAL_SEED_BASE,
        own=rollout_mod.RL_H50K_V4_EVAL_BLOCK)
    assert eval_man["reserved_range_checked"] == list(
        rollout_mod.RL_H50K_V4_EVAL_BLOCK)
    assert [s.seed for s in eval_specs][-1] == 7_805_999
    with pytest.raises(ValueError):
        rollout_mod.make_specs(rollout_mod.RL_H50K_V4_EVAL_SEED_BASE, 1_000,
                               rng_seed=rollout_mod.RL_H50K_V4_EVAL_SEED_BASE)


@needs_checkpoint
def test_a_student_seed_base_starts_and_runs_a_round(tmp_path):
    """The registration, end to end: with the students' row in the guard's
    table, a run on 7,200,000 starts instead of refusing its own block."""
    out = tmp_path / "students"
    cfg = small_cfg(str(out), seed_base=rollout_mod.RL_STUDENTS_SEED_BASE,
                    rounds=1)
    result = T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT,
                     log=quiet, nice=0)
    assert result.rounds_done == 1
    assert result.stopped_by == "", result.stopped_by
    rows = read_log(str(out))
    assert len(rows) == 1
    assert rows[0]["seed_lo"] == rollout_mod.RL_STUDENTS_SEED_BASE


@needs_checkpoint
def test_the_critic_and_the_optimizer_survive_a_save(tmp_path):
    """Everything item 1 of the plan asks for, present in the file."""
    import torch as _t
    out = tmp_path / "rl1"
    T.train(small_cfg(str(out)), str(out), ppo_cfg=SMALL,
            init_path=CHECKPOINT, log=quiet, nice=0)
    blob = _t.load(os.path.join(str(out), "latest.pt"), map_location="cpu",
                   weights_only=False)
    for key in ("state_dict", "value_head", "round", "seed_cursor",
                "entropy_reference", "ppo_cfg", "train_cfg", "rng_state",
                "settings", "step", "engine_version", "feature_version",
                "action_table_hash", "name"):
        assert key in blob, key
    assert blob["round"] == 2
    assert blob["seed_cursor"] == 7_000_020
    assert blob["entropy_reference"] is not None
    assert blob["name"] == "rl_1000_20k"
    assert blob["settings"]["in_ch"] == 27
    # the critic is stored beside state_dict so load_policy's strict load works
    assert set(blob["value_head"]) == {"lin.weight", "lin.bias"}
    assert not any(k.startswith("rl_value.") for k in blob["state_dict"])


@needs_checkpoint
def test_the_optimizer_is_carried_across_rounds(tmp_path):
    """Adam's moments are state, and plan8-B4 item 1 asks for them to be saved.

    An optimiser built per round restarts both moments at zero every time, which
    is a different algorithm from one carried across 40 rounds - and it would
    make a resumed run not match an uninterrupted one, which is the property the
    resume test checks.
    """
    out = tmp_path / "rl1"
    T.train(small_cfg(str(out)), str(out), ppo_cfg=SMALL, init_path=CHECKPOINT,
            log=quiet, nice=0)
    blob = torch.load(os.path.join(str(out), "latest.pt"), map_location="cpu",
                      weights_only=False)
    opt = blob["optimizer"]
    assert opt is not None, "the optimizer state was not saved"
    assert len(opt["param_groups"]) == 2, "actor and critic want different lrs"
    assert opt["state"], "every moment is zero, so nothing was carried"
    moved = [i for i, st in opt["state"].items()
             if float(st["exp_avg_sq"].abs().sum()) > 0.0]
    assert moved, "no second moment accumulated"
    # and the two groups really do carry different learning rates
    lrs = sorted(g["lr"] for g in opt["param_groups"])
    assert lrs == [ppo_mod.DEFAULT_ACTOR_LR, ppo_mod.DEFAULT_VALUE_LR]


# --------------------------------------------------------------- 8b resume

@needs_checkpoint
def test_resume_is_bit_identical_to_an_uninterrupted_run(tmp_path):
    """Two rounds straight through, against one round, a stop, and a resume.

    On CPU. The plan allows CUDA to fall back to CPU for this property and to
    report the CUDA magnitude separately, and that is what this does.
    """
    whole = tmp_path / "whole"
    split = tmp_path / "split"
    cfg = small_cfg(str(whole))

    T.train(cfg, str(whole), ppo_cfg=SMALL, init_path=CHECKPOINT,
            log=quiet, nice=0, max_rounds=2)
    T.train(cfg._replace(rounds=1), str(split), ppo_cfg=SMALL,
            init_path=CHECKPOINT, log=quiet, nice=0, max_rounds=1)
    first = T.find_resume_point(str(split))
    assert first[0] == 1 and first[1] is not None
    T.train(cfg, str(split), ppo_cfg=SMALL, init_path=CHECKPOINT,
            log=quiet, nice=0, max_rounds=1, resume=True)

    # round 2 specifically: `max_rounds` caps *this invocation*, so a resume
    # with max_rounds=2 would run rounds 2 and 3 and the two runs would be at
    # different rounds when compared.
    a, _ = load_policy(os.path.join(str(whole), "step_000002.pt"))
    b, _ = load_policy(os.path.join(str(split), "step_000002.pt"))
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    # Semantics downgraded from bit-exact to a float range: the pipeline is
    # not bit-reproducible across processes - the games match bitwise (the
    # rewards and ranks in `rounds.jsonl` are identical) while the float
    # path diverges from round 1's forward pass, and two Adam steps amplify
    # that to max|d| ~ 2.6e-5 in the weights (measured 2026-10-13, 3 of 60
    # runs failed `torch.equal`). 1e-4 keeps headroom over that noise and
    # still catches a resume that drifts for real: a lost optimiser state or
    # a reshuffled batch lands orders of magnitude above it.
    far = {k: float((sa[k].float() - sb[k].float()).abs().max()) for k in sa}
    far = {k: d for k, d in far.items() if d >= 1e-4}
    assert not far, sorted(far.items(), key=lambda kv: -kv[1])[:6]

    la, lb = read_log(str(whole)), read_log(str(split))
    assert len(la) == len(lb) == 2
    assert [r["seed_lo"] for r in la] == [r["seed_lo"] for r in lb]
    # same float-range semantics as the weights above: the recorded loss has
    # been seen to differ by ~2e-6 between the two runs while everything
    # upstream of it matched, so `abs=0.0` would just move the flake here.
    assert la[1]["stats"]["policy_loss"] == pytest.approx(
        lb[1]["stats"]["policy_loss"], abs=1e-4)


@needs_checkpoint
def test_resume_from_nothing_is_the_same_as_a_fresh_run(tmp_path):
    """`--resume` with no checkpoint must not silently skip round 1."""
    out = tmp_path / "rl1"
    cfg = small_cfg(str(out))
    T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT,
            log=quiet, nice=0, max_rounds=1)
    r = T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT,
                log=quiet, nice=0, max_rounds=2, resume=True)
    # max_rounds caps this invocation, so resuming at round 1 with 2 runs rounds
    # 2 and 3. What matters is that round 2 is not skipped.
    assert r.rounds_done == 3
    rows = read_log(str(out))
    assert [row["round"] for row in rows] == [1, 2, 3]


def test_a_resume_refuses_a_checkpoint_trained_under_another_shape():
    """plan 2026-10-13 item 5b: per_round, seed_base, opponent_pool - and
    the balanced flag, for the same reason those three are checked.

    Any one of them changed is a different experiment: the rounds after the
    resume would train on different games than the rounds before it, with the
    seam logged as one continuous run. The helper sees the checkpoint's
    `train_cfg` as `save_checkpoint` writes it - a plain dict.
    """
    cfg = T.TrainConfig()
    good = dict(cfg._asdict())
    # matching, nothing recorded at all, and an old blob without the flag key
    T._refuse_mismatched_resume({"train_cfg": good}, cfg)
    T._refuse_mismatched_resume({}, cfg)
    T._refuse_mismatched_resume(None, cfg)
    old = dict(good)
    old.pop("balanced_schedule")
    T._refuse_mismatched_resume({"train_cfg": old}, cfg)

    for field, bad in (("per_round", cfg.per_round + 1),
                       ("seed_base", cfg.seed_base + 1),
                       ("opponent_pool", ("chess", "fox")),
                       ("balanced_schedule", True)):
        saved = dict(old)
        saved[field] = bad
        with pytest.raises(ValueError) as exc:
            T._refuse_mismatched_resume({"train_cfg": saved}, cfg)
        assert "resume config mismatch" in str(exc.value)
        assert field in str(exc.value)

    # an old checkpoint is fine to resume *without* the flag...
    # ...but turning the flag on against it is still a splice
    with pytest.raises(ValueError) as exc:
        T._refuse_mismatched_resume(
            {"train_cfg": old}, cfg._replace(balanced_schedule=True))
    assert "balanced_schedule" in str(exc.value)


@needs_checkpoint
def test_the_loop_refuses_a_resume_under_a_different_shape(tmp_path):
    """The refusal wired into `train()`: wrong `per_round` dies after the
    checkpoint loads and before round 2 writes anything, and the matching
    resume still works afterwards."""
    out = str(tmp_path / "rl1")
    cfg = small_cfg(out)
    T.train(cfg, out, ppo_cfg=SMALL, init_path=CHECKPOINT, log=quiet,
            nice=0, max_rounds=1)
    before = read_log(out)
    with pytest.raises(ValueError) as exc:
        T.train(cfg._replace(per_round=20), out, ppo_cfg=SMALL,
                init_path=CHECKPOINT, log=quiet, nice=0, resume=True,
                max_rounds=1)
    assert "per_round" in str(exc.value)
    assert read_log(out) == before, "a refused resume ran a round"
    r = T.train(cfg, out, ppo_cfg=SMALL, init_path=CHECKPOINT, log=quiet,
                nice=0, resume=True, max_rounds=1)
    assert r.rounds_done == 2


# ------------------------------------------------------------- 8c guardrails

@needs_checkpoint
def test_a_guardrail_stops_after_saving_and_says_why(tmp_path):
    """An impossible KL threshold must end the run with round 1 intact.

    `kl_coef` is turned on so there is a KL number to be wrong about, and the
    threshold is set below zero so it cannot pass. The checkpoint must exist
    afterwards - a guardrail that returned before the save would discard the one
    round whose numbers someone will want.
    """
    out = tmp_path / "rl1"
    cfg = small_cfg(str(out), max_kl_anchor=-1.0)
    result = T.train(cfg, str(out), ppo_cfg=ppo_mod.Config(
        epochs=1, minibatch=32, kl_coef=0.05), init_path=CHECKPOINT,
        log=quiet, nice=0, max_rounds=5)

    assert result.rounds_done == 1
    assert "kl_anchor" in result.stopped_by
    assert os.path.exists(os.path.join(str(out), "step_000001.pt"))
    assert os.path.exists(os.path.join(str(out), "latest.pt"))
    rows = read_log(str(out))
    assert len(rows) == 1
    fired = [g["name"] for g in rows[0]["guardrails"] if g["fired"]]
    assert fired == ["kl_anchor"]
    assert any("kl_anchor=" in g["detail"] for g in rows[0]["guardrails"])
    # and the guardrail list is all there, so a report names what did not fire too
    names = {g["name"] for g in rows[0]["guardrails"]}
    assert names == {"nan_or_inf", "kl_anchor", "entropy", "first_ratio_dev",
                     "disk"}


@needs_checkpoint
def test_every_guardrail_is_evaluated_even_after_one_fires():
    """All of them, every time, so a round that trips two reports both."""
    guards = T.check_guardrails(
        {"policy_loss": float("nan"), "value_loss": 1.0, "entropy": 0.1,
         "kl_anchor": 0.9, "first_ratio_dev": 1.0, "explained_variance": 0.0,
         "clip_fraction": 0.1, "grad_norm": 1.0, "advantage_mean": 0.0,
         "advantage_std": 1.0},
        T.TrainConfig(), entropy_reference=1.0, free_bytes=10 << 30)
    fired = {g.name for g in guards if g.fired}
    assert fired == {"nan_or_inf", "kl_anchor", "entropy", "first_ratio_dev"}
    assert len(guards) == 5


@needs_checkpoint
def test_a_nan_in_the_statistics_is_caught():
    guards = {g.name: g for g in T.check_guardrails(
        {"policy_loss": float("inf"), "value_loss": 1.0, "entropy": 1.0,
         "first_ratio_dev": 0.0, "explained_variance": 0.0,
         "clip_fraction": 0.0, "grad_norm": 1.0, "advantage_mean": 0.0,
         "advantage_std": 1.0}, T.TrainConfig(), entropy_reference=1.0,
        free_bytes=10 << 30)}
    assert guards["nan_or_inf"].fired
    assert "policy_loss" in guards["nan_or_inf"].detail


@needs_checkpoint
def test_the_entropy_reference_survives_a_resume(tmp_path):
    """Otherwise a resumed run is judged against a number it never saw."""
    out = tmp_path / "rl1"
    cfg = small_cfg(str(out))
    T.train(cfg._replace(rounds=1), str(out), ppo_cfg=SMALL,
            init_path=CHECKPOINT, log=quiet, nice=0, max_rounds=1)
    first = T._read_blob(os.path.join(str(out), "latest.pt")) \
        if hasattr(T, "_read_blob") else torch.load(
            os.path.join(str(out), "latest.pt"), map_location="cpu",
            weights_only=False)
    ref = first["entropy_reference"]
    assert ref is not None
    T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT, log=quiet,
            nice=0, max_rounds=2, resume=True)
    rows = read_log(str(out))
    assert rows[1]["entropy_reference"] == pytest.approx(ref)


# ---------------------------------------------------------- 8d atomic write

def test_an_interrupted_write_leaves_no_broken_latest(tmp_path):
    """The window this exists to close.

    `os.replace` is atomic within a filesystem, so a reader sees the old file or
    the new one. What it cannot protect against is a *temporary* file left
    behind by a process that died mid-write - which is why the temporary lives
    beside the destination and is removed on the way out.
    """
    good = tmp_path / "latest.pt"
    torch.save({"round": 1, "state_dict": {}}, good)
    before = good.read_bytes()

    real_save = torch.save

    def exploding_save(obj, fh, *a, **k):
        fh.write(b"partial garbage")
        fh.flush()
        raise RuntimeError("killed mid-write")

    torch.save = exploding_save
    try:
        with pytest.raises(RuntimeError):
            T.save_checkpoint(str(good), _tiny_net(), round_no=2,
                              seed_cursor=0, entropy_reference=1.0)
    finally:
        torch.save = real_save

    assert good.read_bytes() == before, "latest.pt was modified"
    leftovers = [p for p in os.listdir(str(tmp_path)) if ".tmp." in p]
    assert not leftovers, leftovers


def test_a_good_write_replaces_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "latest.pt"
    torch.save({"round": 1, "state_dict": {}}, target)
    T.save_checkpoint(str(target), _tiny_net(), round_no=2, seed_cursor=0,
                      entropy_reference=1.0)
    blob = torch.load(str(target), map_location="cpu", weights_only=False)
    assert blob["round"] == 2
    assert not [p for p in os.listdir(str(tmp_path)) if ".tmp." in p]


def test_a_temporary_from_another_run_is_not_mistaken_for_a_checkpoint(tmp_path):
    """`find_resume_point` must not resume from a `.tmp` file."""
    d = tmp_path / "rl1"
    d.mkdir()
    torch.save({"round": 9, "state_dict": {}},
               d / "step_000009.pt.tmp.123")
    assert T.find_resume_point(str(d)) == (0, None)


def test_find_resume_point_prefers_the_newest_complete_round(tmp_path):
    """`latest.pt` is written *after* the numbered file, so a crash in between
    leaves a complete round that `latest.pt` does not name. Refusing to resume
    from it would throw away a round of work over a two-line ordering detail."""
    d = tmp_path / "rl1"
    d.mkdir()
    for n in (1, 2, 3):
        torch.save({"round": n, "state_dict": {}},
                   d / ("step_%06d.pt" % n))
    torch.save({"round": 2, "state_dict": {}}, d / "step_000002.pt.tmp.9")
    round_no, path = T.find_resume_point(str(d))
    assert round_no == 3, (round_no, path)
    assert os.path.basename(path) == "step_000003.pt"


def test_find_resume_point_skips_an_unreadable_file(tmp_path):
    d = tmp_path / "rl1"
    d.mkdir()
    with open(d / "latest.pt", "wb") as fh:
        fh.write(b"not a torch file")
    torch.save({"round": 1, "state_dict": {}}, d / "step_000001.pt")
    round_no, path = T.find_resume_point(str(d))
    assert round_no == 1
    assert os.path.basename(path) == "step_000001.pt"


def _tiny_net():
    """A network small enough to save in a unit test."""
    import rl.policy as p
    from rl.imitation import _shape_settings
    from rl.policy import build_checkpoint_model
    net = build_checkpoint_model(channels=4, blocks=1, in_ch=27)
    net.rl_value = p.ValueHead(4)
    net.key = "tiny"
    return net


# ---------------------------------------------------------- 8e the seed cursor

@needs_checkpoint
def test_consecutive_rounds_get_disjoint_seed_blocks_inside_7m(tmp_path):
    out = tmp_path / "rl1"
    cfg = small_cfg(str(out), per_round=10, rounds=3, games=30)
    T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT, log=quiet,
            nice=0, max_rounds=3)
    rows = read_log(str(out))
    blocks = [(r["seed_lo"], r["seed_hi"]) for r in rows]
    assert blocks == [(7_000_000, 7_000_009),
                      (7_000_010, 7_000_019),
                      (7_000_020, 7_000_029)]
    for lo, hi in blocks:
        assert rollout_mod.RL_SEED_BASE <= lo <= hi <= \
            rollout_mod.RL_SEED_BASE + rollout_mod.RL_SEED_SPAN - 1
    for (_alo, ahi), (blo, _bhi) in zip(blocks, blocks[1:]):
        assert ahi < blo, (ahi, blo)


@needs_checkpoint
def test_the_seed_cursor_advances_past_the_last_seed(tmp_path):
    out = tmp_path / "rl1"
    T.train(small_cfg(str(out)), str(out), ppo_cfg=SMALL,
            init_path=CHECKPOINT, log=quiet, nice=0, max_rounds=2)
    blob = torch.load(os.path.join(str(out), "latest.pt"), map_location="cpu",
                      weights_only=False)
    assert blob["seed_cursor"] == 7_000_020
    assert blob["round"] == 2


# ------------------------------------------------------- 8f the evaluation

@needs_checkpoint
def test_the_evaluation_touches_neither_records_json_nor_the_datasets(tmp_path):
    """The property the B0c guard watches, checked where it is easiest to break.

    An evaluation writes 100 episodes through `play_batch`, which opens games,
    builds seats and runs the personalities. The one thing it must not do is leave
    anything behind.
    """
    records = os.path.join(T.PROJECT_DIR, "records.json")
    before_records = (open(records, "rb").read() if os.path.exists(records)
                      else None)

    def snapshot():
        out = {}
        for name in sorted(os.listdir(os.path.join(T.PROJECT_DIR, "data"))):
            if not name.startswith("h") or name == "humanlog":
                continue
            root = os.path.join(T.PROJECT_DIR, "data", name)
            for dirpath, _d, files in os.walk(root):
                for f in files:
                    p = os.path.join(dirpath, f)
                    st = os.stat(p)
                    out[p] = (st.st_size, st.st_mtime_ns)
        return out

    before_data = snapshot()
    out = tmp_path / "rl1"
    cfg = small_cfg(str(out), eval_every=1, eval_games=4)
    T.train(cfg, str(out), ppo_cfg=SMALL, init_path=CHECKPOINT, log=quiet,
            nice=0, max_rounds=1)
    assert snapshot() == before_data
    after = (open(records, "rb").read() if os.path.exists(records) else None)
    assert after == before_records
    assert os.path.exists(os.path.join(str(out), "eval.jsonl"))
    row = json.loads(open(os.path.join(str(out), "eval.jsonl")).readline())
    assert row["current"]["games"] == 4
    assert row["hc_1000"]["games"] == 4
    assert row["seed_base"] == rollout_mod.RL_SEED_VALID_BASE


@needs_checkpoint
def test_the_evaluation_compares_against_hc_1000_on_the_same_seeds(tmp_path):
    """The two rows must be the same experiment, twice."""
    cfg = small_cfg(str(tmp_path / "rl1"), eval_every=1, eval_games=4)
    specs, man = T.eval_specs(cfg)
    assert len(specs) == 4
    assert specs[0].seed == rollout_mod.RL_SEED_VALID_BASE
    # and the same function every time
    again, man2 = T.eval_specs(cfg)
    assert specs == again and man == man2


# --------------------------------------------------------- the CLI itself

def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "never"
    rc = T.main(["--dry-run", "--games", "20000", "--per-round", "500",
                 "--out", str(out)])
    assert rc == 0
    assert not out.exists()
    said = capsys.readouterr().out
    assert "nothing written" in said
    for token in ("rounds", "seeds", "guardrails", "estimate"):
        assert token in said


def test_dry_run_caps_at_max_rounds(tmp_path, capsys):
    T.main(["--dry-run", "--games", "20000", "--per-round", "500",
            "--out", str(tmp_path / "x"), "--max-rounds", "2"])
    said = capsys.readouterr().out
    assert "capped by --max-rounds" in said


def test_the_dry_run_estimate_carries_the_measured_round_costs(tmp_path,
                                                               capsys):
    """Decree 5c: the plan8 constants (10.75 games/s, 6.7 s per update) were
    4-5x off what a production 500-game round actually costs - measured
    2026-10-13, rollout 55-68 s and update 29-33 s. The estimate is the
    number a run gets budgeted by, so it carries the measured midpoints
    (62 s + 31 s = 93 s per round) and still scales linearly with
    `--per-round`.
    """
    T.main(["--dry-run", "--games", "500", "--per-round", "500",
            "--out", str(tmp_path / "x")])
    assert "~93 s per round (62 s rollout + 31 s update)" in \
        capsys.readouterr().out
    T.main(["--dry-run", "--games", "1000", "--per-round", "1000",
            "--out", str(tmp_path / "y")])
    assert "~186 s per round (124 s rollout + 62 s update)" in \
        capsys.readouterr().out


def test_the_cli_refuses_a_missing_checkpoint(tmp_path):
    with pytest.raises(SystemExit) as exc:
        T.main(["--out", str(tmp_path / "x"),
                "--init", str(tmp_path / "nope.pt")])
    assert "starting weights" in str(exc.value)


def test_the_parser_defaults_match_the_plan():
    args = T.build_parser().parse_args([])
    assert args.games == 20_000
    assert args.per_round == 500
    # `--rounds` is gone: it was always overwritten by games // per_round
    # (plan9a item j), so the flag only ever pretended to matter.
    assert not hasattr(args, "rounds")
    assert args.n_procs == 8
    assert args.rollout_device == "cpu"
    assert args.update_device == "cuda"
    assert args.seed_base == 7_000_000
    assert args.eval_every == 5
    assert args.eval_games == 100
    assert args.nice == 19
    assert args.kl_coef == 0.0


def test_rounds_are_derived_from_games_and_per_round():
    cfg, _ = T._config_from_args(T.build_parser().parse_args(
        ["--games", "20000", "--per-round", "500"]))
    assert cfg.rounds == 40
    cfg, _ = T._config_from_args(T.build_parser().parse_args(
        ["--games", "200", "--per-round", "100"]))
    assert cfg.rounds == 2


def test_a_games_total_that_is_not_whole_rounds_is_refused():
    """The derivation used to floor, so `--games 12501` trained 12,500 and
    said nothing. A dropped tail is a different experiment than the one asked
    for, and the place that would drop it is this line, so the line refuses.
    """
    with pytest.raises(SystemExit) as exc:
        T._config_from_args(T.build_parser().parse_args(
            ["--games", "12501", "--per-round", "500"]))
    assert "12501" in str(exc.value) and "remainder 1" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        T._config_from_args(T.build_parser().parse_args(
            ["--games", "20000", "--per-round", "0"]))
    assert "positive" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        T._config_from_args(T.build_parser().parse_args(
            ["--games", "0", "--per-round", "500"]))
    assert "--games must be positive" in str(exc.value)


def test_the_process_renices_itself():
    """Best effort, but it must actually try - and report what it got."""
    before = open("/proc/self/stat").read().split()[18]
    got = T._renice(19)
    after = open("/proc/self/stat").read().split()[18]
    assert got is None or int(got) >= int(before)
    assert int(after) >= 19 or int(before) >= 19


def test_a_resume_plan_is_reported_by_dry_run(tmp_path, capsys):
    """`--dry-run --resume` on an empty directory must not claim to resume."""
    T.main(["--dry-run", "--resume", "--out", str(tmp_path / "x")])
    said = capsys.readouterr().out
    assert "resuming from" not in said
    assert "rounds to run" in said


def test_opponent_pool_is_read_from_the_command_line():
    cfg, _ = T._config_from_args(T.build_parser().parse_args(
        ["--games", "20000", "--per-round", "500",
         "--opponent-pool", "chess, fox ,wolf"]))
    assert cfg.opponent_pool == ("chess", "fox", "wolf")
    cfg, _ = T._config_from_args(T.build_parser().parse_args(
        ["--games", "20000", "--per-round", "500"]))
    assert cfg.opponent_pool is None


def test_a_bad_pool_is_refused_before_anything_is_opened(tmp_path):
    """plan10 §2's "unknown key errors", at the earliest point that sees it.

    Checked in `train()` rather than in `make_specs` because `--dry-run`
    returns before the loop: a pool validated only there would print a full
    plan and exit 0 for a typo, and a real run would load the resume
    checkpoint and write `current.pt` before finding out.
    """
    out = str(tmp_path / "never-created")
    cfg = small_cfg(out, opponent_pool=("chess", "no_such_seat"))
    with pytest.raises(ValueError) as exc:
        T.train(cfg, out, ppo_cfg=SMALL, init_path="does/not/exist.pt",
                dry_run=True, log=quiet, nice=0)
    assert "no_such_seat" in str(exc.value)
    assert not os.path.exists(out), "a refused run created its output dir"