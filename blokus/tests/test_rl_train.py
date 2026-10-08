"""The training loop: does a run resume bit-identically, and do the guardrails
stop it cleanly?

The property that makes a 40-round run worth starting is that round 7 means the
same thing whether the process survived to reach it or was killed and resumed.
So resume is tested for **bit-identical parameters on CPU**, not for "roughly the
same loss" - a resume that drifts would quietly turn one 20,000-episode run into
several shorter ones with overlapping seed blocks, which is the failure this
whole seed-block machinery exists to prevent.

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

import rl.ppo as ppo_mod
import rl.rl_train as T
import rl.rollout as rollout_mod
import seats
from rl.policy import load_policy

STEP = seats.IMITATION_STEPS[0]
# Where step 1000 was *published* to, not where H-C2 trained: plan9a stage 4
# copied it into `ai/checkpoints/rl_h1000_0k/` and untracked the `data/`
# original, so `IMITATION_CHECKPOINT_DIR` would name a file a fresh clone lacks.
CHECKPOINT_DIR = os.path.dirname(
    seats.imitation_checkpoint(seats.imitation_key(STEP)))
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "step_%06d.pt" % STEP)
SMALL = ppo_mod.Config(epochs=1, minibatch=32, kl_coef=0.0)

needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="no H-C2 checkpoint at %s" % CHECKPOINT)


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
                                     "clip_fraction", "explained_variance"}
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
    differing = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not differing, differing[:6]

    la, lb = read_log(str(whole)), read_log(str(split))
    assert len(la) == len(lb) == 2
    assert [r["seed_lo"] for r in la] == [r["seed_lo"] for r in lb]
    assert la[1]["stats"]["policy_loss"] == pytest.approx(
        lb[1]["stats"]["policy_loss"], abs=0.0)


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


def test_the_cli_refuses_a_missing_checkpoint(tmp_path):
    with pytest.raises(SystemExit) as exc:
        T.main(["--out", str(tmp_path / "x"),
                "--init", str(tmp_path / "nope.pt")])
    assert "starting weights" in str(exc.value)


def test_the_parser_defaults_match_the_plan():
    args = T.build_parser().parse_args([])
    assert args.games == 20_000
    assert args.per_round == 500
    assert args.rounds == 40
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