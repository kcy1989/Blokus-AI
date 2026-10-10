"""This layer answers one question: how does plan10 actually run one step of
four-learner training, and what does it write down afterwards?

One step is two environments and one update per learner:

    save -> same-table x240 (one game, four episodes)
         -> fixed pool  x260, once per learner (1040 episodes)
         -> 500 episodes per learner -> ppo_update -> save -> guard -> log

The two environments are kept apart end to end. A same-table game is played
once and produces four `Episode`s - one per network seat - because all four
learners are in that game and none of them may be dropped; a fixed-pool game is
played four times, once per learner, on the **same** seed, so the four copies
see the same opponents, the same colours and the same opening player and the
difference between them is the policy. plan10's "每步學習者軌跡 240 x 4 + 260 x
4 = 2000" is exactly that bookkeeping.

Five decisions this module owns:

**The learners keep their registered 20k keys.** `rl_h1000_20k` and friends are
roster seats, so `seat_for` can stand a training checkpoint in under its own
name and `Game.setup_seats` can build the rest of the table from the registry.
The fallback `seat_for` used to reach for - `rl_h1000_0k` - left the roster in
plan9 task 4, so a run that did not name its learner would now fail at the
first game rather than train.

**The frozen 20k is never a learner's weights.** The learner is loaded from
`current.pt` by `load_policy` and swapped into its seat after setup; the
frozen copy is a separate module built by `load_brain_at` inside
`setup_seats`. They share nothing but the file they were both read from, and
`sha256` of that file is re-checked every step so a run cannot go on while its
opponents are being rewritten underneath it (plan10 risk 7).

**The merged reward is a *reported* number, not a loss weight.** plan10's "兩種
局各算平均獎勵,相加後除以 2" is the curve's aggregation; the PPO batch is the
plain concatenation of the learner's 240 + 260 episodes, and no episode is
reweighted. Changing the loss would have needed a second decision the plan does
not make.

**Numbered checkpoints every 20 steps, `latest.pt` every step.** That is
plan10's "存檔每 10k 局", and `latest.pt` is what a resume actually reads, so an
interrupted run loses at most one step of granularity for the resume path and
nothing at all for the published save points.

**The progress curve is a fixed-seed evaluation, not the training curve.**
`evaluate` replays one immutable opponent schedule from the validation block
against each learner's current weights and against that learner's own frozen
20k, on the same seeds - a paired difference, so its standard error is the
noise in the comparison rather than in either policy. plan10 risk 1 says the
on-policy curves cannot be read as progress; this one can.
"""
import argparse
import hashlib
import os
import sys
import time
from typing import NamedTuple

from rl import multi
from rl import ppo as ppo_mod
from rl import rl_train as rt
from rl import rollout as rollout_mod

PROJECT_DIR = rt.PROJECT_DIR
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "data", "multi")
DEFAULT_EVIDENCE_DIR = os.path.join(PROJECT_DIR, "eval", "rl-multiple-train")
NAME = "rl_multiple"

NICE = 19

# The command this process was started with, recorded in every milestone so a
# published checkpoint can be reproduced without guessing at flags. Captured in
# `main`, which is the only place a run is started from a command line.
COMMAND = None


class MultiConfig(NamedTuple):
    """The run's shape. Step numbers are **global**: 100 steps is 50k games."""

    start_step: int = 40         # the 20k every learner already is
    end_step: int = 100          # plan10's first milestone
    table_games: int = 240
    fixed_games: int = 260
    n_procs: int = 8
    rollout_device: str = "cpu"
    update_device: str = "cuda"
    rollout_mode: str = "softmax"
    eval_every: int = 10
    eval_games: int = 100
    milestone_every: int = 20
    # Snapshots of the three run logs, written once per milestone into a
    # directory named after the step. Never overwritten: an evidence file that
    # a later step could rewrite is not evidence.
    evidence_dir: str = DEFAULT_EVIDENCE_DIR
    # guardrail thresholds, the same four `rl.rl_train` uses
    max_kl_anchor: float = 0.05
    min_entropy_fraction: float = 0.5
    max_first_ratio_dev: float = 1e-6
    min_free_gb: float = 5.0


def learner_dir(out_dir, learner):
    return os.path.join(out_dir, learner)


def checkpoint_paths(out_dir, learner):
    d = learner_dir(out_dir, learner)
    return {"latest": os.path.join(d, "latest.pt"),
            "current": os.path.join(d, "current.pt")}


def run_paths(out_dir):
    """The run-level files. One log for four learners, not four logs for one."""
    return {"log": os.path.join(out_dir, "rounds.jsonl"),
            "eval": os.path.join(out_dir, "eval.jsonl"),
            "milestones": os.path.join(out_dir, "milestones.jsonl")}


def frozen_paths():
    """The four frozen 20k files, in learner order, through the registry."""
    import seats
    return [seats.rl_checkpoint(k) for k in multi.LEARNER_KEYS]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes():
    return {k: sha256_of(p) for k, p in zip(multi.LEARNER_KEYS,
                                            frozen_paths())}


def _learner_nets(cfg, out_dir, resume):
    """`(nets, blobs)` per learner, each loaded from its own directory."""
    from rl.policy import load_policy
    import torch
    nets, blobs, starts = [], [], []
    for i, learner in enumerate(multi.LEARNERS):
        paths = checkpoint_paths(out_dir, learner)
        resume_from = None
        if resume:
            _round, resume_from = rt.find_resume_point(learner_dir(out_dir,
                                                                  learner))
        if resume_from:
            net, blob = rt.load_checkpoint(resume_from, device=cfg.update_device)
        else:
            net, _meta = load_policy(frozen_paths()[i], device=cfg.update_device)
            blob = {}
        # The registered 20k key, so `seat_for` stands the checkpoint in under
        # a name the roster knows. See the module docstring.
        net.key = multi.LEARNER_KEYS[i]
        nets.append(net)
        blobs.append(blob)
        starts.append(int(blob.get("round") or 0))
    if resume and len(set(starts)) > 1:
        raise RuntimeError("the four learners are at different steps: %r"
                           % dict(zip(multi.LEARNERS, starts)))
    return nets, blobs, (starts[0] if resume else cfg.start_step)


def _make_optimizers(nets, ppo_cfg):
    return [rt._make_optimizer(n, ppo_cfg) for n in nets]


def _restore_optimizers(nets, blobs, optimizers, log):
    for i, (net, blob, opt) in enumerate(zip(nets, blobs, optimizers)):
        state = (blob or {}).get("optimizer")
        if not state:
            continue
        try:
            opt.load_state_dict(state)
        except (ValueError, KeyError) as exc:
            log("WARNING: %s could not restore its optimiser state (%s); this "
                "run will not match an uninterrupted one"
                % (multi.LEARNERS[i], exc))


def _entropy_refs(blobs):
    return [(b or {}).get("entropy_reference") for b in blobs]


def _round_rng(cfg, step, i):
    """Learner `i`'s minibatch order for `step`, a function of both alone."""
    import random
    return random.Random(cfg.start_step * 1_000_000 + step * 10 + i)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _episode_rewards(episodes):
    return [e.rewards[e.learner_seat] for e in episodes]


def _curve(table_eps, fixed_eps):
    """plan10's merged reward: the mean of each half, added, halved."""
    t = _mean(_episode_rewards(table_eps))
    f = _mean(_episode_rewards(fixed_eps))
    merged = None if t is None or f is None else (t + f) / 2.0
    return {"table": t, "fixed": f, "merged": merged}


def evaluate(cfg, out_dir, n_procs=None):
    """`(step-independent) paired evaluation of every learner on one schedule.

    Both rows are `argmax` on the same specs from the validation block: the
    current weights, and the frozen 20k the learner started this run from. The
    paired difference is what the curve reports, with its standard error - the
    two rows share every draw, so the noise left is the policies'.
    """
    specs = multi.make_eval_specs(cfg.eval_games)
    n = cfg.n_procs if n_procs is None else n_procs
    # Each learner's current weights have to be handed to a worker through a
    # file, exactly as the rollout does; `current.pt` is that file.
    rows = {}
    for i, learner in enumerate(multi.LEARNERS):
        cur = rollout_mod.play_batch(
            specs, checkpoint_paths(out_dir, learner)["current"],
            n_procs=min(n, 4), device=cfg.rollout_device, mode="argmax",
            opponent_pool=multi.FIXED_POOL, learner_key=multi.LEARNER_KEYS[i])
        fro = rollout_mod.play_batch(
            specs, frozen_paths()[i], n_procs=min(n, 4),
            device=cfg.rollout_device, mode="argmax",
            opponent_pool=multi.FIXED_POOL, learner_key=multi.LEARNER_KEYS[i])
        c = _episode_rewards(cur)
        f = _episode_rewards(fro)
        diffs = [a - b for a, b in zip(c, f)]
        mean = _mean(diffs)
        if len(diffs) > 1:
            var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            se = (var / len(diffs)) ** 0.5
        else:
            se = None
        rows[learner] = {
            "games": len(diffs),
            "current_reward": _mean(c),
            "frozen_reward": _mean(f),
            "paired_diff": mean,
            "paired_se": se,
            "current_remaining": _mean([e.remaining[e.learner_seat]
                                        for e in cur]),
            "frozen_remaining": _mean([e.remaining[e.learner_seat]
                                       for e in fro]),
            "current_rank": _mean([e.ranks[e.learner_seat] for e in cur]),
            "frozen_rank": _mean([e.ranks[e.learner_seat] for e in fro]),
        }
    return {"eval_games": cfg.eval_games, "seed_base": multi.EVAL_SEED_BASE,
            "pool": list(multi.FIXED_POOL), "mode": "argmax",
            "learners": rows}


def _is_milestone(step, cfg):
    return step % cfg.milestone_every == 0 or step == cfg.end_step


def _milestone_row(step, cfg, out_dir, log):
    import torch
    row = {"step": int(step),
           # 500 games a learner a step, and step 100 is 50k - the number the
           # milestone is named after. Counted from the plan's own step
           # numbering, not from how many steps this process happened to run.
           "games": int(step) * (cfg.table_games + cfg.fixed_games),
           "command": COMMAND,
           # The progress curve's replay recipe: same specs, same pool, same
           # argmax, same frozen 20k as the paired baseline. The specs are a
           # function of the validation block, so this is the whole recipe.
           "eval_replay": ("from rl import multi, multi_train; "
                           "print(multi_train.evaluate(multi_train.MultiConfig("
                           "eval_games=%d), %r))"
                           % (cfg.eval_games, out_dir)),
           "files": {}}
    for i, learner in enumerate(multi.LEARNERS):
        path = os.path.join(learner_dir(out_dir, learner),
                            "step_%06d.pt" % step)
        row["files"][learner] = {"path": os.path.relpath(path, PROJECT_DIR),
                                 "sha256": sha256_of(path),
                                 "key": multi.LEARNER_KEYS[i]}
        blob = torch.load(path, map_location="cpu", weights_only=False)
        row["files"][learner]["step"] = int(blob.get("step") or 0)
    log("milestone at step %d: %s"
        % (step, {k: v["sha256"][:12] for k, v in row["files"].items()}))
    return row


def train(cfg=MultiConfig(), out_dir=None, *, ppo_cfg=None, resume=False,
          max_steps=None, dry_run=False, nice=NICE, log=print):
    """Run the loop. Returns a `MultiTraining`.

    `dry_run` prints the plan and touches nothing. `max_steps` caps the run for
    a smoke test; without it the run goes from the resume point to `end_step`.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    ppo_cfg = ppo_cfg or ppo_mod.Config()
    start_step = cfg.start_step
    if resume:
        probe = [rt.find_resume_point(learner_dir(out_dir, l))
                 for l in multi.LEARNERS]
        rounds = [r for r, _p in probe]
        if len(set(rounds)) > 1:
            raise RuntimeError("the four learners are at different steps: %r"
                               % dict(zip(multi.LEARNERS, rounds)))
        start_step = rounds[0]
    end_step = cfg.end_step if max_steps is None else min(
        cfg.end_step, start_step + int(max_steps))

    if dry_run:
        return _dry_run(cfg, ppo_cfg, out_dir, start_step, end_step, log)

    if cfg.table_games % len(multi.PERMS):
        raise ValueError("table_games must be a whole number of seat orders "
                         "(%d), got %d" % (len(multi.PERMS), cfg.table_games))
    table_rounds = cfg.table_games // len(multi.PERMS)
    rt._renice(nice)
    baseline = frozen_hashes()
    for i, learner in enumerate(multi.LEARNERS):
        os.makedirs(learner_dir(out_dir, learner), exist_ok=True)
    nets, blobs, _ = _learner_nets(cfg, out_dir, resume)
    optimizers = _make_optimizers(nets, ppo_cfg)
    _restore_optimizers(nets, blobs, optimizers, log)
    entropy_refs = _entropy_refs(blobs)

    last = None
    for step in range(start_step + 1, end_step + 1):
        timings = {}
        try:
            paths = [checkpoint_paths(out_dir, l)["current"]
                     for l in multi.LEARNERS]
            # 1. hand the rollout workers files to read
            t0 = time.perf_counter()
            for i, learner in enumerate(multi.LEARNERS):
                rt.save_checkpoint(paths[i], nets[i], round_no=step - 1,
                                   seed_cursor=multi.table_seed(step, 1),
                                   entropy_reference=entropy_refs[i],
                                   optimizer=optimizers[i], train_cfg=cfg,
                                   ppo_cfg=ppo_cfg,
                                   extra={"name": multi.LEARNER_KEYS[i],
                                          "init_checkpoint": frozen_paths()[i],
                                          "learner": learner})
            timings["reload"] = time.perf_counter() - t0

            # 2a. the same-table games: one game, four episodes
            t0 = time.perf_counter()
            table_specs = multi.make_table_specs(step, rounds=table_rounds)
            # `play_table_batch` keys a row by *learner index*; the rest of
            # this module reads a row by learner name.
            table_rows = [
                {multi.LEARNERS[i]: ep for i, ep in row.items()}
                for row in rollout_mod.play_table_batch(
                    table_specs, paths, n_procs=cfg.n_procs,
                    device=cfg.rollout_device, mode=cfg.rollout_mode)]
            timings["table"] = time.perf_counter() - t0

            # 2b. the fixed-pool games: one game per learner, same seed
            t0 = time.perf_counter()
            fixed_specs, fixed_manifest = multi.make_fixed_specs(
                step, games=cfg.fixed_games)
            # One call, not one per learner: same specs, four networks, one
            # process pool. `play_paired_batch` keeps the four copies on the
            # same seed so the difference between them is the policy.
            fixed_rows = rollout_mod.play_paired_batch(
                fixed_specs, paths, multi.LEARNER_KEYS,
                n_procs=cfg.n_procs, device=cfg.rollout_device,
                mode=cfg.rollout_mode, opponent_pool=multi.FIXED_POOL)
            fixed_eps = {learner: [row[i] for row in fixed_rows]
                         for i, learner in enumerate(multi.LEARNERS)}
            timings["fixed"] = time.perf_counter() - t0

            # 3. one update per learner over that learner's own 500 episodes
            t0 = time.perf_counter()
            stats_by_learner, episodes_by_learner = {}, {}
            for i, learner in enumerate(multi.LEARNERS):
                eps = ([row[learner] for row in table_rows]
                       + fixed_eps[learner])
                episodes_by_learner[learner] = eps
                batch = ppo_mod.batch_from_episodes(eps)
                stats_by_learner[learner] = ppo_mod.ppo_update(
                    nets[i], None, batch, ppo_cfg, device=cfg.update_device,
                    rng=_round_rng(cfg, step, i), optimizer=optimizers[i])
                if entropy_refs[i] is None:
                    entropy_refs[i] = stats_by_learner[learner]["entropy"]
            timings["update"] = time.perf_counter() - t0

            # 4. save. Numbered files every `milestone_every` steps (plan10's
            # "存檔每 10k 局"), `latest.pt` every step so a resume is exact.
            t0 = time.perf_counter()
            numbered = _is_milestone(step, cfg)
            for i, learner in enumerate(multi.LEARNERS):
                paths_i = checkpoint_paths(out_dir, learner)
                kwargs = dict(round_no=step,
                              seed_cursor=multi.fixed_seed(step, 0) + 1,
                              entropy_reference=entropy_refs[i],
                              optimizer=optimizers[i], train_cfg=cfg,
                              ppo_cfg=ppo_cfg,
                              extra={"name": multi.LEARNER_KEYS[i],
                                     "init_checkpoint": frozen_paths()[i],
                                     "learner": learner})
                if numbered:
                    rt.save_checkpoint(
                        os.path.join(learner_dir(out_dir, learner),
                                     "step_%06d.pt" % step),
                        nets[i], **kwargs)
                rt.save_checkpoint(paths_i["latest"], nets[i], **kwargs)
                # `current.pt` is what the rollout workers read at the start of
                # the *next* step and what `evaluate` reads now: left at the
                # pre-update weights, an evaluation after the update would score
                # the policy the step began with.
                rt.save_checkpoint(paths_i["current"], nets[i], **kwargs)
            timings["save"] = time.perf_counter() - t0

            row = _step_log(step, cfg, table_rows, fixed_eps,
                            episodes_by_learner, stats_by_learner, timings,
                            fixed_manifest, ppo_cfg, entropy_refs)
            frozen_now = frozen_hashes()
            row["frozen_sha256_unchanged"] = (frozen_now == baseline)
            if frozen_now != baseline:
                # Plan10 risk 7: the frozen pool is what the whole run is
                # measured against. If it moved, every number after this point
                # is against a different opponent set.
                raise RuntimeError("a frozen 20k file changed during training: "
                                   "%r" % {k: (baseline[k], frozen_now[k])
                                           for k in baseline
                                           if baseline[k] != frozen_now[k]})
            for learner in multi.LEARNERS:
                guards = rt.check_guardrails(
                    stats_by_learner[learner], cfg, entropy_refs[
                        multi.LEARNERS.index(learner)])
                row.setdefault("guardrails", {})[learner] = [
                    {"name": g.name, "fired": g.fired, "detail": g.detail}
                    for g in guards]
            if numbered:
                row["milestone"] = _milestone_row(step, cfg, out_dir, log)
                rt.append_jsonl(run_paths(out_dir)["milestones"],
                                row["milestone"])
            rt.append_jsonl(run_paths(out_dir)["log"], row)
            _summary(row, log)
            last = row

            if cfg.eval_every and step % cfg.eval_every == 0:
                t0 = time.perf_counter()
                ev = evaluate(cfg, out_dir)
                ev["step"] = step
                ev["seconds"] = round(time.perf_counter() - t0, 3)
                rt.append_jsonl(run_paths(out_dir)["eval"], ev)

            if numbered:
                # **After** this step's round row and (if it is one) its eval
                # row. Taken earlier - as the 50k run's snapshots were - a
                # milestone's own numbers lived only in `data/`, which is
                # gitignored: step_000100's rounds stopped at 99 and its eval
                # at 90, so the snapshot for the milestone did not contain the
                # milestone's result. The snapshot is the evidence for *this*
                # step, so it has to be taken once this step is on disk.
                _snapshot_evidence(cfg, out_dir, step, log)

            fired = [g for learner in multi.LEARNERS
                     for g in row["guardrails"][learner] if g["fired"]]
            if fired:
                reason = "; ".join("%s:%s" % (learner, g["name"])
                                   for learner in multi.LEARNERS
                                   for g in row["guardrails"][learner]
                                   if g["fired"])
                log("STOPPED after step %d (saved): %s" % (step, reason))
                return MultiTraining(step, reason, last)
        except KeyboardInterrupt:
            log("interrupted at step %d; the last complete step is saved"
                % step)
            return MultiTraining(step - 1, "interrupted", last)
    return MultiTraining(end_step, "", last)


class MultiTraining(NamedTuple):
    steps_done: int
    stopped_by: str
    last_row: object


def _step_log(step, cfg, table_rows, fixed_eps, episodes_by_learner,
              stats_by_learner, timings, fixed_manifest, ppo_cfg,
              entropy_refs):
    all_table = [row[l] for row in table_rows for l in multi.LEARNERS]
    all_fixed = [e for l in multi.LEARNERS for e in fixed_eps[l]]
    row = {
        "name": NAME,
        "step": int(step),
        "table_games": len(table_rows),
        "fixed_games": len(next(iter(fixed_eps.values()))),
        "timings": {k: round(float(v), 3) for k, v in timings.items()},
        # plan10: the two halves are averaged separately and the two means are
        # added and halved. Reported, never used to weight the PPO loss.
        "curves": _curve(all_table, all_fixed),
        "per_learner": {},
        "fixed_manifest": fixed_manifest,
        "ppo_cfg": dict(ppo_cfg._asdict()),
        "rollout": rollout_mod.thread_report(),
        "table_seed_span": [multi.table_seed(step, 1),
                            multi.table_seed(step, multi.ROUNDS_PER_STEP)],
        "fixed_seed_span": [multi.fixed_seed(step, 0),
                            multi.fixed_seed(step, cfg.fixed_games - 1)],
    }
    for i, learner in enumerate(multi.LEARNERS):
        row["per_learner"][learner] = {
            "episodes": len(episodes_by_learner[learner]),
            "table_episodes": len(table_rows),
            "fixed_episodes": len(fixed_eps[learner]),
            "curves": _curve([r[learner] for r in table_rows],
                             fixed_eps[learner]),
            # `ppo_update` also returns arrays; the log takes the scalars it
            # names, exactly as `rl.rl_train.round_log` does.
            "stats": {k: (float(stats_by_learner[learner][k])
                          if stats_by_learner[learner].get(k) is not None
                          and rt._finite(stats_by_learner[learner].get(k))
                          else None)
                      for k in rt._SCALAR_STATS},
            "entropy_reference": entropy_refs[i],
        }
    return row


def _snapshot_evidence(cfg, out_dir, step, log):
    """Copy the run logs into `eval/rl-multiple-train/step_<n>/`, once.

    A milestone is the moment a set of numbers becomes something anyone else
    can read, so the logs that produced it are frozen there rather than left in
    `data/` where the next step keeps appending to them. Re-running the copy
    for a step that is already there would overwrite the snapshot, so it
    refuses instead - the same rule `match.py --out` follows.
    """
    import shutil
    dst = os.path.join(cfg.evidence_dir, "step_%06d" % step)
    if os.path.exists(dst):
        log("evidence for step %d already exists at %s; not overwriting"
            % (step, dst))
        return dst
    os.makedirs(dst, exist_ok=True)
    for name in ("rounds.jsonl", "eval.jsonl", "milestones.jsonl"):
        src = os.path.join(out_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, name))
    log("evidence for step %d -> %s" % (step, os.path.relpath(dst,
                                                              PROJECT_DIR)))
    return dst


def _summary(row, log):
    curves = row["curves"]
    per = " ".join("%s %+.3f/%+.3f" % (l, row["per_learner"][l]["curves"]["table"]
                                       or 0.0,
                                       row["per_learner"][l]["curves"]["fixed"]
                                       or 0.0)
                   for l in multi.LEARNERS)
    log("step %3d | table %6.1fs fixed %6.1fs upd %5.1fs | "
        "reward table %+.3f fixed %+.3f merged %+.3f | %s"
        % (row["step"], row["timings"].get("table", 0.0),
           row["timings"].get("fixed", 0.0), row["timings"].get("update", 0.0),
           curves["table"] or 0.0, curves["fixed"] or 0.0,
           curves["merged"] or 0.0, per))


def _dry_run(cfg, ppo_cfg, out_dir, start_step, end_step, log):
    n = end_step - start_step
    log("plan (nothing written)")
    log("  output          %s" % out_dir)
    log("  learners        %s" % " ".join(
        "%s=%s" % (l, k) for l, k in zip(multi.LEARNERS, multi.LEARNER_KEYS)))
    log("  init            each learner's own frozen 20k")
    log("  steps           %d..%d  (%d steps x 500 games per learner)"
        % (start_step + 1, end_step, n))
    log("  per step        %d same-table games (10 seeds x %d orders, "
        "4 episodes each) + %d fixed-pool games x 4 learners"
        % (cfg.table_games, len(multi.PERMS), cfg.fixed_games))
    log("  trajectories    %d per step, %d per learner"
        % (cfg.table_games * 4 + cfg.fixed_games * 4,
           cfg.table_games + cfg.fixed_games))
    log("  seeds           table %d..%d  fixed %d..%d  eval %d..%d"
        % (multi.table_seed(start_step + 1, 1),
           multi.table_seed(end_step, multi.ROUNDS_PER_STEP),
           multi.fixed_seed(start_step + 1, 0),
           multi.fixed_seed(end_step, cfg.fixed_games - 1),
           multi.EVAL_SEED_BASE, multi.EVAL_SEED_BASE + cfg.eval_games - 1))
    log("  rollout         device=%s n_procs=%d mode=%s T=1.0"
        % (cfg.rollout_device, cfg.n_procs, cfg.rollout_mode))
    log("  update          device=%s (one optimiser per learner)"
        % cfg.update_device)
    log("  ppo             %s" % dict(ppo_cfg._asdict()))
    log("  eval            every %d steps, %d games on a fixed schedule"
        % (cfg.eval_every, cfg.eval_games))
    log("  milestones      every %d steps -> numbered checkpoint + sha256"
        % cfg.milestone_every)
    # Measured on the single-learner loop (10.75 games/s, plan8-B2) and on
    # 6.7 s for a 3-epoch update over 8,800 rows (plan8-B3). Same-table games
    # cost about the same as a fixed-pool one - four networks instead of three
    # personalities, and a forward pass is cheaper than a candidate
    # enumeration - so one rate is used for both halves.
    games = cfg.table_games + cfg.fixed_games * 4
    roll = games / 10.75
    upd = 4 * 6.7 * ((cfg.table_games + cfg.fixed_games) / 500.0)
    per = roll + upd
    log("  estimate        ~%.0f s per step (%.0f s rollout + %.0f s update),"
        " ~%.1f h for %d steps"
        % (per, roll, upd, per * n / 3600.0, n))
    return MultiTraining(end_step, "dry-run", None)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m rl.multi_train",
        description="plan10: four learners, two environments, one update each")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--start-step", type=int, default=None,
                    help="first step to run; default is the resume point, or "
                         "40 (= the 20k every learner already is)")
    ap.add_argument("--end-step", type=int, default=100,
                    help="last step; 100 is plan10's first milestone (50k)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap the number of steps, for a smoke test")
    ap.add_argument("--n-procs", type=int, default=8)
    ap.add_argument("--rollout-device", default="cpu")
    ap.add_argument("--update-device", default="cuda")
    ap.add_argument("--mode", default="softmax",
                    choices=("softmax", "argmax"))
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--milestone-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="print the plan and write nothing")
    ap.add_argument("--nice", type=int, default=NICE)
    ap.add_argument("--actor-lr", type=float, default=None)
    ap.add_argument("--value-lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--minibatch", type=int, default=None)
    ap.add_argument("--entropy-coef", type=float, default=None)
    return ap


def _config_from_args(args):
    cfg = MultiConfig()
    if args.start_step is not None:
        cfg = cfg._replace(start_step=args.start_step)
    cfg = cfg._replace(end_step=args.end_step, n_procs=args.n_procs,
                       rollout_device=args.rollout_device,
                       update_device=args.update_device,
                       rollout_mode=args.mode, eval_every=args.eval_every,
                       eval_games=args.eval_games,
                       milestone_every=args.milestone_every)
    ppo_cfg = ppo_mod.Config()
    for field, value in (("actor_lr", args.actor_lr),
                         ("value_lr", args.value_lr),
                         ("epochs", args.epochs),
                         ("minibatch", args.minibatch),
                         ("entropy_coef", args.entropy_coef)):
        if value is not None:
            ppo_cfg = ppo_cfg._replace(**{field: value})
    return cfg, ppo_cfg


def main(argv=None):
    global COMMAND
    COMMAND = " ".join([os.path.basename(sys.argv[0])] + list(
        argv if argv is not None else sys.argv[1:]))
    args = build_parser().parse_args(argv)
    cfg, ppo_cfg = _config_from_args(args)
    if args.resume and args.start_step is not None:
        raise SystemExit("--resume and --start-step are two ways of saying "
                         "where to begin; pass one")
    rt._install_signal_handlers()
    result = train(cfg, out_dir=args.out, ppo_cfg=ppo_cfg,
                   resume=args.resume, max_steps=args.max_steps,
                   dry_run=args.dry, nice=args.nice)
    print("done: steps=%d stopped_by=%s" % (result.steps_done,
                                            result.stopped_by or "-"))
    # A dry run and a clean finish are both success; a guardrail stop or an
    # interrupt is not, and the exit code is what a supervisor has to go on.
    return 0 if result.stopped_by in ("", "dry-run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
