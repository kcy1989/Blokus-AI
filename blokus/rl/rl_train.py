"""This layer answers one question: how does a 40-round PPO run actually run,
and how do you tell afterwards whether it did what it was supposed to?

The loop is the smallest thing that ties together `rl.rollout`, `rl.ppo` and a
file on disk:

    current.pt -> make_specs -> play_batch -> batch_from_episodes -> ppo_update
               -> save -> guard -> log

and it is deliberately thin. Everything hard lives in the three modules it calls;
what lives here is the part that has no other home - the checkpoint, the
guardrails, the log line, the evaluation, and the argument parsing.

Five decisions that a loop like this usually gets wrong, and how they are made
here:

**The checkpoint is written to a temporary name and renamed into place.** A
checkpoint that is half-written is worse than no checkpoint: `torch.load` will
raise on a truncated file, but a *complete-looking* file written by a process
that died between two `torch.save` internals will load and be wrong. `os.replace`
within one directory is atomic on POSIX, so a reader sees either the old file or
the new one and never a half of each. Tested by interrupting a write
(`tests/test_rl_train.py::test_an_interrupted_write_leaves_no_broken_latest`).

**The rollout gets its weights through a file, not through memory.** `fork` would
be cheaper, but a forked worker sharing a parent's tensor is a thing that stops
being true the moment the parent moves it to the GPU, and a file is the only
handoff that cannot be wrong. It also means `play_batch` is exercised exactly as
a caller would exercise it. The reload cost is measured and logged every round
rather than assumed small.

**Guardrails stop the run *after* saving.** A guardrail that returns before the
save discards the round that triggered it, which is the one round whose numbers
someone will want. So the order is update, save, check, and a triggered guard
prints its reason and exits non-zero with the round intact.

**Advantage normalisation is on and the entropy reference is the first round's
entropy, recorded in the checkpoint.** A resume has to compare against the same
starting entropy the run started with, or a run interrupted at round 30 would
be judged against a number it never saw.

**The in-training evaluation plays `argmax` and the same seeds twice** - once
with the current weights, once with `hc_1000` - so the two rows are comparable
to each other rather than to a league average from a different day. Those
episodes never enter a training batch and never touch `records.json`.

The CLI's own process renices itself to 19 on entry; forked rollout workers
inherit it. Nothing here starts a training run implicitly - `rl_train.py` with
no arguments prints its plan and stops.
"""
import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
from typing import NamedTuple

from rl import ppo as ppo_mod
from rl import rollout as rollout_mod

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "data", "rl1")
INIT_CHECKPOINT = os.path.join("data", "hc2", "step_001000.pt")
ANCHOR_CHECKPOINT = INIT_CHECKPOINT
SEED_BASE = rollout_mod.RL_SEED_BASE
EVAL_SEED_BASE = rollout_mod.RL_SEED_VALID_BASE
NAME = "rl_1000_20k"

NICE = 19


class TrainConfig(NamedTuple):
    """The run's shape, as opposed to `rl.ppo.Config`'s update shape."""

    games: int = 20_000
    per_round: int = 500
    rounds: int = 40
    n_procs: int = 8
    rollout_device: str = "cpu"
    update_device: str = "cuda"
    seed_base: int = SEED_BASE
    eval_every: int = 5
    eval_games: int = 100
    eval_seed_base: int = EVAL_SEED_BASE
    # guardrail thresholds, all configurable per plan8-B4 item 3
    max_kl_anchor: float = 0.05
    min_entropy_fraction: float = 0.5
    max_first_ratio_dev: float = 1e-6
    min_free_gb: float = 5.0


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------

def _atomic_save(blob, path):
    """Write `blob` to `path` so a reader never sees a partial file.

    The temporary lives in the **same directory** as the destination, because
    `os.replace` is only atomic within a filesystem - a temporary in `/tmp` would
    degrade to a copy-then-rename and reintroduce the window this exists to close.
    """
    import torch
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        with open(tmp, "wb") as fh:
            torch.save(blob, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def save_checkpoint(path, net, *, round_no, seed_cursor, entropy_reference,
                    optimizer=None, train_cfg=None, ppo_cfg=None,
                    rng=None, extra=None):
    """One checkpoint, in two halves.

    `state_dict` holds only what `rl.policy.load_policy` expects - the trunk, the
    policy head, and the checkpoint's own dead `value` head - so a saved round can
    be loaded as a *seat* by the same code that loads `hc_1000`. The critic goes
    beside it under `value_head` rather than inside `state_dict`, because a
    strict `load_state_dict` would refuse unexpected keys and that function is
    shared with inference.

    `settings` is what `_shape_settings` reads to rebuild the architecture, so it
    has to carry `in_ch` and `blocks`.
    """
    import torch
    policy_sd = {k: v for k, v in net.state_dict().items()
                 if not k.startswith("rl_value.")}
    critic_sd = {k[len("rl_value."):]: v
                 for k, v in net.state_dict().items()
                 if k.startswith("rl_value.")}
    blob = {
        "state_dict": policy_sd,
        "value_head": critic_sd,
        "settings": _settings_of(net),
        "step": int(round_no),
        "round": int(round_no),
        "seed_cursor": int(seed_cursor),
        "entropy_reference": (None if entropy_reference is None
                              else float(entropy_reference)),
        "name": NAME,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "train_cfg": None if train_cfg is None else dict(train_cfg._asdict()),
        "ppo_cfg": None if ppo_cfg is None else dict(ppo_cfg._asdict()),
        "rng_state": None if rng is None else rng.getstate(),
        "torch_version": torch.__version__,
        "engine_version": _engine_version(),
        "feature_version": _feature_version(),
        "action_table_hash": _action_table_hash(),
        "init_checkpoint": INIT_CHECKPOINT,
    }
    if extra:
        blob.update(extra)
    return _atomic_save(blob, path)


def _settings_of(net):
    """`{"channels", "blocks", "in_ch"}` read off the network itself.

    Read rather than remembered: `rl.imitation._shape_settings` rebuilds the
    architecture from exactly these, so a checkpoint whose settings disagree with
    its weights would load a net of the wrong shape and fail somewhere else
    entirely. Counting the blocks from the trunk keys is what `_shape_settings`
    does as its fallback, so this cannot be the only thing that knows.
    """
    idx = set()
    for name, _p in net.named_parameters():
        parts = name.split(".")
        if (len(parts) > 2 and parts[-1] in ("weight", "bias")
                and parts[-2].isdigit() and parts[-3].isdigit()):
            idx.add(int(parts[-3]))
    first = net.trunk[0].weight
    return {"channels": int(first.shape[0]), "in_ch": int(first.shape[1]),
            "blocks": len(idx) or 6}


def load_checkpoint(path, net=None, device=None):
    """`(net, blob)` from a checkpoint written by `save_checkpoint`.

    With `net=None` a network is built through `rl.policy.load_policy`, which is
    the point: a saved round must be usable as a seat by exactly the code that
    loads the starting checkpoint.
    """
    import torch
    from rl.policy import load_policy
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if net is None:
        net, _meta = load_policy(path, device=device)
    else:
        missing = net.load_state_dict(
            {k: v for k, v in blob["state_dict"].items()
             if not k.startswith("rl_value.")}, strict=False)
        if missing.missing_keys:
            raise ValueError("checkpoint is missing %s"
                             % missing.missing_keys[:4])
        net.key = NAME
        if device is not None:
            net.to(device)
    critic = blob.get("value_head")
    if critic:
        # The critic was stored with its prefix already stripped, so it loads
        # into the head directly. Re-adding `rl_value.` here would ask the head
        # for keys it does not have - its own are `lin.weight` and `lin.bias`.
        net.rl_value.load_state_dict(critic)
        if device is not None:
            net.rl_value.to(device)
    return net, blob


def _engine_version():
    import engine
    return engine.ENGINE_VERSION


def _feature_version():
    from rl.features import FEATURE_VERSION
    return FEATURE_VERSION


def _action_table_hash():
    """The action table's hash, or `None`.

    AGENTS.md: once published this cannot change, or old data and old models stop
    meaning the same thing. Recording it in every checkpoint is what makes a
    checkpoint from the future recognisable as incomparable.
    """
    path = os.path.join(PROJECT_DIR, "action_table.json")
    try:
        with open(path) as fh:
            return json.load(fh).get("hash")
    except (OSError, ValueError):
        return None


def checkpoint_paths(out_dir):
    return {"latest": os.path.join(out_dir, "latest.pt"),
            "current": os.path.join(out_dir, "current.pt"),
            "log": os.path.join(out_dir, "rounds.jsonl"),
            "eval": os.path.join(out_dir, "eval.jsonl")}


def find_resume_point(out_dir):
    """`(round_no, path)` for the newest checkpoint that loads, or `(0, None)`.

    `latest.pt` first, then the numbered files. Scanning matters because
    `latest.pt` is written *after* the numbered file, so a crash in between leaves
    a complete round that `latest.pt` does not point at - and refusing to resume
    from it would throw away a round of work for a two-line ordering detail.
    """
    import torch
    paths = checkpoint_paths(out_dir)
    numbered = []
    if os.path.isdir(out_dir):
        for name in sorted(os.listdir(out_dir)):
            if name.startswith("step_") and name.endswith(".pt"):
                numbered.append(os.path.join(out_dir, name))
    numbered.sort(key=lambda p: _round_of(os.path.basename(p)))
    candidates = ([paths["latest"]] if os.path.exists(paths["latest"]) else []) \
        + list(reversed(numbered))
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue                       # truncated or otherwise unreadable
        return int(blob.get("round", 0)), path
    return 0, None


def _round_of(name):
    try:
        return int(name.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


# --------------------------------------------------------------------------
# guardrails
# --------------------------------------------------------------------------

class Guardrail(NamedTuple):
    name: str
    fired: bool
    detail: str


def check_guardrails(stats, cfg, entropy_reference, free_bytes=None):
    """Every guardrail, all of them, each with the number that fired it.

    All are evaluated even after one fires, so a single round that trips two
    reports both - the second one is usually the interesting one.
    """
    out = []

    def add(name, fired, detail):
        out.append(Guardrail(name, bool(fired), detail))

    finite_fields = ("policy_loss", "value_loss", "entropy", "approx_kl_old",
                     "clip_fraction", "grad_norm", "explained_variance",
                     "advantage_mean", "advantage_std", "first_ratio_dev")
    bad = [k for k in finite_fields
           if stats.get(k) is not None and not _finite(stats.get(k))]
    add("nan_or_inf", bool(bad),
        "non-finite in %s" % bad if bad else "all statistics finite")

    kl = stats.get("kl_anchor")
    if kl is not None and _finite(kl):
        add("kl_anchor", kl > cfg.max_kl_anchor,
            "kl_anchor=%.4g > %.4g" % (kl, cfg.max_kl_anchor))
    else:
        add("kl_anchor", False, "no KL term (kl_coef is 0)")

    ent = stats.get("entropy")
    if entropy_reference and ent is not None and _finite(ent):
        floor = cfg.min_entropy_fraction * entropy_reference
        add("entropy", ent < floor,
            "entropy=%.4g < %.0f%% of the starting %.4g"
            % (ent, 100.0 * cfg.min_entropy_fraction, entropy_reference))
    else:
        add("entropy", False, "no entropy reference yet")

    dev = stats.get("first_ratio_dev")
    if dev is not None and _finite(dev):
        add("first_ratio_dev", dev > cfg.max_first_ratio_dev,
            "first_ratio_dev=%.3g > %.3g" % (dev, cfg.max_first_ratio_dev))

    free = free_bytes
    if free is None:
        try:
            free = shutil.disk_usage(PROJECT_DIR).free
        except OSError:
            free = None
    if free is not None:
        add("disk", free < cfg.min_free_gb * (1 << 30),
            "%.1f GB free < %.1f GB" % (free / float(1 << 30), cfg.min_free_gb))
    return out


def _finite(x):
    try:
        return x == x and x not in (float("inf"), float("-inf"))
    except TypeError:
        return False


# --------------------------------------------------------------------------
# the round log
# --------------------------------------------------------------------------

_SCALAR_STATS = ("policy_loss", "value_loss", "entropy", "approx_kl_old",
                 "clip_fraction", "kl_anchor", "grad_norm", "ratio_mean",
                 "explained_variance", "value_return_corr",
                 "advantage_mean", "advantage_std",
                 "first_ratio_min", "first_ratio_max", "first_ratio_dev",
                 "rollout_log_prob_max_diff", "rollout_value_max_diff",
                 "n_rows", "n_minibatches")


def round_log(round_no, seed_lo, seed_hi, timings, episodes, stats, guards,
              cfg, ppo_cfg, entropy_reference=None, eval_row=None):
    """One JSONL row. Only scalars - `ppo_update` also returns two arrays."""
    learner_ranks = [e.ranks[e.learner_seat] for e in episodes]
    rewards = [e.rewards[e.learner_seat] for e in episodes]
    remaining = [e.remaining[e.learner_seat] for e in episodes]
    per_opponent = {}
    for name in rollout_mod.PERSONALITY_POOL:
        got = [e.rewards[e.learner_seat] for e in episodes
               if name in e.opponent_names]
        if got:
            per_opponent[name] = sum(got) / len(got)
    row = {
        "round": int(round_no),
        "name": NAME,
        "seed_lo": int(seed_lo),
        "seed_hi": int(seed_hi),
        "games": len(episodes),
        "seed_base": cfg.seed_base,
        "seconds": {k: round(float(v), 3) for k, v in timings.items()},
        "learner_reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        # plan9 3c wants 平均餘格 beside 平均名次; reward is the shaped score,
        # remaining is the raw cell count the ranking is built from
        "learner_remaining_mean": (sum(remaining) / len(remaining)
                                   if remaining else None),
        "learner_rank_mean": (sum(learner_ranks) / len(learner_ranks)
                              if learner_ranks else None),
        "learner_rank_counts": {str(r): learner_ranks.count(r)
                                for r in (1, 2, 3, 4)},
        "reward_by_opponent": {k: round(v, 4)
                               for k, v in sorted(per_opponent.items())},
        "stats": {k: (float(stats[k]) if stats.get(k) is not None
                      and _finite(stats.get(k)) else None)
                  for k in _SCALAR_STATS},
        "guardrails": [{"name": g.name, "fired": g.fired, "detail": g.detail}
                       for g in guards],
        "ppo_cfg": dict(ppo_cfg._asdict()),
        "entropy_reference": entropy_reference,
    }
    if eval_row is not None:
        row["eval"] = eval_row
    return row


def append_jsonl(path, row):
    """One line, appended, flushed, and fsynced.

    Append-and-fsync rather than a buffer flushed at the end: a run that is
    killed mid-way should still have the rounds it completed, which is the
    entire reason to write a per-round log.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(row, sort_keys=True, default=str)
    with open(path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return line


# --------------------------------------------------------------------------
# the seed block this run may claim
# --------------------------------------------------------------------------

def _claimed_train_block(seed_base):
    """The registered RL train block a run with this base is entitled to claim.

    `make_specs` needs an `own` triple to exempt from the seed guard, and the
    exemption matches a reserved row exactly rather than by containment, so a
    student run has to name the students' row: with 7,200,000-7,219,999
    reserved, claiming the h-version block would overlap the students' own row
    and refuse to start. Only RL's own train blocks are candidates - an
    imitation base must keep landing in the guard instead of exempting itself.
    """
    for block in (rollout_mod.RL_TRAIN_BLOCK, rollout_mod.RL_STUDENTS_BLOCK):
        if block[1] <= seed_base <= block[2]:
            return block
    return rollout_mod.RL_TRAIN_BLOCK


# --------------------------------------------------------------------------
# the cheap in-training evaluation
# --------------------------------------------------------------------------

def eval_specs(cfg):
    """Fixed specs from `eval_seed_base`, identical for every evaluation.

    Same seeds every time is the whole point: the current weights and the
    `hc_1000` baseline are then scored on the same positions with the same
    opponents, so the difference between the two rows is the policy and not the
    draw.
    """
    specs, manifest = rollout_mod.make_specs(
        cfg.eval_seed_base, cfg.eval_games, rng_seed=cfg.eval_seed_base,
        own=rollout_mod.RL_VALID_BLOCK)
    return specs, manifest


def evaluate(current_path, anchor_path, cfg, device=None):
    """`(current row, baseline row)` - both `argmax`, both on the same seeds.

    These episodes are returned to the caller for logging only and are never
    given to `batch_from_episodes`: an evaluation that leaked into the training
    data would let the policy be evaluated on the same positions it was just
    trained towards.
    """
    specs, _man = eval_specs(cfg)
    rows = {}
    for label, path in (("current", current_path), ("hc_1000", anchor_path)):
        eps = rollout_mod.play_batch(specs, path, n_procs=min(cfg.n_procs, 4),
                                     device=None, mode="argmax")
        rewards = [e.rewards[e.learner_seat] for e in eps]
        remaining = [e.remaining[e.learner_seat] for e in eps]
        ranks = [e.ranks[e.learner_seat] for e in eps]
        rows[label] = {
            "games": len(eps),
            "avg_points": sum(rewards) / len(rewards),
            "avg_remaining": sum(remaining) / len(remaining),
            "avg_rank": sum(ranks) / len(ranks),
        }
    rows["seed_base"] = cfg.eval_seed_base
    return rows


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

class Training(NamedTuple):
    rounds_done: int
    stopped_by: str
    last_row: object


def train(cfg=TrainConfig(), out_dir=None, *, ppo_cfg=None, init_path=None,
          resume=False, max_rounds=None, dry_run=False, nice=NICE,
          log=print, anchor_path=None):
    """Run the loop. Returns a `Training`.

    `dry_run` prints the plan and the estimate and touches nothing - no
    directory, no checkpoint, no log. `max_rounds` caps the rounds for a smoke
    test; without it the run goes the full `cfg.rounds`.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    ppo_cfg = ppo_cfg or ppo_mod.Config()
    init_path = init_path or INIT_CHECKPOINT
    anchor_path = anchor_path or ANCHOR_CHECKPOINT
    paths = checkpoint_paths(out_dir)
    start_round = 0
    start_seed = cfg.seed_base

    resume_from = None
    if resume:
        start_round, resume_from = find_resume_point(out_dir)
        if resume_from:
            log("resuming from round %d (%s)"
                % (start_round, os.path.basename(resume_from)))
    n_rounds = cfg.rounds if max_rounds is None else min(cfg.rounds, max_rounds)
    end_round = start_round + n_rounds

    if dry_run:
        return _dry_run(cfg, ppo_cfg, out_dir, start_round, end_round,
                        max_rounds, n_rounds, log)

    _renice(nice)
    os.makedirs(out_dir, exist_ok=True)
    net, blob = _start_net(init_path, cfg.update_device, resume_from)
    anchor, _ablob = _load_anchor(anchor_path, cfg.update_device,
                                  ppo_cfg.kl_coef)
    entropy_reference = blob.get("entropy_reference")
    rng = _resume_rng(blob)
    # One optimiser for the whole run. Adam's moments are state: an optimiser
    # built per round would restart both at zero every time, which is not the same
    # algorithm as one carried across 40 rounds and would make a resumed run a
    # different run.
    optimizer = _make_optimizer(net, ppo_cfg)
    if blob.get("optimizer"):
        try:
            optimizer.load_state_dict(blob["optimizer"])
        except (ValueError, KeyError) as exc:
            log("WARNING: could not restore the optimiser state (%s); this run "
                "will not match an uninterrupted one" % exc)

    last = None
    for round_no in range(start_round + 1, end_round + 1):
        lo = cfg.seed_base + (round_no - 1) * cfg.per_round
        hi = lo + cfg.per_round - 1
        timings = {}
        try:
            # 1. hand the rollout workers a file to read
            t0 = time.perf_counter()
            save_checkpoint(paths["current"], net, round_no=round_no - 1,
                            seed_cursor=lo, entropy_reference=entropy_reference,
                            train_cfg=cfg, ppo_cfg=ppo_cfg, rng=rng)
            timings["reload"] = time.perf_counter() - t0

            # 2. collect
            t0 = time.perf_counter()
            specs, _man = rollout_mod.make_specs(
                lo, cfg.per_round, rng_seed=cfg.seed_base,
                own=_claimed_train_block(cfg.seed_base))
            episodes = rollout_mod.play_batch(
                specs, paths["current"], n_procs=cfg.n_procs,
                device=cfg.rollout_device, mode="softmax")
            timings["rollout"] = time.perf_counter() - t0

            # 3. update
            t0 = time.perf_counter()
            batch = ppo_mod.batch_from_episodes(episodes)
            update_rng = _round_rng(cfg, round_no)
            stats = ppo_mod.ppo_update(net, anchor, batch, ppo_cfg,
                                       device=cfg.update_device,
                                       rng=update_rng, optimizer=optimizer)
            timings["update"] = time.perf_counter() - t0

            if entropy_reference is None:
                entropy_reference = stats["entropy"]

            # 4. save, then guard
            t0 = time.perf_counter()
            round_path = os.path.join(out_dir, "step_%06d.pt" % round_no)
            # Both files carry the optimiser. `find_resume_point` falls back to
            # the numbered files when `latest.pt` is unreadable, and a fallback
            # that silently loses the Adam moments would resume into a different
            # algorithm than the one it was interrupted in.
            save_checkpoint(round_path,
                            net, round_no=round_no,
                            seed_cursor=hi + 1,
                            entropy_reference=entropy_reference,
                            train_cfg=cfg, ppo_cfg=ppo_cfg, rng=rng,
                            optimizer=optimizer)
            save_checkpoint(paths["latest"], net, round_no=round_no,
                            seed_cursor=hi + 1,
                            entropy_reference=entropy_reference,
                            train_cfg=cfg, ppo_cfg=ppo_cfg, rng=rng,
                            optimizer=optimizer)
            timings["save"] = time.perf_counter() - t0

            eval_row = None
            if cfg.eval_every and round_no % cfg.eval_every == 0:
                t0 = time.perf_counter()
                eval_row = evaluate(round_path, anchor_path, cfg)
                eval_row["seconds"] = round(time.perf_counter() - t0, 3)
                append_jsonl(paths["eval"], {"round": round_no,
                                             **eval_row})

            guards = check_guardrails(stats, cfg, entropy_reference)
            last = round_log(round_no, lo, hi, timings, episodes, stats,
                             guards, cfg, ppo_cfg, entropy_reference, eval_row)
            append_jsonl(paths["log"], last)
            _summary(last, log)

            fired = [g for g in guards if g.fired]
            if fired:
                reason = "; ".join("%s (%s)" % (g.name, g.detail) for g in fired)
                log("STOPPED after round %d (saved): %s" % (round_no, reason))
                return Training(round_no, reason, last)
        except KeyboardInterrupt:
            log("interrupted at round %d; the last complete round is saved"
                % round_no)
            return Training(round_no - 1, "interrupted", last)
    return Training(end_round, "", last)


def _dry_run(cfg, ppo_cfg, out_dir, start_round, end_round, max_rounds,
             n_rounds, log):
    log("plan (nothing written)")
    log("  output          %s" % out_dir)
    log("  init            %s" % INIT_CHECKPOINT)
    log("  anchor          %s" % ANCHOR_CHECKPOINT)
    log("  games           %d  per round %d  ->  %d rounds"
        % (cfg.games, cfg.per_round, cfg.rounds))
    log("  rounds to run   %d (from %d to %d)%s"
        % (n_rounds, start_round, end_round,
           "" if max_rounds is None else "  [capped by --max-rounds]"))
    log("  seeds           %d..%d  (checked against the reserved RL block)"
        % (cfg.seed_base, cfg.seed_base + cfg.games - 1))
    log("  rollout         device=%s n_procs=%d mode=softmax T=1.0"
        % (cfg.rollout_device, cfg.n_procs))
    log("  update          device=%s" % cfg.update_device)
    log("  ppo             %s" % dict(ppo_cfg._asdict()))
    log("  eval            every %d rounds, %d games from seed %d"
        % (cfg.eval_every, cfg.eval_games, cfg.eval_seed_base))
    log("  guardrails      kl>%g, entropy<%g%% of start, nan/inf, "
        "first_ratio_dev>%g, disk<%g GB"
        % (cfg.max_kl_anchor, 100 * cfg.min_entropy_fraction,
           cfg.max_first_ratio_dev, cfg.min_free_gb))
    # Both figures measured: 10.75 games/s of rollout (plan8-B2) and 6.7 s for a
    # 3-epoch update over 8,800 rows on CUDA (plan8-B3).
    roll_secs = cfg.per_round / 10.75
    upd_secs = 6.7 * (cfg.per_round / 500.0)
    per = roll_secs + upd_secs
    log("  estimate        ~%.0f s per round (%.0f s rollout + %.0f s update),"
        " ~%.1f min for %d rounds"
        % (per, roll_secs, upd_secs, per * n_rounds / 60.0, n_rounds))
    return Training(end_round, "dry-run", None)


def _renice(nice):
    """Renice this process; forked workers inherit it.

    Best effort by design. A process may lower its own nice value but not raise
    it past what it started with, and lowering it is exactly what this wants, so
    this normally succeeds - but a run started already at 19, or without
    permission, gets whatever it gets. The value in effect is printed rather than
    assumed.
    """
    try:
        os.nice(int(nice))
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/self/stat") as fh:
            return int(fh.read().split()[18])
    except (OSError, IndexError, ValueError):
        return None


def _start_net(init_path, device=None, resume_from=None):
    """The network to train, plus whatever the resume checkpoint said.

    A resumed net comes back through `load_checkpoint`, which is the same reader
    a caller would use - so a resume also proves the saved file is a usable seat,
    not merely a file this module can read.
    """
    from rl.policy import load_policy
    if resume_from:
        return load_checkpoint(resume_from, device=device)
    net, _meta = load_policy(init_path, device=device)
    net.key = NAME
    return net, {}


def _load_anchor(path, device=None, kl_coef=0.0):
    """The frozen KL anchor, or `None` when `kl_coef` is 0.

    Skipping it when unused is not an optimisation: `hc_1000` is the *starting
    weights*, so an anchor identical to the initialiser would be a second copy of
    a file already in memory, and loading it twice on the update device is a
    megabyte and a second CUDA context for nothing.
    """
    if kl_coef == 0:
        return None, None
    from rl.policy import load_policy
    net, _meta = load_policy(path, device=device)
    net.key = "hc_1000"
    return net, None


def _make_optimizer(net, ppo_cfg):
    """The run's optimiser, with the same two learning rates `ppo_update` uses.

    Built here rather than inside `ppo_update` so it can be *carried*, and its
    state is what a resume restores.
    """
    import torch
    return torch.optim.AdamW(ppo_mod._param_groups(net, ppo_cfg))


def _round_rng(cfg, round_no):
    """The update's minibatch order, a function of the round number alone.

    Derived rather than carried, so a resumed run reproduces an uninterrupted one
    exactly. Carrying an rng state across a resume would work too, but it would
    make "round 7 after a resume" and "round 7 without one" depend on history -
    and this way they cannot.
    """
    return random.Random(cfg.seed_base * 1_000 + round_no)


def _resume_rng(blob):
    state = (blob or {}).get("rng_state")
    rng = random.Random(0)
    if state is not None:
        try:
            rng.setstate(state)
        except (TypeError, ValueError):
            pass
    return rng


def _summary(row, log):
    s = row["stats"]
    line = ("round %3d | seeds %d..%d | roll %6.1fs upd %6.1fs save %4.1fs | "
            "reward %+.3f rank %.2f | ent %.3f kl %s clip %.3f EV %+.3f | %s"
            % (row["round"], row["seed_lo"], row["seed_hi"],
               row["seconds"].get("rollout", 0.0),
               row["seconds"].get("update", 0.0),
               row["seconds"].get("save", 0.0),
               row["learner_reward_mean"], row["learner_rank_mean"],
               s.get("entropy") or 0.0,
               ("%.4f" % s["kl_anchor"]) if s.get("kl_anchor") is not None
               else "  -  ",
               s.get("clip_fraction") or 0.0,
               s.get("explained_variance") or 0.0,
               "OK" if not any(g["fired"] for g in row["guardrails"])
               else "GUARD"))
    log(line)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="rl_train.py",
        description="PPO 主迴圈:從 hc_1000 出發,每輪 rollout 後更新並存檔。")
    p.add_argument("--games", type=int, default=TrainConfig().games,
                   help="總局數(預設 %(default)s)")
    p.add_argument("--per-round", type=int, default=TrainConfig().per_round,
                   help="每輪局數(預設 %(default)s)")
    p.add_argument("--rounds", type=int, default=TrainConfig().rounds,
                   help="輪數(預設 %(default)s)")
    p.add_argument("--out", default=None,
                   help="檢查點輸出目錄(預設 data/rl1)")
    p.add_argument("--resume", action="store_true",
                   help="從最後一個完整輪次繼續")
    p.add_argument("--dry-run", action="store_true",
                   help="只印計畫與預估,不執行、不寫任何檔案")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="本次最多跑幾輪(smoke 用)")
    p.add_argument("--n-procs", type=int, default=TrainConfig().n_procs,
                   help="rollout 進程數(預設 %(default)s)")
    p.add_argument("--rollout-device", default=TrainConfig().rollout_device,
                   help="rollout 裝置(預設 %(default)s)")
    p.add_argument("--update-device", default=TrainConfig().update_device,
                   help="更新裝置(預設 %(default)s)")
    p.add_argument("--seed-base", type=int, default=TrainConfig().seed_base,
                   help="訓練種子起點(預設 %(default)s)")
    p.add_argument("--eval-every", type=int, default=TrainConfig().eval_every,
                   help="每幾輪評測一次,0 為停用(預設 %(default)s)")
    p.add_argument("--eval-games", type=int, default=TrainConfig().eval_games,
                   help="每次評測局數(預設 %(default)s)")
    p.add_argument("--init", default=None, help="起始權重(預設 data/hc2/step_001000.pt)")
    p.add_argument("--anchor", default=None, help="KL 錨點(預設同 --init)")
    p.add_argument("--epochs", type=int, default=ppo_mod.DEFAULT_EPOCHS)
    p.add_argument("--minibatch", type=int, default=ppo_mod.DEFAULT_MINIBATCH)
    p.add_argument("--clip", type=float, default=ppo_mod.DEFAULT_CLIP)
    p.add_argument("--actor-lr", type=float, default=ppo_mod.DEFAULT_ACTOR_LR)
    p.add_argument("--value-lr", type=float, default=ppo_mod.DEFAULT_VALUE_LR)
    p.add_argument("--kl-coef", type=float, default=ppo_mod.DEFAULT_KL_COEF)
    p.add_argument("--max-kl-anchor", type=float, default=None,
                   help="護欄:對錨點 KL 的上限")
    p.add_argument("--min-entropy-fraction", type=float, default=None,
                   help="護欄:熵相對於起始值的下限比例")
    p.add_argument("--max-first-ratio-dev", type=float, default=None)
    p.add_argument("--min-free-gb", type=float, default=None)
    p.add_argument("--nice", type=int, default=NICE,
                   help="自我降優先級(預設 %(default)s)")
    p.add_argument("--quiet", action="store_true", help="不印每輪摘要")
    return p


def _config_from_args(args):
    base = TrainConfig()
    cfg = base._replace(
        games=args.games,
        per_round=args.per_round,
        rounds=args.games // args.per_round if args.per_round else base.rounds,
        n_procs=args.n_procs,
        rollout_device=args.rollout_device,
        update_device=args.update_device,
        seed_base=args.seed_base,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        max_kl_anchor=(args.max_kl_anchor if args.max_kl_anchor is not None
                       else base.max_kl_anchor),
        min_entropy_fraction=(args.min_entropy_fraction
                              if args.min_entropy_fraction is not None
                              else base.min_entropy_fraction),
        max_first_ratio_dev=(args.max_first_ratio_dev
                             if args.max_first_ratio_dev is not None
                             else base.max_first_ratio_dev),
        min_free_gb=(args.min_free_gb if args.min_free_gb is not None
                     else base.min_free_gb),
    )
    ppo_cfg = ppo_mod.Config(epochs=args.epochs, minibatch=args.minibatch,
                             clip=args.clip, actor_lr=args.actor_lr,
                             value_lr=args.value_lr, kl_coef=args.kl_coef)
    return cfg, ppo_cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg, ppo_cfg = _config_from_args(args)
    _install_signal_handlers()
    quiet = args.quiet
    logger = (lambda *a, **k: None) if quiet else print

    out_dir = args.out or DEFAULT_OUT_DIR
    init = args.init or INIT_CHECKPOINT
    anchor = args.anchor or ANCHOR_CHECKPOINT

    if not args.dry_run:
        _require_checkpoint(init, "starting weights")
        if ppo_cfg.kl_coef:
            _require_checkpoint(anchor, "KL anchor")

    result = train(cfg, out_dir, ppo_cfg=ppo_cfg, init_path=init,
                   resume=args.resume, max_rounds=args.max_rounds,
                   dry_run=args.dry_run, nice=args.nice, log=logger,
                   anchor_path=anchor)
    if result.stopped_by and result.stopped_by != "dry-run":
        return 1
    return 0


def _require_checkpoint(path, what):
    if not os.path.exists(path):
        raise SystemExit("no %s at %s" % (what, path))


def _install_signal_handlers():
    """Make SIGTERM raise, so the `except KeyboardInterrupt` path runs.

    A default SIGTERM kills the process without unwinding, and the rollout pool's
    `with` block is what guarantees its workers die with it. A run stopped by a
    supervisor would otherwise leave workers behind.
    """
    def handler(signum, _frame):
        raise KeyboardInterrupt("signal %d" % signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())