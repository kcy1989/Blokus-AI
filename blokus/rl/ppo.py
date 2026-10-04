"""This layer answers one question: given a batch of episodes, what is the PPO
loss, and how do the parameters move under it?

One update, as a function. No training loop, no checkpoint writing, no
schedule, no data collection - the caller brings a net, an anchor net, a batch
and an rng and gets back statistics. A loop that can be written on top of this
without changing it is the point; a loop that has to reach inside it is not.

Three decisions here are worth stating, because each is a place where the
standard-looking version is wrong for this project.

**The old log_probs are recomputed at the start, over the shuffled order, cut
into the update's own minibatches.** Not because recomputing is fashionable -
`plan8.md` fixes it - but because of what it buys. Measured on hc_1000, a
position scored inside a batch of 512 and the same position scored alone differ
by **3.49e-03** in log_prob, which is 0.35% of PPO's ratio before a single
gradient step is taken. The batch is part of the arithmetic.

Cutting the rescore along `order` rather than along the row sequence is what
makes the first minibatch's ratio *exactly* 1. On CUDA, same-shape with different
content is bit-identical - verified, 256 rows matched exactly - so shape alone
would do. **On CPU it is not**: that same comparison comes out one float32 ulp
apart, which is a first-minibatch ratio of 0.9999999. Matching the row *set* as
well as the shape removes the question on both devices.

Rescoring all of it before any step is not an optimisation: done lazily per
minibatch, the second minibatch would be scored by parameters the first one had
already changed.

**The value loss reads `trunk.detach()`.** `rl.policy.ValueHead` zero-initialises
its final layer, so on the very first step the value loss *cannot* reach the
trunk anyway - but "cannot yet" is not "cannot", and after one optimiser step
the weight is non-zero and the trunk would start moving under a loss that has
nothing to do with choosing moves. Detaching makes it structural instead of
incidental. `tests/test_ppo.py::test_the_value_loss_never_moves_the_trunk` moves
the value head off zero first, so it is testing the detach and not the
initialisation.

**The anchor's distribution is a forward pass per minibatch, and only when
`kl_coef != 0`.** KL needs the anchor's probability over *every* legal slot, not
just the chosen action's log_prob, so unlike `old_log_prob` it cannot be
precomputed once and indexed - precomputing it would mean holding `(N, 36400)`
floats, which is 1.28 GB at 8,800 samples. `hc_1000` is frozen for the whole
update, so this is genuinely repeated work; it is paid only when the KL term is
switched on, and it is worth knowing that a rescore already costs 39% of an
update on this machine.

Advantage normalisation is over the **whole batch**, not per minibatch. Per
minibatch would re-centre each minibatch to zero mean, which quietly rescales
the gradient on whichever minibatch happened to have an unusual reward spread,
and it makes the update's meaning depend on how the batch was cut.

Nothing here imports torch at module scope, so `rl` stays importable without a
GPU stack.
"""
import random
from typing import NamedTuple

# The action-space size, imported rather than written out: 91 x 400 is the
# product of `rl.actions`'s orientation table and the board, and a copy of it here
# would be a third thing to keep in step.
from rl.actions import N_ACTIONS

DEFAULT_EPOCHS = 3
DEFAULT_MINIBATCH = 512
DEFAULT_CLIP = 0.1
DEFAULT_ACTOR_LR = 3e-5
DEFAULT_VALUE_LR = 1e-3
DEFAULT_GAMMA = 0.99
DEFAULT_LAM = 0.95
DEFAULT_KL_COEF = 0.0
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.0
DEFAULT_MAX_GRAD_NORM = 0.5


class Config(NamedTuple):
    """Everything a single update can be told to do differently.

    A NamedTuple rather than a dataclass so `cfg._replace(...)` is the cheap way
    to vary one field in a test, and so it is hashable and prints readably in a
    manifest. Defaults are `plan8.md`'s candidate values; they are **candidates**,
    not conclusions, and this module has no opinion about which are right - it
    implements whatever it is handed.
    """

    epochs: int = DEFAULT_EPOCHS
    minibatch: int = DEFAULT_MINIBATCH
    clip: float = DEFAULT_CLIP
    actor_lr: float = DEFAULT_ACTOR_LR
    value_lr: float = DEFAULT_VALUE_LR
    gamma: float = DEFAULT_GAMMA
    lam: float = DEFAULT_LAM
    kl_coef: float = DEFAULT_KL_COEF
    value_coef: float = DEFAULT_VALUE_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    normalize_advantage: bool = True


def compute_gae(rewards, values, gamma=DEFAULT_GAMMA, lam=DEFAULT_LAM):
    """`(advantages, returns)` for one episode's decision sequence.

    `rewards` and `values` have one entry per decision - a pass is not a decision
    and contributes neither. `plan8.md` fixes the terminal reward on the last
    decision and zero in between, so `rewards` is `[0, 0, ..., r]`.

    The value after the final decision is **zero**: the episode is over and there
    is no next position. That is what makes the last advantage `r - V`, and it is
    the whole reason the terminal reward lands where it does.

    With `gamma = lam = 1` the recursion telescopes and `advantage[t]` becomes
    `return[t] - value[t]` exactly. The test asserts that, because it is the one
    property of GAE that can be checked against a definition rather than against
    another implementation of the same recursion.
    """
    import numpy as np
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if r.shape != v.shape:
        raise ValueError("rewards %s and values %s must be the same length"
                         % (r.shape, v.shape))
    n = r.size
    adv = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in range(n - 1, -1, -1):
        next_v = v[t + 1] if t + 1 < n else 0.0
        delta = r[t] + gamma * next_v - v[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv, adv + v


def batch_from_episodes(episodes):
    """Flatten episodes into one training batch.

    Returns a dict of parallel arrays plus the episodes themselves for
    provenance. Positions come from `rl.caches.states_from_rows`, which is the
    same four columns `rl.rollout` wrote and the same ones the imitation shards
    use, so an episode and a shard row are the same object here.

    Each episode is its own time series: GAE never runs across the boundary
    between two games, and `episode_index` says which is which so a caller can
    measure within-episode behaviour.
    """
    import numpy as np
    from rl.caches import states_from_rows

    episodes = list(episodes)
    if not episodes:
        raise ValueError("a batch needs at least one episode")
    rows = {"own_bits": [], "hand_bits": [], "stuck": [], "to_move": []}
    action, old_lp, old_v, n_legal, ply = [], [], [], [], []
    advantage, ret, reward, ep_index = [], [], [], []
    learner_seat, ep_reward, ep_seed = [], [], []
    for i, ep in enumerate(episodes):
        r = ep.state_rows()
        for k in rows:
            rows[k].append(r[k])
        action.extend(int(a) for a in ep.action_index)
        old_lp.extend(float(x) for x in ep.log_prob)
        old_v.extend(float(x) for x in ep.value)
        n_legal.extend(int(x) for x in ep.n_legal)
        ply.extend(int(x) for x in ep.ply)
        ep_reward.append(float(ep.rewards[ep.learner_seat]))
        ep_seed.append(int(ep.seed))
        t = ep.n_decisions
        # the terminal reward on the last decision, zero in between
        rewards = np.zeros(t, dtype=np.float64)
        if t:
            rewards[-1] = float(ep.rewards[ep.learner_seat])
        adv, rtn = compute_gae(rewards, np.asarray(ep.value, dtype=np.float64))
        advantage.extend(adv.tolist())
        ret.extend(rtn.tolist())
        reward.extend(rewards.tolist())
        learner_seat.extend([int(ep.learner_seat)] * t)
        ep_index.extend([i] * t)
    n_rows = len(action)
    return {
        "states": states_from_rows({k: np.concatenate(v) for k, v in rows.items()}),
        "action_index": np.asarray(action, dtype=np.int64),
        "advantage": np.asarray(advantage, dtype=np.float64),
        "return": np.asarray(ret, dtype=np.float64),
        "reward": np.asarray(reward, dtype=np.float64),
        "rollout_log_prob": np.asarray(old_lp, dtype=np.float64),
        "rollout_value": np.asarray(old_v, dtype=np.float64),
        "n_legal": np.asarray(n_legal, dtype=np.int64),
        "ply": np.asarray(ply, dtype=np.int64),
        "episode_index": np.asarray(ep_index, dtype=np.int64),
        "learner_seat": np.asarray(learner_seat, dtype=np.int64),
        "episode_reward": np.asarray(ep_reward, dtype=np.float64),
        "episode_seed": np.asarray(ep_seed, dtype=np.int64),
        "n_episodes": len(episodes),
        "n_rows": n_rows,
    }


def _param_groups(net, cfg):
    """Actor and critic get their own learning rates.

    The actor group is everything except the critic. The checkpoint's **dead**
    `value` head - `Conv2d(64, 4, 1)`, never called by `forward_batched` and
    never trained - is excluded from both: it has no gradient, so it would be
    skipped anyway, and naming it here says so rather than leaving it to be
    noticed.
    """
    critic_ids = {id(p) for p in net.rl_value.parameters()}
    actor, critic = [], []
    for name, p in net.named_parameters():
        if name.startswith("value."):
            continue
        (critic if id(p) in critic_ids else actor).append(p)
    return [{"params": actor, "lr": cfg.actor_lr},
            {"params": critic, "lr": cfg.value_lr}]


def _explained_variance(returns, values):
    """`1 - Var(returns - values) / Var(returns)`.

    1 means the critic predicts the return exactly; 0 means it is no better than
    predicting the mean; negative means it is worse than that. Reported as a
    single number because "is the critic learning" is the question, and four
    numbers would not answer it.
    """
    import numpy as np
    y = np.asarray(returns, dtype=np.float64)
    p = np.asarray(values, dtype=np.float64)
    var_y = float(y.var())
    if var_y <= 0.0:
        return float("nan")
    return float(1.0 - (y - p).var() / var_y)


def clipped_policy_loss(ratio, advantage, clip=DEFAULT_CLIP):
    """`-mean(min(r*A, clip(r, 1+/-eps)*A))` - the PPO surrogate.

    Split out so the clipping can be aimed at directly by a test rather than
    inferred from a gradient norm. Its property that matters: once `r` is
    outside `1 +/- eps` **in the direction that would keep improving the
    objective**, the `min` picks the clipped branch, that branch is constant in
    `r`, and the gradient with respect to `r` is exactly zero. That is the whole
    mechanism by which PPO stops a runaway step.
    """
    import torch
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
    return -torch.min(unclipped, clipped).mean()


def clip_is_active(ratio, advantage, clip=DEFAULT_CLIP):
    """`(mask, )` - True where `min` selected the **clipped** branch.

    Not the same as "the ratio is outside the band": for an unfavourable
    advantage the unclipped branch is the smaller one and the gradient is live
    even outside the band. A test that conflates the two would pass on the
    wrong behaviour.
    """
    import torch
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
    return clipped <= unclipped


def _forward(net, states, device):
    """`(logits, trunk)` for a minibatch, gradients live.

    `rl.policy.forward_batched` is not called here, and the reason is the point
    of the whole file. It returns `net.rl_value(trunk)` with the trunk still
    attached, which is right for inference and wrong for an update: a value loss
    built on that value has a live path to the shared trunk, and the plan
    requires that path to be severed structurally rather than by the critic
    happening to be initialised at zero.

    So the forward is written out - once, in one place - and the value head is
    fed `trunk.detach()`. The **masking is still shared**: `mask_from_states` and
    `apply_legal_mask` come from `rl.imitation`, the same functions
    `forward_batched` and the imitation seat use, so there is still only one
    masking rule in the project.
    """
    import torch
    from rl.caches import features_27
    from rl.imitation import apply_legal_mask, mask_from_states

    mask = mask_from_states(states)
    t = torch.from_numpy(features_27(states))
    if device is not None:
        t = t.to(device)
    trunk = net.trunk(t)
    logits = net.policy(trunk).reshape(len(states), N_ACTIONS).float()
    return apply_legal_mask(logits, mask), trunk


def _rescore(net, states, actions, device, chunk):
    """`(log_probs, values)` under `no_grad`, in chunks of `chunk`.

    The chunk size is the whole reason this is a helper rather than one big call:
    see the module docstring. Same shape in, same shape out, bit-identical.
    """
    import numpy as np
    import torch
    from rl.policy import log_prob_entropy

    n = len(states)
    lp = np.empty(n, dtype=np.float64)
    vv = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            logits, trunk = _forward(net, states[lo:hi], device)
            a, _e = log_prob_entropy(
                logits,
                torch.from_numpy(actions[lo:hi]).to(logits.device))
            v = net.rl_value(trunk.detach())
            lp[lo:hi] = a.detach().cpu().numpy()
            vv[lo:hi] = v.detach().cpu().numpy()
            del logits, trunk
    return lp, vv


def ppo_update(net, anchor_net, batch, cfg=Config(), device=None, rng=None):
    """One PPO update over `batch`, in place. Returns a dict of statistics.

    `net` is modified. `anchor_net` is read-only and is only used when
    `cfg.kl_coef != 0`.

        loss = -min(r*A, clip(r, 1+/-eps)*A) + c_v*(v - return)^2
               - c_e*entropy + c_kl*KL(current || anchor)

    `old_log_prob` and `old_value` are recomputed here at the minibatch's own
    batch size, never taken from the batch. `batch["rollout_*"]` is reported but
    not used as the baseline - see the module docstring.

    `rng` decides the minibatch order and nothing else. Same rng, same batch,
    same thread count gives bit-identical parameters.
    """
    import numpy as np
    import torch
    from rl.policy import kl_to_anchor, log_prob_entropy

    cfg = cfg if isinstance(cfg, Config) else Config(*cfg)
    rng = random.Random(0) if rng is None else rng
    states = list(batch["states"])
    actions = np.asarray(batch["action_index"], dtype=np.int64)
    n = actions.size
    if n == 0:
        raise ValueError("the batch has no decisions in it")
    if cfg.minibatch < 1:
        raise ValueError("minibatch must be at least 1, got %d" % cfg.minibatch)
    if int(batch["advantage"].size) != n or int(batch["return"].size) != n:
        raise ValueError("advantage/return (%d/%d) do not match the %d actions"
                         % (batch["advantage"].size, batch["return"].size, n))

    # ---- the order first, because the baseline has to be cut the same way
    order = list(range(n))
    rng.shuffle(order)

    # ---- the PPO baseline, recomputed on the update device
    #
    # Rescored over `order`'s minibatches, not over the rows in sequence, and
    # that ordering matters by exactly one float32 ulp. The first update
    # minibatch is `order[0:minibatch]`, so if the rescore used rows
    # `0:minibatch` instead it would be a different set of positions - and on
    # CPU, same-shape-different-content is *not* bit-identical the way it is on
    # CUDA. Measured: a first-minibatch ratio of 0.9999999 instead of 1.0.
    #
    # All of it before any parameter moves. Rescoring each minibatch lazily at
    # the top of its own step would be cheaper to reason about and wrong: the
    # second minibatch would then be scored by parameters the first minibatch
    # had already changed, which is not a PPO baseline at all.
    old_lp = np.empty(n, dtype=np.float64)
    old_v = np.empty(n, dtype=np.float64)
    for lo in range(0, n, int(cfg.minibatch)):
        idx = order[lo:lo + int(cfg.minibatch)]
        lp, vv = _rescore(net, [states[i] for i in idx], actions[idx], device,
                          len(idx) or 1)
        old_lp[idx] = lp
        old_v[idx] = vv

    # ---- advantage, centred over the WHOLE batch
    advantage = np.asarray(batch["advantage"], dtype=np.float64).copy()
    if cfg.normalize_advantage:
        sd = float(advantage.std())
        advantage = advantage - advantage.mean()
        if sd > 1e-8:
            advantage = advantage / sd

    dev = None
    if device is not None:
        dev = torch.device(device) if not isinstance(device, torch.device) \
            else device
    t_old_lp = torch.from_numpy(old_lp.astype(np.float32)).to(dev)
    t_ret = torch.from_numpy(
        np.asarray(batch["return"], dtype=np.float32)).to(dev)
    t_adv_all = torch.from_numpy(advantage.astype(np.float32)).to(dev)
    t_actions = torch.from_numpy(actions)

    actor, critic = [], []
    critic_ids = {id(q) for q in net.rl_value.parameters()}
    for name, q in net.named_parameters():
        if name.startswith("value."):        # the checkpoint's dead head
            continue
        (critic if id(q) in critic_ids else actor).append(q)
    optimizer = torch.optim.AdamW([{"params": actor, "lr": cfg.actor_lr},
                                   {"params": critic, "lr": cfg.value_lr}])

    keys = ("policy_loss", "value_loss", "entropy", "approx_kl_old",
            "clip_fraction", "kl_anchor", "grad_norm", "ratio_mean")
    per_epoch = {k: [] for k in keys}
    first_minibatch = None
    for _epoch in range(max(0, int(cfg.epochs))):
        if _epoch:
            rng.shuffle(order)
        agg = {k: [] for k in keys}
        for lo in range(0, n, int(cfg.minibatch)):
            idx = order[lo:lo + int(cfg.minibatch)]
            mb_states = [states[i] for i in idx]
            t_actions_mb = t_actions[idx]

            logits, trunk = _forward(net, mb_states, device)
            lp, entropy = log_prob_entropy(logits, t_actions_mb.to(logits.device))
            a_mb = t_adv_all[idx]
            old_mb = t_old_lp[idx]

            ratio = torch.exp(lp - old_mb.to(logits.device))
            policy_loss = clipped_policy_loss(ratio, a_mb, cfg.clip)
            if first_minibatch is None:
                # the baseline check, kept in the statistics because it is the
                # one number that says whether the rescore's batch size matched
                first_minibatch = (float(ratio.detach().min()),
                                   float(ratio.detach().max()))

            # the critic reads a DETACHED trunk - structural, not incidental
            value_pred = net.rl_value(trunk.detach())
            value_loss = cfg.value_coef * (value_pred - t_ret[idx]).pow(2).mean()

            loss = policy_loss + value_loss - cfg.entropy_coef * entropy.mean()
            kl_mean = None
            if cfg.kl_coef != 0.0:
                if anchor_net is None:
                    raise ValueError("kl_coef is %g but no anchor_net was given"
                                     % cfg.kl_coef)
                # The anchor's whole distribution is needed here, not just the
                # chosen action's log_prob, so it is a forward per minibatch
                # rather than something precomputed once. It is under no_grad
                # and the anchor does not change, so this is repeated work -
                # paid only when kl_coef is non-zero, and it was 39% of an
                # update on this machine.
                with torch.no_grad():
                    a_logits, _a_trunk = _forward(anchor_net, mb_states, device)
                kl = kl_to_anchor(logits, a_logits)
                kl_mean = kl.mean()
                loss = loss + cfg.kl_coef * kl_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                [q for q in net.parameters() if q.grad is not None],
                cfg.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                agg["policy_loss"].append(float(policy_loss))
                agg["value_loss"].append(float(value_loss))
                agg["entropy"].append(float(entropy.mean()))
                agg["approx_kl_old"].append(float((old_mb - lp).mean()))
                agg["clip_fraction"].append(float(
                    ((ratio - 1.0).abs() > cfg.clip).float().mean()))
                agg["grad_norm"].append(gnorm)
                agg["ratio_mean"].append(float(ratio.mean()))
                if kl_mean is not None:
                    agg["kl_anchor"].append(float(kl_mean))
        for k, vals in agg.items():
            per_epoch[k].append(sum(vals) / len(vals) if vals else float("nan"))

    stats = {}
    for k, vals in per_epoch.items():
        stats[k] = vals[-1] if vals else float("nan")
    stats["per_epoch"] = {k: list(v) for k, v in per_epoch.items() if v}
    stats["first_ratio_min"] = first_minibatch[0] if first_minibatch else float("nan")
    stats["first_ratio_max"] = first_minibatch[1] if first_minibatch else float("nan")
    stats["first_ratio_dev"] = (max(abs(first_minibatch[0] - 1.0),
                                    abs(first_minibatch[1] - 1.0))
                                if first_minibatch else float("nan"))
    stats["n_rows"] = n
    stats["n_minibatches"] = -(-n // int(cfg.minibatch))
    stats["n_episodes"] = batch.get("n_episodes")
    stats["old_log_prob"] = old_lp
    stats["old_value"] = old_v
    stats["explained_variance"] = _explained_variance(
        np.asarray(batch["return"], dtype=np.float64), old_v)
    stats["advantage_mean"] = float(advantage.mean())
    stats["advantage_std"] = float(advantage.std())
    stats["rollout_log_prob_max_diff"] = float(np.abs(
        old_lp - np.asarray(batch["rollout_log_prob"],
                            dtype=np.float64)).max()) \
        if "rollout_log_prob" in batch else float("nan")
    stats["rollout_value_max_diff"] = float(np.abs(
        old_v - np.asarray(batch["rollout_value"], dtype=np.float64)).max()) \
        if "rollout_value" in batch else float("nan")
    return stats
