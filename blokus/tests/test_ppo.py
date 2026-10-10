"""The PPO update: does it compute the right loss, and move the right parameters?

The properties here are the ones a training run cannot recover from being wrong
about. A PPO update that is subtly misnormalised does not crash; it just makes
the policy worse in a way that looks like "RL is hard". So each of these pins
something that would otherwise fail quietly:

  * GAE against a hand-computed three-step example, and against its own
    definition at gamma = lambda = 1 where `advantage == return - value`;
  * the first minibatch's ratio is **exactly** 1, which is what the rescore's
    batch size buys, and a zero learning rate moves nothing at all;
  * the value loss cannot reach the trunk - tested with the critic moved off zero
    first, so the test is about the detach and not about the initialisation;
  * the clip really does zero the gradient, and does *not* zero it in the
    direction where the objective would improve;
  * the KL term pulls the policy back toward the anchor.

Every test needs `data/hc2/step_001000.pt` and skips without it. Threads are
pinned to one where bit-identical parameters are asserted: multi-threaded CPU
convolution is not associative, which is the same reason `rl.rollout` pins them.
"""
import copy
import os
import random

import numpy as np
import pytest
import torch

import rl.ppo as P
import rl.rollout as Rr
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

needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="no H-C2 checkpoint at %s" % CHECKPOINT)


@pytest.fixture(scope="module", autouse=True)
def bind_retired_0k():
    """Seat the retired 0k checkpoint for the duration of one test.

    `rl_h1000_0k` left the roster in plan9 task 4, and `--adhoc` is the
    supported way to bring it back for a run - so this makes the same call
    `match.py --adhoc KEY=PATH` makes rather than reaching into `kind_of`.
    Scoped and undone, so the roster every other test sees is the real one.
    """
    if not os.path.exists(CHECKPOINT):
        yield
        return
    seats.register_adhoc(LEARNER_KEY, CHECKPOINT)
    yield
    seats.ADHOC.pop(LEARNER_KEY, None)


@pytest.fixture(autouse=True)
def one_thread():
    """Bit-identical parameters require a fixed thread count."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def checkpoint():
    net, _meta = load_policy(CHECKPOINT)
    net.key = LEARNER_KEY
    return net


@pytest.fixture(scope="module")
def episodes(checkpoint):
    specs, _man = Rr.make_specs(Rr.RL_SEED_BASE, 4, rng_seed=3)
    return Rr.play_batch(specs, CHECKPOINT, n_procs=1, mode="softmax")


@pytest.fixture(scope="module")
def batch(episodes):
    return P.batch_from_episodes(episodes)


def fresh():
    net, _ = load_policy(CHECKPOINT)
    net.key = LEARNER_KEY
    return net


# ------------------------------------------------------------- 7a GAE

def test_compute_gae_matches_a_hand_computed_three_step():
    """Three decisions, rewards `[0, 0, r]`, values `[v0, v1, v2]`, gamma 0.9,
    lambda 0.8.

    The last advantage is `r - v2` because the value after the last decision is
    zero - the episode is over. Working backwards:
        A2 = r - v2
        A1 = (0 + 0.9*v2 - v1) + 0.72*A2
        A0 = (0 + 0.9*v1 - v0) + 0.72*A1
    """
    rewards = [0.0, 0.0, 1.0]
    values = [0.5, 0.25, -0.25]
    gamma, lam = 0.9, 0.8
    adv, ret = P.compute_gae(rewards, values, gamma, lam)

    a2 = rewards[2] - values[2]
    a1 = (rewards[1] + gamma * values[2] - values[1]) + gamma * lam * a2
    a0 = (rewards[0] + gamma * values[1] - values[0]) + gamma * lam * a1
    expect = np.array([a0, a1, a2])
    assert np.allclose(adv, expect, rtol=0, atol=1e-12)
    assert np.allclose(ret, expect + np.asarray(values), rtol=0, atol=1e-12)


def test_gae_at_gamma_and_lambda_one_is_return_minus_value():
    """The one property of GAE checkable against a definition rather than against
    another implementation of the same recursion."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        n = int(rng.integers(1, 9))
        values = rng.normal(0, 0.5, n)
        rewards = np.zeros(n)
        rewards[-1] = rng.normal()
        adv, ret = P.compute_gae(rewards, values, gamma=1.0, lam=1.0)
        # returns[t] is the sum of rewards from t onwards, so with the reward on
        # the last step it is that reward at *every* t - not the reward vector.
        future = np.cumsum(rewards[::-1])[::-1]
        assert np.allclose(ret, future, rtol=0, atol=1e-12)
        assert np.allclose(adv, ret - values, rtol=0, atol=1e-12)


def test_gae_rejects_mismatched_lengths():
    with pytest.raises(ValueError) as exc:
        P.compute_gae([0.0, 1.0], [0.5])
    assert "same length" in str(exc.value)


def test_the_terminal_reward_lands_on_the_last_decision_only(episodes):
    """Every episode contributes exactly one non-zero reward, on its last
    decision - which is what `plan8.md` fixes, and what makes an episode a time
    series rather than a bag."""
    b = P.batch_from_episodes(episodes)
    nonzero = b["reward"][b["reward"] != 0.0]
    assert nonzero.size == len(episodes)
    for i, ep in enumerate(episodes):
        rows = np.flatnonzero(b["episode_index"] == i)
        assert rows.size == ep.n_decisions
        assert b["reward"][rows[-1]] == pytest.approx(
            ep.rewards[ep.learner_seat])
        assert np.all(b["reward"][rows[:-1]] == 0.0)


def test_gae_never_crosses_between_episodes(episodes):
    """Two episodes whose concatenation would give a different answer than
    computing each on its own - which is the bug if the boundary is missed."""
    b = P.batch_from_episodes(episodes)
    joined = np.concatenate([np.asarray(e.log_prob) for e in episodes])
    assert joined.size == b["n_rows"]
    # per-episode first advantage equals that episode's reward minus its first
    # value, since nothing precedes it
    for i, ep in enumerate(episodes):
        rows = np.flatnonzero(b["episode_index"] == i)
        rewards = np.zeros(ep.n_decisions)
        rewards[-1] = ep.rewards[ep.learner_seat]
        adv, _ret = P.compute_gae(rewards, np.asarray(ep.value))
        assert np.allclose(b["advantage"][rows], adv, rtol=0, atol=1e-12)


# ---------------------------------------------------- 7b the zero update

@needs_checkpoint
def test_a_zero_learning_rate_changes_nothing_at_all(episodes, batch):
    """Identity, bit for bit. Every tensor of the state_dict, not a norm."""
    net = fresh()
    before = copy.deepcopy(net.state_dict())
    st = P.ppo_update(net, fresh(), batch,
                      P.Config(epochs=3, minibatch=16,
                               actor_lr=0.0, value_lr=0.0),
                      rng=random.Random(0))
    after = net.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), key


@needs_checkpoint
def test_the_first_minibatch_ratio_is_exactly_one(episodes, batch):
    """What the rescore's batch size buys, and the baseline everything else is
    measured against.

    Exactly 1, not near 1. The same position scored alone and scored inside a
    batch of 512 differ by 3.49e-03 in log_prob on this machine - 0.35% of the
    ratio before a gradient is taken - and a baseline that is already 0.35% away
    from 1 has no clean interpretation.
    """
    net = fresh()
    st = P.ppo_update(net, fresh(), batch,
                      P.Config(epochs=3, minibatch=16),
                      rng=random.Random(0))
    assert st["first_ratio_dev"] == 0.0, (st["first_ratio_min"],
                                           st["first_ratio_max"])
    assert st["first_ratio_min"] == 1.0
    assert st["first_ratio_max"] == 1.0


@needs_checkpoint
def test_the_baseline_is_recomputed_not_taken_from_the_rollout(episodes, batch):
    """And the mismatch with the rollout's own numbers is reported, not hidden.

    `plan8.md` fixes the rollout's log_prob as diagnostic only. The rollout
    scored one state at a time on the rollout's device; the update scores in
    minibatches on the update device. They agree to about 3e-6 here and the
    statistic says so every run, so a device or batching mistake would show up in
    the manifest rather than in a metric three stages later.
    """
    st = P.ppo_update(fresh(), fresh(), batch,
                      P.Config(epochs=1, minibatch=16),
                      rng=random.Random(0))
    assert np.isfinite(st["rollout_log_prob_max_diff"])
    assert st["rollout_log_prob_max_diff"] < 1e-3, \
        st["rollout_log_prob_max_diff"]
    assert np.isfinite(st["rollout_value_max_diff"])


# ------------------------------------------- 7c the value loss stays put

@needs_checkpoint
def test_the_value_loss_moves_only_the_value_head(episodes, batch):
    """The detach, tested with the critic moved off zero first.

    `ValueHead` zero-initialises its last layer, so on a fresh critic the value
    loss cannot reach the trunk anyway (`dL/dpool = W^T dL/dout` is zero while `W`
    is). Testing on a fresh critic would therefore pass whether or not the detach
    exists. This one pushes the value head off zero, so the gradient has somewhere
    to go, and then requires that it does not go.
    """
    net = fresh()
    with torch.no_grad():
        net.rl_value.lin.weight.fill_(0.25)
        net.rl_value.lin.bias.fill_(0.1)
    # sanity: the critic is genuinely off zero now
    with torch.no_grad():
        _logits, trunk = P._forward(net, batch["states"][:4], None)
        assert float(net.rl_value(trunk.detach()).abs().sum()) > 0.0

    before = copy.deepcopy(net.state_dict())
    P.ppo_update(net, fresh(), batch,
                 P.Config(epochs=2, minibatch=16, actor_lr=0.0,
                          value_lr=1e-2),
                 rng=random.Random(0))
    after = net.state_dict()
    moved, still = [], []
    for key in before:
        if torch.equal(before[key], after[key]):
            still.append(key)
        else:
            moved.append(key)
    assert not any(k.startswith("trunk.") for k in moved), \
        "the value loss reached the trunk: %s" % [k for k in moved
                                                 if k.startswith("trunk.")]
    assert not any(k.startswith("policy.") for k in moved), \
        [k for k in moved if k.startswith("policy.")]
    assert any(k.startswith("rl_value.") for k in moved), \
        "the value head did not move at all, so the test proved nothing"
    assert "rl_value.lin.weight" in moved and "rl_value.lin.bias" in moved


# ------------------------------------------------ 7d the favourable direction

@needs_checkpoint
def test_positive_advantage_raises_the_log_prob_of_those_actions(episodes, batch):
    """The sign, checked on the quantity that matters.

    Synthetically all-positive advantages - "every action you took was good" -
    and the update should make those actions more likely. This is the one
    directional property of the loss; everything else is bookkeeping.

    `normalize_advantage` is **off** here, and it has to be. Advantage
    normalisation subtracts the batch mean, so an all-positive batch of *varied*
    advantages becomes roughly half negative, and the update then correctly
    pushes half the actions down - the measured mean change is -0.18 and the
    test would fail while the code was right. Nor can the advantages simply be
    made constant, since centring annihilates a constant. So "all advantages
    positive" and "advantages normalised" are mutually exclusive, and this test
    wants the sign, so it asks for the sign.
    """
    net = fresh()
    # Varied but all positive. A *constant* advantage would be annihilated by
    # `normalize_advantage`, which subtracts the batch mean - the update would
    # correctly do nothing and the test would look like a failure of the sign.
    b = dict(batch)
    b["advantage"] = np.random.default_rng(5).uniform(0.2, 0.8, batch["n_rows"])
    cfg = P.Config(epochs=2, minibatch=16, actor_lr=1e-4, value_lr=0.0,
                   kl_coef=0.0, entropy_coef=0.0, max_grad_norm=0.5,
                   normalize_advantage=False)

    from rl.policy import forward_batched, log_prob_entropy
    with torch.no_grad():
        before_logits, _v = forward_batched(net, b["states"])
        before_lp, _e = log_prob_entropy(
            before_logits,
            torch.from_numpy(b["action_index"]))
    P.ppo_update(net, fresh(), b, cfg, rng=random.Random(0))
    with torch.no_grad():
        after_logits, _v2 = forward_batched(net, b["states"])
        after_lp, _e2 = log_prob_entropy(
            after_logits, torch.from_numpy(b["action_index"]))
    delta = (after_lp - before_lp).numpy()
    print("\n  log_prob change over %d actions: mean %+.5f  min %+.5f"
          % (delta.size, delta.mean(), delta.min()))
    assert delta.mean() > 0.0, delta.mean()
    # Not 100%: the trunk is shared, so a step that raises one action's logit can
    # nudge another's down. Most must rise, and the mean must be clearly positive.
    assert (delta > 0).mean() > 0.7, "%.1f%% of actions got more likely" \
        % (100.0 * (delta > 0).mean())


@needs_checkpoint
def test_negative_advantage_lowers_it(episodes, batch):
    """And the other sign, because a loss that always raises would pass the test
    above if the advantage were ever read as its absolute value."""
    net = fresh()
    b = dict(batch)
    b["advantage"] = -np.random.default_rng(6).uniform(0.2, 0.8, batch["n_rows"])
    cfg = P.Config(epochs=2, minibatch=16, actor_lr=1e-4, value_lr=0.0,
                   kl_coef=0.0, entropy_coef=0.0, max_grad_norm=0.5,
                   normalize_advantage=False)
    from rl.policy import forward_batched, log_prob_entropy
    with torch.no_grad():
        lg0, _ = forward_batched(net, b["states"])
        lp0, _ = log_prob_entropy(lg0, torch.from_numpy(b["action_index"]))
    P.ppo_update(net, fresh(), b, cfg, rng=random.Random(0))
    with torch.no_grad():
        lg1, _ = forward_batched(net, b["states"])
        lp1, _ = log_prob_entropy(lg1, torch.from_numpy(b["action_index"]))
    delta = (lp1 - lp0).numpy()
    assert delta.mean() < 0.0, delta.mean()


# -------------------------------------------------------- 7e the clip

def test_the_clip_zeroes_the_gradient_in_the_favourable_direction():
    """`r = 1.5`, epsilon 0.1, advantage positive: the clipped branch is the
    smaller one, it is constant in `r`, and the gradient is exactly zero.

    That is the mechanism that stops a runaway step, so it is worth asserting
    as an exact zero rather than as a small number.
    """
    r = torch.tensor([1.5], requires_grad=True)
    a = torch.tensor([1.0])
    loss = P.clipped_policy_loss(r, a, clip=0.1)
    assert float(loss) == pytest.approx(-1.1)
    loss.backward()
    assert float(r.grad) == 0.0, float(r.grad)


def test_the_clip_binds_on_the_downward_side_only_against_a_negative_advantage():
    """The mirror of the first case, and the case it is easy to get wrong.

    `r = 0.5` with a **positive** advantage does *not* bind: raising `r` toward 1
    improves the objective, so the unclipped branch is the smaller one and the
    gradient is live at -1. Binding there would stop the policy from recovering
    from an under-step, which is the mirror image of the runaway it is there to
    stop. The bind needs the ratio to be moving *against* the advantage.
    """
    r = torch.tensor([0.5], requires_grad=True)
    P.clipped_policy_loss(r, torch.tensor([1.0]), clip=0.1).backward()
    assert float(r.grad) != 0.0, "a favourable undershoot must stay live"
    assert float(r.grad) == pytest.approx(-1.0)

    # the mirror: r below the band against a *negative* advantage. min picks the
    # clipped branch, -min is +0.9, and the gradient is zero.
    r2 = torch.tensor([0.5], requires_grad=True)
    loss = P.clipped_policy_loss(r2, torch.tensor([-1.0]), clip=0.1)
    assert float(loss) == pytest.approx(0.9)
    loss.backward()
    assert float(r2.grad) == 0.0


def test_the_clip_does_not_zero_the_unfavourable_side():
    """Past the band but pushing the wrong way, the *unclipped* branch is the
    smaller one and the gradient is live.

    This is the case a "clamp and stop" implementation gets wrong: it would freeze
    the policy here too, and the policy could never come back from an overshoot.
    """
    r = torch.tensor([1.5], requires_grad=True)
    a = torch.tensor([-1.0])
    loss = P.clipped_policy_loss(r, a, clip=0.1)
    loss.backward()
    assert float(r.grad) != 0.0
    assert bool(P.clip_is_active(r.detach(), a, 0.1)[0]) is False
    assert bool(P.clip_is_active(torch.tensor([1.5]), torch.tensor([1.0]),
                                0.1)[0]) is True
    # and inside the band the gradient is live in both directions
    r2 = torch.tensor([1.05], requires_grad=True)
    P.clipped_policy_loss(r2, torch.tensor([1.0]), clip=0.1).backward()
    assert float(r2.grad) != 0.0


@needs_checkpoint
def test_clip_fraction_is_reported_and_sensible(episodes, batch):
    """One epoch of a real batch barely clips anything; twenty epochs would clip
    a lot. A clip_fraction that is always zero is a clip that is not working."""
    net = fresh()
    st1 = P.ppo_update(net, fresh(), batch,
                       P.Config(epochs=1, minibatch=16), rng=random.Random(0))
    assert 0.0 <= st1["clip_fraction"] <= 1.0
    net2 = fresh()
    st2 = P.ppo_update(net2, fresh(), batch,
                       P.Config(epochs=20, minibatch=16, actor_lr=1e-4,
                                max_grad_norm=1.0),
                       rng=random.Random(0))
    print("\n  clip_fraction: 1 epoch %.4f -> 20 epochs %.4f"
          % (st1["clip_fraction"], st2["clip_fraction"]))
    assert st2["clip_fraction"] > st1["clip_fraction"]


# ------------------------------------------------------- 7f the KL anchor

@needs_checkpoint
def test_the_kl_term_pulls_the_policy_back_toward_the_anchor(episodes, batch):
    """Same data, same seed, `kl_coef` 0 against 0.05.

    The anchor is a *separate* net, so the policy is free to drift away from it;
    the test is that adding the term reduces the distance it actually drifted.
    """
    common = dict(epochs=4, minibatch=16, actor_lr=3e-4, value_lr=1e-3,
                  entropy_coef=0.0, max_grad_norm=10.0)
    free_net = fresh()
    P.ppo_update(free_net, fresh(), batch, P.Config(kl_coef=0.0, **common),
                 rng=random.Random(1))
    held_net = fresh()
    st = P.ppo_update(held_net, fresh(), batch, P.Config(kl_coef=0.05, **common),
                      rng=random.Random(1))

    from rl.policy import kl_to_anchor, forward_batched
    anchor = fresh()

    def kl_of(net):
        with torch.no_grad():
            lg, _ = forward_batched(net, batch["states"])
            al, _a = forward_batched(anchor, batch["states"])
        return float(kl_to_anchor(lg, al).mean())

    free, held = kl_of(free_net), kl_of(held_net)
    print("\n  KL to anchor after 4 epochs: unconstrained %.6f  "
          "with kl_coef=0.05 %.6f" % (free, held))
    assert held < free, (held, free)
    assert st["kl_anchor"] >= 0.0


def test_a_kl_coefficient_without_an_anchor_is_refused(episodes, batch):
    net = fresh()
    with pytest.raises(ValueError) as exc:
        P.ppo_update(net, None, batch, P.Config(epochs=1, minibatch=16,
                                                kl_coef=0.01),
                     rng=random.Random(0))
    assert "anchor_net" in str(exc.value)


# ----------------------------------------------------- 7g reproducibility

@needs_checkpoint
def test_the_same_rng_and_batch_give_identical_parameters(episodes, batch):
    """Bit for bit, on CPU with the thread count pinned by the fixture."""
    a = fresh()
    b = fresh()
    cfg = P.Config(epochs=3, minibatch=16, kl_coef=0.02, entropy_coef=0.001)
    P.ppo_update(a, fresh(), batch, cfg, rng=random.Random(42))
    P.ppo_update(b, fresh(), batch, cfg, rng=random.Random(42))
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


@needs_checkpoint
def test_a_different_rng_gives_a_different_update(episodes, batch):
    """The control: if the rng did nothing, the previous test would be vacuous."""
    a = fresh()
    b = fresh()
    cfg = P.Config(epochs=3, minibatch=16)
    P.ppo_update(a, fresh(), batch, cfg, rng=random.Random(1))
    P.ppo_update(b, fresh(), batch, cfg, rng=random.Random(2))
    sa, sb = a.state_dict(), b.state_dict()
    assert not all(torch.equal(sa[k], sb[k]) for k in sa)


@needs_checkpoint
def test_an_empty_batch_is_refused(checkpoint):
    empty = {"states": [], "action_index": np.empty(0, dtype=np.int64),
             "advantage": np.empty(0), "return": np.empty(0),
             "reward": np.empty(0), "rollout_log_prob": np.empty(0),
             "rollout_value": np.empty(0)}
    with pytest.raises(ValueError) as exc:
        P.ppo_update(checkpoint, fresh(), empty)
    assert "no decisions" in str(exc.value)


# ------------------------------------------------- 7h end to end, small

@needs_checkpoint
def test_a_small_real_update_moves_parameters_and_stays_finite():
    """20 episodes through `play_batch` and one update, end to end.

    The integration test that the pieces are actually compatible: episodes from
    the rollout, a batch from `batch_from_episodes`, an update from `ppo_update`.
    No NaN, parameters changed, explained variance finite.
    """
    specs, man = Rr.make_specs(Rr.RL_SEED_BASE, 20, rng_seed=99)
    eps = Rr.play_batch(specs, CHECKPOINT, n_procs=4, mode="softmax")
    assert len(eps) == 20
    b = P.batch_from_episodes(eps)
    assert b["n_rows"] == sum(e.n_decisions for e in eps)

    net = fresh()
    before = copy.deepcopy(net.state_dict())
    st = P.ppo_update(net, fresh(), b, P.Config(epochs=3, minibatch=64),
                      rng=random.Random(0))

    changed = [k for k in before
               if not torch.equal(before[k], net.state_dict()[k])]
    assert changed, "nothing moved"
    assert any(k.startswith("policy.") for k in changed)
    assert any(k.startswith("rl_value.") for k in changed)

    for key in ("policy_loss", "value_loss", "entropy", "approx_kl_old",
                "clip_fraction", "grad_norm", "explained_variance"):
        assert np.isfinite(st[key]), (key, st[key])
    assert -2.0 <= st["explained_variance"] <= 1.5, st["explained_variance"]
    # a zero-initialised critic on real data: EV is exactly 0 before it learns
    print("\n  20 episodes -> %d rows; EV %.4f, entropy %.4f, approx KL %.2e"
          % (b["n_rows"], st["explained_variance"], st["entropy"],
             st["approx_kl_old"]))
    print("  episode learner rewards: %s"
          % [round(r, 2) for r in b["episode_reward"][:8]])


@needs_checkpoint
def test_explained_variance_of_a_zero_critic_is_exactly_zero():
    """The baseline reading of the number: a constant predictor is not 'bad', it
    is exactly 0, and anything above that is the critic having learned
    something."""
    returns = np.array([1.0, 2.0, 3.0, 4.0])
    assert P._explained_variance(returns, np.zeros(4)) == 0.0
    assert P._explained_variance(returns, returns) == 1.0
    assert not np.isfinite(P._explained_variance(np.ones(4), np.zeros(4)))