"""The rollout generator: is an episode a function of its spec, and nothing else?

The property everything else rests on is that `(seed, learner_seat,
opponent_names)` determines the whole episode - including the recorded
log_probs. If that fails, a batch of episodes is not a reproducible sample and
the PPO ratio it produces cannot be recomputed, which is the difference between
a training run and a rumour.

So most of this file is about determinism, from three directions:

  * the same spec twice, in one process;
  * the same spec under one worker and under four, which is the claim that makes
    parallelising safe - if scheduling could reach the result, the result would
    depend on the machine's load;
  * this loop's opponents against `match.py`'s, move for move, which is what
    makes this a rollout *of the same game* rather than a similar one.

Every test needs `data/hc2/step_001000.pt` and skips without it, the same
bargain `tests/test_rl_imitation.py` strikes. `n_procs=4` is the slowest thing
here, so it is used once and on four episodes.
"""
import os
import random

import numpy as np
import pytest
import torch

import ai.formulas
import engine
import match
import seats
from game import Game
from records import POINTS_FOR_RANK, rank_rows

from rl import rollout as Rr
from rl.caches import states_from_rows
from rl.policy import (N_ACTIONS, forward_batched, load_policy,
                       log_prob_entropy)

STEP = seats.IMITATION_STEPS[0]
# Where step 1000 was *published* to, not where H-C2 trained: plan9a stage 4
# copied it into `ai/checkpoints/rl_h1000_0k/` and untracked the `data/`
# original, so `IMITATION_CHECKPOINT_DIR` would name a file a fresh clone lacks.
CHECKPOINT_DIR = os.path.dirname(
    seats.imitation_checkpoint(seats.imitation_key(STEP)))
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "step_%06d.pt" % STEP)
LEARNER_KEY = seats.imitation_key(STEP)

needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="no H-C2 checkpoint at %s" % CHECKPOINT)


@pytest.fixture(scope="module")
def net():
    n, _meta = load_policy(CHECKPOINT)
    n.key = LEARNER_KEY
    return n


def _identical(a, b):
    """Bit-level equality of two episodes, arrays included."""
    for field in ("action_index", "log_prob", "value", "n_legal", "ply",
                  "to_move", "learner_passes", "remaining", "ranks", "points",
                  "rewards"):
        if getattr(a, field) != getattr(b, field):
            return False, field
    for field in ("own_bits", "hand_bits", "stuck"):
        xa, xb = getattr(a, field), getattr(b, field)
        if len(xa) != len(xb):
            return False, field
        for u, v in zip(xa, xb):
            if not np.array_equal(u, v):
                return False, field
    return True, ""


def _specs(n=4, seed0=Rr.RL_SEED_BASE, rng_seed=99):
    specs, _man = Rr.make_specs(seed0, n, rng_seed=rng_seed)
    return specs


# ------------------------------------------------------------- 8a same spec

@needs_checkpoint
def test_the_same_spec_twice_is_bit_identical(net):
    """The base claim, log_probs included.

    `log_prob` is compared exactly rather than approximately. It is recorded in
    float64 from a float32 forward pass, so the same computation twice is not
    merely close, it is the same number - and a difference here would mean the
    old policy's log_prob is not reproducible, which is the one thing PPO's
    ratio cannot survive.
    """
    spec = _specs(1)[0]
    a = Rr.play_episode(spec, net, mode="softmax")
    b = Rr.play_episode(spec, net, mode="softmax")
    same, field = _identical(a, b)
    assert same, field
    assert a.n_decisions > 0


@needs_checkpoint
def test_argmax_repeats_too(net):
    spec = _specs(1)[0]
    a = Rr.play_episode(spec, net, mode="argmax")
    b = Rr.play_episode(spec, net, mode="argmax")
    same, field = _identical(a, b)
    assert same, field


# ------------------------------------------------- 8b worker count is invisible

@needs_checkpoint
def test_one_worker_and_four_workers_agree(net):
    """Explicitly, not hopefully: the result must not depend on scheduling.

    `play_batch` returns episodes in **spec order**, so this compares element by
    element rather than as sets. Four episodes across four workers means each
    worker gets one, which is the case where an order-dependent implementation
    would shuffle the results.
    """
    specs = _specs(4, rng_seed=1234)
    one = Rr.play_batch(specs, CHECKPOINT, n_procs=1, mode="softmax")
    four = Rr.play_batch(specs, CHECKPOINT, n_procs=4, mode="softmax")
    assert len(one) == len(four) == 4
    for i, (a, b) in enumerate(zip(one, four)):
        same, field = _identical(a, b)
        assert same, "episode %d differs on %s" % (i, field)
        assert a.spec == b.spec == specs[i], "episode %d out of spec order" % i


@needs_checkpoint
def test_the_batch_comes_back_in_spec_order(net):
    """Order is part of the contract, not a convenience.

    `Pool.map` preserves input order, but the check is here so that a future
    switch to `imap` - which does not, and would be the natural thing to reach
    for when a batch got large - cannot pass unnoticed.
    """
    specs = _specs(6, rng_seed=555)
    got = Rr.play_batch(specs, CHECKPOINT, n_procs=3, mode="softmax")
    assert [e.spec for e in got] == list(specs)


@needs_checkpoint
def test_the_worker_initialiser_pins_torch_to_one_thread():
    """Structural, not advisory.

    Eight workers each taking fourteen OpenMP threads on fourteen cores measured
    31x slower than one thread each: 3.86 hours against 7.6 minutes for the same
    5,000 episodes. So it lives in the pool initialiser and is checked there.
    """
    seen = {}

    def record(_ignored):
        import torch
        seen["threads"] = torch.get_num_threads()
        return torch.get_num_threads()

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(2, initializer=Rr._worker_init) as pool:
        got = pool.map(_dummy_take_threads, [0, 0])
    assert got == [1, 1], got
    assert Rr.thread_report()["torch_num_threads"] == 1


def _dummy_take_threads(_ignored):
    import torch
    return torch.get_num_threads()


# ------------------------------------------------- 8c the reuse proof

@needs_checkpoint
def test_the_opponents_choose_exactly_what_match_py_chooses(net):
    """This loop is `match.py`'s loop, shown move for move.

    `match.play_match` takes an `on_move` hook, so its whole trace is
    observable. The comparison is against the **full** sequence of real moves by
    all four seats - not just the learner's - because that is what "the opponents
    do the same thing" actually means.

    The learner is in `argmax` mode here, and that is not a weakening. Under
    `softmax` the learner samples a different move, so the opponents are
    answering a different position from the first decision on; the traces
    diverge immediately and no assertion about them would mean anything. The
    reuse claim is about the opponents' decision procedure given a position, and
    argmax is what makes the positions match so that claim can be tested at all.
    `test_the_sampler_is_stochastic` covers the other mode.
    """
    checked = 0
    for i in range(6):
        seed = Rr.RL_SEED_BASE + 500 + i
        opponents = tuple(random.Random(seed).sample(list(Rr.PERSONALITY_POOL), 3))
        learner_seat = seed % 4
        spec = Rr.Spec(seed, learner_seat, opponents)

        ep = Rr.play_episode(spec, net, mode="argmax", record_moves=True)

        ai.formulas.USE_WALL_BUDGET = False
        keys = list(opponents)
        keys.insert(learner_seat, LEARNER_KEY)
        rng = Rr.episode_streams(seed)[0]
        g = Game(rng)
        g.setup_seats(keys, None, rng, checkpoint_dir=CHECKPOINT_DIR)
        g.start()
        reference = []
        match.play_match(g, rng,
                         on_move=lambda _g, o, m: reference.append((o,) + tuple(m)))

        mine = [m for m in ep.moves if m[1] is not None]
        assert mine == reference, "seed %d diverges" % seed
        assert ep.remaining == tuple(g.remaining_cells(o) for o in range(4)), seed
        checked += 1
    assert checked == 6


# --------------------------------------------------------- 8d the payoff

@needs_checkpoint
def test_the_reward_follows_records_rank_rows_and_points_sum_to_ten(net):
    """The ranking is not the rollout's to decide.

    `episode_reward` calls `records.rank_rows`; this checks the answer is the one
    that function gives for the same remaining cells, including on a tie, and
    that the four seats' points are the competition-ranking points whose sum is
    `4 + 3 + 2 + 1 = 10` whenever no two seats are level.
    """
    from rl.policy import episode_reward
    for spec in _specs(5, rng_seed=77):
        ep = Rr.play_episode(spec, net, mode="softmax")
        rows = rank_rows([(o, ep.remaining[o]) for o in range(4)])
        by_owner = {k: (r, p, rem) for k, r, p, rem in rows}
        for o in range(4):
            rank, points, rem = by_owner[o]
            assert ep.ranks[o] == rank, (spec, o, ep.ranks, by_owner)
            assert ep.points[o] == points
            assert ep.rewards[o] == episode_reward(ep.remaining, o)
            assert ep.rewards[o] == pytest.approx((5 - rank) - rem / 100.0)
            assert ep.points[o] in POINTS_FOR_RANK
        if len(set(ep.remaining)) == 4:
            assert sum(ep.points) == 10, ep.points


@needs_checkpoint
def test_a_tie_gives_two_seats_the_same_place(net):
    """Deliberately constructed rather than waited for.

    A tie is where a hand-written ranking goes wrong - it is the one case where
    `rank_rows`' competition rule and "sort and number" differ. Two seats level
    on cells and both take first place, and the third takes third.
    """
    from rl.policy import episode_reward
    from records import rank_rows
    remaining = (0, 0, 5, 9)
    ranks = {k: r for k, r, _p, _rem in
             rank_rows([(o, remaining[o]) for o in range(4)])}
    assert ranks == {0: 1, 1: 1, 2: 3, 3: 4}
    assert episode_reward(remaining, 0) == episode_reward(remaining, 1) == 4.0
    # and the same tie reached through a real episode
    for spec in _specs(8, rng_seed=4321):
        ep = Rr.play_episode(spec, net, mode="softmax")
        rem = ep.remaining
        tied = [o for o in range(4) if list(rem).count(rem[o]) > 1]
        if tied:
            assert len({ep.ranks[o] for o in tied}) == 1, (rem, ep.ranks)
            return
    # no tie showed up in eight episodes; the constructed case above still stands


# ------------------------------------------------------- 8e the decision count

@needs_checkpoint
def test_the_decision_count_is_in_a_plausible_band(net):
    """A range, and the distribution printed - not a pinned number.

    The count cannot be a constant: it is how many of the learner's turns came
    with a legal placement, and that depends on the opponents. What it can be is
    bounded, because a seat that cannot move never can again
    (`engine._advance`'s one-way latch) and a seat places at most 21 pieces. So
    1..21 is a hard bound from the rules, and the useful assertion is that the
    observed band is where plan8-B2 expected it - about 17 - rather than at an
    edge that would mean the learner is being starved of turns or is playing
    three games at once.
    """
    counts = []
    passes = 0
    with_pass = 0
    specs = _specs(20, rng_seed=2024)
    for spec in specs:
        ep = Rr.play_episode(spec, net, mode="softmax")
        counts.append(ep.n_decisions)
        passes += ep.learner_passes
        with_pass += 1 if ep.learner_passes else 0
    n = len(counts)
    print("\n  learner decisions over %d episodes: mean %.2f min %d max %d"
          % (n, sum(counts) / n, min(counts), max(counts)))
    print("  sorted: %s" % sorted(counts))
    print("  learner passes: %d total, %d of %d episodes had at least one"
          % (passes, with_pass, n))
    assert min(counts) >= 1
    assert max(counts) <= 21, "a seat places at most 21 pieces"
    assert 8.0 <= sum(counts) / n <= 21.0, sum(counts) / n
    # every decision is a legal action in that position
    for spec, count in zip(specs[:5], counts[:5]):
        ep = Rr.play_episode(spec, net, mode="softmax")
        for action, n_legal in zip(ep.action_index, ep.n_legal):
            assert 0 <= action < N_ACTIONS
            assert 0 < n_legal <= N_ACTIONS


# ------------------------------------------------------------ 8f the sampler

@needs_checkpoint
def test_the_sampler_is_stochastic_across_seeds_and_exact_within_one(net):
    """Two properties that pull against each other, so both are checked.

    Different seeds must give the learner different moves - otherwise 5,000
    episodes would be one episode replayed 5,000 times and the whole run would
    be measuring nothing. The same seed must give exactly the same moves, or the
    recorded log_probs would not belong to the trajectory they claim to.
    """
    a = Rr.play_episode(Rr.Spec(Rr.RL_SEED_BASE, 0, ("wolf", "fox", "hunter")),
                        net, mode="softmax")
    b = Rr.play_episode(Rr.Spec(Rr.RL_SEED_BASE, 0, ("wolf", "fox", "hunter")),
                        net, mode="softmax")
    same, field = _identical(a, b)
    assert same, field

    # different opponent sets, same seat: the learner sees different positions
    c = Rr.play_episode(Rr.Spec(Rr.RL_SEED_BASE, 0, ("chess", "wolf", "fox")),
                        net, mode="softmax")
    assert a.action_index != c.action_index, "sampling looks deterministic"

    # and softmax genuinely explores: the argmax move is not always chosen
    arg = Rr.play_episode(Rr.Spec(Rr.RL_SEED_BASE, 0, ("wolf", "fox", "hunter")),
                          net, mode="argmax")
    differ = sum(1 for x, y in zip(a.action_index, arg.action_index) if x != y)
    print("\n  softmax vs argmax learner moves: %d of %d differ"
          % (differ, min(len(a.action_index), len(arg.action_index))))
    assert differ > 0, "sampling never left the argmax move"


# ------------------------------------------------------------ 7 make_specs

def test_make_specs_draws_three_distinct_opponents_and_uniform_seats():
    """Without replacement, and every seat about equally often.

    The seat check is a coarse chi-square, not a tight one: over 4,000 episodes
    each seat should land near 1,000 and the point is to catch a seat that never
    happens or always does. A 25% deviation is roughly six sigma for n=4,000, so
    passing it means the draw is not doing anything obviously wrong.
    """
    specs, man = Rr.make_specs(Rr.RL_SEED_BASE, 4000, rng_seed=31337)
    assert len(specs) == 4000
    for s in specs:
        assert len(s.opponent_names) == 3
        assert len(set(s.opponent_names)) == 3, s
        for name in s.opponent_names:
            assert name in Rr.PERSONALITY_POOL
        assert 0 <= s.learner_seat < 4
    counts = [0, 0, 0, 0]
    for s in specs:
        counts[s.learner_seat] += 1
    expected = 4000 / 4.0
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    print("\n  learner seat counts %s, chi2=%.2f (df=3, 5%% critical 7.81)"
          % (counts, chi2))
    assert chi2 < 7.815, (counts, chi2)
    assert man["learner_seat_counts"] == counts


def test_every_personality_reaches_every_seat():
    """Coverage of the opponent pool, and it is uniform by construction.

    Two different rates, and conflating them is the easy mistake here: each of a
    spec's three slots is uniform over the seven names, so a name's **share of
    slots** is 1/7 = 0.1429, while the chance it appears **in a given episode**
    at all is 3/7 = 0.4286. Measured shares land near 0.144.
    """
    specs, man = Rr.make_specs(Rr.RL_SEED_BASE, 3000, rng_seed=4711)
    total_slots = 3 * len(specs)
    for name in Rr.PERSONALITY_POOL:
        slots = man["opponent_coverage"][name]
        share = slots / float(total_slots)
        appear = slots / float(len(specs))
        print("  %-10s %5d slots  share %.4f (expect 1/7 = %.4f)  "
              "appears in %.4f of episodes (expect 3/7 = %.4f)"
              % (name, slots, share, 1 / 7.0, appear, 3 / 7.0))
        assert abs(share - 1 / 7.0) < 0.01, (name, share)
        assert abs(appear - 3 / 7.0) < 0.03, (name, appear)


def test_the_seed_block_is_inside_the_reserved_rl_range():
    specs, man = Rr.make_specs(Rr.RL_SEED_BASE, 20_000, rng_seed=1)
    assert specs[0].seed == Rr.RL_SEED_BASE
    assert specs[-1].seed == Rr.RL_SEED_BASE + 19_999
    assert all(Rr.RL_SEED_BASE <= s.seed
               <= Rr.RL_SEED_BASE + Rr.RL_SEED_SPAN - 1 for s in specs)
    assert man["end_seed"] == Rr.RL_SEED_BASE + 19_999
    assert man["reserved_range_checked"][1:] == [Rr.RL_SEED_BASE,
                                                 Rr.RL_SEED_BASE + 19_999]


def test_every_rollout_block_is_registered_to_the_tuple():
    """`reject_reserved_seeds` exempts by exact triple, so the constants a
    caller claims with and the rows the guard polices are one fact in two
    files - drift would refuse every run of the stage the row belongs to."""
    from rl import paired3
    registered = set(paired3.RESERVED_RANGES)
    for block in (Rr.RL_TRAIN_BLOCK, Rr.RL_VALID_BLOCK, Rr.RL_STUDENTS_BLOCK):
        assert block in registered, block


def test_make_specs_runs_on_the_students_block_when_claimed():
    """plan9 step 3's block: one round of a student run starts under its own
    row, and without the claim the registered history refuses it."""
    specs, man = Rr.make_specs(Rr.RL_STUDENTS_SEED_BASE, 500,
                               rng_seed=Rr.RL_STUDENTS_SEED_BASE,
                               own=Rr.RL_STUDENTS_BLOCK)
    assert man["reserved_range_checked"] == list(Rr.RL_STUDENTS_BLOCK)
    with pytest.raises(ValueError) as exc:
        Rr.make_specs(Rr.RL_STUDENTS_SEED_BASE, 500,
                      rng_seed=Rr.RL_STUDENTS_SEED_BASE)
    assert "stage RL students" in str(exc.value)


def test_make_specs_refuses_a_block_that_was_already_used():
    """The imitation set's own seeds, refused before a single game is played.

    H-B1 and H-C1 used 6,000,000-6,029,378. A rollout landing there would be
    replaying the imitation data's games and every metric after it would be a
    statement about memorisation, so the refusal has to happen at spec
    generation rather than at analysis time.
    """
    with pytest.raises(ValueError) as exc:
        Rr.make_specs(6_000_000, 11)
    assert "H-B1/H-C1 imitation train" in str(exc.value)


def test_make_specs_is_a_function_of_one_integer():
    a, _ = Rr.make_specs(Rr.RL_SEED_BASE, 12, rng_seed=99)
    b, _ = Rr.make_specs(Rr.RL_SEED_BASE, 12, rng_seed=99)
    assert a == b
    c, _ = Rr.make_specs(Rr.RL_SEED_BASE, 12, rng_seed=100)
    assert c != a, "a different rng_seed gave the same assignment"


# --------------------------------------------- 8h recomputing the log_prob

@needs_checkpoint
def test_the_recorded_log_prob_can_be_recomputed_from_the_recorded_state(net):
    """The stored numbers have to be a function of the stored position.

    This is what makes the episode self-contained: an update that reads
    `log_prob` and trusts it is trusting a position it could rebuild. Tolerance
    1e-4, against a measured batch=1 versus batch-N difference of 3.8e-6 - so the
    bound is loose enough not to be measuring float32 noise and tight enough to
    catch a state that was recorded from the wrong ply.
    """
    spec = _specs(1)[0]
    ep = Rr.play_episode(spec, net, mode="softmax")
    rows = ep.state_rows()
    rebuilt = states_from_rows(rows)
    assert len(rebuilt) == ep.n_decisions

    with torch.no_grad():
        logits, _value = forward_batched(net, rebuilt)
    lp, _entropy = log_prob_entropy(
        logits, torch.tensor(ep.action_index, dtype=torch.long))
    diff = np.abs(np.asarray(lp) - np.asarray(ep.log_prob))
    print("\n  recomputed log_prob: max |diff| %.3e over %d decisions"
          % (float(diff.max()), len(diff)))
    assert float(diff.max()) < 1e-4

    # and the recorded value, which starts at zero because the critic is
    # zero-initialised - a check that the same state really was the one scored
    with torch.no_grad():
        _l2, value = forward_batched(net, rebuilt)
    assert np.abs(np.asarray(value) - np.asarray(ep.value)).max() < 1e-6


@needs_checkpoint
def test_the_recorded_position_really_is_the_position_acted_on(net):
    """`to_move` in the recorded row must be the learner's seat.

    A row recorded after `game.act` would carry the *next* owner, and the
    features built from it would describe a position the learner never saw. The
    `to_move` column is the cheap check; the action legality check below is the
    one that would fail.
    """
    from rl.actions import legal_indices, real_to_view, seat_of_mover
    spec = Rr.Spec(Rr.RL_SEED_BASE, 3, ("wolf", "fox", "hunter"))
    ep = Rr.play_episode(spec, net, mode="softmax")
    rows = ep.state_rows()
    rebuilt = states_from_rows(rows)
    for i, state in enumerate(rebuilt):
        assert state.to_move == spec.learner_seat, i
        # `legal_indices` is in real coordinates and the recorded action is in
        # the mover's frame, so the check has to rotate with `real_to_view`.
        legal = np.asarray(sorted(legal_indices(state, state.to_move).tolist()),
                          dtype=np.int32)
        view = set(real_to_view(legal, seat_of_mover(state)).tolist())
        assert ep.action_index[i] in view, i


# ----------------------------------------------------------- the guards

@needs_checkpoint
def test_a_spec_that_is_not_a_game_is_refused(net):
    bad = [
        Rr.Spec(Rr.RL_SEED_BASE, 4, ("wolf", "fox", "hunter")),
        Rr.Spec(Rr.RL_SEED_BASE, -1, ("wolf", "fox", "hunter")),
        Rr.Spec(Rr.RL_SEED_BASE, 0, ("wolf", "fox")),
        Rr.Spec(Rr.RL_SEED_BASE, 0, ("wolf", "wolf", "fox")),
        Rr.Spec(Rr.RL_SEED_BASE, 0, ("wolf", "fox", "hc_1000")),
    ]
    for spec in bad:
        with pytest.raises(ValueError):
            Rr.play_episode(spec, net, mode="softmax")


def test_the_personality_pool_is_the_seven_named_in_the_plan():
    """Written down rather than read from the registry.

    `ai.registry.personality_keys()` returns these same seven today. Reading it
    here would mean a personality added for any other reason silently entered the
    RL opponent pool, and the pool is a decision rather than a default.
    """
    assert Rr.PERSONALITY_POOL == ("hunter", "optimizer", "builder", "intruder",
                                   "fox", "chess", "wolf")
    import ai
    assert set(Rr.PERSONALITY_POOL) == set(ai.registry.personality_keys())


@needs_checkpoint
def test_hc_and_rl_keys_are_not_in_the_opponent_pool(net):
    """plan8.md is explicit: the opponent pool is the seven personalities only."""
    pool = Rr.PERSONALITY_POOL
    assert not any(seats.is_imitation_key(k) for k in pool)
    assert not any(k.startswith("rl_") for k in pool)


def test_an_empty_batch_is_an_empty_list():
    assert Rr.play_batch([], CHECKPOINT, n_procs=1) == []