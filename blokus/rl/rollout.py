"""This layer answers one question: what does one episode of RL data look like,
and how is a batch of them produced without the result depending on anything
outside the spec?

A "spec" is `(seed, learner_seat, opponent_names)` and it is the whole input.
Everything else - the seat keys, the colours, who opens, every opponent move -
is derived from it, so the same spec is the same episode on any machine and in
any worker. That is what makes the batch below safe to parallelise: a worker
cannot influence another worker's episode because it cannot influence the seeds.

Three decisions are worth stating up front, because each of them is a place
where the obvious implementation would have been wrong.

**One shared rng for setup and the opponents, a second one for the learner.**
`match.run_league` draws the seats, the colours, the opening player and every
move from a single `random.Random`, and `Game.setup_seats` spends a
deterministic number of draws on it - 7 per personality for its weight profile
(`ai.make_profile` draws seven uniforms), 0 for an imitation seat, and 6 more for
colours and turn order. Reproducing that exact consumption is what lets
`tests/test_rl_rollout.py` prove this loop is `match.py`'s loop by showing the
opponents choose the same moves for the same spec. The learner gets a separate
stream because it samples: if it drew from the shared one, its draws would
shift the opponents' stream, and the opponents' stream is what the reuse proof
holds fixed.

**The learner samples; the opponents do not.** The learner draws from the masked
softmax at temperature 1, which is what makes 5,000 episodes 5,000 different
lines through the tree rather than one line replayed 5,000 times. The
personalities keep their own behaviour, including their mistake rates.

**Every episode runs inside `rl.collect._Budget(False)`.** With the wall-clock
budget on, `ai.chooser.choose_move` decides whether to run the opponent lookahead
by asking `time.perf_counter()`, so wolf, chess and fox - the three with
`uses_lookahead = True` - would answer differently on a loaded machine than on an
idle one. That is acceptable for a game and fatal for a rollout: the reward
would depend on how busy the box was. The context manager is `rl.collect`'s, not
a copy of it.

The learner's pass is not a decision. `engine` guarantees a seat with no legal
placement can never have one later, and `BlokusEnv` already skips those turns;
the count is kept in the episode summary because "how much of this episode was
spent skipping" is worth knowing and the rollout should not throw it away.
"""
import multiprocessing as mp
import os
import random
import sys
from typing import NamedTuple

import numpy as np

import ai
import ai.formulas
import engine
from game import Game
from records import rank_rows
from rl.actions import legal_indices

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The seven rule-based and weighted personalities. Deliberately not
# `ai.registry.personality_keys()`: that returns the same seven today, and
# reading it here would mean a personality added for any other reason silently
# entered the RL opponent pool. The pool is a decision, so it is written down.
PERSONALITY_POOL = ("hunter", "optimizer", "builder", "intruder", "fox",
                    "chess", "wolf")

# The seed block plan8.md reserves for RL training. Checked through
# `rl.paired3.reject_reserved_seeds` rather than trusted.
RL_SEED_BASE = 7_000_000
RL_SEED_VALID_BASE = 7_100_000
RL_SEED_SPAN = 20_000
RL_SEED_VALID_SPAN = 1_000

# The reserved block a call is allowed to claim. Train by default; the
# in-training evaluation claims the validation block instead, because
# `plan8.md` puts its fixed evaluation seeds at 7,100,000 and they have to come
# from somewhere reserved rather than from the training stream.
RL_TRAIN_BLOCK = ("stage RL train", RL_SEED_BASE, RL_SEED_BASE + RL_SEED_SPAN - 1)
RL_VALID_BLOCK = ("stage RL validation", RL_SEED_VALID_BASE,
                  RL_SEED_VALID_BASE + RL_SEED_VALID_SPAN - 1)

DEFAULT_TEMPERATURE = 1.0
LEARNER_MODE = "softmax"


class Spec(NamedTuple):
    """One episode's whole input: `(seed, learner_seat, opponent_names)`."""

    seed: int
    learner_seat: int
    opponent_names: tuple


def episode_streams(seed):
    """`(shared rng, learner rng)` for one episode.

    `seed * 2` and `seed * 2 + 1` are injective for the seeds this module
    issues and cannot collide, so two different episodes never share a stream.
    The shared one carries setup and the opponents, which is the consumption
    pattern `match.run_league` has, and the second carries only the learner's
    action sampling so that drawing an action cannot shift the opponents' stream.
    """
    return random.Random(seed * 2), random.Random(seed * 2 + 1)


def _state_columns(state):
    """The compressed position: enough to rebuild the features and the mask.

    The same four fields `rl.collect` writes per row, so an episode's states go
    straight back through `rl.caches.states_from_rows` with no conversion. 220
    bytes: 4 x 400 bits of stones, 4 x 21 bits of hands, 4 stuck latches and the
    mover. `turn_order` is dropped because nothing in the engine's rules reads
    it - the opening corner comes from `OWNER_CORNER` and the contact rule from
    the owner's own stones.
    """
    return {
        "own_bits": np.frombuffer(
            b"".join(m.to_bytes(50, "little") for m in state.own_bits),
            dtype=np.uint8).reshape(4, 50),
        "hand_bits": np.array(state.hand_bits, dtype=np.uint32),
        "stuck": np.array(state.stuck, dtype=bool),
        "to_move": np.int8(state.to_move),
    }


def _sample_from_masked(logits, rng, temperature=DEFAULT_TEMPERATURE):
    """One action index from the masked softmax, by inverse CDF.

    Written out rather than `torch.multinomial` so the draw comes from the
    episode's own `random.Random` and not from a global or a generator whose
    state would have to be threaded through. The cumulative sum is taken in
    float64: the probabilities are float32 and 36,400 of them summed in float32
    drifts far enough to matter at the tail.
    """
    import torch
    p = torch.softmax(logits.detach().double() / float(temperature), dim=-1)
    p = p.cpu().numpy()
    total = float(p.sum())
    if not (total > 0.0):
        raise ValueError("the masked distribution has no mass")
    target = rng.random() * total
    idx = int(np.searchsorted(np.cumsum(p), target, side="right"))
    return min(idx, p.size - 1)


class Episode:
    """One finished game and the learner's decisions inside it."""

    __slots__ = ("spec", "opponent_names", "learner_seat", "seed",
                 "own_bits", "hand_bits", "stuck", "to_move", "action_index",
                 "log_prob", "value", "n_legal", "ply", "learner_passes",
                 "remaining", "ranks", "points", "rewards", "turns", "moves")

    def __init__(self, spec, opponent_names, learner_seat, seed):
        self.spec = spec
        self.opponent_names = tuple(opponent_names)
        self.learner_seat = int(learner_seat)
        self.seed = int(seed)
        self.own_bits = []
        self.hand_bits = []
        self.stuck = []
        self.to_move = []
        self.action_index = []
        # float64 rather than float32: a float32 log_prob is a log_prob the old
        # policy did not actually produce once it has been rounded, and PPO's
        # ratio is exp(new - old) where that difference is the whole quantity.
        # Storing wide is free at 17 decisions per episode and removes the
        # question.
        self.log_prob = []
        self.value = []
        self.n_legal = []
        self.ply = []
        self.learner_passes = 0
        self.remaining = None
        self.ranks = None
        self.points = None
        self.rewards = None
        self.turns = 0
        # `None` unless asked for: every seat's move, in order, as
        # `(owner, name, oi, x, y)`. Roughly 78 tuples per episode, so it is
        # worth the ~5 KB it costs to keep off by default - and
        # `tests/test_rl_rollout.py` switches it on to compare this loop's
        # opponents against `match.py`'s move for move, which cannot be shown
        # from the learner's decisions alone.
        self.moves = []

    # ------------------------------------------------------------------ sizes

    @property
    def n_decisions(self):
        return len(self.action_index)

    def state_rows(self):
        """The recorded states as `rl.caches.states_from_rows` wants them."""
        n = self.n_decisions
        if n == 0:
            return {"own_bits": np.empty((0, 4, 50), dtype=np.uint8),
                    "hand_bits": np.empty((0, 4), dtype=np.uint32),
                    "stuck": np.empty((0, 4), dtype=bool),
                    "to_move": np.empty(0, dtype=np.int8)}
        return {"own_bits": np.stack(self.own_bits),
                "hand_bits": np.stack(self.hand_bits),
                "stuck": np.stack(self.stuck),
                "to_move": np.array(self.to_move, dtype=np.int8)}

    def nbytes(self):
        """Bytes the recorded arrays hold, for deciding whether to spill to disk.

        Measured on what is actually stored rather than estimated from the
        decision count: the position columns dominate, so a per-episode
        estimate that ignored them would be wrong by an order of magnitude.
        """
        total = 0
        for rows in (self.own_bits, self.hand_bits, self.stuck):
            for r in rows:
                total += r.nbytes
        for col in (self.to_move, self.action_index, self.log_prob, self.value,
                    self.n_legal, self.ply):
            if col:
                total += int(np.asarray(col).nbytes)
        return total

    def summary(self):
        """The per-episode fields a manifest wants."""
        return {
            "seed": self.seed,
            "learner_seat": self.learner_seat,
            "opponent_names": list(self.opponent_names),
            "decisions": self.n_decisions,
            "learner_passes": self.learner_passes,
            "turns": self.turns,
            "remaining": list(self.remaining),
            "ranks": list(self.ranks),
            "points": list(self.points),
            "rewards": list(self.rewards),
            "learner_reward": self.rewards[self.learner_seat],
            "learner_remaining": self.remaining[self.learner_seat],
        }


def play_episode(spec, net, device=None, temperature=DEFAULT_TEMPERATURE,
                 mode=LEARNER_MODE, checkpoint_dir=None, record_moves=False,
                 learner_key=None):
    """Play one episode. Returns an `Episode`.

    `spec.seed` fixes everything: the seat keys, the colours, who opens, and
    every opponent move. Two calls with the same spec produce the same episode
    down to the recorded log_probs.

    `mode` is `"softmax"` - draw from the masked distribution, the reason a
    batch of episodes is not one episode repeated - or `"argmax"`, which plays
    the policy's best move. Argmax is not there because it is a better policy;
    it is there so that `tests/test_rl_rollout.py` can prove the opponents in
    this loop choose exactly what `match.py`'s loop chooses for the same spec.
    Under softmax that comparison cannot hold, and not because either loop is
    wrong: the learner samples a different move, so the opponents are answering a
    different position from move one on.

    `device` must be the device the update will run on. The log_probs recorded
    here are the old policy's, and a policy scored on one device and updated on
    another would have its ratio built from a number it never produced.
    """
    from rl.collect import _Budget
    from rl.imitation import action_to_move, state_from_game
    from rl.policy import episode_reward, forward_batched, log_prob_entropy

    import torch

    if not 0 <= spec.learner_seat < 4:
        raise ValueError("learner_seat %r is not a seat" % (spec.learner_seat,))
    names = tuple(spec.opponent_names)
    if len(names) != 3:
        raise ValueError("an episode has three opponents, got %d" % len(names))
    if len(set(names)) != 3:
        raise ValueError("opponents must be distinct, got %r" % (names,))
    for n in names:
        if n not in PERSONALITY_POOL:
            raise ValueError("%r is not in the personality pool" % (n,))

    game_rng, learner_rng = episode_streams(spec.seed)
    seat_key, needs_swap = seat_for(net, learner_key)
    keys = list(names)
    keys.insert(spec.learner_seat, seat_key)

    ep = Episode(spec, names, spec.learner_seat, spec.seed)
    with _Budget(False), _ThreadBudget(1):
        game = Game(game_rng)
        game.setup_seats(keys, None, game_rng, checkpoint_dir=checkpoint_dir,
                         device=device, mode="argmax")
        if needs_swap:
            # the whole point of the stand-in: this seat plays `net`, not
            # whatever its key named
            game.brains[spec.learner_seat] = as_brain(net, key=seat_key)
        game.start()
        ply = 0
        while game.state == "PLAYING":
            owner = game.current_owner()
            if owner == spec.learner_seat:
                state = state_from_game(game)
                if engine.legal_move_mask(state, owner) == 0:
                    # A pass is not a choice, so it is not a decision and is not
                    # recorded as one. Counted, because how much of an episode
                    # was spent skipping is worth knowing.
                    ep.learner_passes += 1
                    if record_moves:
                        ep.moves.append((owner, None, None, None, None))
                    game.act_pass()
                    continue
                legal = legal_indices(state, owner)
                # No gradients: this is the old policy being *scored*, and PPO's
                # gradient comes from the update pass over the recorded numbers.
                # `rl.policy.forward_batched` deliberately leaves grad enabled for
                # the training path, so the rollout is the side that has to ask
                # for this - and asking here rather than inside the shared
                # function is what keeps both callers honest.
                with torch.no_grad():
                    logits, value = forward_batched(net, [state], device=device)
                    if mode == "argmax":
                        action = int(logits.argmax(dim=-1).item())
                    elif mode == "softmax":
                        action = _sample_from_masked(logits[0], learner_rng,
                                                     temperature)
                    else:
                        raise ValueError("unknown learner mode %r; use softmax "
                                         "or argmax" % (mode,))
                    log_prob, _entropy = log_prob_entropy(
                        logits, torch.tensor([action]))
                cols = _state_columns(state)
                ep.own_bits.append(cols["own_bits"])
                ep.hand_bits.append(cols["hand_bits"])
                ep.stuck.append(cols["stuck"])
                ep.to_move.append(int(state.to_move))
                ep.action_index.append(int(action))
                ep.log_prob.append(float(log_prob[0]))
                ep.value.append(float(value[0]))
                ep.n_legal.append(int(legal.size))
                ep.ply.append(ply)
                name, oi, base = action_to_move(action, state)
                game.act(name, oi, base % 20, base // 20)
                if record_moves:
                    ep.moves.append((owner, name, oi, base % 20, base // 20))
            else:
                move = ai.choose_move(game.board, game.hands[owner].names,
                                      owner, game.brains[owner], game_rng,
                                      other_brains=game.brain_map(),
                                      must_cover=game.must_cover(owner),
                                      reach=game.reach(owner),
                                      other_must_cover={o: game.must_cover(o)
                                                       for o in range(4)},
                                      other_reach={o: game.reach(o)
                                                   for o in range(4)})
                if move:
                    if record_moves:
                        ep.moves.append((owner,) + tuple(move))
                    game.act(*move)
                else:
                    if record_moves:
                        ep.moves.append((owner, None, None, None, None))
                    game.act_pass()
            ply += 1
        ep.turns = ply

    remaining = tuple(game.remaining_cells(o) for o in range(4))
    ep.remaining = remaining
    # `rank_rows` returns the seats **best first**, not in seat order, so its
    # rows have to be written back through their key. Reading them positionally
    # would put the leader's rank on owner 3 and leave `ranks` disagreeing with
    # `remaining` and `rewards`, which are both indexed by owner - the kind of
    # mismatch that reads as a plausible number rather than as an off-by-three.
    ranks = [0] * 4
    points = [0] * 4
    for key, rank, point, _rem in rank_rows([(o, remaining[o]) for o in range(4)]):
        ranks[key] = rank
        points[key] = point
    ep.ranks = tuple(ranks)
    ep.points = tuple(points)
    ep.rewards = tuple(episode_reward(remaining, o) for o in range(4))
    return ep


def as_brain(net, key=None):
    """Give a bare network the attributes a seated brain must expose.

    `ai.chooser.choose_move` reads `other_brains[o].profile` when *another* seat
    runs its opponent lookahead - `ai/chooser.py:242` - so every brain in a game
    needs a `profile` whether or not it ever picks a move itself. A bare
    `TwinHead` from `rl.policy.load_policy` has none, and the result is an
    `AttributeError` raised by a **personality** on the turn after the learner
    moves: nothing about the learner's own code is involved, which is what makes
    it confusing to debug.

    The profile is chess's, because that is what `rl.imitation._chess_profile`
    hands an `ImitationBrain` - so an opponent's lookahead scores the RL learner
    with exactly the weights it would score `hc_1000` with. The RL seat is an
    imitation seat as far as the opponents are concerned, which is what it is.
    """
    from rl.imitation import _chess_profile
    if not hasattr(net, "profile"):
        net.profile = _chess_profile()
    if not hasattr(net, "mistake_rate"):
        net.mistake_rate = 0.0
    if not hasattr(net, "uses_lookahead"):
        net.uses_lookahead = False
    if key is not None and getattr(net, "key", None) is None:
        net.key = key
    return net


def seat_for(net, learner_key=None):
    """`(seat_key, needs_swap)` for the learner.

    `Game.setup_seats` validates every seat key against `seats`, so a network
    with no registered name - which is what a training checkpoint is, since
    `rl_1000_20k` is not in the pool - cannot simply be named. So the learner
    stands in under the first H-C2 seat key and the brain is replaced
    immediately afterwards with `needs_swap` saying so.

    `hc_1000`'s weights are loaded in passing. That is nearly free -
    `rl.imitation._BLOB_CACHE` reads the file once per process, not once per
    game - and it buys not touching `Game.setup_seats`, which is the seat-layout
    code `plan8-A` found should stay the single definition.

    The stand-in is `hc_1000` rather than a personality on purpose: if the swap
    were ever missed, a personality would silently play the learner's moves and
    the run would look fine. A wrong `hc_1000` moves are at least a different
    policy from round one.
    """
    import seats
    candidates = []
    if learner_key is not None:
        candidates.append((learner_key, True))
    net_key = getattr(net, "key", None)
    if net_key:
        candidates.append((net_key, net_key != learner_key))
    for key, swap in candidates:
        try:
            seats.kind_of(key)
        except ValueError:
            continue
        return key, swap
    return seats.imitation_key(seats.IMITATION_STEPS[0]), True


def make_specs(start_seed, n, rng_seed=None, own=None):
    """`(specs, manifest)` for `n` episodes from `start_seed` upwards.

    Each seed contributes one episode. The three opponents are drawn **without
    replacement** from `PERSONALITY_POOL` and the learner's seat is uniform over
    the four, both from a `random.Random` seeded once by `rng_seed` - so the
    whole assignment is a function of one integer and is reproducible without
    replaying the draw.

    The seed block is checked with `rl.paired3.reject_reserved_seeds` before a
    single episode is played. That guard exists because H-B1 and H-C1 used
    6,000,000-6,029,378 and a rollout landing there would be replaying the
    imitation set's own games; this is the one place in the RL path where that
    mistake gets caught, and it is caught before anything is played.

    `rng_seed` defaults to `start_seed`, which is the right answer for a single
    run and the wrong one for a second: two runs over the same seed block with
    the same opponents would be two copies of one experiment. Pass it
    explicitly when generating more than one.

    `own` names the reserved block the caller is claiming, as
    `(label, lo, hi)`; it defaults to the training block. A caller working inside
    the validation block must say so, because `reject_reserved_seeds` cannot tell
    "this run owns these seeds" from "this run is about to walk into the
    imitation data" - it only sees a range and a claim.
    """
    from rl.paired3 import reject_reserved_seeds

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive, got %d" % n)
    start_seed, end = int(start_seed), int(start_seed) + n - 1
    if rng_seed is None:
        rng_seed = start_seed
    reject_reserved_seeds(start_seed, end,
                          own=RL_TRAIN_BLOCK if own is None else own)
    rng = random.Random(int(rng_seed))
    pool = list(PERSONALITY_POOL)
    specs = []
    seat_counts = [0, 0, 0, 0]
    for seed in range(start_seed, end + 1):
        names = tuple(rng.sample(pool, 3))
        seat = rng.randrange(4)
        seat_counts[seat] += 1
        keys = list(names)
        keys.insert(seat, "hc_1000")
        specs.append(Spec(seed, seat, names))
    manifest = {
        "start_seed": start_seed,
        "end_seed": end,
        "games": n,
        "rng_seed": int(rng_seed),
        "personality_pool": list(PERSONALITY_POOL),
        "opponents_per_game": 3,
        "learner_seat_counts": seat_counts,
        "learner_seat_share": [c / float(n) for c in seat_counts],
        "opponent_coverage": {name: _count(name, specs)
                              for name in PERSONALITY_POOL},
        "reserved_range_checked": ["stage RL train",
                                   RL_SEED_BASE,
                                   RL_SEED_BASE + RL_SEED_SPAN - 1],
    }
    return specs, manifest


def _count(name, specs):
    """How many opponent seats across `specs` this personality filled."""
    c = 0
    for s in specs:
        c += sum(1 for x in s.opponent_names if x == name)
    return c


# --------------------------------------------------------------------------
# the batch
# --------------------------------------------------------------------------

_WORKER = {}


class _ThreadBudget:
    """Pin torch to one thread for the duration, and put it back.

    The second half of the reproducibility argument, and the reason the log_probs
    recorded by a worker are the same numbers a single process records. A
    multi-threaded CPU convolution sums in a different order than a single-threaded
    one, so the same position gives slightly different float32 logits:
    **1.43e-06** on `log_prob` between fourteen threads and one, measured on
    hc_1000. That is the same order as the batch=1 versus batch=N floor, and it
    is enough to make `play_batch(n_procs=1)` and `play_batch(n_procs=4)` record
    different last digits.

    Actions, positions and rewards are unaffected - the sampled action and the
    resulting game are identical - so this is not a correctness bug in the
    trajectory. It is a reproducibility one, and the fix is the same shape as
    `rl.collect._Budget`: set a module global for the block, restore it in a
    `finally`. Nested uses save and restore correctly, so pinning once around a
    batch and again around each episode inside it lands back where it started.

    Restoring rather than leaving it at 1 matters because this runs inside a
    training process that will want its threads back for the update.
    """

    def __init__(self, threads=1):
        self.threads = int(threads)
        self.saved = None

    def __enter__(self):
        import torch
        self.saved = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        return self

    def __exit__(self, *exc):
        import torch
        torch.set_num_threads(self.saved)
        return False


def _worker_init():
    """Runs once in each worker process, before it does any work.

    `torch.set_num_threads(1)` belongs here rather than in a docstring, because
    the alternative is a pool of eight processes each trying to use fourteen
    OpenMP threads on fourteen cores: measured on this machine that is a 31x
    slowdown, 3.86 hours for 5,000 episodes against 7.6 minutes. Making it a
    pool initialiser is the difference between being a default and being
    something a caller has to remember.

    The environment variables are belt to that braces - they only matter if
    torch is imported after this point, which for a forked worker it is not.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)


def _worker(job):
    """Play a chunk of specs, loading the weights once per process."""
    specs, weights_path, device, temperature, mode, checkpoint_dir = job
    if _WORKER.get("path") != weights_path:
        import torch
        from rl.policy import load_policy
        net, meta = load_policy(weights_path, device=device)
        key = _key_for(weights_path, meta)
        if key:
            net.key = key
        _WORKER["path"] = weights_path
        _WORKER["net"] = net
        _WORKER["meta"] = meta
    net = _WORKER["net"]
    return [play_episode(s, net, device=device, temperature=temperature,
                         mode=mode, checkpoint_dir=checkpoint_dir)
            for s in specs]


def _key_for(weights_path, meta):
    """The seat key for a checkpoint file, or `None` when it has no name yet.

    `hc_1000` when the file is one of the pool's. A training checkpoint has no
    registered seat name, and that is not this module's to invent - so it returns
    `None` and `seat_for` substitutes the stand-in. Inventing `rl_<step>` here
    would put a name in the logs and in `Game.owner_key` that no seat registry
    knows about, which is how a rollout ends up asking for a seat that does not
    exist.
    """
    base = os.path.basename(weights_path)
    import seats
    for s in seats.IMITATION_STEPS:
        if base == "step_%06d.pt" % s:
            return seats.imitation_key(s)
    return None


def play_batch(specs, weights_path, n_procs=1, device=None,
               temperature=DEFAULT_TEMPERATURE, mode=LEARNER_MODE,
               checkpoint_dir=None):
    """Play every spec, in **spec order**, however the work was scheduled.

    With `n_procs=1` this is a plain loop in this process. Above that, the
    specs are cut into contiguous chunks and one `fork` pool is built with
    `_worker_init` as its initialiser; `Pool.map` returns results in the order
    the inputs were given, so the output does not depend on which episode
    finished first. Each worker loads the weights itself, once - a worker that
    inherited a network through `fork` would be sharing its memory with a parent
    that is about to exit.

    Nothing about the result may depend on `n_procs`, which
    `tests/test_rl_rollout.py` checks rather than hopes. It holds because the
    thread count is pinned on both paths - see `_ThreadBudget` - and not because
    the work is deterministic, which multi-threaded float32 is not.
    """
    specs = list(specs)
    if not specs:
        return []
    if device is not None and n_procs > 1:
        import torch
        if torch.device(device).type == "cuda":
            raise ValueError("device=%r with n_procs=%d would give every worker "
                             "its own CUDA context on one GPU; pass device=None "
                             "and let each worker choose, or run on CPU"
                             % (device, n_procs))
    with _ThreadBudget(1):
        if n_procs <= 1:
            from rl.policy import load_policy
            net, meta = load_policy(weights_path, device=device)
            key = _key_for(weights_path, meta)
            if key:
                net.key = key
            return [play_episode(s, net, device=device,
                                 temperature=temperature, mode=mode,
                                 checkpoint_dir=checkpoint_dir)
                    for s in specs]

        per = max(1, -(-len(specs) // int(n_procs)))
        chunks = [specs[i:i + per] for i in range(0, len(specs), per)]
        jobs = [(chunk, weights_path, device, temperature, mode, checkpoint_dir)
                for chunk in chunks]
        with mp.get_context("fork").Pool(len(jobs),
                                         initializer=_worker_init) as pool:
            results = pool.map(_worker, jobs)
    # `map` preserves input order, and the chunks were cut in order too, so
    # flattening is already spec order. The length check is here because a
    # silently short result would index-align against the wrong spec for every
    # episode after it, which is a far worse failure than an exception.
    out = [ep for chunk in results for ep in chunk]
    if len(out) != len(specs):
        raise RuntimeError("expected %d episodes, got %d"
                           % (len(specs), len(out)))
    return out


def thread_report():
    """What the worker initialiser sets, for a manifest."""
    return {"torch_num_threads": 1,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset")}


if __name__ == "__main__":       # pragma: no cover
    print(__doc__)
    print("personality pool:", PERSONALITY_POOL)
    print("seed base:", RL_SEED_BASE)
    sys.exit(0)