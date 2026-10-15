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
from math import gcd
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

# plan9 step 3: the train block reserved for the three students (o / b / i),
# shared by all three exactly as 3a shares everything else. The label and range
# must match `rl.paired3.RESERVED_RANGES` to the tuple - `reject_reserved_seeds`
# exempts by exact triple, so a mismatch would make every student run refuse to
# start against its own reserved row.
RL_STUDENTS_SEED_BASE = 7_200_000
RL_STUDENTS_BLOCK = ("stage RL students", RL_STUDENTS_SEED_BASE,
                     RL_STUDENTS_SEED_BASE + RL_SEED_SPAN - 1)

# plan10 extension, h-only rerun (v2): `rl_h1000_50k` fell behind its own 20k
# on argmax, so the h learner walks steps 81..100 again from the 40k checkpoint
# under *this* pipeline instead of `rl.multi`'s two environments - the same
# seven-personality pool the 0k->20k walk used.
#
# `seed_base` is the anchor where round 1 *would* have started and is
# deliberately outside the row: `rl_train` derives a round's seeds as
# `seed_base + (round - 1) * per_round`, so rounds 81..100 on a 7,460,000 base
# draw exactly 7,500,000..7,509,999 and nothing else. The row covers that used
# range, not the 40,000 anchor-side seeds no round ever draws - a wider row
# would lock away seeds that were never used and describe a history that never
# happened. Written as literals because the bounds are the evidence.
RL_V2_H_SEED_BASE = 7_460_000
RL_V2_H_BLOCK = ("stage RL v2 h train", 7_500_000, 7_509_999)

# The fixed evaluation set the acceptance measurement is scored on, kept in its
# own row so a bigger schedule cannot silently grow into the next stage's, and
# so it cannot be confused with the in-training 100-game curve's seeds
# (`RL_VALID_BLOCK`) or with the rb80 rerun's (`rl.multi.RB80_EVAL_BLOCK`).
RL_V2_H_EVAL_SEED_BASE = 7_520_000
RL_V2_H_EVAL_SPAN = 1_000
RL_V2_H_EVAL_BLOCK = ("stage RL v2 h eval", RL_V2_H_EVAL_SEED_BASE,
                      RL_V2_H_EVAL_SEED_BASE + RL_V2_H_EVAL_SPAN - 1)

# plan10, the i learner: rounds 41..100 from its own 20k, 30,000 games under a
# caller's frozen pool. Same anchor arithmetic as the h rerun above - `seed_base`
# is where round 1 *would* have started and rounds 41..100 on a 7,150,000 base
# draw exactly 7,170,000..7,199,999. That sits in the gap between `stage RL
# validation`'s top and `stage RL students`' floor, which was never reserved and
# never used, so nothing had to move. Written as literals for the reason the
# row above is.
#
# The base is a fresh anchor rather than the students' 7,200,000 the 20k was
# trained on: the anchor table is matched *before* containment, so a second
# entry for 7,200,000 would shadow `RL_STUDENTS_BLOCK` and every future student
# run would stop being exempted against its own row.
RL_I50K_V2_SEED_BASE = 7_150_000
RL_I50K_V2_BLOCK = ("stage RL i50k v2 train", 7_170_000, 7_199_999)

# The acceptance evaluation, its own row for the reason `RL_V2_H_EVAL_BLOCK`
# has one: a bigger schedule must not grow into the next stage's, and these
# seeds must not read as the in-training curve's.
RL_I50K_V2_EVAL_SEED_BASE = 7_540_000
RL_I50K_V2_EVAL_SPAN = 1_000
RL_I50K_V2_EVAL_BLOCK = ("stage RL i50k v2 eval", RL_I50K_V2_EVAL_SEED_BASE,
                         RL_I50K_V2_EVAL_SEED_BASE + RL_I50K_V2_EVAL_SPAN - 1)

# plan10, the o learner: rounds 41..100 from its own 20k, 30,000 games under a
# caller's frozen pool. Same anchor arithmetic as the two reruns above - a
# 7,530,000 base resumed at round 41 draws exactly 7,550,000..7,579,999.
# Both ranges sit above every row registered so far rather than in the crowded
# 7,1xx,xxx gap, which is left for whatever needs it next.
#
# The base is again chosen so that it is inside no registered row: it is
# matched before containment, so a base that a row also contains would shadow
# that row. 7,530,000 is between `stage RL v2 h eval`'s top and
# `stage RL i50k v2 eval`'s floor, and belongs to neither.
RL_O50K_V2_SEED_BASE = 7_530_000
RL_O50K_V2_BLOCK = ("stage RL o50k v2 train", 7_550_000, 7_579_999)

# The acceptance evaluation, its own row for the reason the three above have
# one.
RL_O50K_V2_EVAL_SEED_BASE = 7_590_000
RL_O50K_V2_EVAL_SPAN = 1_000
RL_O50K_V2_EVAL_BLOCK = ("stage RL o50k v2 eval", RL_O50K_V2_EVAL_SEED_BASE,
                         RL_O50K_V2_EVAL_SEED_BASE + RL_O50K_V2_EVAL_SPAN - 1)

# plan10 v3 (2026-10-13 decree), the o learner again: 60 rounds from a cold
# start (`--init`, no `--resume`) on `rl_o1000_20k`, under the balanced
# schedule and a ten-seat pool without the learner's own start. Round
# numbering restarts at 1, so - unlike the three reruns above - `seed_base` is
# where round 1 actually starts *and is drawn*: rounds 1..60 on a 7,620,000
# base walk exactly 7,620,000..7,649,999, which is why the base sits at the
# floor of the row it claims rather than outside any row. Inside its own row
# an anchor claims nothing but itself; the shape that must stay forbidden is
# an anchor inside some *other* registered row, because the table is consulted
# before containment and would shadow that row's claim. The eval base stays
# the ordinary shape: arithmetic, never drawn by the in-training loop.
RL_O50K_V3_SEED_BASE = 7_620_000
RL_O50K_V3_BLOCK = ("stage RL o50k v3 train", 7_620_000, 7_649_999)

RL_O50K_V3_EVAL_SEED_BASE = 7_660_000
RL_O50K_V3_EVAL_SPAN = 1_000
RL_O50K_V3_EVAL_BLOCK = ("stage RL o50k v3 eval", RL_O50K_V3_EVAL_SEED_BASE,
                         RL_O50K_V3_EVAL_SEED_BASE + RL_O50K_V3_EVAL_SPAN - 1)

# The pre-decreed re-verification (2026-10-13): same pairing, same criterion,
# fresh seeds - 7,670,000..7,670,999 sits above every row registered so far
# and 9,000 clear of the first evaluation's block, so the second measurement
# does not replay a seed the first one scored. Eval bases never enter
# `TRAIN_ANCHOR_BLOCKS`: no training round draws them, and the harness claims
# the triple directly through `make_specs(..., own=...)`.
RL_O50K_V3_EVAL2_SEED_BASE = 7_670_000
RL_O50K_V3_EVAL2_SPAN = 1_000
RL_O50K_V3_EVAL2_BLOCK = ("stage RL o50k v3 eval2",
                          RL_O50K_V3_EVAL2_SEED_BASE,
                          RL_O50K_V3_EVAL2_SEED_BASE + RL_O50K_V3_EVAL2_SPAN
                          - 1)

# plan10 (2026-10-14 decree), the h learner: 60 rounds from a cold start
# (`--init` on `rl_h1000_20k`, no `--resume`) under the balanced schedule,
# 30,000 games. Cold start for the same shape reason as the v3 row: round 1
# draws `seed_base` itself, so the base sits at the floor of the row it claims
# rather than outside every row.
#
# The pool is the caller's ten seats and it *does* hold the learner's own
# frozen start (`rl_h1000_20k`) - a self-play exception the user declared for
# this run, the v2o shape rather than the v3 one. The seat is the second 50k
# walk for h, so `rl_h1000_50k` (the four-at-once run's round 100) is neither
# in the pool nor the output: this run's seat is `rl_h1000_50k_v2`.
#
# No second evaluation row is registered: the decree made the acceptance
# measurement a single 1000-game paired argmax look, so there is no
# pre-decreed re-verification needing its own seeds. A row nobody draws would
# only lock seeds away.
RL_H50K_V2_SEED_BASE = 7_680_000
RL_H50K_V2_BLOCK = ("stage RL h50k v2 train", RL_H50K_V2_SEED_BASE,
                    RL_H50K_V2_SEED_BASE + 30_000 - 1)

RL_H50K_V2_EVAL_SEED_BASE = 7_715_000
RL_H50K_V2_EVAL_SPAN = 1_000
RL_H50K_V2_EVAL_BLOCK = ("stage RL h50k v2 eval", RL_H50K_V2_EVAL_SEED_BASE,
                         RL_H50K_V2_EVAL_SEED_BASE + RL_H50K_V2_EVAL_SPAN
                         - 1)

# The pool-swapped re-measurement the user decreed on 2026-10-14, after the
# §5 criterion run on 7,715,000 came back at +0.006430, below its 1.3 x se
# line. The opponent pool is the seven personalities plus two frozen nets,
# `rl_b1000_50k` and `rl_o1000_50k_v3`, and the learner's own start is gone
# from it, so - unlike the training pool - nothing in this set is the
# candidate itself. It is still not an independent measurement of the
# candidate in general: both of those nets sat in the pool it trained
# against, so a positive number here cannot separate "got stronger" from
# "got used to these two opponents".
#
# Fresh seeds are the whole reason this row exists. A second look on the
# seeds the first measurement scored would be the same 1,000 games twice,
# which is why it sits 9,001 above that block's top rather than inside it -
# the same gap style as `RL_O50K_V3_EVAL2_BLOCK`. Eval bases never enter
# `TRAIN_ANCHOR_BLOCKS`: no training round draws them, and the harness
# claims the triple through `make_specs(..., own=...)`.
RL_H50K_V2_EVAL2_SEED_BASE = 7_725_000
RL_H50K_V2_EVAL2_SPAN = 1_000
RL_H50K_V2_EVAL2_BLOCK = ("stage RL h50k v2 eval2", RL_H50K_V2_EVAL2_SEED_BASE,
                          RL_H50K_V2_EVAL2_SEED_BASE + RL_H50K_V2_EVAL2_SPAN
                          - 1)

# plan10 (2026-10-14 decree), the i learner: 60 rounds from a cold start
# (`--init` on `rl_i1000_20k`, no `--resume`) under the balanced schedule,
# 30,000 games, towards the seat `rl_i1000_50k_v3`. Cold start for the same
# shape reason as the v3 and h-v2 rows: round 1 draws `seed_base` itself, so
# the base sits at the floor of the row it claims rather than outside every
# row.
#
# The pool is the caller's ten seats and it does **not** hold the learner's
# own start - the plan10 default shape, not the h run's declared self-play
# exception - so no game is the learner against its own frozen 20k. It also
# holds no downstream version of the learner: `rl_i1000_50k` (the
# four-at-once run's round 100) is deliberately absent, which is the check
# the earlier i rerun's eleven-seat pool failed.
#
# The seat is named v3 rather than v2 because `rl_i1000_50k_v2` already
# names the earlier resume-based rerun (rounds 41..100, evidence
# `eval/rl-single-train-v2i/`), which scored argmax -0.0297 and was never
# pooled; one name must not point at two sets of weights.
#
# One evaluation row only: the decree made the acceptance measurement a
# single 1000-game argmax look, so softmax is not run and there is no
# re-verification to reserve seeds for.
RL_I50K_V3_SEED_BASE = 7_730_000
RL_I50K_V3_BLOCK = ("stage RL i50k v3 train", RL_I50K_V3_SEED_BASE,
                    RL_I50K_V3_SEED_BASE + 30_000 - 1)

RL_I50K_V3_EVAL_SEED_BASE = 7_765_000
RL_I50K_V3_EVAL_SPAN = 1_000
RL_I50K_V3_EVAL_BLOCK = ("stage RL i50k v3 eval", RL_I50K_V3_EVAL_SEED_BASE,
                         RL_I50K_V3_EVAL_SEED_BASE + RL_I50K_V3_EVAL_SPAN
                         - 1)

# The two mid-step measurements the user decreed on 2026-10-15, after the
# §5 criterion run on 7,765,000 came back at -0.036140, below its line. The
# in-training curve for the same run had peaked at rounds 30-35 (+0.316 and
# +0.334 against the start on the 100-game validation block), so the question
# these answer is whether an *intermediate* checkpoint is stronger than the
# final one - the final is what the criterion scored, and it lost.
#
#   * `step_000035.pt` against the start `rl_i1000_20k`
#   * `step_000030.pt` against `rl_h1000_20k`
#
# Both are 1000-game paired argmax runs on the default seven-personality pool,
# the same shape as the criterion. Fresh seeds are the point of the new row:
# 7,765,000-7,765,999 was already scored by the criterion run, and replaying
# it would make the second look the same 1,000 games twice. This block sits
# 4,001 above that row's top, the gap style `RL_O50K_V3_EVAL2_BLOCK` set.
# The two measurements share the one block, so both reads share a yardstick;
# each is still paired within itself, and the two share nothing else - the
# references differ (`rl_i1000_20k` vs `rl_h1000_20k`), so the two numbers
# say nothing about each other. Eval bases never enter `TRAIN_ANCHOR_BLOCKS`.
RL_I50K_V3_EVAL2_SEED_BASE = 7_770_000
RL_I50K_V3_EVAL2_SPAN = 1_000
RL_I50K_V3_EVAL2_BLOCK = ("stage RL i50k v3 eval2", RL_I50K_V3_EVAL2_SEED_BASE,
                          RL_I50K_V3_EVAL2_SEED_BASE + RL_I50K_V3_EVAL2_SPAN
                          - 1)

# The h run's round-20 measurement, decreed on 2026-10-15 after the i run's
# mid-step look: the i run's round 35 beat its own start (+0.0656) while its
# final weights did not (-0.0361), so the same question is asked of the h run.
# The h in-training curve peaked at rounds 15-20 (+0.154 and +0.150 against
# the start on the 100-game validation block) and its final weights scored
# +0.006430, below the criterion's line, so round 20 is the checkpoint most
# likely to carry the gain the final one lost.
#
# Same shape as the criterion and as the i mid-step measurements: 1000 paired
# argmax games on the default seven-personality pool, this time
# `data/rl_h50k_v3/step_000020.pt` against its own start
# `rl_h1000_20k`. Fresh seeds again - the criterion run scored
# 7,680,000-adjacent eval row 7,715,000 and the pool-swapped look scored
# 7,725,000, so this row sits 9,001 above the highest of them. Eval bases
# never enter `TRAIN_ANCHOR_BLOCKS`.
RL_H50K_V2_EVAL3_SEED_BASE = 7_775_000
RL_H50K_V2_EVAL3_SPAN = 1_000
RL_H50K_V2_EVAL3_BLOCK = ("stage RL h50k v2 eval3", RL_H50K_V2_EVAL3_SEED_BASE,
                          RL_H50K_V2_EVAL3_SEED_BASE + RL_H50K_V2_EVAL3_SPAN
                          - 1)

# Bases `_claimed_train_block` matches on when the run's `seed_base` is not
# inside the row it is entitled to claim. The first three are runs that
# resumed at a round above 1, so their bases are arithmetic rather than
# reserved seeds and sit outside every row. The v3, h-v2 and i-v3 bases are
# the deliberate exception: they start at round 1, so they are drawn from
# round 1 and sit at the floor of their own rows.
# Matched before containment, which is why a base inside a row *other than its
# own* must never appear here - see `RL_I50K_V2_SEED_BASE`, where a base
# inside the students' row would have stolen every student run's claim.
TRAIN_ANCHOR_BLOCKS = ((RL_V2_H_SEED_BASE, RL_V2_H_BLOCK),
                       (RL_I50K_V2_SEED_BASE, RL_I50K_V2_BLOCK),
                       (RL_O50K_V2_SEED_BASE, RL_O50K_V2_BLOCK),
                       (RL_O50K_V3_SEED_BASE, RL_O50K_V3_BLOCK),
                       (RL_H50K_V2_SEED_BASE, RL_H50K_V2_BLOCK),
                       (RL_I50K_V3_SEED_BASE, RL_I50K_V3_BLOCK))

DEFAULT_TEMPERATURE = 1.0
LEARNER_MODE = "softmax"

# Plan10 task 2: a same-table game records one episode per network seat, so
# each needs a sampling stream of its own. These two constants put them in an
# integer range `seed * 2` / `seed * 2 + 1` cannot reach; the stride is 2**24
# and every seed block this module claims is below it, so `(seat, seed)` is
# injective across the whole reserved space.
_LEARNER_STREAM_BASE = 0x1000_0000
_LEARNER_STREAM_STRIDE = 0x100_0000


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


def log_move(eps, move):
    """Append one ply to every episode's move log.

    The log is of the **game**, not of one seat's trajectory, so when one table
    records four episodes they all get the same list - which is what lets a
    caller compare the whole game move for move against `match.py`'s.
    """
    for ep in eps.values():
        ep.moves.append(move)


def play_episode(spec, net, device=None, temperature=DEFAULT_TEMPERATURE,
                 mode=LEARNER_MODE, checkpoint_dir=None, record_moves=False,
                 learner_key=None, opponent_pool=None):
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

    `opponent_pool` replaces the seven personalities as the set opponents are
    drawn from. Left as `None` the three opponents must be **distinct
    personalities**, which is the rule every published batch was recorded under
    and the rule `tests/test_rl_rollout.py` pins. A caller that names a pool
    (plan10's fixed pool of eleven) is saying duplicates are fine and that a
    network may sit opposite the learner - both of which `Game.setup_seats`
    already allows.
    """
    if not 0 <= spec.learner_seat < 4:
        raise ValueError("learner_seat %r is not a seat" % (spec.learner_seat,))
    names = tuple(spec.opponent_names)
    if len(names) != 3:
        raise ValueError("an episode has three opponents, got %d" % len(names))
    if opponent_pool is None:
        if len(set(names)) != 3:
            raise ValueError("opponents must be distinct, got %r" % (names,))
        pool = PERSONALITY_POOL
        pool_name = "the personality pool"
    else:
        pool = tuple(opponent_pool)
        pool_name = "the pool %r" % (pool,)
    for n in names:
        if n not in pool:
            raise ValueError("%r is not in %s" % (n, pool_name))

    seat_key, needs_swap = seat_for(net, learner_key)
    keys = list(names)
    keys.insert(spec.learner_seat, seat_key)
    eps = play_game(keys, {spec.learner_seat: net}, spec.seed,
                    device=device, temperature=temperature, mode=mode,
                    checkpoint_dir=checkpoint_dir, record_moves=record_moves)
    ep = eps[spec.learner_seat]
    # The spec is the caller's, not the one `play_game` rebuilt: a batch's
    # manifest and `tests/test_rl_rollout.py` both read `ep.spec`, and a
    # rebuilt stand-in would answer the same three fields but compare unequal
    # to what was asked for.
    ep.spec = spec
    ep.opponent_names = names
    if needs_swap is False:
        # `seat_for` only says that when the seat key already *is* the learner's
        # own registered key, which would mean `setup_seats` loaded the frozen
        # registry weights into the learner's seat. `play_game` always swaps the
        # handed network in, so this is unreachable - kept as a check because
        # silently training against frozen weights is the failure it would be.
        raise RuntimeError("seat_for refused to stand %r in; the learner would "
                           "have played the frozen registry weights" % (seat_key,))
    return ep


def play_game(keys, seat_nets, seed, device=None,
              temperature=DEFAULT_TEMPERATURE, mode=LEARNER_MODE,
              checkpoint_dir=None, record_moves=False):
    """Play one game and record every seat that has a network.

    Returns `{seat: Episode}` - one Episode per entry in `seat_nets`, each with
    `learner_seat` set to its own seat, so the four go straight to four separate
    `ppo.batch_from_episodes` calls without any repacking.

    `keys` are the four option keys handed to `Game.setup_seats`, in seat order,
    and a key may repeat - two identical frozen policies at one table is a game,
    not an error. `seat_nets` maps a seat index to the network that seat really
    plays; **after** setup that seat's brain is replaced by `as_brain(net)`, so a
    seat in `seat_nets` always plays the network it was handed and never the
    weights its key happens to name. Seats not in `seat_nets` keep whatever brain
    `setup_seats` built for them.

    `seed` fixes everything, exactly as `play_episode` says it does.
    """
    from rl.collect import _Budget
    from rl.imitation import action_to_move, state_from_game
    from rl.policy import episode_reward, forward_batched, log_prob_entropy

    import torch

    keys = list(keys)
    if len(keys) != 4:
        raise ValueError("a game has four seats, got %d keys" % len(keys))
    if not seat_nets:
        raise ValueError("a game needs at least one network seat")
    for s in seat_nets:
        if not 0 <= int(s) < 4:
            raise ValueError("seat %r is not a seat" % (s,))
    if mode not in ("softmax", "argmax"):
        raise ValueError("unknown learner mode %r; use softmax or argmax"
                         % (mode,))

    game_rng, _ = episode_streams(seed)
    streams = _learner_streams(seed, sorted(int(s) for s in seat_nets))

    eps = {}
    for s in sorted(int(s) for s in seat_nets):
        others = tuple(keys[o] for o in range(4) if o != s)
        eps[s] = Episode(Spec(seed, s, others), others, s, seed)

    def record(owner, state, legal, ply):
        """One network seat's decision. Returns True if a move was played."""
        ep = eps[owner]
        with torch.no_grad():
            logits, value = forward_batched(net_by_seat[owner], [state],
                                            device=device)
            if mode == "argmax":
                action = int(logits.argmax(dim=-1).item())
            else:
                action = _sample_from_masked(logits[0], streams[owner],
                                             temperature)
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
            log_move(eps, (owner, name, oi, base % 20, base // 20))
        return True

    net_by_seat = {int(s): n for s, n in seat_nets.items()}
    with _Budget(False), _ThreadBudget(1):
        game = Game(game_rng)
        game.setup_seats(keys, None, game_rng, checkpoint_dir=checkpoint_dir,
                         device=device, mode="argmax")
        for s, net in net_by_seat.items():
            # the whole point of the stand-in: this seat plays `net`, not
            # whatever its key named
            game.brains[s] = as_brain(net, key=keys[s])
        game.start()
        ply = 0
        while game.state == "PLAYING":
            owner = game.current_owner()
            if owner in net_by_seat:
                state = state_from_game(game)
                if engine.legal_move_mask(state, owner) == 0:
                    # A pass is not a choice, so it is not a decision and is not
                    # recorded as one. Counted, because how much of an episode
                    # was spent skipping is worth knowing.
                    eps[owner].learner_passes += 1
                    if record_moves:
                        log_move(eps, (owner, None, None, None, None))
                    game.act_pass()
                    continue
                record(owner, state, legal_indices(state, owner), ply)
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
                        log_move(eps, (owner,) + tuple(move))
                    game.act(*move)
                else:
                    if record_moves:
                        log_move(eps, (owner, None, None, None, None))
                    game.act_pass()
            ply += 1
        for ep in eps.values():
            ep.turns = ply

    remaining = tuple(game.remaining_cells(o) for o in range(4))
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
    rewards = tuple(episode_reward(remaining, o) for o in range(4))
    for ep in eps.values():
        ep.remaining = remaining
        ep.ranks = tuple(ranks)
        ep.points = tuple(points)
        ep.rewards = rewards
    return eps


def _learner_streams(seed, seats):
    """`{seat: rng}` for the network seats' own action sampling.

    One network seat keeps `seed * 2 + 1`, the stream every published batch was
    recorded with. Switching it would change every trajectory the existing
    evidence rests on and buys nothing, so the single-seat path is frozen.

    Two or more get a generator each, keyed injectively by `(seat, seed)` in a
    range `seed * 2` and `seed * 2 + 1` can never reach - the setup and opponent
    stream must not be able to collide with a sampling stream, or a learner's
    draw would shift the colours. The stride is `2**24` and every seed this
    module claims is below it, so two seats cannot collide either.
    """
    if len(seats) == 1:
        return {seats[0]: random.Random(seed * 2 + 1)}
    return {s: random.Random(_LEARNER_STREAM_BASE + s * _LEARNER_STREAM_STRIDE
                             + int(seed))
            for s in seats}


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
    with exactly the weights it would score `rl_h1000_0k` with. The RL seat is an
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

    `rl_h1000_0k`'s weights are loaded in passing. That is nearly free -
    `rl.imitation._BLOB_CACHE` reads the file once per process, not once per
    game - and it buys not touching `Game.setup_seats`, which is the seat-layout
    code `plan8-A` found should stay the single definition.

    The stand-in is the first H-C2 seat - `rl_h1000_0k`, which the retired
    `hc_1000` used to resolve to - rather than a personality on purpose: if the
    swap were ever missed, a personality would silently play the learner's moves
    and the run would look fine. A wrong stand-in's moves are at least a
    different policy from round one.

    Since plan9 task 4 that name has left the roster, so the fallback is no
    longer a seat on its own: it is only reached when neither `learner_key` nor
    the network's own `key` is one. Checking it here rather than letting
    `Game.setup_seats` reject it keeps the failure where the cause is - the
    caller forgot to name or bind the learner - instead of surfacing as a
    roster complaint about a seat nobody just asked for. Binding the file with
    `seats.register_adhoc` makes the old name a seat again and the fallback
    keeps answering.
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
    key = seats.imitation_key(seats.IMITATION_STEPS[0])
    try:
        seats.kind_of(key)
    except ValueError:
        raise ValueError(
            "no seat key to stand the learner in under: %r left the roster "
            "when plan9 task 4 retired the 0k checkpoints, and neither "
            "learner_key nor net.key names a seat. Pass learner_key, set "
            "net.key, or bind the file with seats.register_adhoc" % (key,))
    return key, True


def resolve_opponent_pool(opponent_pool):
    """`(pool, distinct)` - the seat keys to draw from, and how to draw them.

    `None` is the seven personalities drawn **without** replacement: that is
    the rule every published batch was recorded under, so the default cannot be
    expressed as a parameter default - it has to stay a separate branch whose
    draw sequence is untouched. A caller's pool is drawn **with** replacement,
    plan10 §1 and §4's rule, because a pool that mixes personalities with
    frozen policies may legitimately repeat an opponent and one seat named
    three times should appear three times.

    Every key is checked against `seats.automated_options()` through
    `canonical_key`, so an alias is accepted the way it is everywhere else, a
    retired or misspelled key is refused here rather than three seats into a
    round, and a human seat cannot be frozen into a training pool at all.
    Canonicalisation is only the *check*: the caller's own spelling is what the
    specs record, so `spec.names`, `opponent_coverage` and `play_episode`'s
    membership test all see the same string.
    """
    import seats
    if opponent_pool is None:
        return list(PERSONALITY_POOL), True
    pool = list(opponent_pool)
    if not pool:
        raise ValueError("opponent_pool must name at least one seat")
    known = seats.automated_options()
    canonical = []
    for key in pool:
        resolved = seats.canonical_key(key)
        if resolved not in known:
            raise ValueError(
                "unknown seat key %r in opponent_pool; it is not an "
                "automated option" % (key,))
        canonical.append(resolved)
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate seat key in opponent_pool: %r"
                         % (opponent_pool,))
    return pool, False


# --------------------------------------------------------------------------
# the balanced schedule (--balanced-schedule)
# --------------------------------------------------------------------------

def balanced_cells(round_no, n_pool, games):
    """`cells[s][p]` for one round: how often pool seat `p` sits opposite
    learner seat `s` this round, as a closed form of the round number.

    The schedule is a function of `(round_no, n_pool, games)` alone - no state,
    no replay of earlier rounds - so a resumed run continues the same sequence
    bit-for-bit and two runs with the same round numbers agree.

    The arithmetic. Each learner seat plays `games / 4` games of three
    opponents, so the row target is `W = 3 * games / 4`, which is why the
    schedule demands `games % 4 == 0`. Writing `W = base * n_pool + rho`, cell
    `(s, p)` carries +1 in round `q` when its phase steps over the `rho/n_pool`
    threshold, i.e. when `(t + a[s][p]) % n_pool >= n_pool - rho` with
    `t = ((q - 1) * rho) % n_pool`. Phases are

        a[s][p] = (p + n_pool * s // 4) % n_pool

    and the two properties the decree wants fall out of that choice:

      * every row of `a` is a permutation of `0..n_pool-1`, so each learner
        seat's row sum is exactly `W` in every round;
      * the four shifts are Beatty points of the 4-partition, which balances
        both the per-round column sums and the *cumulative* ones - after `q`
        rounds a column's total is a fixed constant plus the count of its
        phases at or above one threshold, and every threshold the schedule can
        ask about comes out balanced to a spread of at most one.

    `_check_balanced` re-verifies both gates on every call: the formulas are
    short enough to be wrong for a pool size nobody has tried, and a schedule
    that is merely *close* would still train - it would just quietly be the
    uniform draw again.

    Raises `ValueError` for a shape the construction cannot serve: games not
    divisible by 4, or fewer than 3 pool seats (three *distinct* opponents per
    game is the rule).
    """
    games, n_pool = int(games), int(n_pool)
    if games % 4:
        raise ValueError("balanced schedule needs games divisible by 4, "
                         "got %d" % games)
    if n_pool < 3:
        raise ValueError("balanced schedule needs at least 3 pool seats, "
                         "got %d" % n_pool)
    row_target = 3 * games // 4
    base, rho = divmod(row_target, n_pool)
    t = ((int(round_no) - 1) * rho) % n_pool
    phases = _balanced_phases(n_pool)
    cells = [[base + (1 if (t + phases[s][p]) % n_pool >= n_pool - rho else 0)
              for p in range(n_pool)] for s in range(4)]
    _check_balanced(n_pool, games, cells, phases, row_target)
    return cells


def _balanced_phases(n_pool):
    """The four phase rows, each a permutation of `0..n_pool-1`."""
    return [[(p + (n_pool * s) // 4) % n_pool for p in range(n_pool)]
            for s in range(4)]


def _check_balanced(n_pool, games, cells, phases, row_target):
    """Refuse a shape whose per-round *or* cumulative balance does not hold.

    Two gates because they protect two different claims. The per-round gate is
    the round itself: row sums exact, pool-column sums spread <= 1, cell values
    within a column spread <= 1. The threshold gate is what keeps the
    *cumulative* matrix inside one: after `q` rounds the total for pool seat
    `p` is a constant plus `#{s : a[s][p] >= n_pool - t}` for some `t` the
    round number reaches, so balancing every reachable threshold balances every
    prefix, not merely the ones somebody happened to test.
    """
    for s in range(4):
        if sorted(phases[s]) != list(range(n_pool)):
            raise ValueError("balanced schedule does not verify: phase row %d "
                             "is not a permutation" % s)
        if sum(cells[s]) != row_target:
            raise ValueError("balanced schedule does not verify: learner seat "
                             "%d plays %d of %d opponent slots"
                             % (s, sum(cells[s]), row_target))
    for p in range(n_pool):
        col = [cells[s][p] for s in range(4)]
        if max(col) - min(col) > 1:
            raise ValueError("balanced schedule does not verify: pool seat %d "
                             "sits opposite seat counts %s in one round"
                             % (p, col))
    cols = [sum(cells[s][p] for s in range(4)) for p in range(n_pool)]
    if max(cols) - min(cols) > 1:
        raise ValueError("balanced schedule does not verify: pool columns %s"
                         % cols)
    rho = row_target % n_pool
    period = n_pool // gcd(rho, n_pool) if rho else 1
    for k in range(period):
        t = (k * rho) % n_pool
        prof = [sum(1 for s in range(4) if phases[s][p] >= n_pool - t)
                for p in range(n_pool)]
        if max(prof) - min(prof) > 1:
            raise ValueError(
                "balanced schedule does not verify for %d pool seats with %d "
                "games: cumulative pool totals %s at threshold %d"
                % (n_pool, games, prof, n_pool - t))


def _seat_triples(counts, games, rng):
    """`counts` split into `games` triples of three *distinct* pool seats.

    The constructive side of Gale-Ryser: seats largest-count first, each into
    the still-emptiest games that do not already hold it. `counts` comes from
    `balanced_cells`, so `sum(counts) == 3 * games` and `max(counts) <= games`
    already hold; the checks below state the contract anyway, because a triple
    with a repeated seat is a different game and the cell matrix would be
    lying about it.

    `rng` only breaks ties between equally-counted seats. Which seats pair up
    is the cell matrix's answer, not the rng's - but the tie-break still has to
    move between rounds, or the constant-cell shape (pool 10, 1000 games) would
    deal the identical 1,000 games every single round.
    """
    n_pool = len(counts)
    if sum(counts) != 3 * games:
        raise ValueError("seat needs %d opponent slots over %d games, got %s"
                         % (3 * games, games, sum(counts)))
    if games and max(counts) > games:
        raise ValueError("pool seat with %d slots cannot fit in %d games "
                         "without repeating inside one game"
                         % (max(counts), games))
    game_cap = [3] * games
    members = [[] for _ in range(games)]
    order = sorted(range(n_pool), key=lambda p: (-counts[p], rng.random()))
    for p in order:
        need = counts[p]
        if not need:
            continue
        cands = sorted((g for g in range(games) if p not in members[g]),
                       key=lambda g: (-game_cap[g], g))
        if len(cands) < need:
            raise ValueError("no room for pool seat %d in %d games"
                             % (p, games))
        for g in cands[:need]:
            members[g].append(p)
            game_cap[g] -= 1
    if any(game_cap):
        raise ValueError("games left unfilled: %s" % game_cap)
    triples = [tuple(m) for m in members]
    for t in triples:
        if len(set(t)) != 3:
            raise ValueError("game without three distinct opponents: %s" % (t,))
    return triples


def _balanced_assignments(cells, games, rng):
    """Shuffled `(learner_seat, triple-of-pool-indices)` for every game.

    The flat list is `games` long and each seat owns exactly `games / 4` of it;
    the shuffle is what stops a seed block from being a run of seat-0 games
    followed by seat-1 games, and it runs off the same per-round rng as the
    tie-breaks, so `(rng_seed, round_no)` still decides everything.
    """
    assignments = []
    per_seat = games // 4
    for seat in range(4):
        for triple in _seat_triples(cells[seat], per_seat, rng):
            assignments.append((seat, triple))
    rng.shuffle(assignments)
    return assignments


def make_specs(start_seed, n, rng_seed=None, own=None, opponent_pool=None,
               balanced=False, round_no=None):
    """`(specs, manifest)` for `n` episodes from `start_seed` upwards.

    Each seed contributes one episode. The three opponents are drawn from
    `PERSONALITY_POOL` **without** replacement, or from `opponent_pool`
    **with** replacement - see `resolve_opponent_pool`. The learner's seat is
    uniform over the four, both from a `random.Random` seeded once by
    `rng_seed` - so the whole assignment is a function of one integer and is
    reproducible without replaying the draw.

    Leaving `opponent_pool` as `None` keeps the draw sequence bit-for-bit what
    it was before the parameter existed: the pool is resolved *before* the
    loop and the loop still makes exactly two calls to `rng` per seed.

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

    `balanced=True` swaps the draw for one round of the balanced schedule (see
    `balanced_cells`): `round_no` - 1-based, the training loop's round number -
    selects the cell matrix, each game gets three distinct pool seats from it,
    every learner seat plays exactly `n / 4` games, and the game order is
    shuffled from an rng mixed from `rng_seed` and `round_no`, so the
    assignment is still a function of integers and replays exactly. The
    manifest then also carries `balanced`, `round_no` and `cells`. Off is the
    default and off is the old draw, byte for byte: `eval_specs` never passes
    the flag at all, so the in-training evaluation stays comparable across
    both.
    """
    from rl.paired3 import reject_reserved_seeds

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive, got %d" % n)
    # The pool is a configuration error and the seed range is a claim, so the
    # pool is checked first: a caller who misspelled a key should be told that
    # and not sent off to pick a different seed block first.
    pool, distinct = resolve_opponent_pool(opponent_pool)
    cells = None
    if balanced:
        if round_no is None:
            raise ValueError("balanced=True needs the round number whose "
                             "schedule to build")
        # Deliberately before the seed guard: a shape error is configuration,
        # and reporting it after a claim error would put the wrong fact first.
        cells = balanced_cells(round_no, len(pool), n)
    start_seed, end = int(start_seed), int(start_seed) + n - 1
    if rng_seed is None:
        rng_seed = start_seed
    own_block = RL_TRAIN_BLOCK if own is None else own
    reject_reserved_seeds(start_seed, end, own=own_block)
    specs = []
    seat_counts = [0, 0, 0, 0]
    if balanced:
        # Mixed rather than seeded with rng_seed alone: two rounds of one run
        # must differ, and the multiplier is not `_round_rng`'s (seed * 1000)
        # so the game order never lines up with the minibatch order by
        # construction.
        rng = random.Random(int(rng_seed) * 1_000_003 + int(round_no))
        assignments = _balanced_assignments(cells, n, rng)
        for i, (seat, idxs) in enumerate(assignments):
            seat_counts[seat] += 1
            specs.append(Spec(start_seed + i, seat,
                              tuple(pool[j] for j in idxs)))
    else:
        rng = random.Random(int(rng_seed))
        for seed in range(start_seed, end + 1):
            if distinct:
                names = tuple(rng.sample(pool, 3))
            else:
                names = tuple(rng.choice(pool) for _ in range(3))
            seat = rng.randrange(4)
            seat_counts[seat] += 1
            specs.append(Spec(seed, seat, names))
    manifest = {
        "start_seed": start_seed,
        "end_seed": end,
        "games": n,
        "rng_seed": int(rng_seed),
        # The key is historical: it holds the pool the draw came from, which
        # for the default is `PERSONALITY_POOL` and for a caller is that
        # caller's pool. Renaming it would move the field on every published
        # manifest, and nothing reads it by name.
        "personality_pool": list(pool),
        "opponents_per_game": 3,
        "learner_seat_counts": seat_counts,
        "learner_seat_share": [c / float(n) for c in seat_counts],
        "opponent_coverage": {name: _count(name, specs) for name in pool},
        # the block this call claimed, which is the row the guard exempted
        "reserved_range_checked": list(own_block),
    }
    if balanced:
        # Only under the flag: a default manifest must stay exactly the shape
        # every published one already has.
        manifest["balanced"] = True
        manifest["round_no"] = int(round_no)
        manifest["cells"] = [list(row) for row in cells]
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


def _learner_net(weights_path, device):
    """The one network a worker keeps for `weights_path`, loading it once.

    A worker that inherited a network through `fork` would be sharing its memory
    with a parent that is about to exit, so every worker reads the file itself.
    Keyed by path rather than by "the last one", because a same-table worker
    holds four at a time and a single-slot cache would reload all four for every
    game.
    """
    cache = _WORKER.setdefault("nets", {})
    if weights_path not in cache:
        from rl.policy import load_policy
        net, meta = load_policy(weights_path, device=device)
        key = _key_for(weights_path, meta)
        if key:
            net.key = key
        cache[weights_path] = net
    return cache[weights_path]


def _worker(job):
    """Play a chunk of specs, loading the weights once per process."""
    (specs, weights_path, device, temperature, mode, checkpoint_dir, pool,
     learner_key) = job
    net = _learner_net(weights_path, device)
    return [play_episode(s, net, device=device, temperature=temperature,
                         mode=mode, checkpoint_dir=checkpoint_dir,
                         opponent_pool=pool, learner_key=learner_key)
            for s in specs]


def _paired_worker(job):
    """Play a chunk of specs once per learner, loading each learner's weights.

    The four copies of one fixed-pool game share the spec, and therefore the
    seed, the opponents, the colours and the opening player - which is what
    makes them paired. They differ only in which network sat down.
    """
    (specs, weight_paths, device, temperature, mode, checkpoint_dir, pool,
     learner_keys) = job
    nets = tuple(_learner_net(p, device) for p in weight_paths)
    out = []
    for s in specs:
        out.append({i: play_episode(s, net, device=device,
                                    temperature=temperature, mode=mode,
                                    checkpoint_dir=checkpoint_dir,
                                    opponent_pool=pool,
                                    learner_key=learner_keys[i])
                    for i, net in enumerate(nets)})
    return out


def _table_worker(job):
    """Play a chunk of same-table specs, loading each learner's weights once."""
    specs, weight_paths, device, temperature, mode = job
    nets = tuple(_learner_net(p, device) for p in weight_paths)
    out = []
    for spec in specs:
        seat_nets = {seat: nets[spec.order[seat]] for seat in range(4)}
        eps = play_game(list(spec.keys), seat_nets, spec.seed, device=device,
                        temperature=temperature, mode=mode)
        # keyed by *learner*, not by seat: the caller groups a step's episodes
        # per learner, and a seat-keyed dict would have to be remapped anyway.
        out.append({spec.order[seat]: ep for seat, ep in eps.items()})
    return out


def _key_for(weights_path, meta):
    """The seat key for a checkpoint file, or `None` when it has no name yet.

    `rl_h1000_0k` / `hc_2000` / `hc_10000` when the file is one of the pool's.
    A training checkpoint has no registered seat name, and that is not this
    module's to invent - so it returns `None` and `seat_for` substitutes the
    stand-in. Inventing `rl_<step>` here would put a name in the logs and in
    `Game.owner_key` that no seat registry knows about, which is how a rollout
    ends up asking for a seat that does not exist.
    """
    base = os.path.basename(weights_path)
    import seats
    for s in seats.IMITATION_STEPS:
        if base == "step_%06d.pt" % s:
            return seats.imitation_key(s)
    return None


def play_batch(specs, weights_path, n_procs=1, device=None,
               temperature=DEFAULT_TEMPERATURE, mode=LEARNER_MODE,
               checkpoint_dir=None, opponent_pool=None, learner_key=None):
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

    `opponent_pool` is handed straight to `play_episode`: `None` for the seven
    personalities, or an explicit pool for plan10's fixed eleven.
    `learner_key` names the seat the learner stands in under - required now
    that `seat_for`'s fallback left the roster, and the reason a training
    checkpoint can be played without binding a retired seat.
    """
    specs = list(specs)
    if not specs:
        return []
    _refuse_cuda(device, n_procs)
    with _ThreadBudget(1):
        if n_procs <= 1:
            net = _learner_net(weights_path, device)
            return [play_episode(s, net, device=device,
                                 temperature=temperature, mode=mode,
                                 checkpoint_dir=checkpoint_dir,
                                 opponent_pool=opponent_pool,
                                 learner_key=learner_key)
                    for s in specs]
        per = max(1, -(-len(specs) // int(n_procs)))
        chunks = [specs[i:i + per] for i in range(0, len(specs), per)]
        jobs = [(chunk, weights_path, device, temperature, mode, checkpoint_dir,
                 opponent_pool, learner_key) for chunk in chunks]
        results = _pool_map(_worker, jobs)
    # `map` preserves input order, and the chunks were cut in order too, so
    # flattening is already spec order. The length check is here because a
    # silently short result would index-align against the wrong spec for every
    # episode after it, which is a far worse failure than an exception.
    out = [ep for chunk in results for ep in chunk]
    if len(out) != len(specs):
        raise RuntimeError("expected %d episodes, got %d"
                           % (len(specs), len(out)))
    return out


def play_table_batch(specs, weight_paths, n_procs=1, device=None,
                     temperature=DEFAULT_TEMPERATURE, mode=LEARNER_MODE):
    """Play every same-table spec, returning one row per spec of
    `{learner: Episode}`.

    `weight_paths` is the four learners' checkpoint files in **learner** order,
    and `spec.order` says which learner sits at which seat - so a worker holds
    all four networks and one game produces four episodes, one per learner.
    `spec.keys` is the seat key each seat is set up under, which is what makes a
    table's `Game.owner_key` readable and what a replay would have to name.

    Same ordering guarantee as `play_batch`: the result is in spec order
    whatever the scheduling did, and nothing about it depends on `n_procs`.
    """
    specs = list(specs)
    if not specs:
        return []
    if len(weight_paths) != 4:
        raise ValueError("a table needs four weight paths, got %d"
                         % len(weight_paths))
    _refuse_cuda(device, n_procs)
    with _ThreadBudget(1):
        if n_procs <= 1:
            nets = tuple(_learner_net(p, device) for p in weight_paths)
            rows = []
            for spec in specs:
                seat_nets = {seat: nets[spec.order[seat]] for seat in range(4)}
                eps = play_game(list(spec.keys), seat_nets, spec.seed,
                                device=device, temperature=temperature,
                                mode=mode)
                rows.append({spec.order[seat]: ep for seat, ep in eps.items()})
            return rows
        per = max(1, -(-len(specs) // int(n_procs)))
        chunks = [specs[i:i + per] for i in range(0, len(specs), per)]
        jobs = [(chunk, tuple(weight_paths), device, temperature, mode)
                for chunk in chunks]
        results = _pool_map(_table_worker, jobs)
    out = [row for chunk in results for row in chunk]
    if len(out) != len(specs):
        raise RuntimeError("expected %d tables, got %d"
                           % (len(specs), len(out)))
    return out


def play_paired_batch(specs, weight_paths, learner_keys, n_procs=1, device=None,
                      temperature=DEFAULT_TEMPERATURE, mode=LEARNER_MODE,
                      checkpoint_dir=None, opponent_pool=None):
    """Play every spec **once per learner**, returning one row per spec of
    `{learner: Episode}`.

    Same specs, same seeds, four different networks - the pairing plan10 asks
    for on its fixed-pool games. One process pool for all four learners rather
    than one per learner: a pool costs a fork per worker and a model load per
    worker, and four pools in a row paid both four times over.
    """
    specs = list(specs)
    if not specs:
        return []
    if len(weight_paths) != 4 or len(learner_keys) != 4:
        raise ValueError("a paired batch needs four paths and four keys, got "
                         "%d and %d" % (len(weight_paths), len(learner_keys)))
    _refuse_cuda(device, n_procs)
    with _ThreadBudget(1):
        if n_procs <= 1:
            nets = tuple(_learner_net(p, device) for p in weight_paths)
            return [{i: play_episode(s, net, device=device,
                                     temperature=temperature, mode=mode,
                                     checkpoint_dir=checkpoint_dir,
                                     opponent_pool=opponent_pool,
                                     learner_key=learner_keys[i])
                     for i, net in enumerate(nets)} for s in specs]
        per = max(1, -(-len(specs) // int(n_procs)))
        chunks = [specs[i:i + per] for i in range(0, len(specs), per)]
        jobs = [(chunk, tuple(weight_paths), device, temperature, mode,
                 checkpoint_dir, opponent_pool, tuple(learner_keys))
                for chunk in chunks]
        results = _pool_map(_paired_worker, jobs)
    out = [row for chunk in results for row in chunk]
    if len(out) != len(specs):
        raise RuntimeError("expected %d paired games, got %d"
                           % (len(specs), len(out)))
    return out


def _refuse_cuda(device, n_procs):
    if device is not None and n_procs > 1:
        import torch
        if torch.device(device).type == "cuda":
            raise ValueError("device=%r with n_procs=%d would give every worker "
                             "its own CUDA context on one GPU; pass device=None "
                             "and let each worker choose, or run on CPU"
                             % (device, n_procs))


def _pool_map(worker, jobs):
    with mp.get_context("fork").Pool(len(jobs),
                                     initializer=_worker_init) as pool:
        return pool.map(worker, jobs)


def thread_report():
    """What the worker initialiser sets, for a manifest."""
    return {"torch_num_threads": 1,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset")}


if __name__ == "__main__":       # pragma: no cover
    print(__doc__)
    print("personality pool:", PERSONALITY_POOL)
    print("seed base:", RL_SEED_BASE)
    sys.exit(0)