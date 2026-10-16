"""This layer answers one question: what are plan10's four learners seated at,
and which seed decides it?

plan10 trains four policies at once - h / o / b / i, each continuing from its
own frozen 20k - under two environments per step:

  * **same-table** (動態同伴): all four sit at one table, so every game yields
    four learner trajectories and the opponents are each other. 240 games a
    step, which is all 4! = 24 seat orders crossed with 10 rounds, **one seed
    per round** shared by the 24 orders. That sharing is the design, not a
    shortcut: with four network seats `Game.setup_seats` draws the colours and
    the opening player and nothing else, so one seed means one setup and the 24
    games differ only in who sits where - a blocked comparison over seat
    arrangement. plan10's own risk 9 names the cost ("每步種子數 10 組").
  * **fixed pool** (對固定池): the learner is alone against three opponents
    drawn **with replacement** from `FIXED_POOL`. 260 games a step, and each is
    played four times - once per learner, all four sitting in the **same**
    seat - so the four copies share the seed and therefore the opponents, the
    colours and the opening player. 260 x 4 = 1040 trajectories.

  240 x 4 + 260 x 4 = 2000 learner trajectories a step, 500 of them each.

Everything here is a pure function of `(step, index)`: no rng is carried, and
`make_table_specs(step)` regenerates a step's layout without replaying the
steps before it. The seed blocks are registered in `rl.paired3.RESERVED_RANGES`
and mirrored here as `*_BLOCK` tuples, because `reject_reserved_seeds` exempts a
claim by **exact triple** - a label or a bound that drifts would make every run
refuse to start against its own reserved row.
"""
import itertools
from typing import NamedTuple

from rl.rollout import Spec

# The four learners, in the order every table, every checkpoint directory and
# every log row uses. Learner index, not seat: `TableSpec.order` maps between
# the two.
LEARNERS = ("h", "o", "b", "i")
LEARNER_KEYS = ("rl_h1000_20k", "rl_o1000_20k", "rl_b1000_20k",
                "rl_i1000_20k")
# The same four learners' published 50k, in the same order. Not the learners'
# weights during training - this is the milestone a later run is compared
# against, and the name it is registered under.
LEARNER_KEYS_50K = ("rl_h1000_50k", "rl_o1000_50k", "rl_b1000_50k",
                    "rl_i1000_50k")

# The fixed pool: the seven personalities plus the four frozen 20k. Written out
# rather than derived from `seats.automated_options()`, because the pool is a
# decision and reading it from the roster would let a future roster change walk
# into a training run. Sorted by key - the same rule `pool_order` uses - so the
# order is a function of the set and not of who typed it.
FIXED_POOL = ("builder", "chess", "fox", "hunter", "intruder", "optimizer",
              "rl_b1000_20k", "rl_h1000_20k", "rl_i1000_20k",
              "rl_o1000_20k", "wolf")

# Every seat order, once. `itertools.permutations` is deterministic and the
# order of this tuple is part of the protocol: a spec's index within a round is
# `PERMS.index(order)`, so changing it would relabel every table the evidence
# was recorded on.
PERMS = tuple(itertools.permutations(range(4)))

ROUNDS_PER_STEP = 10            # plan10: 24 orders x 10 rounds = 240 games
GAMES_PER_STEP = 260            # plan10: fixed-pool games, x 4 learners

# plan10's seed layout, for the whole 200-step plan (steps 41..200 is 20k ->
# 100k; a step is 500 games a learner, so 100 steps is 50k).
#
# Each stream gets **two** reserved rows rather than one long one, and the
# split is at step 100. `reject_reserved_seeds` exempts a claim by exact
# triple, so a block constant that is widened past its registered row stops
# matching it - and the *old* steps then overlap their own row and refuse to
# start. Two literals are the price of keeping steps 1..100 replayable under
# the same constants the 50k evidence was recorded on.
#
# The blocks below are therefore literals, not `base + span`, and the plan's
# length is policed by `PLAN_STEPS` in `check_step` rather than by the size of
# any block. `*_SEED_BASE` stays because the seed formulas read it.
PLAN_STEPS = 200               # 20k -> 100k: 200 global steps of 500 games
TABLE_SEED_BASE = 7_300_000
FIXED_SEED_BASE = 7_310_000
EVAL_SEED_BASE = 7_400_000

TABLE_BLOCK = ("stage RL multiple table", 7_300_000, 7_300_999)
TABLE2_BLOCK = ("stage RL multiple table 2", 7_301_000, 7_301_999)
FIXED_BLOCK = ("stage RL multiple fixed", 7_310_000, 7_335_999)
FIXED2_BLOCK = ("stage RL multiple fixed 2", 7_336_000, 7_361_999)
EVAL_BLOCK = ("stage RL multiple validation", 7_400_000, 7_400_999)
# The large-sample evaluation of the 50k and 75k policies, on its own schedule
# so it cannot be confused with the 100-game progress curve's seeds.
BIG_EVAL_BLOCK = ("stage RL multiple big eval", 7_410_000, 7_411_999)
BIG_EVAL_SEED_BASE = 7_410_000

# The rollback rerun: steps 81..100 re-walked from the 40k checkpoint on seeds
# the original run never touched, to tell a one-off bad update apart from
# something the 80 -> 100 dynamics do every time.
#
# `*_SEED_BASE` here is the seed **step 1** would have had, so the existing
# `(step - 1) * stride` formula is reused untouched and only the anchor moves.
# It is arithmetic, never a claimed seed - what the guard sees is the range
# steps 81..100 actually produce, registered verbatim in `rl.paired3`. Both
# anchors therefore sit inside `stage RL multiple fixed 2`, which is harmless
# and deliberate: nothing ever asks the guard about them.
#
#   table  base 7_361_200  ->  steps 81..100 claim 7_362_000..7_362_199
#   fixed  base 7_342_200  ->  steps 81..100 claim 7_363_000..7_368_199
#   eval   base 7_369_000  ->  400 games claim     7_369_000..7_369_399
RB80_TABLE_SEED_BASE = 7_361_200
RB80_FIXED_SEED_BASE = 7_342_200

RB80_TABLE_BLOCK = ("stage RL rb80 table", 7_362_000, 7_362_199)
RB80_FIXED_BLOCK = ("stage RL rb80 fixed", 7_363_000, 7_368_199)
RB80_EVAL_BLOCK = ("stage RL rb80 big eval", 7_369_000, 7_369_399)

MULTI_BLOCKS = (TABLE_BLOCK, TABLE2_BLOCK, FIXED_BLOCK, FIXED2_BLOCK,
                EVAL_BLOCK, BIG_EVAL_BLOCK,
                RB80_TABLE_BLOCK, RB80_FIXED_BLOCK, RB80_EVAL_BLOCK)


def _table_block(step, base=None):
    """Which table row this step claims: by seed when a base is given, else by
    step, because the plan's first hundred and second hundred have their own."""
    if base is not None:
        return RB80_TABLE_BLOCK
    return TABLE_BLOCK if step <= 100 else TABLE2_BLOCK


def _fixed_block(step, base=None):
    if base is not None:
        return RB80_FIXED_BLOCK
    return FIXED_BLOCK if step <= 100 else FIXED2_BLOCK


class TableSpec(NamedTuple):
    """One same-table game: a round's seed and who sits where.

    `order[seat]` is the learner index at that seat and `keys[seat]` the option
    key that seat is set up under - which is that learner's own registered 20k,
    so a table's `Game.owner_key` reads as the policy each seat *started* from
    even though every one of them is replaced by its current weights.
    """

    seed: int
    order: tuple
    keys: tuple


def table_seed(step, round_no, base=None):
    """The seed of one round of one step. A pure function, not a cursor.

    `base` is the seed step 1 would have had; left alone it is the plan's own,
    and the answer is byte-identical to before this parameter existed.
    """
    if not 1 <= round_no <= ROUNDS_PER_STEP:
        raise ValueError("round %r is not one of %d per step"
                         % (round_no, ROUNDS_PER_STEP))
    base = TABLE_SEED_BASE if base is None else int(base)
    return base + (int(step) - 1) * ROUNDS_PER_STEP + round_no - 1


def fixed_seed(step, game, base=None):
    """The seed of one fixed-pool game of one step. See `table_seed`."""
    if not 0 <= game < GAMES_PER_STEP:
        raise ValueError("game %r is not one of the %d per step"
                         % (game, GAMES_PER_STEP))
    base = FIXED_SEED_BASE if base is None else int(base)
    return base + (int(step) - 1) * GAMES_PER_STEP + game


def check_table_step(step, max_step=PLAN_STEPS, base=None):
    """Reject a step whose table seeds would fall outside their row.

    Split from `check_fixed_step` because the two streams have their own
    anchors: the plan's, and the rollback rerun's, which differ. One `base`
    for both would make `make_table_specs` bound the table stream against the
    fixed row - and that is exactly how the first draft of this failed.
    """
    step = int(step)
    if not 1 <= step <= int(max_step):
        raise ValueError("step %r is outside 1..%d" % (step, max_step))
    hi = table_seed(step, ROUNDS_PER_STEP, base)
    block = _table_block(step, base)
    if hi > block[2]:
        raise ValueError("step %d needs table seeds up to %d, past the "
                         "reserved block's %d" % (step, hi, block[2]))
    return step


def check_fixed_step(step, max_step=PLAN_STEPS, base=None):
    """Reject a step whose fixed-pool seeds would fall outside their row."""
    step = int(step)
    if not 1 <= step <= int(max_step):
        raise ValueError("step %r is outside 1..%d" % (step, max_step))
    hi = fixed_seed(step, GAMES_PER_STEP - 1, base)
    block = _fixed_block(step, base)
    if hi > block[2]:
        raise ValueError("step %d needs fixed seeds up to %d, past the "
                         "reserved block's %d" % (step, hi, block[2]))
    return step


def check_step(step, max_step=PLAN_STEPS, base=None, table_base=None):
    """Both streams. `base` anchors the fixed one, `table_base` the table's;
    either may be left as the plan's own."""
    check_table_step(step, max_step, table_base)
    return check_fixed_step(step, max_step, base)


def check_range(lo_step, hi_step, base=None, table_base=None):
    """`check_step` over every step a run would touch, in one call.

    A run validates each step as it gets there, so a range that runs past the
    plan fails *after* the good steps have already been written. Checking the
    whole span up front turns that into a refusal before anything is touched -
    which is also what `--dry` needs, since dry-run never generates a spec.
    """
    lo_step, hi_step = int(lo_step), int(hi_step)
    if hi_step < lo_step:
        raise ValueError("step range %d..%d ends before it starts"
                         % (lo_step, hi_step))
    for step in range(lo_step, hi_step + 1):
        check_step(step, base=base, table_base=table_base)
    return hi_step


def _claim(block, lo, hi):
    """Run the reserved-seed guard over `[lo, hi]`, claiming `block`.

    Same guard `make_specs` uses, for the same reason: a range that lands
    inside another stage's reserved row is a mistake to catch before a single
    game is played, not after a thousand are on disk.
    """
    from rl.paired3 import reject_reserved_seeds
    reject_reserved_seeds(lo, hi, own=block)


def make_table_specs(step, rounds=ROUNDS_PER_STEP, base=None):
    """Every same-table game of one step, in `(round, seat order)` order.

    One seed per round, shared by all 24 orders - see the module docstring for
    why that is the design rather than an omission. The result is a function of
    `step` alone: no rng is seeded, and the permutation order is `PERMS`.
    `base` moves the whole stream to another reserved row; left alone the
    answer is byte-identical to before it existed.
    """
    check_table_step(step, base=base)
    rounds = int(rounds)
    if not 1 <= rounds <= ROUNDS_PER_STEP:
        raise ValueError("rounds must be within 1..%d, got %d"
                         % (ROUNDS_PER_STEP, rounds))
    _claim(_table_block(step, base), table_seed(step, 1, base),
           table_seed(step, rounds, base))
    specs = []
    for r in range(rounds):
        seed = table_seed(step, r + 1, base)
        for order in PERMS:
            keys = tuple(LEARNER_KEYS[o] for o in order)
            specs.append(TableSpec(seed, order, keys))
    return specs


def make_fixed_specs(step, games=GAMES_PER_STEP, rng_seed=None, base=None):
    """`(specs, manifest)` for one step's fixed-pool games.

    Three opponents are drawn **with replacement** from `FIXED_POOL` - plan10
    says 有放回, so the same frozen policy can sit twice at one table and a
    learner can draw its own starting weights as an opponent. The seat is
    `game % 4`, so a learner plays each of the four seats exactly 65 times in a
    step and the four learners' copies of a game share it; that sharing is what
    makes the copies paired.

    `rng_seed` defaults to the step's first seed, following `make_specs`. Pass
    it explicitly to get a second, different opponent schedule over the same
    board seeds. `base` moves the stream to another reserved row - see
    `make_table_specs`.
    """
    import random
    check_fixed_step(step, base=base)
    games = int(games)
    if not 1 <= games <= GAMES_PER_STEP:
        raise ValueError("games must be within 1..%d, got %d"
                         % (GAMES_PER_STEP, games))
    _claim(_fixed_block(step, base), fixed_seed(step, 0, base),
           fixed_seed(step, games - 1, base))
    rng = random.Random(fixed_seed(step, 0, base) if rng_seed is None
                        else int(rng_seed))
    specs = []
    coverage = {k: 0 for k in FIXED_POOL}
    for g in range(games):
        names = tuple(rng.choice(FIXED_POOL) for _ in range(3))
        for n in names:
            coverage[n] += 1
        specs.append(Spec(fixed_seed(step, g, base), g % 4, names))
    manifest = {
        "step": int(step),
        "pool": list(FIXED_POOL),
        "opponents_per_game": 3,
        "with_replacement": True,
        "seat_rule": "game % 4",
        "rng_seed": (fixed_seed(step, 0, base) if rng_seed is None
                     else int(rng_seed)),
        "opponent_coverage": coverage,
        "seed_lo": fixed_seed(step, 0, base),
        "seed_hi": fixed_seed(step, games - 1, base),
    }
    return specs, manifest


def make_eval_specs(games, rng_seed=None, seed_base=None, block=None):
    """The **fixed** opponent schedule a progress curve is measured on.

    Same generator shape as `make_fixed_specs` but anchored to a validation
    block and independent of the step, so every evaluation in a run sees the
    same opponents, the same seats and the same opening player. That is the
    whole point: plan10's risk 1 says the on-policy curves cannot be compared
    across time, and §6 asks for a fixed seed set and a fixed pool.

    `seed_base` / `block` move the whole schedule somewhere else - the 50k
    milestone's large-sample evaluation uses `BIG_EVAL_BLOCK`, so its 400-game
    schedule cannot be confused with the 100-game curve's. Both default to the
    validation row, which is exactly the schedule the existing evidence was
    recorded on; the specs a default call returns are unchanged.
    """
    import random
    games = int(games)
    base = EVAL_SEED_BASE if seed_base is None else int(seed_base)
    block = EVAL_BLOCK if block is None else tuple(block)
    span = block[2] - block[1] + 1
    if not 1 <= games <= span:
        raise ValueError("games must be within 1..%d, got %d" % (span, games))
    if not block[1] <= base <= block[2] - games + 1:
        raise ValueError("seed_base %d with %d games runs past block %r"
                         % (base, games, block))
    _claim(block, base, base + games - 1)
    rng = random.Random(base if rng_seed is None else int(rng_seed))
    specs = []
    for g in range(games):
        names = tuple(rng.choice(FIXED_POOL) for _ in range(3))
        specs.append(Spec(base + g, g % 4, names))
    return specs
