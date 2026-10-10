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

# The three seed blocks plan10 reserves. 100 steps is the whole plan (150k);
# this run stops at step 100.
TABLE_SEED_BASE = 7_300_000
TABLE_SEED_SPAN = 1_000         # 100 steps x 10 rounds
FIXED_SEED_BASE = 7_310_000
FIXED_SEED_SPAN = 26_000        # 100 steps x 260 games
EVAL_SEED_BASE = 7_400_000
EVAL_SEED_SPAN = 1_000

TABLE_BLOCK = ("stage RL multiple table", TABLE_SEED_BASE,
               TABLE_SEED_BASE + TABLE_SEED_SPAN - 1)
FIXED_BLOCK = ("stage RL multiple fixed", FIXED_SEED_BASE,
               FIXED_SEED_BASE + FIXED_SEED_SPAN - 1)
EVAL_BLOCK = ("stage RL multiple validation", EVAL_SEED_BASE,
              EVAL_SEED_BASE + EVAL_SEED_SPAN - 1)


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


def table_seed(step, round_no):
    """The seed of one round of one step. A pure function, not a cursor."""
    if not 1 <= round_no <= ROUNDS_PER_STEP:
        raise ValueError("round %r is not one of the %d per step"
                         % (round_no, ROUNDS_PER_STEP))
    return TABLE_SEED_BASE + (int(step) - 1) * ROUNDS_PER_STEP + round_no - 1


def fixed_seed(step, game):
    """The seed of one fixed-pool game of one step."""
    if not 0 <= game < GAMES_PER_STEP:
        raise ValueError("game %r is not one of the %d per step"
                         % (game, GAMES_PER_STEP))
    return FIXED_SEED_BASE + (int(step) - 1) * GAMES_PER_STEP + game


def check_step(step, max_step=100):
    """Reject a step whose seeds would fall outside the reserved blocks."""
    step = int(step)
    if not 1 <= step <= int(max_step):
        raise ValueError("step %r is outside 1..%d" % (step, max_step))
    hi = fixed_seed(step, GAMES_PER_STEP - 1)
    if hi > FIXED_BLOCK[2]:
        raise ValueError("step %d needs seeds up to %d, past the reserved "
                         "block's %d" % (step, hi, FIXED_BLOCK[2]))
    return step


def _claim(block, lo, hi):
    """Run the reserved-seed guard over `[lo, hi]`, claiming `block`.

    Same guard `make_specs` uses, for the same reason: a range that lands
    inside another stage's reserved row is a mistake to catch before a single
    game is played, not after a thousand are on disk.
    """
    from rl.paired3 import reject_reserved_seeds
    reject_reserved_seeds(lo, hi, own=block)


def make_table_specs(step, rounds=ROUNDS_PER_STEP):
    """Every same-table game of one step, in `(round, seat order)` order.

    One seed per round, shared by all 24 orders - see the module docstring for
    why that is the design rather than an omission. The result is a function of
    `step` alone: no rng is seeded, and the permutation order is `PERMS`.
    """
    check_step(step)
    rounds = int(rounds)
    if not 1 <= rounds <= ROUNDS_PER_STEP:
        raise ValueError("rounds must be within 1..%d, got %d"
                         % (ROUNDS_PER_STEP, rounds))
    _claim(TABLE_BLOCK, table_seed(step, 1),
           table_seed(step, rounds))
    specs = []
    for r in range(rounds):
        seed = table_seed(step, r + 1)
        for order in PERMS:
            keys = tuple(LEARNER_KEYS[o] for o in order)
            specs.append(TableSpec(seed, order, keys))
    return specs


def make_fixed_specs(step, games=GAMES_PER_STEP, rng_seed=None):
    """`(specs, manifest)` for one step's fixed-pool games.

    Three opponents are drawn **with replacement** from `FIXED_POOL` - plan10
    says 有放回, so the same frozen policy can sit twice at one table and a
    learner can draw its own starting weights as an opponent. The seat is
    `game % 4`, so a learner plays each of the four seats exactly 65 times in a
    step and the four learners' copies of a game share it; that sharing is what
    makes the copies paired.

    `rng_seed` defaults to the step's first seed, following `make_specs`. Pass
    it explicitly to get a second, different opponent schedule over the same
    board seeds.
    """
    import random
    check_step(step)
    games = int(games)
    if not 1 <= games <= GAMES_PER_STEP:
        raise ValueError("games must be within 1..%d, got %d"
                         % (GAMES_PER_STEP, games))
    _claim(FIXED_BLOCK, fixed_seed(step, 0), fixed_seed(step, games - 1))
    rng = random.Random(fixed_seed(step, 0) if rng_seed is None
                        else int(rng_seed))
    specs = []
    coverage = {k: 0 for k in FIXED_POOL}
    for g in range(games):
        names = tuple(rng.choice(FIXED_POOL) for _ in range(3))
        for n in names:
            coverage[n] += 1
        specs.append(Spec(fixed_seed(step, g), g % 4, names))
    manifest = {
        "step": int(step),
        "pool": list(FIXED_POOL),
        "opponents_per_game": 3,
        "with_replacement": True,
        "seat_rule": "game % 4",
        "rng_seed": fixed_seed(step, 0) if rng_seed is None else int(rng_seed),
        "opponent_coverage": coverage,
        "seed_lo": fixed_seed(step, 0),
        "seed_hi": fixed_seed(step, games - 1),
    }
    return specs, manifest


def make_eval_specs(games, rng_seed=None):
    """The **fixed** opponent schedule the progress curve is measured on.

    Same generator shape as `make_fixed_specs` but anchored to the validation
    block and independent of the step, so every evaluation in a run sees the
    same opponents, the same seats and the same opening player. That is the
    whole point: plan10's risk 1 says the on-policy curves cannot be compared
    across time, and §6 asks for a fixed seed set and a fixed pool.
    """
    import random
    games = int(games)
    if not 1 <= games <= EVAL_SEED_SPAN:
        raise ValueError("games must be within 1..%d, got %d"
                         % (EVAL_SEED_SPAN, games))
    _claim(EVAL_BLOCK, EVAL_SEED_BASE, EVAL_SEED_BASE + games - 1)
    rng = random.Random(EVAL_SEED_BASE if rng_seed is None else int(rng_seed))
    specs = []
    for g in range(games):
        names = tuple(rng.choice(FIXED_POOL) for _ in range(3))
        specs.append(Spec(EVAL_SEED_BASE + g, g % 4, names))
    return specs
