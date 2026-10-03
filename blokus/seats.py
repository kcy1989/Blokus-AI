"""Seat options, and the setup that turns four choices into a game.

A game has four seats. Each seat picks an *option* - who or what plays it - and
a *colour*. The two are independent, and that independence is the whole design:

  * An option may be repeated without limit. Four wolves, four humans or two
    recording humans are all legal. Options are never consumed by being chosen,
    so nothing has to be removed from the menu.
  * A colour may not. It is the one thing the four seats genuinely compete for,
    so a colour that one seat has taken is dropped from the other three menus.

The colour is only a label. A seat's *corner* is fixed by its owner id through
`config.OWNER_CORNER`, so which corner somebody sits at follows from which
colour they were given, and a seat left on "random" is dealt one of whatever
colours are left over.

Turn order is a separate, third random decision, and it is deliberately not
tied to either of the others. `draw_turn_order` returns a rotation of the one
clockwise cycle, which is the only kind of order the rules allow - a shuffle
would sometimes put two neighbouring turns on opposite corners. Only the corner
the rotation begins from is random, and that is precisely "who goes first".

This module imports neither torch nor pygame: `rl.imitation` is imported inside
`build_brain` so a game with no imitation seat never loads torch, and the UI,
the league and the tests can all use the same seat model.
"""
from ai import make_brain
from ai.registry import personality_keys
from config import CLOCKWISE_OWNERS, COLORS, OWNER_CORNER, PLAYER_OWNER

# The four colours, in the order the menus list them.
COLOR_NAMES = tuple(COLORS)

# What kind of contestant a seat holds. The kind, not the key, is what decides
# whether the seat needs a brain, may be logged, or waits for a human.
KIND_AI = "ai"
KIND_IMITATION = "imitation"
KIND_HUMAN = "human"
KIND_HUMAN_LOG = "human_log"

# The per-seat colour sentinel: "deal me one of what is left".
RANDOM = "random"

HUMAN_KEY = "human"
HUMAN_LOG_KEY = "human_log"

# Which H-B2 checkpoints are offered as players, in the order the menu lists
# them. Read-only at inference: the optimiser state in those files is ignored,
# and so is the fact that 38,000 was the last step trained.
IMITATION_STEPS = (2000, 10000, 22000, 38000)
IMITATION_CHECKPOINT_DIR = "data/hb2"

SEATS = 4


def imitation_key(step):
    """The option key for the checkpoint saved at `step`."""
    return "step_%d" % int(step)


def imitation_step(key):
    """The step an option key names. Inverse of `imitation_key`."""
    if not key.startswith("step_"):
        raise ValueError("%r is not an imitation option" % (key,))
    return int(key[len("step_"):])


def seat_options(include_humans=True):
    """Every option key, in menu order: the AI pool, then the imitation steps,
    then (optionally) the two human seats.

    11 without humans, 13 with them.
    """
    out = [k for k in personality_keys()]
    out += [imitation_key(s) for s in IMITATION_STEPS]
    if include_humans:
        out += [HUMAN_KEY, HUMAN_LOG_KEY]
    return tuple(out)


def kind_of(key, include_humans=True):
    """Which kind of contestant `key` names."""
    if key in personality_keys():
        return KIND_AI
    if key in (imitation_key(s) for s in IMITATION_STEPS):
        return KIND_IMITATION
    if key == HUMAN_KEY:
        return KIND_HUMAN
    if key == HUMAN_LOG_KEY:
        return KIND_HUMAN_LOG
    raise ValueError("no such seat option: %r" % (key,))


def is_human_kind(kind):
    """A seat a person plays, whether or not it is recorded."""
    return kind in (KIND_HUMAN, KIND_HUMAN_LOG)


def assign_colours(wanted, rng=None):
    """Resolve four per-seat colour wishes into four distinct colours.

    `wanted` is four entries, each either a colour name or `None`/`RANDOM` for
    "whatever is left". Specified colours must be distinct; unspecified seats
    are dealt the remaining colours in seat order. With `rng=None` the dealing
    is deterministic, which is what makes this testable.
    """
    wanted = list(wanted)
    if len(wanted) != SEATS:
        raise ValueError("a game has %d seats, got %d colours"
                         % (SEATS, len(wanted)))
    for c in wanted:
        if c not in (None, RANDOM) and c not in COLOR_NAMES:
            raise ValueError("no such colour: %r" % (c,))
    fixed = [c for c in wanted if c not in (None, RANDOM)]
    if len(set(fixed)) != len(fixed):
        dupes = sorted({c for c in fixed if fixed.count(c) > 1})
        raise ValueError("colours must be distinct, %r given more than once"
                         % (dupes,))
    rest = [c for c in COLOR_NAMES if c not in fixed]
    if rng is not None:
        rng.shuffle(rest)
    out = []
    for c in wanted:
        out.append(rest.pop(0) if c in (None, RANDOM) else c)
    return out


def colour_choices(taken, include_random=True):
    """The colours still on offer to one seat.

    A colour another seat holds is not on offer, which is the one thing the
    menus do remove. `RANDOM` is always available and is what a seat defaults to.
    """
    out = [RANDOM] if include_random else []
    out += [c for c in COLOR_NAMES if c not in taken]
    return out


def draw_turn_order(rng):
    """A random rotation of the one clockwise cycle.

    The direction is fixed; only the starting corner is random, and the corner
    it starts from is the player who moves first. This cannot be a shuffle:
    that would sometimes make two neighbouring turns sit on opposite corners,
    which is not the direction the rules define.

    The rotation is anchored on `PLAYER_OWNER` before the draw rather than on
    the draw itself. Adding a constant to a uniform draw is still uniform, so
    this is just as random - but it is the *same* random, which is why a seeded
    game still replays to the same board it always did. Dropping the anchor
    shifted every rotation by two and silently moved every position fixture in
    the test suite.
    """
    start = (CLOCKWISE_OWNERS.index(PLAYER_OWNER)
             + rng.randrange(SEATS)) % SEATS
    return [CLOCKWISE_OWNERS[(start + i) % SEATS] for i in range(SEATS)]


def corner_of(owner):
    """The corner a seat owns. Fixed by owner id, not by colour."""
    return OWNER_CORNER[owner]


def build_brain(key, rng, kind=None, checkpoint_dir=IMITATION_CHECKPOINT_DIR,
                device=None, mode="argmax"):
    """The contestant for one seat: a Brain for an AI or an imitation player,
    `None` for a human.

    `rl.imitation` is imported here rather than at module scope so that a game
    with no imitation seat never loads torch. A human seat has no brain, which
    is what makes "how many humans are there" answerable by counting non-None
    brains only after the seat kinds are known - hence `kind` being passed in.
    """
    kind = kind_of(key) if kind is None else kind
    if kind == KIND_AI:
        return make_brain(key, rng)
    if kind == KIND_IMITATION:
        from rl.imitation import load_brain
        return load_brain(imitation_step(key), checkpoint_dir=checkpoint_dir,
                          device=device, mode=mode)
    return None


def describe_seats(owner_key, colors):
    """A short human-readable line per seat, for logs and for match output."""
    from config import I
    out = []
    for o in range(SEATS):
        try:
            name = I[owner_key[o]]
        except KeyError:
            name = owner_key[o]
        out.append("座位%d %s %s" % (o + 1, name, colors[o]))
    return out