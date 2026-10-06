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
# A trained policy. Its own kind rather than KIND_IMITATION so that
# `imitation_only` keeps meaning the checkpoints that were trained by imitation,
# which is what the name says and what the three `hc_*` seats are.
KIND_RL = "rl"
KIND_HUMAN = "human"
KIND_HUMAN_LOG = "human_log"
# Not a kind of contestant but a deferred choice: a seat left on this is dealt
# one of the eleven automated options when the game starts, the same way a
# "random" colour is dealt one of what is left. It is resolved before anything
# looks at the seat, so a game in progress never contains one.
KIND_RANDOM_AI = "random_ai"

# The per-seat colour sentinel: "deal me one of what is left".
RANDOM = "random"
# The per-seat option sentinel: "deal me one of the automated options".
RANDOM_AI_KEY = "random_ai"

HUMAN_KEY = "human"
HUMAN_LOG_KEY = "human_log"

# Which H-C2 checkpoints are offered as players, in the order the menu lists
# them. Read-only at inference: the optimiser state in those files is ignored,
# and so is the fact that 10,000 was the last step trained.
#
# H-C2 is the run that regenerated the imitation data on the exact-latch engine
# (reports/hc2_report.md). H-B2's checkpoints are still on disk in `data/hb2`
# and are still loadable by step number; they are simply no longer *named*, so
# that a stale key cannot silently resolve to a checkpoint trained on data whose
# `stuck` column disagreed with the rules.
IMITATION_STEPS = (1000, 2000, 10000)
IMITATION_CHECKPOINT_DIR = "data/hc2"

# The option key each step is registered under, and its inverse. A table rather
# than a prefix, because two naming conventions coexist on purpose: step 1000 is
# the 0k starting point of the `rl_h1000` chain and is named for it, while 2000
# and 10000 keep the `hc_` prefix they were published under. `rl.imitation` asks
# this module for a key rather than inventing one, so an old `step_*` name has
# no path to a network - and `imitation_key` now refuses a step that has no
# registered name instead of manufacturing `hc_9999` for it.
IMITATION_KEYS = {1000: "rl_h1000_0k", 2000: "hc_2000", 10000: "hc_10000"}
IMITATION_KEY_STEPS = {key: step for step, key in IMITATION_KEYS.items()}

# Trained policies, as `(option key, checkpoint file)` pairs.
#
# Written out as a table rather than derived, because a trained policy's identity
# *is* its file. H-C2's three checkpoints live in one directory under one
# filename convention, so a step number identifies a file and `imitation_key`
# looks up the name it was registered under. `data/rl1` has no such convention:
# `step_000040.pt` there means round 40, not an imitation step, and `latest.pt`
# is a moving target that would make the pool entry quietly mean a different
# policy after every retrain. So the file is named, and pinning it is the
# friction that keeps a league result interpretable.
#
# `rl_h1000_20k` = PPO from the 0k checkpoint, 20,000 episodes. The `h` names
# the personality that 0k model was distilled from: `data/hc1/manifest.json`
# records `teacher = hunter`, and the checkpoint's own `init_checkpoint` field
# records `data/hc2/step_001000.pt` - the file the seat `rl_h1000_0k` plays,
# which plan9a stage 1 renamed out of its old `hc_1000` spelling. The training
# curve is in `data/rl1/rounds.jsonl`, which is where the 40 rounds of 500
# games are.
RL_SEATS = (
    ("rl_h1000_20k", "data/rl1/step_000040.pt"),
)

# The key this seat carried before plan9 step 1 renamed it, kept resolving to the
# same policy. An alias rather than a second row in `RL_SEATS`: a second row would
# put the same weights in the pool twice, taking the pool from eleven options to
# twelve and giving every league a doubled chance of drawing one model - which
# would silently change what a pool size means.
#
# The alias resolves on the way *in* and never appears in `automated_options()`,
# so the pool still lists eleven and the leaderboard still has one row per
# policy. What it does not solve is a leaderboard that already holds the old
# name: a result recorded through the alias lands in its own row, because
# `records.json` is keyed by the name a game was played under and knows nothing
# about aliases. `records.json` was renamed in step 1 rather than left to split.
RL_ALIASES = {
    "rl_1000_20k": "rl_h1000_20k",
}

# Every alias, the imitation ones and the trained-policy ones alike. This is
# what `canonical_key` reads, and it is *built* from `RL_ALIASES` rather than
# written beside it so the two cannot drift: two tables would mean a name
# retired in one of them resolves in one place and not another, and which place
# that is would depend on which table the reader happened to look at.
#
# `hc_1000` is the spelling the 0k checkpoint was published and measured under;
# it stays valid for ever, because `--subject hc_1000` names two committed
# evidence batches and `records.json` still has a row under it.
ALIASES = dict(RL_ALIASES, **{"hc_1000": IMITATION_KEYS[1000]})

RL_KEY_PREFIX = "rl_"

SEATS = 4


def canonical_key(key):
    """The registered name for `key`, following the alias table.

    Everything that has to *recognise* a seat goes through here, so an alias is
    accepted anywhere a key is: `--pool rl_1000_20k`, `--subject hc_1000` and
    `build_brain("rl_1000_20k", ...)` all name the seat their canonical name
    names. Deliberately one hop - an alias naming another alias would need a
    loop and nothing currently needs one.
    """
    return ALIASES.get(key, key)


def rl_keys():
    """The option keys naming trained policies, in table order."""
    return tuple(key for key, _path in RL_SEATS)


def rl_checkpoint(key):
    """The checkpoint file for a trained-policy key.

    Raises for anything else, including a key of the right shape that is not
    registered - `rl_9999` would otherwise be a name that resolves to nothing, or
    worse, to whatever a future table happens to put at that slot.
    """
    key = canonical_key(key)
    for registered, path in RL_SEATS:
        if registered == key:
            return path
    raise ValueError("%r is not a trained-policy seat" % (key,))


def is_rl_key(key):
    """Is this an option key naming a trained policy in the pool?

    Set membership for the same reason `is_imitation_key` is: a prefix test would
    accept `rl_9999`, which has the right shape and no file behind it. Aliases
    count, or an old command line would stop resolving.
    """
    return key in _RL_KEYS or canonical_key(key) in _RL_KEYS


_RL_KEYS = frozenset(rl_keys())


def imitation_key(step):
    """The option key the checkpoint saved at `step` is registered under.

    Raises for a step that has no registered name. The prefix scheme this
    replaced would have happily returned `hc_9999` - a key of the right shape
    with no file behind it - which is the exact thing `is_imitation_key` exists
    to refuse.
    """
    key = IMITATION_KEYS.get(int(step))
    if key is None:
        raise ValueError("step %r has no registered imitation option; the pool "
                         "names %s" % (step, sorted(IMITATION_KEYS)))
    return key


def imitation_step(key):
    """The step an option key names. Inverse of `imitation_key`.

    Goes through `canonical_key`, so the retired `hc_1000` still answers 1000 -
    which is what lets a game set up with the old spelling reach the same
    weights as one set up with `rl_h1000_0k`.
    """
    step = IMITATION_KEY_STEPS.get(canonical_key(key))
    if step is None:
        raise ValueError("%r is not an imitation option" % (key,))
    return step


def is_imitation_key(key):
    """Is this an option key naming a checkpoint that is actually in the pool?

    Deliberately a set membership rather than a prefix test: `hc_9999` has the
    right shape and no file behind it, and the old `step_*` names have the wrong
    shape and a perfectly good file behind them. Matching the set is what stops
    either of those from being treated as a playable seat. Aliases count, the
    same way `is_rl_key` counts them, so `hc_1000` keeps answering True now that
    the seat it names is registered as `rl_h1000_0k`.
    """
    return key in _IMITATION_KEYS or canonical_key(key) in _IMITATION_KEYS


_IMITATION_KEYS = frozenset(imitation_key(s) for s in IMITATION_STEPS)


def automated_options():
    """The eleven options something can be played by: seven personalities, three
    checkpoints and one trained policy. This is the pool `RANDOM_AI_KEY` draws
    from, and the same pool the league draws its four seats from."""
    out = [k for k in personality_keys()]
    out += [imitation_key(s) for s in IMITATION_STEPS]
    out += list(rl_keys())
    return tuple(out)


def seat_options(include_humans=True):
    """Every option a seat can be *set to*, in menu order: the AI pool, then
    the imitation steps, then the trained policies, then (optionally) the two
    human seats.

    11 without humans, 13 with them. The deferred "random AI" is not here,
    because it is not something a seat stays set to - `seat_menu_options` is
    what the picker offers, and it is this plus that one.
    """
    out = list(automated_options())
    if include_humans:
        out += [HUMAN_KEY, HUMAN_LOG_KEY]
    return tuple(out)


def seat_menu_options():
    """What the seat screen offers: the thirteen, plus "random AI" at the end.

    Fourteen entries, because the deferred choice is a fourth thing a seat can
    say, sitting alongside the eleven automated options and the two human ones.
    """
    return seat_options(True) + (RANDOM_AI_KEY,)


def resolve_random_ai(keys, rng):
    """Deal any `RANDOM_AI_KEY` seat one of the eleven, one draw per seat.

    Each draw is independent, so two "random AI" seats may land on the same
    option - which is the same rule the colours follow. The result is a concrete
    option for every seat: a game that has started never contains a deferred
    choice, which is what keeps `owner_key`, the seat kinds, the log and the
    leaderboard all talking about something specific.
    """
    pool = automated_options()
    out = []
    for k in keys:
        out.append(rng.choice(pool) if k == RANDOM_AI_KEY else k)
    return out


def kind_of(key, include_humans=True):
    """Which kind of contestant `key` names.

    `RANDOM_AI_KEY` answers `KIND_RANDOM_AI`, which is a request rather than a
    contestant - `resolve_random_ai` has to turn it into one before a game
    starts.
    """
    if key == RANDOM_AI_KEY:
        return KIND_RANDOM_AI
    if key in personality_keys():
        return KIND_AI
    if is_imitation_key(key):
        return KIND_IMITATION
    if is_rl_key(key):
        return KIND_RL
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


def build_brain(key, rng, kind=None, checkpoint_dir=None,
                device=None, mode="argmax"):
    """The contestant for one seat: a Brain for an AI or an imitation player,
    `None` for a human.

    `rl.imitation` is imported here rather than at module scope so that a game
    with no imitation seat never loads torch. A human seat has no brain, which
    is what makes "how many humans are there" answerable by counting non-None
    brains only after the seat kinds are known - hence `kind` being passed in.
    """
    kind = kind_of(key) if kind is None else kind
    if kind == KIND_RANDOM_AI:
        raise ValueError(
            "seat option %r is a deferred choice; call resolve_random_ai "
            "before building a game" % (key,))
    if checkpoint_dir is None:
        # Resolved here rather than as a default argument, which would bind the
        # directory once at import and quietly ignore a later rebinding - the
        # same trap `records.Records.__init__` sets with its path.
        checkpoint_dir = IMITATION_CHECKPOINT_DIR
    if kind == KIND_AI:
        return make_brain(key, rng)
    if kind == KIND_IMITATION:
        from rl.imitation import load_brain
        # The seat key is handed to the brain rather than rebuilt from the step:
        # `choose_move` traces and `test_seats` both require
        # `brains[owner].key == owner_key[owner]`, and two places formatting the
        # same name is two places to forget.
        return load_brain(imitation_step(key), checkpoint_dir=checkpoint_dir,
                          device=device, mode=mode, key=key)
    if kind == KIND_RL:
        from rl.imitation import load_brain_at
        # `checkpoint_dir` is deliberately not consulted: it names where the
        # H-C2 checkpoints live, and a trained policy is not one of those. Its
        # file is named in `RL_SEATS`, which is the only place that decides which
        # weights this key means.
        return load_brain_at(rl_checkpoint(key), key=key, device=device,
                             mode=mode)
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