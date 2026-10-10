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
import hashlib
import os
import re

from ai import make_brain
from ai.registry import (aliases as registry_aliases,
                         automated_options as registry_automated_options,
                         checkpoint as registry_checkpoint,
                         entry as registry_entry,
                         keys as registry_keys,
                         personality_keys)
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
#
# Since plan9a stage 4 this used to be only where H-C2 **trained** - step 1000
# was published into `ai/checkpoints/rl_h1000_0k/` while 2000 and 10000 stayed
# here. Plan9 task 4 retired all six imitation seats and deleted that published
# copy, so the directory is again the one place the three files live, and the
# only authority for their path. No registered key loads from it: these three
# are read by `rl/` (by step + directory) and by a `--adhoc` replay, not by a
# seat.
IMITATION_STEPS = (1000, 2000, 10000)
IMITATION_CHECKPOINT_DIR = "data/hc2"

# The option key each H-C2 step is registered under - the direction the
# training code asks in (`imitation_key(step)`). The other direction, key ->
# step, stopped being a table in plan9a stage 6: `imitation_step` reads the
# checkpoint filename, because four seats sit at step 1000 and no table keyed
# by step could name them all. A table rather
# than a prefix, because two naming conventions coexist on purpose: step 1000 is
# the 0k starting point of the `rl_h1000` chain and is named for it, while 2000
# and 10000 keep the `hc_` prefix they were published under. `rl.imitation` asks
# this module for a key rather than inventing one, so an old `step_*` name has
# no path to a network - and `imitation_key` now refuses a step that has no
# registered name instead of manufacturing `hc_9999` for it.
IMITATION_KEYS = {1000: "rl_h1000_0k", 2000: "hc_2000", 10000: "hc_10000"}

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
#
# Since plan9a stage 4 the *load* path is `ai/registry.json`'s `checkpoint` -
# `ai/checkpoints/rl_h1000_20k/step_000040.pt`, beside the code that reads it.
# What this table still names is where the policy was *trained*, which is what
# `source` records and what `tests/test_registry_json.py` cross-checks against.
# The table itself stays: which option keys are trained policies is still
# decided here, because that is a fact about the code and not about a file.
RL_SEATS = (
    ("rl_h1000_20k", "data/rl1/step_000040.pt"),
    ("rl_o1000_20k", "data/rl_o1000/step_000040.pt"),
    ("rl_b1000_20k", "data/rl_b1000/step_000040.pt"),
    ("rl_i1000_20k", "data/rl_i1000/step_000040.pt"),
    # plan10 milestone 1 (2026-10-10): the four learners after 60 more rounds
    # of 500 games each, trained at once under eval/rl-multiple-train/. The
    # path is where the policy was *trained*; `ai/registry.json`'s `checkpoint`
    # is where it is *published* and what `rl_checkpoint` reads. The 20k rows
    # above stay: each 50k policy continues from its own frozen 20k, and both
    # are separate seats so a league can put them against each other.
    ("rl_h1000_50k", "data/multi/h/step_000100.pt"),
    ("rl_o1000_50k", "data/multi/o/step_000100.pt"),
    ("rl_b1000_50k", "data/multi/b/step_000100.pt"),
    ("rl_i1000_50k", "data/multi/i/step_000100.pt"),
)

# Every alias, the imitation ones and the trained-policy ones alike. This is
# what `canonical_key` reads, and since plan9a stage 2 it is *read out of*
# `ai/registry.json` rather than assembled here, so there is one place in the
# repository where a name can be retired: two tables would mean a name retired
# in one of them resolves in one place and not another, and which place that is
# would depend on which table the reader happened to look at.
#
# `hc_1000` is the spelling the 0k checkpoint was published and measured under;
# it stays valid for ever, because `--subject hc_1000` names two committed
# evidence batches and `records.json` still has a row under it.
ALIASES = dict(registry_aliases())

# The old spelling of `rl_h1000_20k`, from before plan9 step 1 renamed it, kept
# resolving to the same policy. An alias rather than a second row in
# `RL_SEATS`: a second row would put the same weights in the pool twice, taking
# the pool up by one option and giving every league a doubled chance of drawing
# one model - which would silently change what a pool size means.
#
# A *view* of `ALIASES` rather than a table of its own - the aliases whose target
# is one of `RL_SEATS`' own rows - so that asking "which old names name a trained
# policy" does not become a second place to forget when a name goes.
#
# The alias resolves on the way *in* and never appears in `automated_options()`,
# so the pool still lists each policy once and the leaderboard still has one
# row per policy. What it does not solve is a leaderboard that already holds the old
# name: a result recorded through the alias lands in its own row, because
# `records.json` is keyed by the name a game was played under and knows nothing
# about aliases. `records.json` was renamed in step 1 rather than left to split.
RL_ALIASES = {a: v for a, v in ALIASES.items() if v in dict(RL_SEATS)}

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
    """The checkpoint file a trained-policy key loads from.

    The path comes from `ai/registry.json` as of plan9a stage 4 - the weights
    live in `ai/checkpoints/` beside the code that reads them - while
    `RL_SEATS` above keeps naming where the policy was trained. A key bound by
    `register_adhoc` (`match.py --adhoc`) comes first: the table is per-process
    and never enters the roster, but this stays the one place that decides
    which weights a key means, and the batch file records the table so the
    evidence still says which bytes were measured.

    Raises for anything else, including a key of the right shape that is not
    registered - `rl_9999` would otherwise be a name that resolves to nothing, or
    worse, to whatever a future table happens to put at that slot.
    """
    key = canonical_key(key)
    if key in ADHOC:
        return ADHOC[key]["path"]
    if not is_rl_key(key):
        raise ValueError("%r is not a trained-policy seat" % (key,))
    return registry_checkpoint(key)


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


_STEP_FILENAME = re.compile(r"^step_(\d{6})\.pt$")


def imitation_step(key):
    """The step an option key names, read off its checkpoint filename.

    Since plan9a stage 6 (decision beta) the filename is the only authority:
    `rl_h1000_0k` and the three students are all at step 1000, so a table keyed
    by step could not name them all, and the registry already owns the path.
    `imitation_key(step)` keeps pointing the other way, along the H-C2 chain,
    where a step really is unique.

    Goes through `canonical_key`, so the retired `hc_1000` still answers 1000 -
    which is what lets a game set up with the old spelling reach the same
    weights as one set up with `rl_h1000_0k`.

    A checkpoint whose name is not H-C2's `step_%06d.pt` convention raises
    rather than falling back to a number: a default here would make the seat
    claim a step that loads a different file than the one named.
    """
    key = canonical_key(key)
    if not is_imitation_key(key):
        raise ValueError("%r is not an imitation option" % (key,))
    path = registry_checkpoint(key)
    match = _STEP_FILENAME.match(os.path.basename(path))
    if match is None:
        raise ValueError("%r: checkpoint %r is not a step_NNNNNN.pt file"
                         % (key, path))
    return int(match.group(1))


def imitation_checkpoint(key):
    """The checkpoint file an imitation key loads from.

    The path used to come from `ai/registry.json`. Plan9 task 4 removed the last
    imitation row, so this now refuses every key - which is the point: an
    imitation seat is not something the product can be set up with any more.
    The files themselves are back under `data/hc2` (see `IMITATION_STEPS`), and
    `rl/` reads them by step and directory rather than through here.
    """
    key = canonical_key(key)
    if not is_imitation_key(key):
        raise ValueError("%r is not an imitation-checkpoint seat" % (key,))
    return registry_checkpoint(key)


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


# Membership of the imitation family is `ai/registry.json`'s `family` field
# rather than `IMITATION_KEYS`, since plan9a stage 6 registered three students
# that no step number can name. `IMITATION_STEPS` still describes H-C2's chain
# and nothing else, which is what the training code and its tests want.
_IMITATION_KEYS = frozenset(k for k in registry_keys()
                            if registry_entry(k)["family"] == "imitation")


def automated_options():
    """The eleven options something can be played by: seven personalities and
    four trained policies. This is the pool
    `RANDOM_AI_KEY` draws from, and the same pool the league draws its four
    seats from.

    The list itself comes from `ai/registry.json` as of plan9a stage 2 - the
    roster is data now, not code - and so does which keys are imitation seats.
    What is still decided here is `kind_of`: it answers from
    `personality_keys`, the registry's `family` field and `RL_SEATS`, and
    `tests/test_registry_json.py` fails if the JSON and those ever disagree.

    **The order is by key**, through `ai.registry.pool_order`, and it is the
    only pool order in the project. `rng.choice` draws an index, so this tuple
    is the index-to-key mapping every same-seed game depends on; key order is a
    function of the set of keys, so adding, removing or reordering a roster row
    changes it only when the set changes.
    """
    return registry_automated_options()


def seat_options(include_humans=True):
    """Every option a seat can be *set to*, in menu order: the eleven
    automated options by key, then (optionally) the two human seats.

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
    say, sitting alongside the eleven automated options and the two human
    ones.
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


# Checkpoints bound for this process only, by `match.py --adhoc KEY=PATH`.
# Deliberately *not* part of the roster: these keys never enter
# `automated_options()`, so the UI menus, `resolve_random_ai` and the default
# league pool cannot draw them - a league only gets them when the command line
# names them. `kind_of` and `rl_checkpoint` consult the table, and the batch
# file records it (`payload["adhoc"]`, path + md5) because the files themselves
# live under `data/` and are not in the repository.
ADHOC = {}


def register_adhoc(key, path):
    """Bind `key` to the checkpoint file at `path` for this process.

    Refuses keys that already exist as seats - shadowing `hunter` or a
    registered checkpoint would make the same spelling mean two different
    things depending on the command line - and refuses missing files, so a
    typo fails before the first game rather than in the middle of the batch.
    Re-registering the same key with the same file is a no-op, because a test
    session or a wrapper may set the same binding twice.
    """
    key = canonical_key(key)
    if (key in (RANDOM_AI_KEY, HUMAN_KEY, HUMAN_LOG_KEY)
            or key in personality_keys() or is_imitation_key(key)
            or is_rl_key(key)):
        raise ValueError(
            "%r is already a seat option; --adhoc is only for checkpoint "
            "files that are not in the roster" % (key,))
    if not key:
        raise ValueError("--adhoc needs a non-empty key")
    if not os.path.isfile(path):
        raise ValueError("--adhoc %s: no such checkpoint file %r" % (key, path))
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    entry = {"path": path, "md5": h.hexdigest()}
    prev = ADHOC.get(key)
    if prev is not None and prev != entry:
        raise ValueError(
            "--adhoc %s was already bound to %s with a different file"
            % (key, prev["path"]))
    ADHOC[key] = entry
    return entry


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
    if canonical_key(key) in ADHOC:
        # An --adhoc key is a single trained-policy file, which is exactly
        # what the KIND_RL branch of `build_brain` loads - through
        # `rl_checkpoint`, which reads the same table. Registration refuses
        # keys the roster already owns, so this cannot shadow one.
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
    if kind == KIND_AI:
        return make_brain(key, rng)
    if kind == KIND_IMITATION:
        from rl.imitation import load_brain
        if checkpoint_dir is None:
            # Resolved here rather than as a default argument, which would bind
            # the directory once at import and quietly ignore a later
            # rebinding - the same trap `records.Records.__init__` sets with its
            # path. It is per *key* rather than one constant for the three.
            # Plan9 task 4 put every file back under `data/` and retired the
            # seats, so this branch is unreachable today - it is kept so that
            # re-registering an imitation row does not also need a rewrite of
            # how the path is found.
            checkpoint_dir = os.path.dirname(imitation_checkpoint(key))
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
        # file is `ai/registry.json`'s `checkpoint`, read through
        # `rl_checkpoint` - the only place that decides which weights this key
        # means since plan9a stage 4.
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