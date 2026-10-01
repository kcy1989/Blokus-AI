"""Personality registry: maps a key to a Brain class and a weight profile.

Adding a personality takes three steps: define `KEY` / `PROFILE_SPEC`
(weighted type) or a `Brain` subclass (rule-based type) in your own module,
then register it here.
"""
from . import builder, chess, fox, hunter, intruder, optimizer, wolf
from .base import (BUILDER_KEY, HUNTER_KEY, INTRUDER_KEY, OPTIMIZER_KEY,
                   RULE_KEYS)
from .base import make_profile as _make_profile

# (key, Brain class, profile spec) for the weighted personalities. The order
# decides the draw order of `personality_keys()`, and also the UI order.
WEIGHTED_SPECS = (
    (wolf.KEY, wolf.WolfBrain, wolf.PROFILE_SPEC),
    (chess.KEY, chess.ChessBrain, chess.PROFILE_SPEC),
    (fox.KEY, fox.FoxBrain, fox.PROFILE_SPEC),
)

# Rule-based personalities = the ones whose objective function is not a
# weighted sum. They still carry a profile spec, used only for the coarse
# screen in stage 1 and for scoring when they are predicted.
RULE_BRAIN_CLASSES = {
    INTRUDER_KEY: intruder.IntruderBrain,
    OPTIMIZER_KEY: optimizer.OptimizerBrain,
    BUILDER_KEY: builder.BuilderBrain,
    HUNTER_KEY: hunter.HunterBrain,
}

PROFILE_SPECS = {key: spec for key, _cls, spec in WEIGHTED_SPECS}

WEIGHTED_BRAIN_CLASSES = {key: cls for key, cls, _spec in WEIGHTED_SPECS}


def personality_keys():
    """The keys of all personalities, in a fixed order."""
    return tuple(key for key, _cls, _spec in WEIGHTED_SPECS) + RULE_KEYS


def make_profile(key, rng):
    """The profile spec for `key`, plus a random perturbation. Only the
    weighted personalities have a profile spec."""
    return _make_profile(PROFILE_SPECS[key], rng)


def draw_personalities(rng, n=3):
    """Draw n **mutually distinct** personalities."""
    return rng.sample(list(personality_keys()), n)


def make_brain(key, rng, game_seed=None):
    """Build a personality.

    Rule-based personalities are handed chess's profile spec (not their own),
    used only for the coarse screen in stage 1 and for scoring when they are
    predicted; their own objective functions live entirely in their respective
    Brain classes.
    `game_seed` is passed only to a class that advertises `accepts_game_seed`,
    and `rng` only alongside it. A driver that knows the game's seed passes the
    seed; a caller with none - a normal game, where `Game` exposes only an `rng` -
    lets Hunter derive one from that rng's state without consuming from it. Every
    other personality ignores both, so adding the arguments changed nothing about
    how they are built.
    """
    cls = RULE_BRAIN_CLASSES.get(key)
    if cls is not None:
        profile = make_profile(chess.KEY, rng)
        if getattr(cls, "accepts_game_seed", False):
            return cls(key, profile, game_seed=game_seed, seed_rng=rng)
        return cls(key, profile)
    return WEIGHTED_BRAIN_CLASSES[key](key, make_profile(key, rng))
