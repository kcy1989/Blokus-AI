"""Personality registry: maps a key to a Brain class and a weight profile.

Adding a personality takes three steps: define `KEY` / `PROFILE_SPEC`
(weighted type) or a `Brain` subclass (rule-based type) in your own module,
then register it here.
"""
from . import builder, chess, fox, intruder, optimizer, wolf
from .base import (BUILDER_KEY, INTRUDER_KEY, OPTIMIZER_KEY, RULE_KEYS)
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
}

PROFILE_SPECS = {key: spec for key, _cls, spec in WEIGHTED_SPECS}

WEIGHTED_BRAIN_CLASSES = {key: cls for key, cls, _spec in WEIGHTED_SPECS}


def personality_keys():
    """The keys of all six personalities, in a fixed order."""
    return tuple(key for key, _cls, _spec in WEIGHTED_SPECS) + RULE_KEYS


def make_profile(key, rng):
    """The profile spec for `key`, plus a random perturbation. Only the
    weighted personalities have a profile spec."""
    return _make_profile(PROFILE_SPECS[key], rng)


def draw_personalities(rng, n=3):
    """Draw n **mutually distinct** personalities."""
    return rng.sample(list(personality_keys()), n)


def make_brain(key, rng):
    """Build a personality.

    Rule-based personalities are handed chess's profile spec (not their own),
    used only for the coarse screen in stage 1 and for scoring when they are
    predicted; their own objective functions live entirely in their respective
    Brain classes.
    """
    cls = RULE_BRAIN_CLASSES.get(key)
    if cls is not None:
        return cls(key, make_profile(chess.KEY, rng))
    return WEIGHTED_BRAIN_CLASSES[key](key, make_profile(key, rng))
