"""The AI package: the personalities + shared formulas + the move pipeline.

Architecture (three layers, bottom to top):

    formulas.py   Shared formulas. Bit geometry (ODIRS / legality / placement
                  extrapolation), quantitative metrics (extension, playable
                  squares, live components, packing), and the weight table.
    base.py       The Brain interface and the profile type. Candidate
                  enumeration, opponent lookahead and the draw are not here.
    <persona>.py  Wolf/chess/fox (weighted) and intruder/optimizer/builder
                  (rule-based).
    chooser.py    The shared move pipeline, which knows no personality.
    registry.py   The key -> Brain class / weight profile lookup table.

This module re-exports every name from the original single-file `ai.py`, so
after `import ai` the old spellings such as `ai.choose_move`, `ai.ODIRS` and
`ai.IntruderBrain` still work unchanged.
"""
from . import formulas
from .base import Brain, Profile, WeightedBrain
from .base import (BUILDER_KEY, HUNTER_KEY, INTRUDER_KEY, OPTIMIZER_KEY,
                   RULE_KEYS)
from .builder import BuilderBrain
from .builder import (BUILDER_SCAN, BUILDER_W_LOST, BUILDER_W_SIZE,
                      BUILDER_W_TOTAL)
from .chess import ChessBrain
from .fox import FoxBrain
from .hunter import HunterBrain, BOOK, BOOK_STEPS, book_candidates
from .intruder import IntruderBrain, INTRUDER_SCAN
from .optimizer import OptimizerBrain, OPTIMIZER_SCAN, URGENT_PLACES
from .registry import (PROFILE_SPECS, RULE_BRAIN_CLASSES, WEIGHTED_BRAIN_CLASSES,
                      WEIGHTED_SPECS, draw_personalities, make_brain,
                      make_profile, personality_keys)
from .wolf import WolfBrain
from .chooser import _opponent_pool, choose_move
from .formulas import (B, CENTER, LEAPERS, N, ODIRS, OPP_POOL_K, Reach,
                       SEAL_MIN, SHORTLIST_CORE, SQ_CAP, STRETCHERS, W_CROSS,
                       W_EXT, W_SEAL, W_SIZE, W_SQUARES, W_STRATEGIC,
                       W_VERTICES, _adjoining_bases, _corner_delta, _free_bases,
                       _legal_bases,                        _score_move, board_context, board_feats,
                       contact_vertices, cross_anchors, crossing_bonus,
                       crossing_context, extension, fill_components,
                       fillable_closure, has_crossed, is_leaper, key_cells,
                       move_geometry, own_reach, pack_lost, place_geometry,
                       place_state, placement_counts, size_bonus)


__all__ = [
    # move choice
    "choose_move", "_opponent_pool",
    # registry
    "make_brain", "make_profile", "draw_personalities", "personality_keys",
    "PROFILE_SPECS", "RULE_BRAIN_CLASSES", "WEIGHTED_BRAIN_CLASSES",
    "WEIGHTED_SPECS", "RULE_KEYS", "INTRUDER_KEY", "OPTIMIZER_KEY",
    "BUILDER_KEY", "HUNTER_KEY",
    # base
    "Brain", "WeightedBrain", "Profile",
    # personalities
    "WolfBrain", "ChessBrain", "FoxBrain", "IntruderBrain", "OptimizerBrain",
    "BuilderBrain", "HunterBrain", "BOOK", "BOOK_STEPS", "book_candidates",
    # formulas
    "formulas", "B", "N", "CENTER", "ODIRS", "Reach", "board_context",
    "board_feats", "contact_vertices", "cross_anchors", "crossing_bonus",
    "crossing_context", "extension", "fill_components", "fillable_closure",
    "has_crossed", "is_leaper", "key_cells", "move_geometry", "own_reach",
    "pack_lost", "place_geometry", "place_state", "placement_counts",
    "size_bonus", "LEAPERS", "STRETCHERS", "SEAL_MIN", "SQ_CAP", "W_CROSS",
    "W_EXT", "W_SEAL", "W_SIZE", "W_SQUARES", "W_STRATEGIC", "W_VERTICES",
    "_adjoining_bases", "_corner_delta", "_free_bases", "_legal_bases",
    "_score_move", "SHORTLIST_CORE", "OPP_POOL_K",
    # scan caps
    "INTRUDER_SCAN", "OPTIMIZER_SCAN", "URGENT_PLACES", "BUILDER_SCAN",
    "BUILDER_W_LOST", "BUILDER_W_SIZE", "BUILDER_W_TOTAL",
]
