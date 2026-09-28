"""AI 套件：六個人格 + 共用公式 + 選棋流水線。

架構（三層，由下往上）：

    formulas.py   共用的計算公式。位元幾何（ODIRS／合法性／落子推演）、
                  量化指標（延伸度、可放空格、活區塊、裝箱）、權重表。
    base.py       Brain 介面與權重檔型別。候選枚舉、對手預判、抽籤都不在這裡。
    <人格>.py     狼／棋手／狐狸（權重式）與入侵者／優化者／築城者（規則式）。
    chooser.py    共用的選棋流水線，不認識任何人格。
    registry.py   key → Brain 類別／權重檔的對照表。

這個模組把原本單檔 `ai.py` 的所有名字重新匯出，所以 `import ai` 之後
`ai.choose_move`、`ai.ODIRS`、`ai.IntruderBrain` 之類的舊寫法都照舊可用。
"""
from . import formulas
from .base import Brain, Profile, WeightedBrain
from .base import BUILDER_KEY, INTRUDER_KEY, OPTIMIZER_KEY, RULE_KEYS
from .builder import BuilderBrain
from .builder import (BUILDER_SCAN, BUILDER_W_LOST, BUILDER_W_SIZE,
                      BUILDER_W_TOTAL)
from .chess import ChessBrain
from .fox import FoxBrain
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
    # 選棋
    "choose_move", "_opponent_pool",
    # 註冊
    "make_brain", "make_profile", "draw_personalities", "personality_keys",
    "PROFILE_SPECS", "RULE_BRAIN_CLASSES", "WEIGHTED_BRAIN_CLASSES",
    "WEIGHTED_SPECS", "RULE_KEYS", "INTRUDER_KEY", "OPTIMIZER_KEY",
    "BUILDER_KEY",
    # 基底
    "Brain", "WeightedBrain", "Profile",
    # 人格
    "WolfBrain", "ChessBrain", "FoxBrain", "IntruderBrain", "OptimizerBrain",
    "BuilderBrain",
    # 公式
    "formulas", "B", "N", "CENTER", "ODIRS", "Reach", "board_context",
    "board_feats", "contact_vertices", "cross_anchors", "crossing_bonus",
    "crossing_context", "extension", "fill_components", "fillable_closure",
    "has_crossed", "is_leaper", "key_cells", "move_geometry", "own_reach",
    "pack_lost", "place_geometry", "place_state", "placement_counts",
    "size_bonus", "LEAPERS", "STRETCHERS", "SEAL_MIN", "SQ_CAP", "W_CROSS",
    "W_EXT", "W_SEAL", "W_SIZE", "W_SQUARES", "W_STRATEGIC", "W_VERTICES",
    "_adjoining_bases", "_corner_delta", "_free_bases", "_legal_bases",
    "_score_move", "SHORTLIST_CORE", "OPP_POOL_K",
    # 掃描上限
    "INTRUDER_SCAN", "OPTIMIZER_SCAN", "URGENT_PLACES", "BUILDER_SCAN",
    "BUILDER_W_LOST", "BUILDER_W_SIZE", "BUILDER_W_TOTAL",
]
