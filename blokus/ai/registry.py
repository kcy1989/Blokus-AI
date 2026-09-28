"""人格註冊表：把 key 對到 Brain 類別與權重檔。

新增一個人格只要三步：在自己的模組定義 `KEY`／`PROFILE_SPEC`（權重式）或
`Brain` 子類別（規則式），然後在這裡登記。
"""
from . import builder, chess, fox, intruder, optimizer, wolf
from .base import (BUILDER_KEY, INTRUDER_KEY, OPTIMIZER_KEY, RULE_KEYS)
from .base import make_profile as _make_profile

# 權重式人格的 (key, Brain 類別, 權重檔)。順序決定 `personality_keys()` 的抽籤
# 順序，也決定 UI 的顯示順序。
WEIGHTED_SPECS = (
    (wolf.KEY, wolf.WolfBrain, wolf.PROFILE_SPEC),
    (chess.KEY, chess.ChessBrain, chess.PROFILE_SPEC),
    (fox.KEY, fox.FoxBrain, fox.PROFILE_SPEC),
)

# 規則式人格 = 目標函式不是權重加總的那幾種。它們仍然帶一份權重檔，只用在第一
# 階段的粗篩與被預判時的評分。
RULE_BRAIN_CLASSES = {
    INTRUDER_KEY: intruder.IntruderBrain,
    OPTIMIZER_KEY: optimizer.OptimizerBrain,
    BUILDER_KEY: builder.BuilderBrain,
}

PROFILE_SPECS = {key: spec for key, _cls, spec in WEIGHTED_SPECS}

WEIGHTED_BRAIN_CLASSES = {key: cls for key, cls, _spec in WEIGHTED_SPECS}


def personality_keys():
    """全部六個人格的 key，順序固定。"""
    return tuple(key for key, _cls, _spec in WEIGHTED_SPECS) + RULE_KEYS


def make_profile(key, rng):
    """`key` 對應的權重檔加上隨機擾動。只有權重式人格有權重檔。"""
    return _make_profile(PROFILE_SPECS[key], rng)


def draw_personalities(rng, n=3):
    """抽 n 個**互不相同**的人格。"""
    return rng.sample(list(personality_keys()), n)


def make_brain(key, rng):
    """建立一個人格。

    規則式人格帶的是 chess 的權重檔（不是它自己的），只用在第一階段的粗篩與
    被預判時的評分；它們自己的目標函式完全在各自的 Brain 類別裡。
    """
    cls = RULE_BRAIN_CLASSES.get(key)
    if cls is not None:
        return cls(key, make_profile(chess.KEY, rng))
    return WEIGHTED_BRAIN_CLASSES[key](key, make_profile(key, rng))
