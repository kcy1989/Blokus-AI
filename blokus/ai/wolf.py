"""狼：激進的搶角者。

`w_corner` 是六個權重裡最高的（4.0），`w_large` 也很高（3.0），所以它會在角
位還空著時毫不猶豫地搶下來，並且傾向一次下最大的棋塊。`w_block` 中等
（2.0），`w_defend` 很低（0.3）——它不太理會自己的棋被貼住這件事。

    python3 -c "import ai; print(ai.PROFILE_SPECS['wolf'])"
"""
from .base import WeightedBrain

KEY = "wolf"

# (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
PROFILE_SPEC = (4.0, 1.0, 2.0, 0.3, 0.5, 3.0, 0.15)


class WolfBrain(WeightedBrain):
    """搶四角、擺大塊。評分路徑與其他權重式人格完全相同，差別只在權重。"""

    key = KEY
