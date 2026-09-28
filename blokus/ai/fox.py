"""狐狸：防守型，壓制接合處、把控邊界。

`w_open` 最高（2.0）——它最重視留白，所以會挑邊界而不是把棋塞進自己的口袋
裡。`w_defend`（3.0）與 `w_block`（3.0）都是最高，意思是要把自己的棋擠在對手
的棋上、或擋在對手與空間之間。相對地 `w_corner` 只有 0.5（狼 4.0），它幾乎不
搶角。

    python3 -c "import ai; print(ai.PROFILE_SPECS['fox'])"
"""
from .base import WeightedBrain

KEY = "fox"

# (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
PROFILE_SPEC = (0.5, 1.0, 3.0, 3.0, 2.0, 0.5, 0.25)


class FoxBrain(WeightedBrain):
    """壓制接合、把控邊界。評分路徑與其他權重式人格完全相同。"""

    key = KEY
