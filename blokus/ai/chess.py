"""棋手：平衡的佈局者。

六個權重全部給 1.0，所以它沒有偏好，每個面向都照自己的判斷權重加總。`w_open`
只有 1.0（狼 0.5、狐狸 2.0），`w_corner` 也只有 1.0，所以它既不搶角也不貪空
地。這是預設人格，也是規則式人格被預判時借用的權重檔。

    python3 -c "import ai; print(ai.PROFILE_SPECS['chess'])"
"""
from .base import WeightedBrain

KEY = "chess"

# (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
PROFILE_SPEC = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05)


class ChessBrain(WeightedBrain):
    """穩健佈局、權重均衡。評分路徑與其他權重式人格完全相同。"""

    key = KEY
