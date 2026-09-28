"""Wolf: an aggressive corner snatcher.

`w_corner` is the highest of the six weights (4.0) and `w_large` is high as
well (3.0), so it grabs a free corner without hesitation and tends to play
the largest piece in one go. `w_block` is moderate (2.0) and `w_defend` is
very low (0.3) -- it largely does not care about its own pieces being
smothered by the opponent's.

    python3 -c "import ai; print(ai.PROFILE_SPECS['wolf'])"
"""
from .base import WeightedBrain

KEY = "wolf"

# (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
PROFILE_SPEC = (4.0, 1.0, 2.0, 0.3, 0.5, 3.0, 0.15)


class WolfBrain(WeightedBrain):
    """Snatches corners, lays down big pieces. The scoring path is exactly
    the same as the other weighted personalities; only the weights differ."""

    key = KEY
