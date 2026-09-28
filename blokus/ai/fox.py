"""Fox: defensive, presses on the seams and controls the edges.

`w_open` is the highest (2.0) -- it cares most about leaving space open, so
it picks the edges rather than stuffing pieces into its own pocket.
`w_defend` (3.0) and `w_block` (3.0) are both the highest, meaning it wants
to press its own pieces against the opponent's, or to stand between the
opponent and the open space. By contrast `w_corner` is only 0.5 (wolf 4.0),
so it almost never grabs corners.

    python3 -c "import ai; print(ai.PROFILE_SPECS['fox'])"
"""
from .base import WeightedBrain

KEY = "fox"

# (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
PROFILE_SPEC = (0.5, 1.0, 3.0, 3.0, 2.0, 0.5, 0.25)


class FoxBrain(WeightedBrain):
    """Presses on the seams, controls the edges. The scoring path is exactly
    the same as the other weighted personalities."""

    key = KEY
