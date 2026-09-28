"""Chess: a balanced layout player.

All six weights are set to 1.0, so it has no preference and simply sums the
weighted terms by its own judgement. `w_open` is only 1.0 (wolf 0.5, fox 2.0)
and `w_corner` is also only 1.0, so it neither grabs corners nor hoards open
space. This is the default personality, and also the profile spec that the
rule-based personalities borrow when they are predicted.

    python3 -c "import ai; print(ai.PROFILE_SPECS['chess'])"
"""
from .base import WeightedBrain

KEY = "chess"

# (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
PROFILE_SPEC = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05)


class ChessBrain(WeightedBrain):
    """Steady layout, balanced weights. The scoring path is exactly the same
    as the other weighted personalities."""

    key = KEY
