"""Golden baseline: whole games replayed move for move.

`tests/golden/golden.json` was recorded from the pre-refactor tree
(`pre-refactor-baseline`). Every case is replayed through the *same*
`golden_cases.replay` the generator used, and the whole game - turn order,
colours, every ply, and the final remaining counts - must come back
identical. This is the per-step check the refactor runs against: a change
that alters any move here has altered behaviour, whatever the other tests
say.

Personalities only, no checkpoint, no `data/`, no `records.json` - see the
module docstring of `tests/golden/golden_cases.py`.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")
if GOLDEN_DIR not in sys.path:
    sys.path.insert(0, GOLDEN_DIR)

from golden_cases import CASES, replay  # noqa: E402

with open(os.path.join(GOLDEN_DIR, "golden.json"), encoding="utf-8") as _fh:
    GOLDEN = json.load(_fh)


def test_every_case_has_a_recorded_game():
    assert sorted(GOLDEN) == sorted(case[0] for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_golden_game_replays_move_for_move(case):
    assert replay(case) == GOLDEN[case[0]]
