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
    _DOC = json.load(_fh)

META = _DOC["meta"]
GAMES = _DOC["games"]


def test_every_case_has_a_recorded_game():
    assert sorted(GAMES) == sorted(case[0] for case in CASES)


def test_the_baseline_records_where_it_came_from():
    """The file is only meaningful next to the commit it was recorded from."""
    assert META["generated_from_head"]
    assert META["generated_at_utc"]
    assert META["cases"] == len(CASES)
    assert META["plies"] == sum(len(g["moves"]) for g in GAMES.values())


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_golden_game_replays_move_for_move(case):
    """Whole-game equality, not just the result.

    `replay` returns turn order, colours, seat keys, the full per-ply list
    (`[owner, name, oi, x, y]`, or `[owner, "PASS"]`), the final remaining
    counts and the final state; this compares all of it. A change that moves
    a single ply, or reseats a colour, fails here.
    """
    assert replay(case) == GAMES[case[0]]
