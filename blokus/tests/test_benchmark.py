"""D2: the benchmark must be reproducible, and its ranking must be fair.

Two things are easy to get wrong in a benchmark and both produce numbers that
look plausible:

  * **Seat bias.** If seats are not rotated, whichever personality happens to
    sit in seat 0 also sits in the corner that moves first in every game. A
    result that is really "seat 0 is lucky" reads as "this personality is
    strong".
  * **Tie-breaking by seat.** If joint scores are separated by seat number, the
    benchmark reports a difference that is not in the game - exactly the bias it
    is trying to measure away.

So the tie rule is pinned here directly, and reproducibility is checked by
running the tool twice and diffing.
"""
import importlib.util
import itertools
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TOOL = os.path.join(_ROOT, "tools", "benchmark.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("bench_tool", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BENCH = _load_tool()


# ------------------------------------------------------------- tied ranks

@pytest.mark.parametrize("cells,expected", [
    # No ties: 1, 2, 3, 4.
    ([10, 20, 30, 40], [1.0, 2.0, 3.0, 4.0]),
    # Two joint first: they share places 1 and 2, so each gets 1.5.
    ([5, 5, 9, 20], [1.5, 1.5, 3.0, 4.0]),
    # All four level: everyone is joint first, sharing 1..4 -> 2.5 each.
    ([7, 7, 7, 7], [2.5, 2.5, 2.5, 2.5]),
    # A three-way tie for first out of four.
    ([1, 1, 1, 9], [2.0, 2.0, 2.0, 4.0]),
    # Tie for second, which is the case the plan calls out.
    ([3, 8, 8, 12], [1.0, 2.5, 2.5, 4.0]),
    # Tie for third.
    ([1, 2, 5, 5], [1.0, 2.0, 3.5, 3.5]),
])
def test_tied_places_are_split_evenly(cells, expected):
    assert BENCH.average_ranks(cells) == expected


def test_tie_split_never_consults_the_seat_number():
    """Swapping seats must not change anyone's rank."""
    base = [4, 9, 9, 15]
    ranks = BENCH.average_ranks(base)
    for perm in itertools.permutations(base):
        assert BENCH.average_ranks(list(perm)) == \
            [ranks[base.index(v)] for v in perm]


def test_averages_are_1_to_4_and_sum_is_10():
    """Four players, average place must average to 2.5 whatever the outcome."""
    for cells in ([1, 2, 3, 4], [5, 5, 5, 5], [0, 100, 50, 50], [2, 2, 9, 9]):
        ranks = BENCH.average_ranks(cells)
        assert min(ranks) >= 1.0
        assert max(ranks) <= 4.0
        assert abs(sum(ranks) - 10.0) < 1e-9, (cells, ranks)


# --------------------------------------------------------- reproducibility

def _run(args):
    proc = subprocess.run([sys.executable, _TOOL] + args,
                          capture_output=True, text=True, cwd=_ROOT)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


SMALL = ["--personalities", "wolf", "chess", "fox", "optimizer",
         "--games", "1", "--permutations", "3", "--no-timing"]


def test_same_arguments_give_byte_identical_output():
    a = _run(SMALL)
    b = _run(SMALL)
    assert a == b, (a, b)


def test_scores_reproduce_even_with_the_timing_column_present():
    """The wall-clock column may drift; the scores must not."""
    names = ("wolf", "chess", "fox", "optimizer", "intruder", "builder")

    def scores(text):
        out = {}
        for line in text.splitlines():
            f = line.split()
            # key games avg_place avg_cells wins [ms/move]
            if len(f) >= 5 and f[0] in names:
                out[f[0]] = f[1:5]
        return out
    a = scores(_run(SMALL[:-1]))
    b = scores(_run(SMALL[:-1]))
    assert a == b, (a, b)
    assert a, "no score rows parsed"


# ------------------------------------------------------------- validation

def test_it_rejects_a_bad_personalities_list():
    for args in (["--personalities", "wolf", "chess", "fox"],
                 ["--personalities", "wolf", "chess", "fox", "fox"],
                 ["--personalities", "wolf", "chess", "fox", "nope"]):
        proc = subprocess.run([sys.executable, _TOOL] + args + ["--games", "0",
                              "--permutations", "1"],
                              capture_output=True, text=True, cwd=_ROOT)
        assert proc.returncode != 0, args


def test_it_rotates_seatings():
    """Every seating must be exercised, or the benchmark measures seats."""
    g = BENCH.make_game(["wolf", "chess", "fox", "intruder"], 7, 0)
    seen = set()
    for rotation in range(4):
        assert BENCH.make_game(["wolf", "chess", "fox", "intruder"],
                               7, rotation).turn_order[0] is not None
        seen.add(tuple(BENCH.make_game(["wolf", "chess", "fox", "intruder"],
                                       7, rotation).turn_order))
    assert len(seen) == 4, seen
