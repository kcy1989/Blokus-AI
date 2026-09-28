"""Tests for piece definitions and orientations."""
import random

from pieces import MASTER, Hand
from config import PIECES

EXPECT = {
    "I1": 1,
    "I2": 2,
    "I3": 2,
    "V3": 4,
    "I4": 2,
    "L4": 8,
    "O4": 1,
    "T4": 4,
    "Z4": 4,
    "F5": 8,
    "I5": 2,
    "L5": 8,
    "N5": 8,
    "P5": 8,
    "T5": 4,
    "U5": 4,
    "V5": 4,
    "W5": 4,
    "X5": 1,
    "Y5": 8,
    "Z5": 4,
}


def test_orientation_counts():
    for name, expected in EXPECT.items():
        assert len(MASTER[name]["orientations"]) == expected, name
    assert len(MASTER) == 21


def test_piece_sizes():
    assert MASTER["I1"]["size"] == 1
    assert MASTER["I2"]["size"] == 2
    assert MASTER["I3"]["size"] == MASTER["V3"]["size"] == 3
    assert MASTER["I4"]["size"] == MASTER["L4"]["size"] == MASTER["O4"]["size"] == 4
    assert all(MASTER[n]["size"] == 5 for n in ["F5", "I5", "L5", "N5", "P5", "T5", "U5", "V5", "W5", "X5", "Y5", "Z5"])


def test_total_cell_counts():
    total = 0
    for n, p in MASTER.items():
        assert len(set(p["cells"])) == p["size"]
        total += p["size"]
    assert total == 12 * 5 + 5 * 4 + 2 * 3 + 1 * 2 + 1 * 1


def test_orientation_normalized_and_dedup():
    for name, p in MASTER.items():
        for oi in p["orientations"]:
            xs = [x for x, _ in oi]
            ys = [y for _, y in oi]
            assert min(xs) == 0 and min(ys) == 0
            assert len(oi) == p["size"]
        assert len(set(p["orientations"])) == len(p["orientations"])
    # symmetric pieces dedup correctly
    assert MASTER["I1"]["orientations"][0] == frozenset({(0, 0)})
    assert MASTER["O4"]["orientations"][0] == frozenset({(0, 0), (1, 0), (0, 1), (1, 1)})
    assert MASTER["X5"]["orientations"][0] == frozenset({(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)})


def test_symmetry_dedup():
    # symmetric: I1, O4, X5 -> 1; strips -> 2; T4 -> 4; asymmetric -> 8
    for name in ["I1", "O4", "X5"]:
        assert len(MASTER[name]["orientations"]) == 1, name
    for name in ["I2", "I3", "I5"]:
        assert len(MASTER[name]["orientations"]) == 2, name
    assert len(MASTER["T4"]["orientations"]) == 4
    for name in ["L4", "L5", "F5", "N5", "P5", "Y5"]:
        assert len(MASTER[name]["orientations"]) == 8, name


def test_hand_remove():
    h = Hand([p["name"] for p in PIECES])
    assert len(h) == 21
    n = MASTER["I1"]["name"] if "name" in MASTER else "I1"
    h.remove("I1")
    assert len(h) == 20
    assert "I1" not in h.names
    assert "F5" in h.names


def test_hand_random():
    rng = random.Random(0)
    names = [p["name"] for p in PIECES]
    for _ in range(5):
        h = Hand(names[:])
        pick = rng.choice(h.names)
        h.remove(pick)
        assert len(h) == 20
        cells = MASTER[pick]
        for x, y in MASTER[pick]["cells"]:
            assert x >= 0
