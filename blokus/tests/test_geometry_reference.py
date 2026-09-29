"""A1: the first-move corner rule, checked against a brute-force reference.

The first move of a game may only cover the player's own start corner. That is
the `cbit` branch of `formulas._adjoining_bases`, which runs exactly once per
game per player - so a wrong answer there would be very hard to notice in play
while quietly corrupting the candidate list for owner 0, 2 and 3.

These tests pin that branch to an independent brute-force enumeration.
"""
import random

import pytest

import ai.formulas as F
import board
from board import ALL, N
from config import B, OWNER_CORNER
from pieces import MASTER

ALL_CORNER_BITS = {owner: 1 << (cx + cy * B)
                   for owner, (cx, cy) in OWNER_CORNER.items()}


def brute_first(od, cbit):
    """Bases where at least one cell of the piece lands on the corner."""
    out = 0
    for b in od["bases"]:
        if (od["m"] << b) & cbit:
            out |= 1 << b
    return out


def brute_legal(od, empt, reach, cbit):
    """Bases that are on the board, all empty, and satisfy the contact rule."""
    out = 0
    for b in od["bases"]:
        shifted = od["m"] << b
        if (empt & shifted) != shifted:
            continue
        if cbit:
            if not shifted & cbit:
                continue
        elif reach is not None:
            if not shifted & reach.need or shifted & reach.avoid:
                continue
        out |= 1 << b
    return out


def all_orientations():
    for name in sorted(MASTER):
        for oi, od in sorted(F.ODIRS[name].items()):
            yield name, oi, od


def test_first_move_bases_all_corners():
    for owner, cbit in sorted(ALL_CORNER_BITS.items()):
        for name, oi, od in all_orientations():
            got = F._adjoining_bases(od, None, cbit)
            assert got == brute_first(od, cbit), (owner, name, oi)
            assert F._legal_bases(od, ALL, None, cbit) == brute_first(od, cbit), \
                (owner, name, oi)


def test_first_move_branch_is_not_vacuous():
    """Every corner must admit at least one placement.

    A branch that returns 0 for a corner would make that player auto-pass on
    its opening move, so each corner needs real coverage - not just agreement
    with the reference on a mask that is empty for both.
    """
    for owner, cbit in sorted(ALL_CORNER_BITS.items()):
        total = 0
        for _name, _oi, od in all_orientations():
            total += F._legal_bases(od, ALL, None, cbit).bit_count()
        assert total > 0, owner


def test_legal_bases_matches_brute_force_on_a_partial_board():
    """Same check, but with pieces on the board so `empt` actually bites."""
    rng = random.Random(20240501)
    corners = list(ALL_CORNER_BITS.values())
    for trial in range(300):
        empt = ALL
        for _ in range(rng.randint(1, 40)):
            empt &= ~(1 << rng.randrange(N))
        cbit = rng.choice(corners) if trial % 2 else 0
        for _name, _oi, od in all_orientations():
            got = F._legal_bases(od, empt, None, cbit)
            assert got == brute_legal(od, empt, None, cbit), \
                (trial, _name, _oi, cbit)


def test_legal_bases_with_reach_matches_brute_force():
    """The ordinary (non-first-move) branch against brute force."""
    from board import Reach
    rng = random.Random(20240502)
    for trial in range(200):
        own = 0
        for _ in range(rng.randint(1, 12)):
            own |= 1 << rng.randrange(N)
        empt = ALL & ~own
        for _ in range(rng.randint(0, 20)):
            empt &= ~(1 << rng.randrange(N))
        if not own:
            continue
        need = board.dilate_diag(own) & empt
        avoid = board.dilate(own) & empt
        reach = Reach(need, avoid)
        for _name, _oi, od in all_orientations():
            got = F._legal_bases(od, empt, reach, 0)
            assert got == brute_legal(od, empt, reach, 0), \
                (trial, _name, _oi)


def test_orientation_cells_are_normalised():
    """Every orientation is shifted so its minimum x and y are both 0.

    `rot_base` in the rotation tables later depends on this, and a piece whose
    bounding box does not start at the origin would shift by a silent offset.
    """
    for name in sorted(MASTER):
        for oi, cells in enumerate(MASTER[name]["orientations"]):
            assert min(x for x, _ in cells) == 0, (name, oi)
            assert min(y for _, y in cells) == 0, (name, oi)


def test_action_table_frozen():
    """The published action table must still match the piece definitions.

    `action_table.json` is the contract between the game and any future neural
    net: index i means piece `pieces[i]`. If it ever drifts, old datasets and
    old models silently become garbage, so the hash is pinned here.
    """
    import hashlib
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "action_table.json")
    with open(path) as fh:
        table = json.load(fh)

    names = sorted(MASTER)
    actions = [(n, oi) for n in names
               for oi in range(len(MASTER[n]["orientations"]))]
    assert table["pieces"] == names
    assert [tuple(a) for a in table["actions"]] == actions

    blob = json.dumps({"pieces": names, "actions": actions})
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    assert table["hash"] == digest, (table["hash"], digest)

    assert len(names) == 21
    assert len(actions) == 91
    assert sum(MASTER[n]["size"] for n in names) == 89

    # total_placements is derived, not stored blindly: recompute it two ways -
    # from the raw piece geometry (what the freeze tool does) and from the
    # precomputed ODIRS table (what the game actually uses) - and require them
    # to agree with each other and with the published value.
    from config import B
    raw = 0
    for name in names:
        for cells in MASTER[name]["orientations"]:
            w = max(x for x, _ in cells) + 1
            h = max(y for _, y in cells) + 1
            raw += (B - w + 1) * (B - h + 1)
    precomputed = sum(od["valid"].bit_count() for _n, _oi, od in all_orientations())
    assert raw == precomputed, (raw, precomputed)
    assert table["total_placements"] == raw
