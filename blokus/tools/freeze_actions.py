"""Freeze the action space into `action_table.json`.

Run this once per project root:

    python3 tools/freeze_actions.py

It writes the contract between the game and anything that indexes actions
(a dataset, a network, a replay log): piece index i means `pieces[i]`, and
action j means `actions[j] = (piece_name, orientation_index)`.

The file is *published data*. `test_action_table_frozen` recomputes the hash
from the live piece definitions, so if a piece ever changes shape the test
fails loudly instead of silently invalidating every stored dataset and model.

Nothing here imports pygame or the AI: this must keep working as a pure data
tool.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pieces import MASTER  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "action_table.json")


def build_table():
    """Assemble the table and its derived counts, asserting the invariants."""
    names = sorted(MASTER)                       # piece bit index 0..20
    table = [(n, oi) for n in names
             for oi in range(len(MASTER[n]["orientations"]))]

    # Counts are computed, never hard-coded, so a piece edit shows up here.
    assert len(names) == 21, "expected 21 piece types, got %d" % len(names)
    assert len(table) == 91, "expected 91 orientations, got %d" % len(table)
    total_cells = sum(MASTER[n]["size"] for n in names)
    assert total_cells == 89, "expected 89 cells, got %d" % total_cells

    # total_placements: how many (orientation, base) pairs exist in total, i.e.
    # the full unfiltered action space before any legality test. Computed from
    # the raw piece data so this tool stays a pure data script with no game or
    # AI imports; `test_action_table_frozen` cross-checks it against the
    # precomputed `ODIRS` table so the two definitions cannot drift apart.
    from config import B
    total_placements = 0
    for n in names:
        for cells in MASTER[n]["orientations"]:
            w = max(x for x, _ in cells) + 1
            h = max(y for _, y in cells) + 1
            total_placements += (B - w + 1) * (B - h + 1)

    return names, table, total_cells, total_placements


def digest_of(names, table):
    """Stable content hash. Changing it invalidates all published data."""
    blob = json.dumps({"pieces": names, "actions": table})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def render(names, table, total_cells, total_placements, digest):
    """Serialise for review: one line per piece and per action.

    The *hash* is taken over `digest_of`, not over these bytes, so the layout
    stays free to change for readability without invalidating published data.
    """
    lines = ['{']
    lines.append(' "pieces": %s,' % json.dumps(names))
    lines.append(' "actions": [')
    body = [json.dumps([n, oi]) for n, oi in table]
    lines.extend("  %s%s" % (row, "," if i < len(body) - 1 else "")
                 for i, row in enumerate(body))
    lines.append(' ],')
    lines.append(' "total_cells": %d,' % total_cells)
    lines.append(' "total_placements": %d,' % total_placements)
    lines.append(' "hash": %s' % json.dumps(digest))
    lines.append('}')
    return "\n".join(lines) + "\n"


def main():
    names, table, total_cells, total_placements = build_table()
    digest = digest_of(names, table)
    with open(OUT_PATH, "w") as fh:
        fh.write(render(names, table, total_cells, total_placements, digest))

    print("wrote %s" % os.path.relpath(OUT_PATH, PROJECT_ROOT))
    print("  pieces           %d" % len(names))
    print("  orientations     %d" % len(table))
    print("  cells            %d" % total_cells)
    print("  total_placements %d" % total_placements)
    print("  hash             %s" % digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
