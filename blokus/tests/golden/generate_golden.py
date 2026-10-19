"""Regenerate `tests/golden/golden.json` from the current tree.

    .venv-test/bin/python tests/golden/generate_golden.py

Not collected by pytest (the name is not `test_*`). Run it only when the
baseline is *meant* to move - a deliberate rules change, not a refactor.
A refactor that changes any recorded move is a refactor that changed
behaviour.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from golden_cases import CASES, replay  # noqa: E402


def _head():
    """The HEAD the baseline was recorded from, or "unknown".

    Recorded rather than assumed: the file is only meaningful next to the
    commit it describes, and the reader should not have to guess which one
    that was.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    games = {case[0]: replay(case) for case in CASES}
    total = sum(len(g["moves"]) for g in games.values())
    payload = {
        "meta": {
            "generated_from_head": _head(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "cases": len(games),
            "plies": total,
            "wall_budget": "off (see golden_cases.py)",
        },
        "games": games,
    }
    path = os.path.join(HERE, "golden.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s (%d cases, %d plies)" % (path, len(games), total))


if __name__ == "__main__":
    main()
