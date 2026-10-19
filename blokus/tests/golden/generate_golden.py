"""Regenerate `tests/golden/golden.json` from the current tree.

    .venv-test/bin/python tests/golden/generate_golden.py

Not collected by pytest (the name is not `test_*`). Run it only when the
baseline is *meant* to move - a deliberate rules change, not a refactor.
A refactor that changes any recorded move is a refactor that changed
behaviour.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from golden_cases import CASES, replay  # noqa: E402


def main():
    out = {case[0]: replay(case) for case in CASES}
    path = os.path.join(HERE, "golden.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    total = sum(len(g["moves"]) for g in out.values())
    print("wrote %s (%d cases, %d plies)" % (path, len(out), total))


if __name__ == "__main__":
    main()
