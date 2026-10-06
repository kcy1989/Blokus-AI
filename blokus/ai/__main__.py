"""`python -m ai --check`: the registry's health check.

The entry point lives on the *package* rather than on `ai.registry`, because
`ai/__init__.py` re-exports `ai.registry` - so `python -m ai.registry` would
execute that module a second time, beside the copy the package already
imported, and runpy warns about the duplicate before a line of it runs. Running
the package instead imports `ai` once and then executes `ai/__main__.py`,
which nobody has imported yet, and there is nothing to duplicate.
"""
import sys

from .registry import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
