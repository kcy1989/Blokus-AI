"""The seven personalities, each in its own module.

Plan9a stage 5 moved them here from the top of `ai/`; the only edits were
import paths (`..base`, `..formulas`) and the `module` field in
`ai/registry.json`, which `tests/test_registry_json.py` compares against
`cls.__module__`. `ai/__init__.py` still re-exports every brain class, so
`ai.HunterBrain` and the rest keep working, and `hunter`'s import of
`optimizer` stays a one-dot import because both moved together.

This package is imported while `ai/__init__.py` is still executing, so
`from .. import formulas` resolves against a half-initialised `ai` exactly the
way `from . import formulas` did before the move.
"""
