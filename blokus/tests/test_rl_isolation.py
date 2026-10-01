"""The import graph is part of the contract, not a convention.

Two claims, both checked in a clean subprocess because checking them in this
process would prove nothing - everything here has already imported everything
else by the time pytest runs:

    import engine       must not pull in torch
    import rl.actions   must not pull in pygame, or torch

The reason is practical rather than aesthetic. A data collection run needs
numpy and nothing else; a headless training run needs torch; the human game
needs pygame. Keeping the three apart is what lets any of them run without the
other two, and `rl.smoke_gpu` importing torch is fine precisely because nothing
else in `rl/` does.
"""
import subprocess
import sys
import os


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=HERE)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_the_engine_does_not_pull_in_torch():
    """The rules must stay usable on a machine that has no GPU stack at all."""
    assert _run("import sys, engine; print('torch' in sys.modules)") == "False"


def test_importing_rl_actions_pulls_in_neither_pygame_nor_torch():
    """`rl.actions` is the module a data collection run imports, and it has to
    work with neither pygame nor torch installed."""
    code = ("import sys, rl.actions; "
            "print('pygame' in sys.modules, 'torch' in sys.modules)")
    assert _run(code) == "False False"


def test_the_other_rl_modules_are_clean_too():
    """Same rule for `features` and `env`: the whole runtime path, not just one
    module, stays free of the display and the GPU."""
    for mod in ("rl.features", "rl.env"):
        code = ("import sys, %s; "
                "print('pygame' in sys.modules, 'torch' in sys.modules)" % mod)
        assert _run(code) == "False False", mod


def test_importing_engine_still_does_not_pull_in_numpy():
    """The E3 check, restated here so the dependency story lives in one file:
    numpy is imported inside the two functions that need it."""
    assert _run("import sys, engine; print('numpy' in sys.modules)") == "False"


def test_importing_rl_smoke_gpu_also_defers_its_torch():
    """Even the one module allowed to need torch does not import it at module
    level, so `rl` stays importable in an interpreter without it. Calling
    `gpu_facts()` is what pulls torch in, and that is the call that should fail
    loudly rather than the import that should."""
    code = ("import sys, rl.smoke_gpu; print('torch' in sys.modules)")
    assert _run(code) == "False"


def test_rl_smoke_gpu_can_reach_torch_where_torch_is_installed():
    """The mirror image, skipped when torch is absent. This is the check that
    runs in `.venv-rl`, and it is the reason `rl/smoke_gpu.py` is a function
    around the import rather than an import at the top."""
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch is not installed in this interpreter")
    code = ("import rl.smoke_gpu as s; "
            "print(s.gpu_facts()['torch_version'])")
    assert _run(code).startswith("2.")