"""Reinforcement-learning scaffolding. Stage G builds the parts; nothing here
learns yet.

The import graph is a rule, not a convention. `rl.actions`, `rl.features` and
`rl.env` must stay importable without `pygame` and without `torch`, because
`tests/test_rl_isolation.py` checks it in a clean subprocess and a data
collection run should not need a GPU to import. Only `rl.smoke_gpu` and, later,
whatever trains, may touch torch.

`engine` is the single source of rules. Legal moves are always computed on a
real-coordinate `State` and only then mapped into the mover's frame through
`rl.actions.LUT_ROT`; calling the engine's rules on a normalised view is not
done anywhere in this package. See `reports/g_report.md` section G5 for why.
"""

__all__ = ("actions", "features", "env", "collect")
