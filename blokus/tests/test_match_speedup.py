"""`match.py` --jobs parallelism correctness tests.

Every test here only checks that parallel execution reproduces the sequential
result bit-for-bit. No timing claims are made.
"""
import os

import pytest

from match import (expand_pool, run_league, _play_game_batch)


def _pool():
    return expand_pool("all")


def _game_points(rows):
    """Per-game per-seat `(remaining, points)` from `run_league` rows."""
    return [
        [(row[2], row[4]) for row in game]
        for game in rows
    ]


def _run(games, seed, jobs, paired_rng=True, pool=None):
    if pool is None:
        pool = _pool()
    return run_league(games, seed=seed, records=None, options=pool,
                      mode="argmax", paired_rng=paired_rng, subject=None,
                      jobs=jobs)


def test_jobs_equals_one_matches_sequential():
    """`--jobs 1` must be bit-for-bit identical to no `--jobs` at all."""
    games, seed = 24, 20260928
    pool = _pool()
    rows_jobs1 = _run(games, seed, jobs=1, paired_rng=True, pool=pool)
    rows_seq = run_league(games, seed=seed, records=None, options=pool,
                          mode="argmax", paired_rng=True, subject=None)
    assert _game_points(rows_jobs1) == _game_points(rows_seq)
    # Also full row equality (option, colour, remaining, rank, points).
    assert rows_jobs1 == rows_seq


def test_jobs_n_equals_jobs_1_after_aggregation():
    """`--jobs 4` must equal `--jobs 1` after in-order reassembly."""
    games, seed = 30, 20260928
    pool = _pool()
    rows_j4 = _run(games, seed, jobs=4, paired_rng=True, pool=pool)
    rows_j1 = _run(games, seed, jobs=1, paired_rng=True, pool=pool)
    assert _game_points(rows_j4) == _game_points(rows_j1)
    assert rows_j4 == rows_j1


def test_jobs_different_job_counts_agree():
    """Various job counts all produce the same aggregated result."""
    games, seed = 20, 7
    pool = _pool()
    base = _run(games, seed, jobs=1, paired_rng=True, pool=pool)
    for jobs in (2, 3, 5):
        assert _run(games, seed, jobs=jobs, paired_rng=True, pool=pool) == base


def test_jobs_refuses_without_paired_rng():
    """`--jobs` without `--paired-rng` must raise ValueError."""
    pool = _pool()
    with pytest.raises(ValueError) as exc:
        run_league(5, seed=20260928, records=None, options=pool,
                   mode="argmax", paired_rng=False, subject=None, jobs=2)
    assert "--jobs requires --paired-rng" in str(exc.value)


def test_worker_sets_env_vars():
    """The worker function pins OMP/MKL/OpenBLAS to 1 thread."""
    before = {k: os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    try:
        _play_game_batch(20260928, [0], _pool(), "argmax", None, True, None)
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
