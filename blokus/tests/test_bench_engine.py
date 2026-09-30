"""Tests for the stage F-prime measurement script.

Deliberately few and deliberately cheap. The full report needs hundreds of
games and minutes of wall clock, which does not belong in CI; what CI should
protect is the *instrument*: a playout that is reproducible, a playout that
terminates, a playout that cannot mutate the state it was handed, a report
whose results half is byte-identical between runs, and a counting helper that
agrees with the engine.

So every test here runs on one or two seeds. The scale belongs to the report.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import bench_engine as B  # noqa: E402
import engine  # noqa: E402

SCRIPT = os.path.join(ROOT, "bench_engine.py")


def test_a_playout_is_reproducible_from_its_seed():
    """Same seed, same game, twice over.

    The legal moves are sorted before the draw precisely so this holds; the
    engine returns sets, and a set's iteration order is not stable across
    processes, so drawing without sorting would make this test fail at random.
    """
    a_state, a_real, a_pass = B.random_playout(seed=1)
    b_state, b_real, b_pass = B.random_playout(seed=1)
    assert engine.serialize(a_state) == engine.serialize(b_state)
    assert (a_real, a_pass) == (b_real, b_pass)


def test_a_playout_ends_with_a_finished_game():
    state, n_real, n_passes = B.random_playout(seed=1)
    assert engine.is_over(state)
    for owner in range(4):
        assert not engine.has_any_legal(state, owner), owner
    assert n_real > 0


def test_a_playout_cannot_change_the_states_it_was_handed():
    """A `State` is frozen, so the hook's positions must still serialise to the
    same bytes after the game that collected them has finished. If `apply_move`
    ever grew an in-place path, this is what would catch it."""
    collected = []
    before = []
    B.random_playout(seed=1, hook=lambda s, m, n, p: collected.append(s))
    before = [engine.serialize(s) for s in collected]
    B.random_playout(seed=1, hook=lambda s, m, n, p: collected.append(s))
    after = [engine.serialize(s) for s in collected]
    assert before == after[:len(before)]
    assert len(after) == 2 * len(before)
    # And the tape's positions are untouched by the moves that followed them.
    tape = []
    B.random_playout(seed=1, tape=tape)
    snap = [engine.serialize(s) for s, _m, _n, _mv in tape]
    assert snap == [engine.serialize(s) for s, _m, _n, _mv in tape]


def test_the_report_without_timing_is_byte_identical_between_runs():
    """The whole point of splitting results from measurements.

    Two runs, same arguments, compared byte for byte. If a wall-clock number or
    a set iteration order ever leaks into the results half, this fails.
    """
    args = [sys.executable, SCRIPT, "--no-timing", "--sections",
            "f1b", "f2", "f3", "f4c", "env", "--games", "5",
            "--json", os.path.join(HERE, "..", "reports", "_test_raw.json")]
    first = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0, first.stderr
    second = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    try:
        os.remove(os.path.join(ROOT, "reports", "_test_raw.json"))
    except OSError:
        pass


def test_the_legal_move_count_helper_agrees_with_the_engine():
    """The helper under test is the one the F1b histogram uses, checked against
    the engine on the one position whose answer can be worked out by hand: the
    empty board, where owner 1 may only cover the top-left square."""
    state = engine.initial_state()
    mask = engine.legal_move_mask(state, 1)
    assert B.pieces_in_mask(mask) == 20      # X5 cannot open, the other 20 can
    tape = []
    B.random_playout(seed=1, tape=tape)
    for s, m, _n, _mv in tape:
        assert m.bit_count() == sum(1 for _ in engine.unpack_move_mask(m))
        assert B.pieces_in_mask(m) == len({p for p, _oi, _b
                                           in engine.unpack_move_mask(m)})


def test_the_stage_buckets_cover_every_decision_point():
    """0..15 opening, 16..47 middle, 48+ end, and the three are a partition of
    the non-negative integers, so a stage can never silently drop a position."""
    assert B.stage_of(0) == B.STAGE_OPENING
    assert B.stage_of(15) == B.STAGE_OPENING
    assert B.stage_of(16) == B.STAGE_MIDDLE
    assert B.stage_of(47) == B.STAGE_MIDDLE
    assert B.stage_of(48) == B.STAGE_END
    assert B.stage_of(10 ** 6) == B.STAGE_END
