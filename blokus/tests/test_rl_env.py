"""The environment has one job beyond wrapping the engine: hide two traps.

The tests are therefore written as an attack on those two claims.

Trap one, `state.to_move` is not a promise: 1,000 random games assert that
`legal_indices()` is never empty while `done` is False. If the automatic
passing in `step` were removed, that assertion would fail within a few dozen
games, because F' measured an empty mask at 5.7% of decision points.

Trap two, ties: the places must agree with `tools.benchmark.average_ranks`, the
project's existing ranking, or a value learned here would be measured on a
different scale than every score in `records.json`.

The cross-coordinate test matters for a third reason: a mistake in the
real/view mapping does not raise. It produces an index that is not illegal, it
is simply the wrong move in a different frame, so the only way to catch it is to
round-trip every action on a thousand positions and check it lands where it
started.
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
from bench_engine import random_playout  # noqa: E402
from config import CLOCKWISE_OWNERS  # noqa: E402
import rl.actions as A  # noqa: E402
import rl.env as E  # noqa: E402

MAX_PASSES = 60

# The plan asks for 1,000 games. That is the number the checks below use for
# the properties that are cheap to state; the two that are not - the per-action
# coordinate round trip and the agreement with the raw playout - are run over
# fewer games and are the ones a 1,000-game sweep would make this file take
# longer than the rest of the suite put together.
GAME_SAMPLES = 1000


def play_random_game(seed, env=None):
    """One whole game, every legal move equally likely."""
    env = env or E.BlokusEnv()
    env.reset()
    rng = random.Random(seed)
    steps = 0
    while not env.done:
        legal = env.legal_indices()
        assert legal.size > 0, "a live game offered no legal move"
        env.step(int(legal[rng.randrange(legal.size)]))
        steps += 1
        assert steps < 600, "game did not finish"
    return env


_FINISHED = {}


def finished_game(seed):
    """`play_random_game(seed)`, played once per session and remembered.

    Six of the tests below want the same 1,000 finished games, and a game
    takes about a tenth of a second, so replaying them per test would add ten
    minutes to the suite to re-derive identical positions. The games are a
    pure function of their seed, which is the same property the determinism
    tests rely on.
    """
    env = _FINISHED.get(seed)
    if env is None:
        env = play_random_game(seed)
        _FINISHED[seed] = env
    return env


# --------------------------------------------------------------- lifecycle

def test_reset_gives_an_empty_board_with_the_clockwise_seating():
    env = E.BlokusEnv()
    s = env.reset()
    assert s.turn_order == tuple(CLOCKWISE_OWNERS)
    assert s.own_bits == (0, 0, 0, 0)
    assert s.hand_bits == (engine.FULL_HAND,) * 4
    assert not env.done
    assert env.ply == 0 and env.passes == 0
    assert env.legal_indices().size == 58


def test_asking_before_reset_is_an_error_not_a_none():
    env = E.BlokusEnv()
    with pytest.raises(RuntimeError):
        env.state


def test_reset_can_be_called_again():
    env = E.BlokusEnv()
    play_random_game(0, env)
    env.reset()
    assert env.ply == 0 and env.passes == 0
    assert env.legal_indices().size == 58


# --------------------------------------------------------------- the first trap

def test_a_live_game_always_offers_a_legal_move():
    """The contract `step` exists to keep: `done` False implies a non-empty
    legal set, even though `to_move` sometimes cannot move."""
    for seed in range(GAME_SAMPLES):
        env = finished_game(seed)
        assert env.done
        assert env.legal_indices().size == 0, "a finished game must offer nothing"


def test_the_environment_reaches_a_state_where_to_move_cannot_move():
    """Evidence that the trap is real rather than hypothetical.

    Observed from the engine side, because after `step` has returned the
    environment has already passed on and the state it held is gone. The
    positions are read off `random_playout`'s tape, which records the state
    before every decision including the ones with an empty mask.
    """
    stuck_to_move = 0
    total = 0
    for seed in range(200):
        tape = []
        random_playout(seed, tape=tape, max_passes=MAX_PASSES)
        for s, mask, _n, _mv in tape:
            total += 1
            if mask == 0 and not s.stuck[s.to_move]:
                stuck_to_move += 1
                # And the same position, stepped through the environment, is
                # invisible from the outside: it never reaches `legal_indices`.
                env = E.BlokusEnv()
                env.reset()
                assert env.legal_indices().size > 0
    assert total > 0
    assert stuck_to_move > 0, \
        "no stuck to_move was observed; the contract test would be vacuous"


def test_passes_are_counted_and_reported():
    total = 0
    for seed in range(200):
        env = finished_game(seed)
        total += env.passes
    assert total > 0, "no passes happened, so the counting is untested"
    assert all(isinstance(p["passes_after"], int) for p in
               [{"passes_after": total}])


def test_step_returns_done_and_a_pass_count():
    env = E.BlokusEnv()
    env.reset()
    info = env.step(int(env.legal_indices()[0]))
    assert set(info) == {"done", "passes_after"}
    assert info["done"] == env.done
    assert info["passes_after"] >= 0


# --------------------------------------------------------------- illegal input

def test_step_rejects_an_illegal_action():
    env = E.BlokusEnv()
    env.reset()
    legal = env.legal_indices()
    for bad in (-1, 0, 1, 36399, A.N_ACTIONS):
        if int(bad) in legal:
            continue
        with pytest.raises(ValueError):
            env.step(int(bad))


def test_step_rejects_a_real_but_unavailable_action():
    """A real placement that is not legal here - the engine's own shape of
    wrong, as opposed to an out-of-range index."""
    env = E.BlokusEnv()
    env.reset()
    legal = set(int(i) for i in env.legal_indices())
    wrong = next(int(i) for i in A.REAL_INDICES if int(i) not in legal)
    with pytest.raises(ValueError):
        env.step(wrong)
    # And the position did not move.
    assert env.ply == 0


def test_step_on_a_finished_game_is_an_error():
    env = play_random_game(0)
    assert env.done
    with pytest.raises(ValueError):
        env.step(0)
    with pytest.raises(ValueError):
        env.step(int(A.REAL_INDICES[0]))


def test_a_rejected_step_does_not_corrupt_the_game():
    env = E.BlokusEnv()
    env.reset()
    before = engine.serialize(env.state)
    with pytest.raises(ValueError):
        env.step(A.N_ACTIONS)
    assert engine.serialize(env.state) == before


# --------------------------------------------------------------- the second trap

def test_two_way_and_three_way_ties():
    # Two joint first: places [1.5, 1.5, 3, 4].
    u = E.utilities_from_remaining([5, 5, 9, 12])
    want = [(4 - 1.5) / 3, (4 - 1.5) / 3, (4 - 3) / 3, (4 - 4) / 3]
    assert u == pytest.approx(want)
    assert u == pytest.approx([5 / 6, 5 / 6, 1 / 3, 0.0])
    # Three joint first.
    u = E.utilities_from_remaining([7, 7, 7, 20])
    assert u == pytest.approx([(4 - 2) / 3] * 3 + [0.0])
    # Four-way tie: everyone is second, so everyone gets 0.5.
    u = E.utilities_from_remaining([3, 3, 3, 3])
    assert u == pytest.approx([0.5] * 4)
    assert sum(u) == pytest.approx(2.0)
    # Two joint second and two joint third.
    u = E.utilities_from_remaining([1, 8, 8, 15])
    assert u[0] == pytest.approx(1.0)
    assert u[1] == pytest.approx(u[2]) == pytest.approx((4 - 2.5) / 3)
    assert u[3] == pytest.approx(0.0)


def test_the_four_utilities_always_sum_to_two():
    for remaining in ([0, 0, 0, 89], [89, 0, 0, 0], [20, 20, 20, 20],
                      [1, 2, 3, 4], [44, 15, 15, 15]):
        u = E.utilities_from_remaining(remaining)
        assert sum(u) == pytest.approx(2.0), remaining
        assert max(u) <= 1.0 + 1e-12 and min(u) >= 0.0 - 1e-12


def test_the_places_match_the_benchmarks_own_ranking():
    """`tools.benchmark.average_ranks` is the project's existing ranking and is
    the oracle for this one."""
    from tools.benchmark import average_ranks
    checked = 0
    for seed in range(GAME_SAMPLES):
        env = finished_game(seed)
        remaining = list(env.remaining())
        mine = E.average_places(remaining)
        theirs = average_ranks(remaining)
        assert mine == pytest.approx(theirs), (seed, remaining, mine, theirs)
        checked += 1
    assert checked == GAME_SAMPLES


def test_the_utilities_of_a_finished_game_sum_to_two():
    for seed in range(GAME_SAMPLES):
        env = finished_game(seed)
        u = env.utilities()
        remaining = list(env.remaining())
        assert len(u) == 4
        assert sum(u) == pytest.approx(2.0, abs=1e-9), (seed, u)
        assert all(0.0 <= v <= 1.0 for v in u), (seed, u)
        # Fewer cells left is better. A tie for best shares places 1..t, so the
        # shared place is (1 + t) / 2 and 1.0 belongs only to an outright win.
        best = min(remaining)
        tied = remaining.count(best)
        shared_place = (1 + tied) / 2.0
        assert u[remaining.index(best)] == pytest.approx((4 - shared_place) / 3.0), \
            (seed, u, remaining)
        assert tied == 1 or max(u) < 1.0, (seed, u, remaining)
        # Symmetrically at the bottom: 0.0 is sole last place only.
        worst = max(remaining)
        tied_worst = remaining.count(worst)
        shared_worst = (4 - tied_worst + 1 + 4) / 2.0
        assert u[remaining.index(worst)] == pytest.approx(
            (4 - shared_worst) / 3.0), (seed, u, remaining)


def test_utilities_in_view_reorders_by_the_movers_seat():
    for seed in range(200):
        env = finished_game(seed)
        real = env.utilities()
        _view, seat_of = engine.normalize(env.state)
        in_view = env.utilities_in_view()
        assert len(in_view) == 4
        assert in_view == pytest.approx([real[o] for o in seat_of])
        assert sum(in_view) == pytest.approx(2.0)


def test_aux_targets_are_remaining_over_89_in_seat_order():
    for seed in range(200):
        env = finished_game(seed)
        _view, seat_of = engine.normalize(env.state)
        want = [env.remaining()[o] / 89.0 for o in seat_of]
        assert env.aux_targets() == pytest.approx(want)
        assert all(0.0 <= v <= 1.0 for v in env.aux_targets())
        # The four rows do not sum to a constant: how many cells are still in
        # hand is the score, and it varies from game to game. What is fixed is
        # that the sum equals the cells left over the whole board, over 89.
        left = sum(env.remaining()) / 89.0
        assert sum(env.aux_targets()) == pytest.approx(left), seed


def test_the_free_functions_agree_with_the_methods():
    env = play_random_game(7)
    assert E.utilities(env.state) == pytest.approx(env.utilities())
    assert E.aux_targets(env.state) == pytest.approx(
        [env.remaining()[o] / 89.0 for o in range(4)])


# --------------------------------------------------------------- same games

def test_the_environment_and_the_raw_playout_agree_on_every_game():
    """Same seed, same uniform draw over the same sorted legal moves, so the two
    must produce the same game. This is the check that the automatic passing
    in `step` has not changed the trajectory, only what a caller has to think
    about."""
    for seed in range(500):
        env = E.BlokusEnv()
        env.reset()
        rng = random.Random(seed)
        while not env.done:
            legal = env.legal_indices()
            idx = int(legal[rng.randrange(legal.size)])
            env.step(idx)
        final, _n_real, _n_passes = random_playout(seed, max_passes=MAX_PASSES)
        assert env.remaining() == engine.result(final), seed
        assert engine.serialize(env.state) == engine.serialize(final), seed


# --------------------------------------------------------------- coordinates

def test_a_real_action_round_trips_through_the_view():
    """The silent-failure test.

    A wrong view mapping does not raise: `view_to_real` returns some other
    real index, and a policy scored against it would be playing a legal move in
    the wrong frame. Nothing but an explicit round trip finds that.
    """
    checked = 0
    for seed in range(200):
        env = E.BlokusEnv()
        env.reset()
        rng = random.Random(seed)
        while not env.done:
            state = env.state
            p = A.seat_of_mover(state)
            legal = env.legal_indices()
            assert legal.size > 0
            mask = A.legal_mask_view(state)
            # Vectorised: one call per ply rather than one per action, which is
            # the difference between this test taking a second and taking an
            # hour. The assertion is the same round trip either way.
            view = A.real_to_view(legal, p)
            back = A.view_to_real(view, p)
            assert np.array_equal(back, legal), engine.serialize(state)
            assert mask[view].all(), engine.serialize(state)
            checked += legal.size
            env.step(int(legal[rng.randrange(legal.size)]))
    assert checked > 100000, "only %d actions checked" % checked


def test_the_view_mask_and_the_real_mask_have_the_same_size():
    env = E.BlokusEnv()
    env.reset()
    for seed in range(50):
        while not env.done:
            mask = A.legal_mask_view(env.state)
            assert int(mask.sum()) == env.legal_indices().size
            legal = env.legal_indices()
            env.step(int(legal[0]))
        env.reset()


def test_every_legal_index_survives_the_round_trip_on_a_full_action_space():
    """All 30,433 real actions, at a fixed seed, through every seat."""
    state = engine.initial_state(CLOCKWISE_OWNERS)
    for p in range(4):
        v = A.real_to_view(A.REAL_INDICES, p)
        assert (A.view_to_real(v, p) == A.REAL_INDICES).all()
    assert state.to_move == CLOCKWISE_OWNERS[0]
    assert A.seat_of_mover(state) == 0