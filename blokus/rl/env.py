"""This layer answers one question: what does a caller have to do to play one
game, and what does a finished game pay out?

Two traps live inside the engine and are hidden here rather than left for every
caller to rediscover.

The first is that `state.to_move` is not a promise. `_advance` scans owners in
id order and stops at the first who can still move, so the player handed the
turn may have no legal move at all - stage F' measured this at 5.7% of decision
points and 84% of games. A loop that reads `to_move`, asks for the legal moves
and steps would occasionally be handed an empty list. `step` therefore passes
on until somebody can actually play, and the contract it offers is the useful
one: if `done` is False, `legal_indices()` is not empty.

The second is ties. `Game.standings` breaks a tie by seat number, which is
deterministic and also biased, so it is not used here. Places come from the
competition-ranking rule that `tools.benchmark.average_ranks` implements,
which shares a place evenly across everyone who tied. Utilities are
`(4 - place) / 3`, so first place is 1.0, last is 0.0, and the four of them
always add to 2.0 - a constant that makes a value estimate comparable across
games however the places fell.

Legal moves come from `rl.actions.legal_indices`, computed on the real
coordinates and rotated afterwards. The engine's rules are never asked about a
normalised view.
"""
import numpy as np

import engine
from config import CLOCKWISE_OWNERS
from rl.actions import index_to_move, legal_indices, seat_of_mover

# A pass is not a choice, so a run of them has to end. Stage F' never saw more
# than three in a row; 100 is three orders of magnitude of headroom and still a
# number small enough that reaching it means the game is broken rather than
# unlucky.
MAX_CONSECUTIVE_PASSES = 100

_EMPTY = np.empty(0, dtype=np.int32)


def average_places(remaining):
    """Competition ranking with tied places split evenly, fewest cells first.

    The same rule as `tools.benchmark.average_ranks`, written out rather than
    imported: that function is this project's ranking oracle and `plan3.md`
    asks for the ranking here to be checked against it, which an import would
    make vacuous. `tests/test_rl_env.py` compares the two over 1,000 games.
    """
    order = sorted(range(len(remaining)), key=lambda i: remaining[i])
    places = [0.0] * len(remaining)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and remaining[order[j + 1]] == remaining[order[i]]:
            j += 1
        shared = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            places[order[k]] = shared
        i = j + 1
    return places


def utilities_from_remaining(remaining):
    """`(4 - place) / 3` per player, in the order they were given."""
    return [(4.0 - p) / 3.0 for p in average_places(remaining)]


def utilities(state):
    """The four utilities of a finished position, indexed by real owner id.

    A free function as well as a method: a caller holding a bare `State` - a
    replay, a dataset row - wants the same numbers without building an
    environment first.
    """
    return utilities_from_remaining(engine.result(state))


def aux_targets(state):
    """Remaining cells over 89, indexed by real owner id."""
    return [engine.remaining_cells(state, o) / 89.0 for o in range(4)]


class BlokusEnv:
    """One game of Blokus over `engine`, addressed by real action index.

    Deliberately not a gym environment. It is the smallest object that hides
    the two traps above, and nothing here rewards, terminates on a timer or
    resets itself: this stage builds no policy, so there is nothing to plug in
    yet.
    """

    def __init__(self):
        self._state = None
        self._n_passes = 0
        self._ply = 0

    # -------------------------------------------------------------- lifecycle

    def reset(self):
        """An empty board with the plain clockwise seating.

        The seating is the fixed clockwise order rather than a random
        permutation. `engine.initial_state` already defaults to it, and
        rotating the seating per game would only make two games differ in a
        way the features are designed to be blind to.
        """
        self._state = engine.initial_state(CLOCKWISE_OWNERS)
        assert self._state.turn_order == tuple(CLOCKWISE_OWNERS)
        self._n_passes = 0
        self._ply = 0
        return self._state

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("call reset() before asking for the state")
        return self._state

    @property
    def done(self):
        return engine.is_over(self.state)

    @property
    def passes(self):
        """How many turns this game has spent on passes."""
        return self._n_passes

    @property
    def ply(self):
        """How many pieces have been played."""
        return self._ply

    # -------------------------------------------------------------- the move

    def legal_indices(self):
        """Sorted legal action indices in real coordinates; empty when done."""
        if self.done:
            return _EMPTY
        return legal_indices(self.state)

    def step(self, real_index):
        """Play `real_index`, then pass on for as long as nobody can move.

        Returns `{"done": bool, "passes_after": int}`. `passes_after` is the
        running total for the whole game rather than the count for this call,
        because "how much of this game was spent skipping" is the number worth
        having.
        """
        state = self.state
        if engine.is_over(state):
            raise ValueError("the game is over; call reset() first")
        available = legal_indices(state)
        idx = int(real_index)
        if idx not in available:
            raise ValueError("action %d is not legal here (%d legal)"
                             % (idx, available.size))
        self._state = engine.apply_move(state, index_to_move(idx))
        self._ply += 1
        passes = 0
        while not engine.is_over(self._state) \
                and engine.legal_move_mask(self._state) == 0:
            self._state = engine.pass_turn(self._state)
            passes += 1
            if passes > MAX_CONSECUTIVE_PASSES:
                raise RuntimeError(
                    "passed %d times in a row without the game finishing; the "
                    "position is not progressing" % passes)
        self._n_passes += passes
        return {"done": engine.is_over(self._state),
                "passes_after": self._n_passes}

    # -------------------------------------------------------------- payoff

    def utilities(self):
        """`(4,)` floats indexed by real owner id, summing to 2.0."""
        return utilities(self.state)

    def utilities_in_view(self):
        """The same four numbers in the mover's seat order.

        A network sees one position at a time and its output row 0 has to mean
        "me", so the payoff has to be reordered the same way. The seat order
        comes from `engine.normalize`, which is where it is defined.
        """
        _view, seat_of = engine.normalize(self.state)
        util = utilities(self.state)
        return [util[o] for o in seat_of]

    def aux_targets(self):
        """`(4,)` remaining cells over 89, in the mover's seat order."""
        _view, seat_of = engine.normalize(self.state)
        return [engine.remaining_cells(self.state, o) / 89.0 for o in seat_of]

    def remaining(self):
        """`(4,)` remaining cells indexed by real owner id."""
        return engine.result(self.state)

    def seat_of_mover(self):
        return seat_of_mover(self.state)

    def __repr__(self):
        if self._state is None:
            return "BlokusEnv(unreset)"
        n_legal = 0 if self.done else self.legal_indices().size
        return ("BlokusEnv(ply=%d, to_move=%d, legal=%d, done=%s)"
                % (self._ply, self._state.to_move, n_legal, self.done))