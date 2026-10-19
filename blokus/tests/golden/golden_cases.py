"""The golden baseline's cases and the one function that plays them.

A golden game is an end-to-end pin: given a fixed seed and a fixed seat
layout, the whole game must come out move for move the same. It is the
strongest check a refactor can be put against, because it exercises the
seats, the RNG draws, the rules and the AI pipeline together and compares
the result to bytes recorded before the refactor.

Two deliberate properties:

  * **Personalities only.** No key here names a checkpoint, so playing a
    case loads no weights and imports no torch. The baseline must run in a
    CPU-only interpreter and must not read `data/` or `records.json`.
  * **The wall-clock budget is off.** `formulas.USE_WALL_BUDGET` makes the
    AI skip opponent lookahead once a move takes too long, which is a
    function of machine load rather than of the position - a golden game
    recorded with it on would differ between a quiet and a busy machine.
    `replay` drops the cap for the duration of the game and puts it back,
    so the pinned path is the reproducible one (`SIM_EVAL_BUDGET` still
    bounds the cost).

`replay` is the single definition of "play this case". Both the generator
and the test call it, so a case cannot be recorded one way and checked
another.

**Two blind spots, written down rather than discovered later:**

  * **The wall-clock cap never fires here.** With the budget off, the
    `elapsed >= WALL_BUDGET -> skip opponent lookahead` branch of
    `ai/chooser.py` is not exercised by any case. No test covers it today
    either; it is a phase-2 backlog item, not something to add now.
  * **No case holds an rl or imitation seat.** Every key below is a
    personality, so `build_brain` never takes the network path and the
    per-seat RNG consumption of a checkpoint seat is not pinned. Before
    phase 2 touches `Game.setup_seats` (the `brain_factory` injection), add
    a pin that builds a seat with a **stub** brain and asserts the RNG
    consumption order - otherwise a change there can move the shared-stream
    turn-order draw without any case noticing.
"""
import random

import ai
from ai import formulas as F
from game import Game

# (id, mode, keys, colours, seed).
#
#   mode "match"  -> `Game.setup_match()`, which draws four distinct
#                    personalities from the game's own rng. The draw is part
#                    of what the case pins.
#   mode "seats"  -> `Game.setup_seats(keys, colours)`, with rng left as
#                    `None` so it uses the game's rng - the same call the UI
#                    makes, and the same stream the AI draws from.
#
# The spread is on purpose: three random draws, all-rule-based, all-weighted
# with a repeated seat, repeated rule-based seats, and one with hand-picked
# colours.
CASES = (
    ("match_1", "match", None, None, 1),
    ("match_42", "match", None, None, 42),
    ("match_20260928", "match", None, None, 20260928),
    ("rules_seed7", "seats",
     ("hunter", "optimizer", "builder", "intruder"), None, 7),
    ("weighted_seed13", "seats",
     ("wolf", "chess", "fox", "wolf"), None, 13),
    ("dupes_seed99", "seats",
     ("intruder", "intruder", "hunter", "hunter"), None, 99),
    ("fixed_colours_seed2026", "seats",
     ("builder", "wolf", "hunter", "chess"),
     ("blue", "green", "red", "yellow"), 2026),
)

# A game is a few dozen plies; this only stops a rules regression from
# looping forever rather than pinning a length.
GUARD = 2000


def replay(case):
    """Play one case to the end and return its recorded shape.

    The return value is a plain JSON-able dict, so the same structure is
    what the generator writes and what the test compares against.
    """
    _cid, mode, keys, colours, seed = case
    previous_budget = F.USE_WALL_BUDGET
    F.USE_WALL_BUDGET = False
    try:
        game = Game(random.Random(seed))
        if mode == "match":
            game.setup_match(keys)
        else:
            game.setup_seats(keys, colours)
        game.start()
        moves = []
        guard = 0
        while game.state == "PLAYING" and guard < GUARD:
            guard += 1
            owner = game.current_owner()
            move = ai.choose_move(
                game.board, game.hands[owner].names, owner, game.brains[owner],
                game.rng, other_brains=game.brain_map(),
                must_cover=game.must_cover(owner), reach=game.reach(owner),
                other_must_cover={x: game.must_cover(x) for x in range(4)},
                other_reach={x: game.reach(x) for x in range(4)})
            if move is None:
                moves.append([owner, "PASS"])
                game.act_pass()
            else:
                name, oi, x, y = move
                moves.append([owner, name, oi, x, y])
                game.act(name, oi, x, y)
        return {
            "turn_order": list(game.turn_order),
            "colors": [game.colors[o] for o in range(4)],
            "owner_key": [game.owner_key[o] for o in range(4)],
            "moves": moves,
            "remaining": [game.remaining_cells(o) for o in range(4)],
            "state": game.state,
        }
    finally:
        F.USE_WALL_BUDGET = previous_budget
