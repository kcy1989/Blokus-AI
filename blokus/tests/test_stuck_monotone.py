"""C1: a player who has no legal move never gets one back.

`Game._all_stuck` re-checks all four players after every single move, which is
the second-largest call-chain in the profile. C1 wants to mark a player stuck
once and never re-check them. That is only sound if being stuck is permanent,
so this test tries hard to break that assumption *before* the flag exists.

Why it should hold, for any player p:
  * p's contact rule depends only on p's own stones (need = diagonal contact
    with p, avoid = edge contact with p), so it cannot change unless p moves;
  * a player with no legal move never moves, so their stones never change;
  * `empt` only shrinks as the game goes on, so no empty square comes back.

The check therefore is: every time the engine declares a player stuck, re-test
that player with a fresh, independent legality query on every later move, and
assert they never regain a move. A single counterexample kills C1.
"""
import os
import random

import ai
from game import Game

# The brute-force re-check is the expensive part: a stuck player has no early
# exit, so every re-check costs a full scan of all placements. The full 1000
# game run is the C1 gate and takes ~20 minutes, so it is opt-in:
#
#     BLOKUS_STUCK_GAMES=1000 python3 -m pytest tests/test_stuck_monotone.py
#
# The default is a quick smoke run that still exercises real endgames, because
# that is where a false "stuck is permanent" would show up.
GAMES = int(os.environ.get("BLOKUS_STUCK_GAMES", "120"))


def _independent_has_legal(g, owner):
    """Deliberately *not* `Game.has_legal`.

    This re-derives legality through `Board.has_legal_move`, the original
    brute-force path, so the check cannot be fooled by whatever the engine
    decides to cache.
    """
    from pieces import MASTER
    cells_iter = [MASTER[n]["orientations"] for n in g.hands[owner].names]
    return g.board.has_legal_move(owner, cells_iter, g.must_cover(owner),
                                  g.reach(owner))


def test_a_stuck_player_never_regains_a_legal_move():
    """The C1 precondition. One counterexample in GAMES games fails it."""
    keys = list(ai.personality_keys())
    stuck_events = 0
    for seed in range(GAMES):
        g = Game(random.Random(seed))
        g.setup_match(random.Random(seed).sample(keys, 4))
        g.start()
        ever_stuck = [False] * 4
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            # Re-test anyone already declared stuck. This is the whole point of
            # the test: the engine may skip them, we may not.
            for o in range(4):
                if ever_stuck[o] and _independent_has_legal(g, o):
                    raise AssertionError(
                        "seed %d turn %d: owner %d regained a legal move after "
                        "being stuck" % (seed, g.turn_count, o))
            owner = g.current_owner()
            if not g.has_legal(owner):
                ever_stuck[owner] = True
                stuck_events += 1
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], g.rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner),
                                reach=g.reach(owner),
                                other_must_cover={x: g.must_cover(x)
                                                  for x in range(4)},
                                other_reach={x: g.reach(x) for x in range(4)})
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)
    # The flag is only worth adding if players actually do get stuck, otherwise
    # there is nothing to skip.
    assert stuck_events > 0, "no player was ever stuck in %d games" % GAMES


def test_stuck_flag_matches_recomputation_exactly():
    """The cached flag must agree with a fresh `has_legal` in both directions.

    C1 trades a repeated scan for a remembered answer, so the memory has to be
    right. This replays real games and checks, at every turn, that
    `stuck[o] == not has_legal(o)` for every seat.

Both directions matter now. A flag set too early would silently skip a
    player who still had a move. A flag left unset when the seat had no move
    used to be harmless - `Game._all_stuck` used to return at the first seat
    that could still move, so seats later in the loop were never tested that
    pass - but `stuck` feeds four network feature channels, so "I can still
    move" for a seat that cannot is a lie the model is trained on. plan7-A
    removed the short-circuit, which is what makes the exact equivalence hold:
    `stuck[o]` must equal `not has_legal(o)`, with no third state.
    """
    keys = list(ai.personality_keys())
    checked = 0
    for seed in range(25):
        g = Game(random.Random(seed))
        g.setup_match(random.Random(seed).sample(keys, 4))
        g.start()
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            for o in range(4):
                can = g.has_legal(o)
                assert bool(g.stuck[o]) == (not can), (
                    seed, g.turn_count, o,
                    "stuck=%s but has_legal=%s; the latch must be exact"
                    % (g.stuck[o], can))
                checked += 1
            owner = g.current_owner()
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], g.rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner),
                                reach=g.reach(owner),
                                other_must_cover={x: g.must_cover(x)
                                                  for x in range(4)},
                                other_reach={x: g.reach(x) for x in range(4)})
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)
    # A vacuous pass would be an equality checked zero times, so say how many
    # comparisons actually ran.
    assert checked > 0, "no seat was ever checked"


def test_stuck_flag_is_cleared_on_reset_and_start():
    """A new game must not inherit the previous game's stuck seats."""
    keys = list(ai.personality_keys())
    g = Game(random.Random(0))
    g.setup_match(random.Random(0).sample(keys, 4))
    g.start()
    guard = 0
    while g.state == "PLAYING" and guard < 600:
        guard += 1
        owner = g.current_owner()
        mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                            g.brains[owner], g.rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner),
                            reach=g.reach(owner),
                            other_must_cover={x: g.must_cover(x)
                                              for x in range(4)},
                            other_reach={x: g.reach(x) for x in range(4)})
        if mv is None:
            g.act_pass()
        else:
            g.act(*mv)
    assert any(g.stuck), "expected at least one stuck seat at the end"
    g.start()
    assert g.stuck == [False] * 4
    g.reset()
    assert g.stuck == [False] * 4


def test_stuck_is_monotone_within_a_game():
    """Cheaper companion: once stuck, always stuck, in a single long game.

    Complements the 1000-game test by going deep on one board: a single game
    exercises a crowded endgame, where C1 would be most likely to be wrong.
    """
    keys = list(ai.personality_keys())
    for seed in range(20):
        g = Game(random.Random(10_000 + seed))
        g.setup_match(random.Random(10_000 + seed).sample(keys, 4))
        g.start()
        stuck = [False] * 4
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            for o in range(4):
                if stuck[o]:
                    assert not _independent_has_legal(g, o), \
                        (seed, g.turn_count, o)
            owner = g.current_owner()
            if not g.has_legal(owner):
                stuck[owner] = True
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], g.rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner),
                                reach=g.reach(owner),
                                other_must_cover={x: g.must_cover(x)
                                                  for x in range(4)},
                                other_reach={x: g.reach(x) for x in range(4)})
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)
