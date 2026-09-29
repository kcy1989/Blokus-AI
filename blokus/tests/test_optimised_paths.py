"""C2/C3/C4: the optimised paths must agree with the obvious code.

The full-size runs live in `tools/verify_same_moves.py`:

    python3 tools/verify_same_moves.py --check-has-legal 10000   # C2
    python3 tools/verify_same_moves.py --check-candidates 1000  # C3
    python3 tools/verify_same_moves.py --impl frozen --games 1000 --record base.json
    python3 tools/verify_same_moves.py --impl current --games 1000 --baseline base.json

Those take a long time because the reference is deliberately the slow,
obviously-correct implementation. This file keeps the same three properties
covered on a small sample, so the routine suite still catches a regression
immediately instead of only after a two-hour run.
"""
import os
import random
import threading

import ai
import ai.formulas as F
from game import Game
from pieces import MASTER

GAMES = int(os.environ.get("BLOKUS_EQUIV_GAMES", "12"))


def _play_one(seed):
    keys = list(ai.personality_keys())
    g = Game(random.Random(seed))
    g.setup_match(random.Random(seed).sample(keys, 4))
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
    return g


def _step_wise(seed):
    """Yield the game at every turn so a caller can inspect intermediate state."""
    keys = list(ai.personality_keys())
    g = Game(random.Random(seed))
    g.setup_match(random.Random(seed).sample(keys, 4))
    g.start()
    guard = 0
    while g.state == "PLAYING" and guard < 600:
        guard += 1
        yield g
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


# ------------------------------------------------------------------- C2

def test_has_legal_bitmask_matches_the_brute_force_reference():
    """C2: `Game.has_legal` must equal `Board.has_legal_move` at every step."""
    checks = 0
    for seed in range(GAMES):
        for g in _step_wise(seed):
            for o in range(4):
                cells_iter = [MASTER[n]["orientations"]
                              for n in g.hands[o].names]
                old = g.board.has_legal_move(o, cells_iter, g.must_cover(o),
                                             g.reach(o))
                new = g.has_legal(o)
                checks += 1
                assert old == new, (seed, g.turn_count, o, old, new)
    assert checks > 0


# ------------------------------------------------------------------- C3

def test_candidate_list_is_identical_element_for_element():
    """C3: the bitmask walk must produce the old list, in the old order."""
    import ai.chooser as CH
    from tools.verify_same_moves import brute_candidates
    rows = 0
    for seed in range(GAMES):
        for g in _step_wise(seed):
            owner = g.current_owner()
            old = brute_candidates(g.board, g.hands[owner].names, owner,
                                   g.brains[owner], g.must_cover(owner),
                                   g.reach(owner))
            empt = g.board.empty_bits
            open_a, block, defend = F.board_feats(g.board.grid)
            new = CH._candidates(g.board, list(g.hands[owner].names), owner,
                                 g.brains[owner].profile,
                                 g.must_cover(owner), g.reach(owner),
                                 open_a, block, defend, g.board.corner_regions,
                                 g.board.borders, empt)
            rows += len(old)
            assert old == new, (seed, g.turn_count, owner, len(old), len(new))
    assert rows > 0


def test_candidate_order_is_still_piece_then_oi_then_base():
    """Hard rule 4: never reorder the enumeration.

    `cands.sort` is stable, so the order the candidates are appended in decides
    every tie. The bitmask walk must keep visiting piece, then orientation,
    then base ascending - the same order the old per-base scan produced.
    """
    import ai.chooser as CH
    from pieces import MASTER as M
    for seed in range(3):
        for g in _step_wise(seed):
            owner = g.current_owner()
            empt = g.board.empty_bits
            open_a, block, defend = F.board_feats(g.board.grid)
            cands = CH._candidates(g.board, list(g.hands[owner].names), owner,
                                   g.brains[owner].profile,
                                   g.must_cover(owner), g.reach(owner),
                                   open_a, block, defend,
                                   g.board.corner_regions, g.board.borders,
                                   empt)
            # Independent rank: every (piece, oi, base) triple has a fixed
            # position in "piece, then oi, then base ascending" order. The
            # candidate list must be strictly increasing by that rank, which
            # holds no matter which bases happen to be legal.
            rank = {}
            for name in g.hands[owner].names:
                for oi in range(len(M[name]["orientations"])):
                    cells = M[name]["orientations"][oi]
                    mx = max(x for x, _ in cells)
                    my = max(y for _, y in cells)
                    for y in range(F.B - my):
                        for x in range(F.B - mx):
                            rank[(name, oi, x + y * F.B)] = len(rank)
            ranks = [rank[(n, oi, b)] for _s, n, oi, b in cands]
            assert ranks == sorted(ranks), (seed, g.turn_count, owner)
            assert len(set(ranks)) == len(ranks), (seed, g.turn_count, owner)


# ------------------------------------------------------------------- C4

def _record_games(games, switch):
    F.USE_WALL_BUDGET = switch
    try:
        out = []
        for seed in range(games):
            keys = list(ai.personality_keys())
            g = Game(random.Random(seed))
            g.setup_match(random.Random(seed).sample(keys, 4))
            g.start()
            rec = []
            while g.state == "PLAYING":
                o = g.current_owner()
                mv = ai.choose_move(g.board, g.hands[o].names, o, g.brains[o],
                                    g.rng, other_brains=g.brains,
                                    must_cover=g.must_cover(o), reach=g.reach(o),
                                    other_must_cover={x: g.must_cover(x)
                                                      for x in range(4)},
                                    other_reach={x: g.reach(x) for x in range(4)})
                rec.append((g.turn_count, o, mv))
                if mv is None:
                    g.act_pass()
                else:
                    g.act(*mv)
            out.append(rec)
        return out
    finally:
        F.USE_WALL_BUDGET = True


def test_wall_budget_switch_is_off_by_default_and_reversible():
    assert F.USE_WALL_BUDGET is True, "default must keep the historical behaviour"


def test_without_the_wall_cap_results_do_not_depend_on_machine_load():
    """C4: the point of the switch is reproducibility, so prove it under load."""
    a = _record_games(6, False)
    b = _record_games(6, False)
    assert a == b, "two unloaded runs differed"

    stop = [False]

    def burn():
        while not stop[0]:
            sum(i * i for i in range(20000))

    threads = [threading.Thread(target=burn) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        loaded = _record_games(6, False)
    finally:
        stop[0] = True
        for t in threads:
            t.join()
    assert a == loaded, "a loaded machine changed the moves"
