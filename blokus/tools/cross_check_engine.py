"""E1 acceptance: play the real `Game` and mirror every move in `engine.py`.

`tests/test_engine_cross.py` pins the second engine against `Game` at a size
that fits in CI. This tool runs the same comparison at the scale the plan asks
for - 10,000 games - and reports where the two disagree, because the whole
point of a second implementation is that a disagreement is possible and must be
found before any data is built on top of it.

    python3 tools/cross_check_engine.py --games 10000

At every single step it compares:

  * the current player's complete legal move set
  * all four `owner_bits`
  * all four `stuck` latches
  * `to_move`

and at the end of each game the final remaining-cell counts and the game-over
verdict.

The legal-move oracle is the existing `ai.formulas._legal_bases` bitmask path,
not `engine.py`'s own code. `--brute-sample` additionally re-derives a sample of
those sets from `Board.can_place`, which loops over every base and every cell, so
the fast oracle is anchored to brute force inside the same run.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
from ai.formulas import ODIRS, _legal_bases  # noqa: E402
import engine  # noqa: E402
from config import B  # noqa: E402
from game import Game  # noqa: E402
from pieces import MASTER  # noqa: E402


def game_legal_move_mask(g, owner):
    """Every legal move for `owner`, packed the same way `engine` packs it."""
    empt = g.board.empty_bits
    reach = g.reach(owner)
    must_cover = g.must_cover(owner)
    cbit = 1 << (must_cover[0] + must_cover[1] * B) if must_cover else 0
    out = 0
    for name in g.hands[owner].names:
        piece_idx = engine.PIECE_IDX[name]
        for oi, od in ODIRS[name].items():
            m = _legal_bases(od, empt, reach, cbit)
            while m:
                low = m & -m
                out |= engine._pack(piece_idx, oi, low.bit_length() - 1)
                m ^= low
    return out


def brute_legal_move_mask(g, owner):
    """The same set, found by scanning every base and every cell.

    Obviously correct and far too slow to run per step, so it is only used to
    audit a sample.
    """
    out = 0
    b = g.board
    for name in g.hands[owner].names:
        piece_idx = engine.PIECE_IDX[name]
        for oi, cells in enumerate(MASTER[name]["orientations"]):
            for y in range(B):
                for x in range(B):
                    if b.can_place(x, y, cells, owner, g.must_cover(owner),
                                   g.reach(owner)):
                        out |= engine._pack(piece_idx, oi, x + y * B)
    return out


def make_game(seed):
    keys = list(ai.personality_keys())
    g = Game(random.Random(seed))
    g.setup_match(random.Random(seed).sample(keys, 4))
    g.start()
    return g


def choose(g, owner):
    return ai.choose_move(g.board, g.hands[owner].names, owner, g.brains[owner],
                          g.rng, other_brains=g.brains,
                          must_cover=g.must_cover(owner),
                          reach=g.reach(owner),
                          other_must_cover={x: g.must_cover(x) for x in range(4)},
                          other_reach={x: g.reach(x) for x in range(4)})


class Report:
    def __init__(self):
        self.diffs = []
        self.steps = 0
        self.legal_checks = 0
        self.legal_moves = 0
        self.brute_checked = 0

    def fail(self, msg):
        if len(self.diffs) < 12:
            self.diffs.append(msg)
        return False


def cross_check(games, brute_sample=0, verbose_every=0):
    rep = Report()
    brute_left = brute_sample
    t0 = time.perf_counter()
    for seed in range(games):
        g = make_game(seed)
        s = engine.initial_state(g.turn_order)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            rep.steps += 1
            o = g.current_owner()

            if s.to_move != o:
                rep.fail("seed=%d turn=%d: to_move %s vs game %s"
                         % (seed, g.turn_count, s.to_move, o))
            if s.stuck != tuple(g.stuck):
                rep.fail("seed=%d turn=%d: stuck %s vs game %s"
                         % (seed, g.turn_count, s.stuck, g.stuck))
            if tuple(s.own_bits) != tuple(g.board.owner_bits):
                rep.fail("seed=%d turn=%d: owner_bits differ"
                         % (seed, g.turn_count))

            if brute_left > 0:
                brute_left -= 1
                rep.brute_checked += 1
                if game_legal_move_mask(g, o) != brute_legal_move_mask(g, o):
                    rep.fail("seed=%d turn=%d: the fast oracle disagrees with "
                             "brute force" % (seed, g.turn_count))

            mine = engine.legal_move_mask(s, o)
            theirs = game_legal_move_mask(g, o)
            rep.legal_checks += 1
            rep.legal_moves += mine.bit_count()
            if mine != theirs:
                rep.fail("seed=%d turn=%d owner=%d: legal move set differs "
                         "(%d vs %d moves)"
                         % (seed, g.turn_count, o, mine.bit_count(),
                            theirs.bit_count()))

            mv = choose(g, o)
            if mv is None:
                g.act_pass()
                s = engine.pass_turn(s)
            else:
                name, oi, x, y = mv
                emv = (engine.PIECE_IDX[name], oi, x + y * B)
                if not mine & engine.move_bit(emv[0], oi, emv[2]):
                    rep.fail("seed=%d turn=%d: the engine rejected the game's "
                             "own move %r" % (seed, g.turn_count, mv))
                    return rep
                g.act(*mv)
                s = engine.apply_move(s, emv)

        if engine.is_over(s) != (g.state == "GAME_OVER"):
            rep.fail("seed=%d: is_over %s vs game state %s"
                     % (seed, engine.is_over(s), g.state))
        if engine.result(s) != tuple(g.remaining_cells(o) for o in range(4)):
            rep.fail("seed=%d: final remaining %s vs %s"
                     % (seed, engine.result(s),
                        tuple(g.remaining_cells(o) for o in range(4))))

        if verbose_every and (seed + 1) % verbose_every == 0:
            print("  %6d/%d games, %d steps, %.0fs elapsed"
                  % (seed + 1, games, rep.steps, time.perf_counter() - t0),
                  flush=True)

    return rep


def main():
    ap = argparse.ArgumentParser(description="cross-check engine.py against Game")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--brute-sample", type=int, default=200,
                    help="audit the fast oracle against can_place this often")
    ap.add_argument("--verbose-every", type=int, default=500)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rep = cross_check(args.games, args.brute_sample, args.verbose_every)
    print("games          : %d" % args.games)
    print("steps compared : %d" % rep.steps)
    print("legal sets     : %d checks, %d legal moves total, %d brute-force "
          "audits" % (rep.legal_checks, rep.legal_moves, rep.brute_checked))
    print("elapsed        : %.0fs" % (time.perf_counter() - t0))
    if rep.diffs:
        print("DIFFERENCES (%d shown):" % len(rep.diffs))
        for d in rep.diffs:
            print("  " + d)
        return 1
    print("IDENTICAL: engine.py and Game agree on every step of every game")
    return 0


if __name__ == "__main__":
    sys.exit(main())
