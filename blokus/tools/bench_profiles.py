"""Per-personality move timings. Measurement only - no game code is changed.

`cProfile` answers "which function", but not "which personality is expensive",
and the two point at different fixes: a slow rule-based brain and a slow
weighted brain have nothing to do with each other. This wraps `choose_move`
from the outside and times it per `brain.key`, so the engine keeps running
exactly as it does in `match.py`.

    python3 tools/bench_profiles.py --games 20
    python3 tools/bench_profiles.py --games 20 --seed 7
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
import game  # noqa: E402
from game import Game  # noqa: E402

TIMES = defaultdict(float)
TURNS = defaultdict(int)

_real_choose = ai.choose_move


def timed_choose(board, hand_names, owner, brain, rng, **kw):
    """Time one decision, attributed to the personality that made it.

    Wall clock is the right unit: anything that cares about latency (a live
    game, an MCTS node budget) pays wall clock, and a counting proxy would
    have to guess which counter matches it.
    """
    t0 = time.perf_counter()
    try:
        return _real_choose(board, hand_names, owner, brain, rng, **kw)
    finally:
        dt = time.perf_counter() - t0
        TIMES[brain.key] += dt
        TURNS[brain.key] += 1


def play_one(g):
    """Play one game to the end; return the four remaining-cell counts."""
    guard = 0
    while g.state == "PLAYING" and guard < 600:
        guard += 1
        o = g.current_owner()
        mv = ai.choose_move(g.board, g.hands[o].names, o, g.brains[o], g.rng,
                            other_brains=g.brains,
                            must_cover=g.must_cover(o), reach=g.reach(o),
                            other_must_cover={x: g.must_cover(x) for x in range(4)},
                            other_reach={x: g.reach(x) for x in range(4)})
        if mv is None:
            g.act_pass()
        else:
            g.act(*mv)
    return [g.remaining_cells(o) for o in range(4)]


def main():
    ap = argparse.ArgumentParser(description="per-personality move timings")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the seeds so two runs are comparable")
    args = ap.parse_args()

    # Patch every reference the game holds, so nothing bypasses the timer.
    ai.choose_move = timed_choose
    ai.chooser.choose_move = timed_choose
    game.choose_move = timed_choose

    keys = list(ai.personality_keys())
    t0 = time.perf_counter()
    cells = 0
    for i in range(args.games):
        if args.seed is None:
            g = Game(random.Random())
            g.setup_match(random.Random().sample(keys, 4))
        else:
            seed = args.seed + i
            g = Game(random.Random(seed))
            g.setup_match(random.Random(seed).sample(keys, 4))
        g.start()
        cells += sum(play_one(g))
    elapsed = time.perf_counter() - t0

    print("games %d in %.2fs  (%.3f s/game)" % (args.games, elapsed,
                                                elapsed / args.games))
    print("cells left over: %d" % cells)
    print()
    print("%-10s %7s %10s" % ("key", "turns", "ms/turn"))
    total_turns = sum(TURNS.values())
    for key in sorted(TIMES, key=lambda k: -TIMES[k]):
        turns = TURNS[key]
        print("%-10s %7d %10.2f" % (key, turns, TIMES[key] / turns * 1000))
    print()
    print("total turns %d, mean %.2f ms/turn"
          % (total_turns, sum(TIMES.values()) / total_turns * 1000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
