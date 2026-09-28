"""All-AI league: each game draws four personalities to compete, and the
results feed the leaderboard.

This is a tool for answering "which personality is actually strongest"; it does
not affect the flow of a normal player game.

    python3 match.py --games 100            # record into records.json
    python3 match.py --games 20 --dry       # run only, write nothing
    python3 match.py --games 100 --reset    # clear the leaderboard first
"""
import argparse
import random
import sys

import ai
from config import PERSONALITY_ORDER
from game import Game
from records import Records


def play_match(game, rng, on_move=None):
    """Play one game to the end, returning [(personality, remaining) x 4].

    `on_move(game, owner, move)` can be hooked in to check something after each
    turn (verifying the corner-contact rule move by move, say), the same way
    `tests/test_simulation.py` does it.
    """
    game.start()
    while game.state == "PLAYING":
        owner = game.current_owner()
        move = ai.choose_move(game.board, game.hands[owner].names, owner,
                              game.brains[owner], rng, other_brains=game.brains,
                              must_cover=game.must_cover(owner),
                              reach=game.reach(owner),
                              other_must_cover={o: game.must_cover(o)
                                                for o in range(4)},
                              other_reach={o: game.reach(o) for o in range(4)})
        if move:
            if on_move is not None:
                on_move(game, owner, move)
            game.act(*move)
        else:
            game.act_pass()
    return [(game.owner_key[o], game.remaining_cells(o)) for o in range(4)]


def run_league(games=100, seed=0, records=None, on_game=None, on_move=None):
    """Run `games` all-AI matches, recording each into the leaderboard, and
    return that Records.

    With `records=None` no file is touched at all (which is how the tests use
    it).
    """
    rng = random.Random(seed)
    rows = []
    for i in range(games):
        g = Game(rng)
        g.setup_match()
        standings = play_match(g, rng, on_move)
        rows.append(standings)
        if records is not None:
            records.record(standings)
        if on_game is not None:
            on_game(i + 1, standings)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="全 AI 比賽")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--reset", action="store_true",
                    help="先清空排行榜再開始")
    ap.add_argument("--dry", action="store_true", help="不寫 records.json")
    args = ap.parse_args(argv)

    records = None
    if not args.dry:
        records = Records()
        if args.reset:
            records.reset()
            print("排行榜已清空")
    tally = dict.fromkeys(PERSONALITY_ORDER, 0)
    left = dict.fromkeys(PERSONALITY_ORDER, 0.0)

    def report(i, standings):
        for key, rem in standings:
            tally[key] += 1
            left[key] += rem

    rows = run_league(args.games, args.seed, records, on_game=report)
    assert len(rows) == args.games
    print("%d 局完成" % args.games)
    for k in PERSONALITY_ORDER:
        if tally[k]:
            print("  %-9s 出場 %3d 局，平均餘 %.1f 格"
                  % (k, tally[k], left[k] / tally[k]))
    if records is not None:
        for key, games, pts, rem in records.rows(order=PERSONALITY_ORDER):
            print("  %-9s %3d 局　平均 %.2f 分　平均餘 %.1f 格"
                  % (key, games, pts, rem))
        print("已寫入 %s" % records.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
