"""全 AI 比賽：每局從人格池抽四個互比，結果餵進排行榜。

這是給「人格到底誰比較強」用的工具，不影響玩家對局的流程。

    python3 match.py --games 100            # 記進 records.json
    python3 match.py --games 20 --dry       # 只跑，不寫檔
    python3 match.py --games 100 --reset    # 先清空排行榜再跑
"""
import argparse
import random
import sys

import ai
from config import PERSONALITY_ORDER
from game import Game
from records import Records


def play_match(game, rng, on_move=None):
    """把一局跑到結束，回傳 [(人格, 剩餘格數) × 4]。

    `on_move(game, owner, move)` 可以在每手之後插一個檢查（例如逐手驗證
    角對角規則），跟 `tests/test_simulation.py` 的做法一樣。
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
    """跑 `games` 局全 AI 比賽，逐局記進排行榜，回傳那個 Records。

    `records=None` 時不碰任何檔案（測試就是這樣用的）。
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
