"""All-AI league: each game draws four seats from the eleven automated options,
and the results feed the leaderboard.

This is a tool for answering "which option is actually strongest"; it does not
affect the flow of a normal player game.

The eleven are the seven personalities plus the four H-B2 imitation steps. The
two human seats are excluded: a league of people cannot be replayed from a
seed, and a seat with no brain would have nothing to choose with.

    python3 match.py --games 100            # record into records.json
    python3 match.py --games 20 --dry       # run only, write nothing
    python3 match.py --games 100 --reset    # clear the leaderboard first
    python3 match.py --games 5 --steps      # only the imitation options
"""
import argparse
import json
import os
import random
import sys

import ai
import seats as seats_mod
from game import Game
from records import Records


def league_options():
    """The eleven options a league seat can be drawn from."""
    return seats_mod.seat_options(include_humans=False)


def play_match(game, rng, on_move=None):
    """Play one game to the end, returning `[(seat option, colour, remaining)]`.

    The colour is part of the result, not decoration: the same option can be
    drawn twice in one game on two different corners, and only the pair
    identifies which seat a row is talking about.

    `on_move(game, owner, move)` can be hooked in to check something after each
    turn (verifying the corner-contact rule move by move, say), the same way
    `tests/test_simulation.py` does it.
    """
    game.start()
    while game.state == "PLAYING":
        owner = game.current_owner()
        move = ai.choose_move(game.board, game.hands[owner].names, owner,
                              game.brains[owner], rng,
                              other_brains=game.brain_map(),
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
    return [(game.owner_key[o], game.colors[o], game.remaining_cells(o))
            for o in range(4)]


def run_league(games=100, seed=0, records=None, on_game=None, on_move=None,
               options=None, mode="argmax", checkpoint_dir=None, device=None):
    """Run `games` league matches, recording each into the leaderboard.

    Every seat is drawn independently, so options repeat freely - which is what
    lets a league answer "is the imitation better than the personalities" rather
    than only "which four personalities cooperate best". Colours and the opening
    player are decided by `setup_seats`, the same code the UI uses, so a league
    game and a played game are laid out the same way.

    With `records=None` no file is touched at all, which is how the tests use
    it.
    """
    options = league_options() if options is None else tuple(options)
    if checkpoint_dir is None:
        checkpoint_dir = seats_mod.IMITATION_CHECKPOINT_DIR
    rng = random.Random(seed)
    rows = []
    for i in range(games):
        g = Game(rng)
        keys = [rng.choice(options) for _ in range(4)]
        g.setup_seats(keys, [None] * 4, rng, checkpoint_dir=checkpoint_dir,
                      device=device, mode=mode)
        standings = play_match(g, rng, on_move)
        rows.append(standings)
        if records is not None:
            records.record([(k, rem) for k, _c, rem in standings])
        if on_game is not None:
            on_game(i + 1, standings)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="AI 比賽")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--reset", action="store_true",
                    help="先清空排行榜再開始")
    ap.add_argument("--dry", action="store_true", help="不寫 records.json")
    ap.add_argument("--steps", action="store_true",
                    help="只讓四個模仿選項上場")
    ap.add_argument("--mode", default="argmax", choices=("argmax", "softmax"),
                    help="模仿選項的選步方式")
    ap.add_argument("--device", default=None, help="模仿選項的 torch device")
    ap.add_argument("--json", default=None, help="把逐局結果寫成 JSON")
    args = ap.parse_args(argv)

    options = league_options()
    if args.steps:
        options = tuple(o for o in options
                        if o.startswith("step_"))

    records = None
    if not args.dry:
        records = Records()
        if args.reset:
            records.reset()
            print("排行榜已清空")
    tally = dict.fromkeys(options, 0)
    left = dict.fromkeys(options, 0.0)

    def report(_i, standings):
        for key, _colour, rem in standings:
            tally[key] = tally.get(key, 0) + 1
            left[key] = left.get(key, 0.0) + rem

    rows = run_league(args.games, args.seed, records, on_game=report,
                      options=options, mode=args.mode, device=args.device)
    assert len(rows) == args.games
    print("%d 局完成" % args.games)
    for k in options:
        if tally[k]:
            print("  %-12s 出場 %3d 局，平均餘 %.1f 格"
                  % (k, tally[k], left[k] / tally[k]))
    if records is not None:
        for key, games, pts, rem in records.rows(order=options):
            print("  %-12s %3d 局　平均 %.2f 分　平均餘 %.1f 格"
                  % (key, games, pts, rem))
        print("已寫入 %s" % records.path)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"seed": args.seed, "mode": args.mode,
                       "options": list(options),
                       "games": [[[k, c, r] for k, c, r in row] for row in rows]},
                      fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print("逐局結果已寫入 %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
