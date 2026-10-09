"""Record each seat's first N piece selections, then stop the game there.

This layer answers a descriptive question only: which pieces do a seat reach
for in the opening. It writes counts. Nothing here tests anything, and the
distributions it produces may not be used as evidence for a claim - plan9 step
4 asks for the shape of the opening and says in the same breath that it proves
nothing.

The games are laid out exactly as `match.run_league` lays them out - same
`paired_streams(seed, i)`, same `subject_draw`, same `Game(game_rng)`, same
`setup_seats`, same `play_match` - so an opening recorded here is the opening
the corresponding batch played. The one difference is that the game is
abandoned once N selections are recorded: the rest of the game is not what is
being measured, and stopping early is what keeps a 3000-game run affordable.

`play_match` calls `on_move` only for a placed piece, never for a pass, so the
hook sees exactly the piece selections and the ply counter counts selections,
not turns. The hook fires *before* the move is applied, so the Nth selection is
recorded and the game stops with it unplayed - which loses nothing, because
nothing after the Nth selection is part of the measurement.

Usage (the twelve-seat batch shape, one student per run):

    .venv-rl/bin/python openings.py \
        --pool rl_h1000_0k,rl_o1000_0k,rl_b1000_0k,rl_i1000_0k,hunter,optimizer,builder,intruder,fox,chess,wolf,rl_h1000_20k \
        --subject rl_h1000_20k --seed 20261005 --games 3000 --mode argmax \
        --stop-after 10 --device cuda \
        --out eval/rl-single-train/openings-12-h.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import match  # noqa: E402
import seats  # noqa: E402
from game import Game  # noqa: E402

NOTE = ("descriptive counts of the first N piece selections per seat option; "
        "passes are not counted, each game stops after N selections, and no "
        "test is performed on these numbers")


class _Stop(Exception):
    """Raised from the move hook once the opening has been recorded."""


def new_tally(stop_after):
    """The per-run counters: appearances, selections by option and by ply.

    `game_moves` is the current game's own counter - the stop is per game, and
    a shared running total would stop every game after the first one had spent
    the budget.
    """
    return {
        "appearances": {},
        "openings": {},
        "moves": 0,
        "game_moves": 0,
        "games": 0,
        "games_with_moves": 0,
        "stop_after": stop_after,
    }


def make_hook(tally):
    """The `on_move` hook: record, then stop the game at the Nth selection."""
    stop_after = tally["stop_after"]

    def hook(game, owner, move):
        tally["game_moves"] += 1
        tally["moves"] += 1
        ply = tally["game_moves"]
        if ply <= stop_after:
            key = game.owner_key[owner]
            entry = tally["openings"].setdefault(
                key, {"moves": 0, "pieces": {}, "by_ply": {}})
            entry["moves"] += 1
            entry["pieces"][move[0]] = entry["pieces"].get(move[0], 0) + 1
            by_ply = entry["by_ply"].setdefault(str(ply), {})
            by_ply[move[0]] = by_ply.get(move[0], 0) + 1
        if tally["game_moves"] >= stop_after:
            raise _Stop

    return hook


def run_batch(pool_raw, pool, pool_order, subject, seed, games, mode, device,
              stop_after, adhoc, progress=500):
    """Play `games` openings and return the payload (no file touched here).

    The loop is `match.run_league`'s loop with `play_match` wrapped in the
    recording hook, kept in step with it deliberately: a game's layout must be
    the batch's layout, or the openings would describe games nobody played.
    """
    if seed is None:
        raise ValueError("a seed is required")
    rng_mode = match.rng_mode_of(True, subject)
    tally = new_tally(stop_after)
    hook = make_hook(tally)

    for index in range(games):
        tally["game_moves"] = 0
        game_rng, seat_rngs = match.paired_streams(seed, index)
        game = Game(game_rng)
        if subject is None:
            keys = [game_rng.choice(pool) for _ in range(4)]
        else:
            _seat, keys = match.subject_draw(game_rng, pool, subject)
        game.setup_seats(keys, [None] * 4, game_rng, seat_rngs=seat_rngs,
                         device=device, mode=mode)
        try:
            match.play_match(game, game_rng, hook, seat_rngs=seat_rngs)
        except _Stop:
            pass
        tally["games"] += 1
        if tally["game_moves"] >= stop_after:
            tally["games_with_moves"] += 1
        for owner in range(4):
            key = game.owner_key[owner]
            tally["appearances"][key] = tally["appearances"].get(key, 0) + 1
        if progress and (index + 1) % progress == 0:
            print("  %d/%d 局" % (index + 1, games), flush=True)

    return {
        "pool_raw": pool_raw,
        "pool": pool,
        "pool_size": len(pool),
        "pool_order": pool_order,
        "seed": seed,
        "games": games,
        "mode": mode,
        "rng_mode": rng_mode,
        "subject": subject,
        "adhoc": {k: dict(v) for k, v in adhoc.items()},
        "stop_after_moves": stop_after,
        "games_with_moves": tally["games_with_moves"],
        "appearances": tally["appearances"],
        "openings": tally["openings"],
        "note": NOTE,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="記錄每個席位開局前 N 手選塊，第 N 手後停局")
    ap.add_argument("--pool", required=True,
                    help="逗號分隔，無空白；可含 preset 與 --adhoc 的 key")
    ap.add_argument("--pool-order", choices=("sorted", "literal"),
                    default="sorted", help="池順序，與 match.py 同義")
    ap.add_argument("--subject", default=None,
                    help="每局必有一席的選項；需 --paired-rng，這裡恆為成對")
    ap.add_argument("--adhoc", action="append", default=[],
                    metavar="KEY=PATH",
                    help="把未註冊的檢查點當席位用，可重複")
    ap.add_argument("--seed", type=int, required=True, help="種子")
    ap.add_argument("--games", type=int, default=3000, help="局數（預設 3000）")
    ap.add_argument("--mode", choices=("argmax", "softmax"), default="argmax",
                    help="模仿選項的選步方式（預設 argmax）")
    ap.add_argument("--device", default="cuda",
                    help="模仿選項的 torch device（預設 cuda）")
    ap.add_argument("--stop-after", type=int, default=10,
                    help="錄到第幾手選塊後停局（預設 10）")
    ap.add_argument("--out", required=True, help="輸出檔；已存在則拒絕寫入")
    args = ap.parse_args(argv)

    if os.path.exists(args.out):
        raise SystemExit("%s already exists; refusing to overwrite it"
                         % args.out)
    for spec in args.adhoc:
        if "=" not in spec:
            raise SystemExit("--adhoc expects KEY=PATH, got %r" % (spec,))
        key, _sep, path = spec.partition("=")
        if not key or not path:
            raise SystemExit("--adhoc expects KEY=PATH, got %r" % (spec,))
        seats.register_adhoc(key, path)
    pool = match.expand_pool(args.pool, order=args.pool_order)
    subject = match.validate_subject(args.subject, pool, True,
                                     canonicalise=(args.pool_order != "literal"))

    print("池        : %s（%d 席）" % (args.pool, len(pool)))
    print("subject   : %s" % (subject or "(無，四席皆隨機抽)"))
    print("種子/局數 : %d / %d，mode %s，第 %d 手後停局"
          % (args.seed, args.games, args.mode, args.stop_after))
    if seats.ADHOC:
        for key in sorted(seats.ADHOC):
            print("  adhoc %-16s %s" % (key, seats.ADHOC[key]["path"]))
    print()

    payload = run_batch(args.pool, pool, args.pool_order, subject, args.seed,
                        args.games, args.mode, args.device, args.stop_after,
                        seats.ADHOC)

    directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(directory, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print("已寫入 %s（%d 局，其中 %d 局錄滿 %d 手，共 %d 手選塊）"
          % (args.out, payload["games"], payload["games_with_moves"],
             args.stop_after,
             sum(v["moves"] for v in payload["openings"].values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
