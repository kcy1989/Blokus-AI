"""All-AI league: each game draws four seats from the ten automated options,
and the results feed the leaderboard.

This is a tool for answering "which option is actually strongest"; it does not
affect the flow of a normal player game.

The ten are the seven personalities plus the three H-C2 imitation steps. The
two human seats are excluded: a league of people cannot be replayed from a
seed, and a seat with no brain would have nothing to choose with.

    python3 match.py --games 100            # record into records.json
    python3 match.py --games 20 --dry       # run only, write nothing
    python3 match.py --games 100 --reset    # clear the leaderboard first
    python3 match.py --games 5 --pool imitation_only   # only the checkpoints
    python3 match.py --games 5 --pool no_imitation,hunter
"""
import argparse
import json
import math
import os
import random
import sys

import ai
import seats as seats_mod
from config import PROJECT_DIR
from game import Game
from records import POINTS_FOR_RANK, Records, rank_rows

# Where a batch evaluation is written. Under `data/`, which `.gitignore`
# excludes, and deliberately **not** `records.json`: that file is the
# cumulative leaderboard of the human player's own games, and a benchmark run
# would flood it with thousands of machine-vs-machine results.
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "data", "match")

# The three shorthands `--pool` accepts. `all` is the whole automated pool, and
# is spelled out as the ordered tuple rather than a filter so that
# `expand_pool("all")` is the *same list object contents* as the default - the
# same length in the same order, which is what makes `rng.choice` consume the
# random stream identically and keeps a default run bit-for-bit identical to one
# that was given `--pool all`.
#
# Which entries are the imitation seats is asked of `seats.kind_of` rather than
# tested against a string prefix. A prefix test is exactly what goes stale the
# moment the key naming changes, and it fails *silently*: a prefix that matches
# nothing turns `imitation_only` into an empty pool and `no_imitation` into the
# whole pool. That is not hypothetical - the pool moved from `step_*` to `hc_*`
# when H-B2's checkpoints were replaced by H-C2's.
def _is_imitation(key):
    return seats_mod.kind_of(key) == seats_mod.KIND_IMITATION


_ORDER = seats_mod.automated_options()
POOL_PRESETS = {
    "all": _ORDER,
    "no_imitation": tuple(k for k in _ORDER if not _is_imitation(k)),
    "imitation_only": tuple(k for k in _ORDER if _is_imitation(k)),
}


def league_options():
    """The eleven options a league seat can be drawn from."""
    return seats_mod.seat_options(include_humans=False)


def expand_pool(text):
    """`--pool` text -> the deduplicated, ordered list of options to draw from.

    A comma-separated list of items, each either a preset name or one option
    name. Whitespace around items is allowed, and so is the whole string being
    quoted, because that is what a shell hands over when the commas are inside
    quotes.

    **Every distinct option in the result carries the same weight, 1/n.** Items
    are a union, not a multiset: naming `hunter` as well as `no_imitation` does
    not give `hunter` two chances, because `hunter` is already in that preset.
    Repeating an item, or listing a preset and a name that overlaps it, changes
    nothing.

    The order is `seats.automated_options()` order, always - not the order the
    items were given in, and not the traversal order of a set. Two `--pool`
    arguments with the same contents therefore expand to the same list and play
    the same games from the same seed.

    An unknown name is an error naming what is legal, never a silent omission:
    a typo that dropped an option would quietly change what is being measured.
    """
    if text is None:
        return list(_ORDER)
    body = text.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'":
        body = body[1:-1].strip()
    items = [p.strip() for p in body.split(",")]
    items = [p for p in items if p]
    if not items:
        raise ValueError(
            "--pool was given nothing to expand; expected a comma-separated "
            "list of %s or one of %s"
            % (", ".join(_ORDER), ", ".join(sorted(POOL_PRESETS))))
    chosen = set()
    for item in items:
        if item in POOL_PRESETS:
            chosen.update(POOL_PRESETS[item])
        elif item in _ORDER:
            chosen.add(item)
        else:
            raise ValueError(
                "unknown --pool item %r. Presets: %s. Options: %s"
                % (item, ", ".join(sorted(POOL_PRESETS)), ", ".join(_ORDER)))
    # Ordered by the fixed option order, so the input order cannot matter.
    pool = [k for k in _ORDER if k in chosen]
    if not pool:
        raise ValueError("--pool %r expanded to an empty pool" % (text,))
    return pool


def play_match(game, rng, on_move=None):
    """Play one game to the end.

    Returns one row per **seat**: `(option, colour, remaining, rank, points)`.
    The colour is part of the result, not decoration: the same option can be
    drawn twice in one game on two different corners, and only the pair
    identifies which seat a row is about. Rank and points are per seat too, so
    two seats on one option each carry their own - which is the whole reason
    this is a list and not something keyed by option.

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
    # Ranked by the seat option, exactly as the leaderboard folds it, so a row
    # here and that entry cannot disagree.
    #
    # `rank_rows` returns the seats best-first, so it has to be walked alongside
    # the same ordering rather than zipped with seat order. The two sorts use the
    # same key and are both stable over seat order, so even when two seats tie
    # on (remaining, option) they stay paired - and a tie gives them the same
    # rank and the same points anyway.
    standings = [(game.owner_key[o], game.remaining_cells(o)) for o in range(4)]
    order = sorted(range(4), key=lambda o: (standings[o][1], standings[o][0]))
    out = [None] * 4
    for o, (_key, rank, points, _rem) in zip(order, rank_rows(standings)):
        out[o] = (game.owner_key[o], game.colors[o], standings[o][1],
                  rank, points)
    return out


def summarise(rows):
    """Per-option statistics over seat appearances.

    Every row is one seat, so an option that held two seats in a game is
    counted twice - once for each seat's own rank, points and remaining squares.
    That is the same accounting `records.py` folds into the leaderboard, and the
    point of a pool benchmark is to compare options on equal terms: averages are
    per appearance, so they stay on the 0-4 and 0-89 scales however many seats
    an option happened to hold.

    `points_stderr` is the sample standard deviation of the points divided by
    the square root of the number of appearances, so two options can be told
    apart from their averages alone. With a single appearance there is no spread
    to measure and it is reported as 0.0.
    """
    acc = {}
    for key, _colour, remaining, _rank, points in rows:
        rec = acc.setdefault(key, {"appearances": 0, "points": [],
                                   "remaining": []})
        rec["appearances"] += 1
        rec["points"].append(points)
        rec["remaining"].append(remaining)
    out = []
    for key in sorted(acc):
        rec = acc[key]
        pts, rem = rec["points"], rec["remaining"]
        n = len(pts)
        mean_p = sum(pts) / n
        mean_r = sum(rem) / n
        if n > 1:
            var = sum((p - mean_p) ** 2 for p in pts) / (n - 1)
            stderr = math.sqrt(var) / math.sqrt(n)
        else:
            stderr = 0.0
        out.append({
            "option": key,
            "appearances": n,
            "avg_points": mean_p,
            "avg_remaining": mean_r,
            "points_stderr": stderr,
        })
    out.sort(key=lambda r: (-r["avg_points"], r["avg_remaining"], r["option"]))
    return out


def default_out_path(seed, games, pool_raw):
    """Where a batch evaluation is written when `--out` is not given."""
    safe = "".join(c if c.isalnum() else "_" for c in (pool_raw or "all"))
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    return os.path.join(DEFAULT_OUT_DIR,
                        "seed%d-g%d-pool%s.json" % (seed, games, safe[:40]))


def run_league(games=100, seed=0, records=None, on_game=None, on_move=None,
               options=None, mode="argmax", checkpoint_dir=None, device=None):
    """Run `games` league matches, recording each into the leaderboard.

    Every seat is drawn independently from `options`, so options repeat freely -
    which is what lets a league answer "is the imitation better than the
    personalities" rather than only "which four personalities cooperate best".
    Colours and the opening player are decided by `setup_seats`, the same code
    the UI uses, so a league game and a played game are laid out the same way.

    Returns a list of games, each a list of four seat rows as
    `play_match` returns them - the game's rows, not a flat list, because four
    seats is the unit that was played.

    With `records=None` no file is touched at all, which is how the tests use
    it.
    """
    options = league_options() if options is None else tuple(options)
    if not options:
        raise ValueError("run_league needs at least one option to draw from")
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
            records.record([(k, rem) for k, _c, rem, _rk, _p in standings])
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
    ap.add_argument("--pool", default=None,
                    help="逗號分隔,無空白。項目可為 preset 或單一 AI 名稱:"
                         " preset 有 all(%d 個)、no_imitation(%d 個人格)、"
                         "imitation_only(%d 個模仿者);名稱為 "
                         % (len(POOL_PRESETS["all"]),
                            len(POOL_PRESETS["no_imitation"]),
                            len(POOL_PRESETS["imitation_only"]))
                         + ", ".join(seats_mod.automated_options()) + "。"
                         "展開後取聯集並去重,每個不同的 AI 權重相等(各 1/n);"
                         "重複的項目不增加權重。順序固定為選項的既定順序,"
                         "與輸入順序無關。不給則等同 all。")
    ap.add_argument("--out", default=None,
                    help="批量評測輸出檔(JSON);預設寫到 data/match/ 下,"
                         "不寫入 records.json")
    ap.add_argument("--mode", default="argmax", choices=("argmax", "softmax"),
                    help="模仿選項的選步方式")
    ap.add_argument("--device", default=None, help="模仿選項的 torch device")
    args = ap.parse_args(argv)

    pool_raw = args.pool
    try:
        pool = expand_pool(pool_raw)
    except ValueError as exc:
        # A bad pool is a usage error, not a crash: argparse prints the usage
        # and exits 2, which is what a mistyped `--pool` deserves.
        ap.error(str(exc))

    records = None
    if not args.dry:
        records = Records()
        if args.reset:
            records.reset()
            print("排行榜已清空")
    rows = run_league(args.games, args.seed, records, options=pool,
                      mode=args.mode, device=args.device)

    flat = [r for game in rows for r in game]
    summary = summarise(flat)
    appearances = {r["option"]: r["appearances"] for r in summary}

    print("%d 局完成（每局 4 席）" % args.games)
    print("  --pool 原始字串 : %s" % (pool_raw if pool_raw else "(未指定 → all)"))
    print("  展開後的池      : %s" % ", ".join(pool))
    print("  種子            : %d" % args.seed)
    print("  局數            : %d" % args.games)
    print("  席次總數        : %d" % len(flat))
    print()
    print("  各選項席次:")
    for key in pool:
        print("    %-12s %4d 席" % (key, appearances.get(key, 0)))
    print()
    print("  選項            席次   平均分   平均餘格   分數標準誤")
    for row in summary:
        print("    %-12s %5d %8.3f %10.1f %11.4f"
              % (row["option"], row["appearances"], row["avg_points"],
                 row["avg_remaining"], row["points_stderr"]))

    out_path = args.out or default_out_path(args.seed, args.games, pool_raw)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "pool_raw": pool_raw,
            "pool": pool,
            "pool_size": len(pool),
            "seed": args.seed,
            "games": args.games,
            "seats_per_game": 4,
            "mode": args.mode,
            "appearances": appearances,
            "summary": summary,
            "games_detail": [[[k, c, r, rk, p] for k, c, r, rk, p in game]
                             for game in rows],
        }, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("批量評測輸出已寫入 %s" % out_path)

    if records is not None:
        print()
        for key, games, pts, rem in records.rows(order=pool):
            print("  %-12s %3d 席次　平均 %.2f 分　平均餘 %.1f 格"
                  % (key, games, pts, rem))
        print("排行榜已寫入 %s" % records.path)
    else:
        print("排行榜未寫入（--dry）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
