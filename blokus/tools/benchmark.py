"""Fixed-seat benchmark: who is actually strongest, without the seat bias.

`match.py` draws four personalities at random each game, which is the right
shape for "let everyone compete" but a poor instrument for *comparing* them:
seat order, the random starting corner and the personality draw all vary
together, so a run of N games mixes the personality effect with the noise of
which seat it happened to draw.

This tool removes that: you name the four personalities, every seating
permutation of them is played, and each game gets a fixed seed. Same arguments
give byte-identical output.

    python3 tools/benchmark.py --personalities wolf chess fox intruder
    python3 tools/benchmark.py --personalities wolf chess fox optimizer \\
        --games 8 --permutations 24
    python3 tools/benchmark.py --personalities wolf chess fox builder --csv out.csv

Tied scores are split evenly (two players joint second each get 2.5).
`Game.standings` is *not* reused for ranking, because it breaks ties by seat
number - which is exactly the bias this tool exists to remove. `standings`
itself is left alone, since the UI depends on it.

Reproducibility: with the same arguments the *scores* are bit-identical - same
games, same seats, same remaining cells, same ranks. The `ms/move` column is a
wall-clock measurement and therefore never reproduces; it is printed in a
separate block and `--no-timing` omits it entirely if you need a report that
compares byte for byte.
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
from config import B, CLOCKWISE_OWNERS, PLAYER_OWNER  # noqa: E402
from game import MATCH_SEATS, Game  # noqa: E402


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def average_ranks(remaining):
    """Competition ranking with **tied places split evenly**.

    `remaining` is a list of remaining-cell counts. Lower is better, so the
    smallest count is rank 1. Ties share the average of the places they span:
    with [5, 5, 9] the two fives are joint first and get (1 + 2) / 2 = 1.5,
    and the 9 is third. This is the standard "1224" convention, and unlike
    `Game.standings` it never consults a seat number.
    """
    order = sorted(range(len(remaining)), key=lambda i: remaining[i])
    ranks = [0.0] * len(remaining)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and remaining[order[j + 1]] == remaining[order[i]]:
            j += 1
        # places i+1 .. j+1, shared evenly
        shared = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def make_game(keys, seed, rotation):
    """A game with the personalities seated in a fixed order.

    `rotation` picks the starting corner deterministically, so the same seed
    always produces the same first player - the same requirement C4 exists to
    satisfy, applied up front rather than by luck.
    """
    g = Game(random.Random(seed))
    g.setup_match(list(keys))
    g.rng = random.Random(seed)
    g.brains = {o: ai.make_brain(keys[o], random.Random(seed * 4 + o))
                for o in range(MATCH_SEATS)}
    start = (CLOCKWISE_OWNERS.index(PLAYER_OWNER) + rotation) % 4
    g.turn_order = [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]
    g.turn_pos = 0
    g.start()
    return g


def play(keys, seed, rotation, per_move_times):
    g = make_game(keys, seed, rotation)
    guard = 0
    while g.state == "PLAYING" and guard < 600:
        guard += 1
        owner = g.current_owner()
        t0 = time.perf_counter()
        mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                            g.brains[owner], g.rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner),
                            reach=g.reach(owner),
                            other_must_cover={x: g.must_cover(x)
                                              for x in range(4)},
                            other_reach={x: g.reach(x) for x in range(4)})
        dt = time.perf_counter() - t0
        per_move_times.setdefault(keys[owner], []).append(dt)
        if mv is None:
            g.act_pass()
        else:
            g.act(*mv)
    return [g.remaining_cells(o) for o in range(4)]


def main():
    ap = argparse.ArgumentParser(description="fixed-seat benchmark")
    ap.add_argument("--personalities", nargs="+",
                    default=["wolf", "chess", "fox", "optimizer"],
                    help="exactly four distinct personality keys")
    ap.add_argument("--games", type=int, default=4,
                    help="games per seating permutation")
    ap.add_argument("--permutations", type=int, default=24,
                    help="how many seatings to use (0 = all 24)")
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-timing", action="store_true",
                    help="omit the wall-clock column so the report is fully "
                         "byte-reproducible")
    args = ap.parse_args()

    keys = list(args.personalities)
    if len(keys) != MATCH_SEATS:
        ap.error("need exactly %d personalities" % MATCH_SEATS)
    if len(set(keys)) != MATCH_SEATS:
        ap.error("personalities must be distinct")
    unknown = [k for k in keys if k not in ai.personality_keys()]
    if unknown:
        ap.error("unknown personalities: %s" % ", ".join(unknown))

    all_perms = list(itertools.permutations(range(MATCH_SEATS)))
    if args.permutations and args.permutations < len(all_perms):
        # Evenly spaced over the permutation list, so a subset still covers the
        # space rather than just the first N.
        step = len(all_perms) / float(args.permutations)
        picked = [all_perms[int(i * step)] for i in range(args.permutations)]
        perms = sorted(set(picked))
    else:
        perms = all_perms

    remaining = {k: [] for k in keys}
    ranks = {k: [] for k in keys}
    per_move = {k: [] for k in keys}
    n_games = 0

    for perm in perms:
        seated = [keys[i] for i in perm]
        for g_index in range(args.games):
            seed = args.seed + 1009 * n_games
            rotation = n_games % 4
            cells = play(seated, seed, rotation, per_move)
            places = average_ranks(cells)
            for seat, key in enumerate(seated):
                remaining[key].append(cells[seat])
                ranks[key].append(places[seat])
            n_games += 1

    wins = {k: 0 for k in keys}
    for k in keys:
        wins[k] = sum(1 for r in ranks[k] if r == 1.0)

    order = sorted(keys, key=lambda k: (mean(ranks[k]), mean(remaining[k]), k))
    lines = []
    lines.append("personalities : %s" % " ".join(keys))
    lines.append("seatings      : %d of %d" % (len(perms), len(all_perms)))
    lines.append("games         : %d (seed base %d)" % (n_games, args.seed))
    lines.append("")
    head = "%-10s %7s %10s %11s %6s" % ("key", "games", "avg_place",
                                        "avg_cells", "wins")
    if not args.no_timing:
        head += " %9s" % "ms/move"
    lines.append(head)
    rows = []
    for k in order:
        row = {
            "key": k,
            "games": len(ranks[k]),
            "avg_place": mean(ranks[k]),
            "avg_remaining_cells": mean(remaining[k]),
            "wins": wins[k],
        }
        if not args.no_timing:
            row["ms_per_move"] = mean(per_move[k]) * 1000
        rows.append(row)
        line = "%-10s %7d %10.3f %11.2f %6d" % (
            k, row["games"], row["avg_place"], row["avg_remaining_cells"],
            row["wins"])
        if not args.no_timing:
            line += " %9.2f" % row["ms_per_move"]
        lines.append(line)
    if not args.no_timing:
        lines.append("")
        lines.append("(ms/move is wall-clock: a measurement, not a result. "
                     "Scores above reproduce exactly;")
        lines.append(" use --no-timing for a byte-identical report.)")
    report = "\n".join(lines)
    print(report)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nwrote %s" % args.csv)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"personalities": keys, "games": n_games,
                       "rows": rows}, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
