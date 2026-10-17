#!/usr/bin/env python3
"""Task 3: AI batch calibration with seat effect pilot.

Runs seat-effect pilot (400 games, strict rotation) + batch calibration
(Scheme B, ~2750 games, adjacent pairs), fits blokuselo MLE, writes
ai/ratings.json.
"""
import hashlib
import json
import math
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import match
from blokuselo import rate
import seats as seats_mod

SEED = 20261016
N_WORKERS = 12
K_ELO = 173.7178


def paired_stream_seed(*parts):
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def play_one_game(args):
    """args = (game_index, [key0,key1,key2,key3], seed, mode, device)."""
    game_index, keys, seed, mode, device = args
    game_rng = random.Random(paired_stream_seed(seed, game_index, "setup"))
    seat_rngs = [random.Random(paired_stream_seed(seed, game_index, "seat", o))
                 for o in range(4)]
    g = match.Game(game_rng)
    g.setup_seats(keys, [None] * 4, game_rng, seat_rngs=seat_rngs,
                  checkpoint_dir=None, device=device, mode=mode)
    standings = match.play_match(g, game_rng, None, seat_rngs=seat_rngs)
    # standings: [(option, colour, remaining, rank, points), ...] per seat
    game_rows = [(key, rank) for key, _colour, remaining, rank, _points in standings]
    seat_remaining = [(key, remaining) for key, _colour, remaining, _rank, _points in standings]
    return game_index, game_rows, seat_remaining


def run_batch(assignments, seed, mode="argmax", device=None, n_workers=N_WORKERS):
    """Run game assignments in parallel via spawn workers."""
    args_list = [(gi, keys, seed, mode, device) for gi, keys in assignments]
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(play_one_game, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            game_index, game_rows, seat_remaining = fut.result()
            results[game_index] = (game_rows, seat_remaining)
    return results


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    options = sorted(seats_mod.automated_options())
    print(f"Automated options ({len(options)}): {options}")

    # ============================================================
    # Step 1: Seat effect pilot (400 games, strict seat rotation)
    # ============================================================
    print("\n=== Step 1: Seat effect pilot (400 games) ===")
    pilot_assignments = []
    for rep in range(100):
        draw_rng = random.Random(paired_stream_seed(SEED, rep, "pilot_draw"))
        keys = []
        while len(keys) < 4:
            k = draw_rng.choice(options)
            if k not in keys:
                keys.append(k)
        for rot in range(4):
            rotated = keys[rot:] + keys[:rot]
            gi = rep * 4 + rot
            pilot_assignments.append((gi, rotated))

    pilot_results = run_batch(pilot_assignments, SEED)

    pos_remaining = defaultdict(list)
    for gi in range(len(pilot_assignments)):
        _game_rows, seat_remaining = pilot_results[gi]
        for seat_idx, (_key, remaining) in enumerate(seat_remaining):
            pos_remaining[seat_idx].append(remaining)

    print("Seat effect pilot results:")
    means, ses = {}, {}
    for pos in range(4):
        vals = pos_remaining[pos]
        mean = sum(vals) / len(vals)
        means[pos] = mean
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            ses[pos] = math.sqrt(var) / math.sqrt(len(vals))
        else:
            ses[pos] = 0.0
        print(f"  Seat {pos}: mean_remaining={mean:.3f} SE={ses[pos]:.4f} n={len(vals)}")

    seat_effect = False
    max_diff = 0.0
    max_pair = None
    for p1 in range(4):
        for p2 in range(p1 + 1, 4):
            diff = abs(means[p1] - means[p2])
            combined_se = math.sqrt(ses[p1] ** 2 + ses[p2] ** 2)
            if diff > max_diff:
                max_diff = diff
                max_pair = (p1, p2)
            if diff > 2 * combined_se:
                print(f"  SEAT EFFECT: Seat {p1} vs {p2}: diff={diff:.3f} > 2*SE={2 * combined_se:.3f}")
                seat_effect = True
    if not seat_effect:
        p1, p2 = max_pair
        print(f"  No significant seat effect (max diff={max_diff:.3f} between seats {p1},{p2})")

    # ============================================================
    # Step 2: Preliminary calibration to get seat ordering (200 games)
    # ============================================================
    print("\n=== Step 2: Preliminary calibration (200 games) ===")
    prelim_assignments = []
    for i in range(200):
        draw_rng = random.Random(paired_stream_seed(SEED, 10000 + i, "prelim_draw"))
        keys = []
        while len(keys) < 4:
            k = draw_rng.choice(options)
            if k not in keys:
                keys.append(k)
        prelim_assignments.append((i, keys))

    prelim_results = run_batch(prelim_assignments, SEED + 1)
    prelim_game_data = []
    for i in range(200):
        game_rows, _ = prelim_results[i]
        prelim_game_data.append(game_rows)

    prelim_mu = rate(prelim_game_data, anchor="optimizer")
    sorted_seats = sorted(prelim_mu.keys(), key=lambda k: -prelim_mu[k][0])
    print("Preliminary ranking (strongest first):")
    for i, seat in enumerate(sorted_seats):
        mu, sigma = prelim_mu[seat]
        print(f"  {i + 1:2d}. {seat:20s} mu={mu:+.5f} sigma={sigma:.5f}")

    # ============================================================
    # Step 3: Main calibration - Scheme B, adjacent pairs (~2750 games)
    # ============================================================
    print("\n=== Step 3: Main calibration (Scheme B, adjacent pairs) ===")
    n_seats = len(sorted_seats)
    chain_types = []
    for i in range(n_seats - 3):
        chain_types.append(sorted_seats[i:i + 4])

    print(f"Chain types (4-seat adjacent chains): {len(chain_types)}")
    for i, chain in enumerate(chain_types):
        print(f"  Chain {i}: {' > '.join(chain)}")

    target_games = 2750
    n_games_per_chain = target_games // len(chain_types)
    remainder = target_games % len(chain_types)

    calib_assignments = []
    gi = 0
    for ci, chain in enumerate(chain_types):
        n_games = n_games_per_chain + (1 if ci < remainder else 0)
        for _ in range(n_games):
            calib_assignments.append((gi, list(chain)))
            gi += 1

    print(f"Total calibration games: {len(calib_assignments)}")
    calib_results = run_batch(calib_assignments, SEED + 2)

    calib_game_data = []
    for i in range(len(calib_assignments)):
        game_rows, _ = calib_results[i]
        calib_game_data.append(game_rows)

    # ============================================================
    # Step 4: Final MLE fit on all games
    # ============================================================
    print("\n=== Step 4: Final MLE fit ===")
    all_game_data = prelim_game_data + calib_game_data
    final_mu = rate(all_game_data, anchor="optimizer")

    print("Final calibration results (strongest first):")
    results = []
    for seat in sorted(final_mu.keys(), key=lambda k: -final_mu[k][0]):
        mu, sigma = final_mu[seat]
        elo = 1000 + K_ELO * mu
        results.append((seat, mu, sigma, elo))
        print(f"  {seat:20s} mu={mu:+.6f} sigma={sigma:.6f} Elo≈{elo:.1f}")

    # ============================================================
    # Step 5: Write ai/ratings.json
    # ============================================================
    print("\n=== Step 5: Write ai/ratings.json ===")
    ratings = {}
    for seat, mu, sigma, elo in results:
        entry = {
            "mu": mu,
            "sigma": sigma,
            "elo": round(elo),
            "games_played": 0,  # will fill below
        }
        # sha256 for RL checkpoints; null for personalities
        try:
            ckpt_path = seats_mod.rl_checkpoint(seat)
            entry["checkpoint"] = ckpt_path
            entry["sha256"] = sha256_file(ckpt_path)
        except ValueError:
            entry["checkpoint"] = None
            entry["sha256"] = None
        ratings[seat] = entry

    # Count games played per seat
    seat_game_count = defaultdict(int)
    for game_rows in all_game_data:
        for key, _rank in game_rows:
            seat_game_count[key] += 1
    for seat in ratings:
        ratings[seat]["games_played"] = seat_game_count.get(seat, 0)

    # Calibration metadata
    calib_meta = {
        "seed": SEED,
        "k_elo": K_ELO,
        "anchor": "optimizer",
        "pilot_games": 400,
        "prelim_games": 200,
        "calib_games": len(calib_assignments),
        "total_games": len(all_game_data),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seat_effect_detected": seat_effect,
    }

    output = {
        "metadata": calib_meta,
        "ratings": ratings,
    }

    out_path = "ai/ratings.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # sha256 of the calibration file itself
    calib_sha256 = sha256_file(out_path)
    print(f"Written: {out_path}")
    print(f"Calibration file sha256: {calib_sha256}")
    print(f"Metadata: {json.dumps(calib_meta, ensure_ascii=False)}")

    return calib_sha256, results


if __name__ == "__main__":
    calib_sha256, results = main()
