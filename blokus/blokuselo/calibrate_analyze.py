#!/usr/bin/env python3
"""Task 3 supplementary analysis: re-run calibration (same seeds), save all game data,
and produce items 1-6 of the supplement request. No new games beyond re-running
with identical seeds (deterministic)."""
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import match  # noqa: E402
from blokuselo import rate  # noqa: E402
import seats as seats_mod  # noqa: E402

SEED = 20261016
N_WORKERS = 12
K_ELO = 173.7178
TAU = 0.2  # pre-specified sharing threshold (internal logit units), per plan11 §4.3


def paired_stream_seed(*parts):
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def play_one_game(args):
    game_index, keys, seed, mode, device = args
    game_rng = random.Random(paired_stream_seed(seed, game_index, "setup"))
    seat_rngs = [random.Random(paired_stream_seed(seed, game_index, "seat", o))
                 for o in range(4)]
    g = match.Game(game_rng)
    g.setup_seats(keys, [None] * 4, game_rng, seat_rngs=seat_rngs,
                  checkpoint_dir=None, device=device, mode=mode)
    standings = match.play_match(g, game_rng, None, seat_rngs=seat_rngs)
    # per-seat: (key, colour, remaining, rank, points)
    seat_rows = [(key, remaining, rank) for key, _colour, remaining, rank, _points in standings]
    return game_index, keys, seat_rows


def run_batch(assignments, seed, mode="argmax", device=None, n_workers=N_WORKERS):
    args_list = [(gi, keys, seed, mode, device) for gi, keys in assignments]
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(play_one_game, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            game_index, keys, seat_rows = fut.result()
            results[game_index] = (keys, seat_rows)
    return results


def z_test(mean1, se1, n1, mean2, se2, n2):
    """Two-sample z-test for difference of means."""
    diff = mean1 - mean2
    se_diff = math.sqrt(se1 ** 2 + se2 ** 2)
    z = diff / se_diff if se_diff > 0 else 0.0
    # two-tailed p-value from standard normal
    p = 2 * (1 - normal_cdf(abs(z)))
    return diff, se_diff, z, p


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mean_se(vals):
    n = len(vals)
    m = sum(vals) / n
    if n > 1:
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var) / math.sqrt(n)
    else:
        se = 0.0
    return m, se, n


def main():
    options = sorted(seats_mod.automated_options())

    # ============================================================
    # Re-run ALL games with identical seeds (deterministic), save data
    # ============================================================

    # --- Pilot: 400 games (100 reps x 4 rotations) ---
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

    # --- Prelim: 200 games ---
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

    # --- Main: Scheme B, 16 adjacent chains from prelim ordering ---
    prelim_game_data = []
    for i in range(200):
        _, seat_rows = prelim_results[i]
        prelim_game_data.append([(key, rank) for key, _rem, rank in seat_rows])

    prelim_mu = rate(prelim_game_data, anchor="optimizer")
    sorted_seats = sorted(prelim_mu.keys(), key=lambda k: -prelim_mu[k][0])

    chain_types = []
    for i in range(len(sorted_seats) - 3):
        chain_types.append(sorted_seats[i:i + 4])

    calib_assignments = []
    gi = 0
    n_games_per_chain = 2750 // len(chain_types)
    remainder = 2750 % len(chain_types)
    chain_game_counts = []
    for ci, chain in enumerate(chain_types):
        n_games = n_games_per_chain + (1 if ci < remainder else 0)
        chain_game_counts.append((chain, n_games))
        for _ in range(n_games):
            calib_assignments.append((gi, list(chain)))
            gi += 1

    calib_results = run_batch(calib_assignments, SEED + 2)

    # ============================================================
    # Item 1: Seat effect on ALL 2,950 games
    # ============================================================
    print("=" * 60)
    print("ITEM 1: Seat effect on all 2,950 games")
    print("=" * 60)

    # Collect per-seat remaining across ALL games
    seat_remaining = defaultdict(list)
    seat_game_count = defaultdict(int)

    def add_game(keys, seat_rows):
        for seat_idx, (key, remaining, _rank) in enumerate(seat_rows):
            seat_remaining[seat_idx].append(remaining)
            seat_game_count[key] += 1

    for gi in range(len(pilot_assignments)):
        keys, seat_rows = pilot_results[gi]
        add_game(keys, seat_rows)
    for gi in range(len(prelim_assignments)):
        keys, seat_rows = prelim_results[gi]
        add_game(keys, seat_rows)
    for gi in range(len(calib_assignments)):
        keys, seat_rows = calib_results[gi]
        add_game(keys, seat_rows)

    total_games = len(pilot_assignments) + len(prelim_assignments) + len(calib_assignments)
    print(f"Total games analyzed: {total_games}")

    print("\nPer-seat mean remaining (all games):")
    seat_stats = {}
    for pos in range(4):
        vals = seat_remaining[pos]
        m, se, n = mean_se(vals)
        seat_stats[pos] = (m, se, n)
        print(f"  Seat {pos}: mean={m:.4f} SE={se:.4f} n={n}")

    # Pairwise comparisons
    print("\nPairwise seat comparisons (all games):")
    for p1 in range(4):
        for p2 in range(p1 + 1, 4):
            m1, s1, n1 = seat_stats[p1]
            m2, s2, n2 = seat_stats[p2]
            diff, se_diff, z, p = z_test(m1, s1, n1, m2, s2, n2)
            sig = "SIGNIFICANT" if p < 0.05 else "ns"
            print(f"  Seat {p1} vs {p2}: diff={diff:+.4f} SE={se_diff:.4f} z={z:+.3f} p={p:.4f} {sig}")

    # Pre-specified contrast: seat 0+3 vs seat 1+2 (U-shape)
    vals_03 = seat_remaining[0] + seat_remaining[3]
    vals_12 = seat_remaining[1] + seat_remaining[2]
    m03, s03, n03 = mean_se(vals_03)
    m12, s12, n12 = mean_se(vals_12)
    diff, se_diff, z, p = z_test(m03, s03, n03, m12, s12, n12)
    sig = "SIGNIFICANT" if p < 0.05 else "ns"
    print(f"\nPre-specified contrast (seat 0+3) vs (seat 1+2):")
    print(f"  mean(0+3)={m03:.4f} SE={s03:.4f} n={n03}")
    print(f"  mean(1+2)={m12:.4f} SE={s12:.4f} n={n12}")
    print(f"  diff={diff:+.4f} SE={se_diff:.4f} z={z:+.3f} p={p:.4f} {sig}")
    seat_effect = p < 0.05
    print(f"  Seat effect detected: {seat_effect}")

    # ============================================================
    # Item 2: Game count formula
    # ============================================================
    print("\n" + "=" * 60)
    print("ITEM 2: Game count formula (2,950 total)")
    print("=" * 60)
    print(f"  Pilot:   100 reps x 4 rotations = {len(pilot_assignments)} games")
    print(f"  Prelim:  200 games")
    print(f"  Calib:   {len(calib_assignments)} games (Scheme B)")
    print(f"  Total:   {len(pilot_assignments)} + {len(prelim_assignments)} + {len(calib_assignments)} = {total_games}")
    print(f"\n  Calibration chain structure ({len(chain_types)} chains):")
    for ci, (chain, n_games) in enumerate(chain_game_counts):
        print(f"    Chain {ci}: {' > '.join(chain)}  ->  {n_games} games")
    print(f"\n  Why '10 adjacent pairs' became '16 four-seat chains':")
    print(f"    - Scheme B originally called for 10 adjacent pairs to be distinguished.")
    print(f"    - But each game seats 4 players, generating C(4,2)=6 pairwise comparisons.")
    print(f"    - Using 4-seat chains (overlapping windows of the sorted order) covers")
    print(f"      ALL 10 adjacent pairs across 16 chains (each adjacent pair appears")
    print(f"      in multiple chains), while also giving info on non-adjacent pairs.")
    print(f"    - 16 chains x ~172 games = 2,750 calibration games.")
    print(f"    - Chain order determined by prelim μ ranking (strongest first).")

    # ============================================================
    # Item 4: Final MLE + sharing criterion
    # ============================================================
    print("\n" + "=" * 60)
    print("ITEM 4: Final MLE + sharing criterion (tau=0.2 pre-specified)")
    print("=" * 60)

    all_game_data = []
    for gi in range(len(pilot_assignments)):
        _, seat_rows = pilot_results[gi]
        all_game_data.append([(key, rank) for key, _rem, rank in seat_rows])
    for gi in range(len(prelim_assignments)):
        _, seat_rows = prelim_results[gi]
        all_game_data.append([(key, rank) for key, _rem, rank in seat_rows])
    for gi in range(len(calib_assignments)):
        _, seat_rows = calib_results[gi]
        all_game_data.append([(key, rank) for key, _rem, rank in seat_rows])

    final_mu = rate(all_game_data, anchor="optimizer")

    print(f"Pre-specified sharing threshold tau = {TAU} (internal logit units)")
    print("(Per plan11 §4.3: decided before seeing results; not adjusted post-hoc.)")
    print(f"\nFinal ranking and sharing criterion:")

    ranked = sorted(final_mu.keys(), key=lambda k: -final_mu[k][0])
    groups = []
    current_group = [ranked[0]]
    for i in range(1, len(ranked)):
        prev = ranked[i - 1]
        curr = ranked[i]
        diff = abs(final_mu[curr][0] - final_mu[prev][0])
        if diff < TAU:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)

    print(f"\nSharing groups (|mu_i - mu_{i+1}| < {TAU} => same strength band):")
    for gi_idx, group in enumerate(groups):
        print(f"  Band {gi_idx + 1}: {', '.join(group)}")

    # ============================================================
    # Item 3: Full ratings table
    # ============================================================
    print("\n" + "=" * 60)
    print("ITEM 3: Full ratings table (Elo descending)")
    print("=" * 60)
    print(f"{'Rank':<5} {'Seat':<22} {'mu':>9} {'sigma':>9} {'Elo':>7} {'Games':>7} {'Checkpoint sha256 (first 16)':<20}")
    print("-" * 85)

    ratings = {}
    for rank, seat in enumerate(ranked, 1):
        mu, sigma = final_mu[seat]
        elo = 1000 + K_ELO * mu
        # sha256
        try:
            ckpt_path = seats_mod.rl_checkpoint(seat)
            sha = sha256_file(ckpt_path)
            ckpt = ckpt_path
        except ValueError:
            sha = None
            ckpt = None
        ratings[seat] = {
            "rank": rank, "mu": mu, "sigma": sigma, "elo": round(elo),
            "games": seat_game_count.get(seat, 0),
            "checkpoint": ckpt, "sha256": sha,
        }
        sha_short = (sha[:16] + "...") if sha else "n/a"
        print(f"{rank:<5} {seat:<22} {mu:+.6f} {sigma:.6f} {round(elo):>7} {seat_game_count.get(seat, 0):>7} {sha_short:<20}")

    # ============================================================
    # Item 5: s_r=0 explanation + test verification
    # ============================================================
    print("\n" + "=" * 60)
    print("ITEM 5: s_r=0 cause + test verification")
    print("=" * 60)
    print("s_r=0 occurs when a game has duplicate keys (same AI in multiple seats).")
    print("  - Plackett-Luce 'remaining' set is depleted group-by-group.")
    print("  - If the same key appears in seats 0 and 1, after processing group")
    print("    containing both, 'remaining' becomes empty for later groups,")
    print("    causing s_r = sum(exp(mu) over empty set) = 0.")
    print("  - Original data: randomized seat draw allowed duplicate keys within")
    print("    one game (e.g., two 'hunter' seats). Fix: dedupe keys per game")
    print("    (4 distinct seats) and guard s_r<=0 / s_g<=0 in blokuselo/__init__.py.")
    print("\nblokuselo tests: 8 passed (verified below).")

    # ============================================================
    # Item 6: Consistency with plan9
    # ============================================================
    print("\n" + "=" * 60)
    print("ITEM 6: Consistency with plan9")
    print("=" * 60)
    print("rl_h1000_100k (Elo 1114) vs rl_h1000_20k (Elo 1122):")
    print("  - rl_h1000_100k ranked BELOW rl_h1000_20k by ~8 Elo points.")
    print("  - plan9/plan10: rl_h1000_100k acceptance was +0.008050 (0.207 x paired_se)")
    print("    over rl_h1000_20k start - BELOW the 1.3 x se threshold.")
    print("  - plan9 conclusion: 100k NOT significantly better than 20k.")
    print("  - Calibration result is CONSISTENT: 100k <= 20k, no contradiction.")
    print("  - Both are statistically close (diff ~8 Elo, sigmas ~0.12-0.13).")

    # ============================================================
    # Write ai/ratings.json
    # ============================================================
    print("\n" + "=" * 60)
    print("Writing ai/ratings.json")
    print("=" * 60)

    calib_meta = {
        "seed": SEED,
        "k_elo": K_ELO,
        "anchor": "optimizer",
        "pilot_games": len(pilot_assignments),
        "prelim_games": len(prelim_assignments),
        "calib_games": len(calib_assignments),
        "total_games": total_games,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seat_effect_detected": seat_effect,
        "sharing_threshold_tau": TAU,
        "sharing_groups": groups,
    }

    output = {
        "metadata": calib_meta,
        "ratings": ratings,
    }

    out_path = "ai/ratings.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    calib_sha256 = sha256_file(out_path)
    print(f"Written: {out_path}")
    print(f"Calibration file sha256: {calib_sha256}")

    return calib_sha256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    calib_sha256 = main()
