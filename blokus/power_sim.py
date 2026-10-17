#!/usr/bin/env python3
"""Power simulation for calibration design with seat effects.

Fixed version:
- Reports raw estimate and error intervals separately (no contradiction).
- Re-centers fitted mu by subtracting fitted optimizer mu before comparing to truth.
- Uses noisy prelim ordering to build chains (mimics real procedure).
- Explicit min-distinguishable-diff formula.
- Reports actual games per seat.
"""
import math
import random
import statistics
from blokuselo import rate_with_seat_effects

SEEDS = list(range(50000, 50020))  # 20 seeds
K_ELO = 173.7178
N_CHAINS = 19
CYCLES_PER_CHAIN = 36

KNOWN_MU = {
    "rl_h1000_30k": 1.185, "rl_o1000_50k_v3": 1.147,
    "rl_h1000_100k": 0.582, "rl_h1000_20k": 0.566, "rl_b1000_20k": 0.530,
    "rl_o1000_20k": 0.321, "rl_i1000_20k": 0.245, "rl_b1000_50k": 0.155,
    "hunter": 0.089, "optimizer": 0.0, "rl_h1000_50k": -0.016,
    "rl_i1000_37.5k": -0.209, "builder": -0.347, "rl_i1000_50k": -0.450,
    "rl_o1000_50k": -0.607, "intruder": -0.684,
    "wolf": -2.450, "chess": -2.452, "fox": -2.486,
}
SEATS = sorted(KNOWN_MU.keys(), key=lambda k: -KNOWN_MU[k])
N_SEATS = len(SEATS)
SEAT_EFFECT = {0: 0.1, 1: 0.0, 2: 0.0, 3: 0.0}  # seat 3 = baseline


def gumbel(rng):
    return -math.log(-math.log(rng.random()))


def build_cyclic_chains(seats):
    n = len(seats)
    return [[seats[(i + j) % n] for j in range(4)] for i in range(n)]


def latin_square_games(chain, n_cycles, rng):
    games = []
    for _ in range(n_cycles):
        for rot in range(4):
            games.append(list(chain[rot:] + chain[:rot]))
    return games


def synth_pl_game(keys, seat_effect, rng):
    scores = {}
    seat_of = {}
    for seat_idx, key in enumerate(keys):
        seat_of[key] = seat_idx
        scores[key] = KNOWN_MU[key] + seat_effect[seat_idx] + gumbel(rng)
    order = sorted(keys, key=lambda k: -scores[k])
    return [(k, seat_of[k], i + 1) for i, k in enumerate(order)]


def build_prelim_ordering(seed, use_noisy=True):
    """Build prelim ordering from synthetic data.
    If use_noisy: run prelim games, fit, order by fitted mu.
    If not: order by true known mu."""
    rng = random.Random(seed)
    if use_noisy:
        prelim_games = []
        for _ in range(50):
            combo = rng.sample(SEATS, 4)
            for rot in range(4):
                prelim_games.append(synth_pl_game(combo[rot:] + combo[:rot],
                                                  SEAT_EFFECT, rng))
        prelim_mu, _ = rate_with_seat_effects(prelim_games, anchor="optimizer")
        return sorted(prelim_mu.keys(), key=lambda k: -prelim_mu[k][0])
    else:
        return sorted(SEATS, key=lambda k: -KNOWN_MU[k])


def run_one_simulation(seed, use_noisy_prelim=True):
    rng = random.Random(seed)

    # Build chain order from prelim (noisy or true)
    chain_order = build_prelim_ordering(seed, use_noisy=use_noisy_prelim)
    chains = build_cyclic_chains(chain_order)

    all_fit_games = []

    # Pilot: 100 random combos x 4 rotations = 400 games
    for _ in range(100):
        combo = rng.sample(SEATS, 4)
        for rot in range(4):
            all_fit_games.append(synth_pl_game(combo[rot:] + combo[:rot],
                                               SEAT_EFFECT, rng))

    # Main: 19 cyclic chains x 36 cycles x 4 games = 2,736 games
    for chain in chains:
        for keys in latin_square_games(chain, CYCLES_PER_CHAIN, rng):
            all_fit_games.append(synth_pl_game(keys, SEAT_EFFECT, rng))

    # Fit with seat effects
    mu_result, seat_result = rate_with_seat_effects(
        all_fit_games, anchor="optimizer")

    # Re-center: subtract fitted optimizer mu from all fitted mu
    opt_mu = mu_result["optimizer"][0]
    mu_errors = {}
    sigmas = {}
    for k in SEATS:
        fitted_centered = mu_result[k][0] - opt_mu
        true_centered = KNOWN_MU[k] - KNOWN_MU["optimizer"]
        mu_errors[k] = fitted_centered - true_centered
        sigmas[k] = mu_result[k][1]

    # Seat coefficient recovery
    seat_errors = {s: seat_result[s][0] - SEAT_EFFECT[s] for s in range(3)}
    seat_estimates = {s: seat_result[s][0] for s in range(3)}
    seat_ses = {s: seat_result[s][1] for s in range(3)}

    # Min distinguishable mu diff: 1.96 * sqrt(2) * median_sigma
    median_sigma = statistics.median(sigmas.values())
    min_diff = 1.96 * math.sqrt(2) * median_sigma

    # Count games per seat
    seat_game_count = {k: 0 for k in SEATS}
    for g in all_fit_games:
        for key, _s, _r in g:
            seat_game_count[key] += 1

    diff_03 = seat_estimates[0] - 0.0  # seat 3 baseline = 0

    return {
        "mu_errors": mu_errors,
        "sigmas": sigmas,
        "seat_estimates": seat_estimates,
        "seat_errors": seat_errors,
        "seat_ses": seat_ses,
        "min_diff": min_diff,
        "median_sigma": median_sigma,
        "diff_03": diff_03,
        "seat_game_count": seat_game_count,
        "chain_order": chain_order,
        "n_games": len(all_fit_games),
    }


def pct(vals, p):
    s = sorted(vals)
    idx = int(p * (len(s) - 1))
    return s[idx]


def main():
    results = [run_one_simulation(s) for s in SEEDS]

    print("=" * 72)
    print("Power simulation v2: 20 seats, seat effects, noisy prelim ordering")
    print("=" * 72)
    print(f"Seeds: {len(SEEDS)}")
    print(f"Design: Pilot 400 + Prelim 200 (excluded) + Main {N_CHAINS}x{CYCLES_PER_CHAIN}x4")
    print(f"Seat effects: seat0=+0.10, seat1=0, seat2=0, seat3=0(baseline)")
    print(f"Prelim ordering: {'noisy (fitted)' if True else 'true'}")
    print()

    # --- Mu error stats (re-centered) ---
    print("--- Mu recovery error AFTER re-centering by fitted optimizer (mean, p5, p95) ---")
    for key in SEATS:
        errs = [r["mu_errors"][key] for r in results]
        mean_e = sum(errs) / len(errs)
        s_errs = sorted(errs)
        std_e = statistics.stdev(errs) if len(errs) > 1 else 0.0
        print(f"  {key:20s} mean={mean_e:+.4f}  std={std_e:.4f}  "
              f"p5={s_errs[1]:+.4f}  p95={s_errs[18]:+.4f}")

    # --- Seat coefficient recovery ---
    print("\n--- Seat coefficient recovery (raw estimate + error interval) ---")
    for s in range(3):
        ests = [r["seat_estimates"][s] for r in results]
        errs = [r["seat_errors"][s] for r in results]
        ses = [r["seat_ses"][s] for r in results]
        mean_est = sum(ests) / len(ests)
        mean_err = sum(errs) / len(errs)
        mean_se = sum(ses) / len(ses)
        s_ests = sorted(ests)
        s_errs = sorted(errs)
        print(f"  Seat {s}: true={SEAT_EFFECT[s]:+.3f}")
        print(f"    estimate: mean={mean_est:+.4f}  90% int=[{s_ests[1]:+.4f}, {s_ests[18]:+.4f}]")
        print(f"    error:    mean={mean_err:+.4f}  90% int=[{s_errs[1]:+.4f}, {s_errs[18]:+.4f}]  mean_SE={mean_se:.4f}")

    # Seat 0 vs 3
    diffs_03 = [r["diff_03"] for r in results]
    print(f"\n--- Seat 0 vs Seat 3 diff (true=+0.10 mu) ---")
    print(f"  mean={sum(diffs_03)/len(diffs_03):+.4f} mu")
    s_d = sorted(diffs_03)
    print(f"  90% interval=[{s_d[1]:+.4f}, {s_d[18]:+.4f}] mu")

    # --- Sigma ---
    print("\n--- Per-seat sigma (median, p5, p95) ---")
    for key in SEATS:
        sigmas = [r["sigmas"][key] for r in results]
        med = statistics.median(sigmas)
        s_s = sorted(sigmas)
        print(f"  {key:20s} median={med:.4f}  p5={s_s[1]:.4f}  p95={s_s[18]:.4f}")

    # --- Min distinguishable ---
    min_diffs = [r["min_diff"] for r in results]
    print(f"\n--- Min distinguishable mu diff ---")
    print(f"  Formula: 1.96 * sqrt(2) * median_sigma")
    print(f"  median={statistics.median(min_diffs):.4f} mu "
          f"({statistics.median(min_diffs) * K_ELO:.0f} Elo)")
    s_md = sorted(min_diffs)
    print(f"  interval=[{s_md[1]:.4f}, {s_md[18]:.4f}] mu")
    print(f"  median_sigma used: {statistics.median([r['median_sigma'] for r in results]):.4f}")

    # --- Games per seat ---
    print("\n--- Games per seat (Main + Pilot) ---")
    all_counts = {}
    for key in SEATS:
        counts = [r["seat_game_count"][key] for r in results]
        all_counts[key] = statistics.mean(counts)
    sorted_counts = sorted(all_counts.items(), key=lambda x: -x[1])
    for key, cnt in sorted_counts:
        print(f"  {key:20s} avg_games={cnt:.1f}")
    max_key, max_val = sorted_counts[0]
    min_key, min_val = sorted_counts[-1]
    print(f"  max={max_key}: {max_val:.1f}, min={min_key}: {min_val:.1f}, ratio={max_val/min_val:.3f}")

    # --- Chain order check ---
    print(f"\n--- Chain order (seed 50000) ---")
    print(f"  {results[0]['chain_order']}")

    return results


if __name__ == "__main__":
    main()
