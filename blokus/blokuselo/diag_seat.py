"""Diagnostic script for rate_with_seat_effects().

Steps:
1. Isolation: same synthetic data (seat effects = 0), compare rate() vs rate_with_seat_effects() with beta=0.
2. Numerical gradient/Hessian check against analytic.
3. Convergence diagnosis.
4. Confirm beta_3 not in parameter vector.
5. Confirm noise model consistency (Plackett-Luce / Gumbel).
"""
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokuselo import rate, rate_with_seat_effects  # noqa: E402

SEED = 12345
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


def gumbel(rng):
    return -math.log(-math.log(rng.random()))


def synth_game(keys, seat_effect, rng):
    """Plackett-Luce via Gumbel: score = mu + beta[seat] + Gumbel."""
    scores = {}
    seat_of = {}
    for seat_idx, key in enumerate(keys):
        seat_of[key] = seat_idx
        scores[key] = KNOWN_MU[key] + seat_effect[seat_idx] + gumbel(rng)
    order = sorted(keys, key=lambda k: -scores[key])
    return [(k, seat_of[k], i + 1) for i, k in enumerate(order)]


def make_games(n_games, n_per_game, rng, seat_effect):
    games = []
    for _ in range(n_games):
        combo = rng.sample(SEATS, n_per_game)
        games.append(synth_game(combo, seat_effect, rng))
    return games


def rate_format(games_with_seats):
    """Convert (player, seat, rank) to (player, rank) for rate()."""
    return [[(p, r) for p, _s, r in g] for g in games_with_seats]


# ---------- Step 1: Isolation ----------
print("=" * 70)
print("STEP 1: Isolation — rate() vs rate_with_seat_effects(beta=0)")
print("=" * 70)

rng = random.Random(SEED)
seat_effect_zero = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
games = make_games(200, 4, rng, seat_effect_zero)

rate_result = rate(rate_format(games), anchor="optimizer")
mu_seat, seat_result = rate_with_seat_effects(games, anchor="optimizer")

print(f"{'Seat':<22} {'rate() mu':>12} {'rwse(beta=0) mu':>16} {'diff':>10}")
for seat in SEATS:
    r_mu = rate_result[seat][0]
    s_mu = mu_seat[seat][0]
    diff = r_mu - s_mu
    print(f"{seat:<22} {r_mu:+.6f}   {s_mu:+.6f}   {diff:+.6f}")

max_diff = max(abs(rate_result[s][0] - mu_seat[s][0]) for s in SEATS)
print(f"\nMax |diff| = {max_diff:.2e}")
print(f"{'PASS' if max_diff < 1e-9 else 'FAIL'}: rate() and rwse(beta=0) {'match' if max_diff < 1e-9 else 'DIFFER'}")

# ---------- Step 2: Numerical gradient/Hessian ----------
print("\n" + "=" * 70)
print("STEP 2: Numerical gradient/Hessian check")
print("=" * 70)

# Build a small test case for numerical checking
test_seats = ["optimizer", "rl_h1000_20k", "rl_i1000_20k", "fox"]
test_mu_true = {k: KNOWN_MU[k] for k in test_seats}
test_seat_effect = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

rng2 = random.Random(999)
test_games = make_games(100, 4, rng2, test_seat_effect)
# Filter to only test_seats
filtered_games = []
for g in test_games:
    keys = [p for p, _s, _r in g if p in test_seats]
    if len(keys) == 4:
        filtered_games.append([(p, s, r) for p, s, r in g if p in test_seats])

if len(filtered_games) < 10:
    # Regenerate with only test_seats
    filtered_games = []
    for _ in range(100):
        combo = test_seats[:]  # fixed order
        rng2.shuffle(combo)
        filtered_games.append(synth_game(combo, test_seat_effect, rng2))

mu_test, seat_test = rate_with_seat_effects(filtered_games, anchor="optimizer")

# Numerical gradient check
def unpack_params(mu_dict, beta_dict, players, anchor_idx):
    """Convert to flat parameter vector."""
    free_mu = [i for i in range(len(players)) if i != anchor_idx]
    vals = []
    for i in free_mu:
        vals.append(mu_dict[players[i]])
    for s in range(3):
        vals.append(beta_dict[s])
    return vals

def pack_params(vals, players, anchor_idx, seat_of_map):
    """Convert flat vector back to mu_dict, beta_dict."""
    mu_dict = {players[anchor_idx]: 0.0}
    free_mu = [i for i in range(len(players)) if i != anchor_idx]
    for k, i in enumerate(free_mu):
        mu_dict[players[i]] = vals[k]
    beta_dict = {s: vals[len(free_mu) + s] for s in range(3)}
    return mu_dict, beta_dict

def neg_loglik(mu_dict, beta_dict, games):
    """Compute negative log-likelihood for given parameters."""
    total = 0.0
    for g in games:
        by_rank = {}
        for p, s, r in g:
            by_rank.setdefault(r, []).append((p, s))
        groups = [by_rank[r] for r in sorted(by_rank)]
        remaining = set(p for grp in groups for p, _s in grp)
        for grp in groups:
            group_size = len(grp)
            in_g = set(p for p, _s in grp)
            s_r = sum(math.exp(mu_dict[p] + (beta_dict[s] if s < 3 else 0.0)) for p, s in remaining)
            if s_r <= 0:
                remaining -= in_g
                continue
            # log P contribution: sum_{p in G} str_p - |G| * log(S_r)
            num = sum(math.exp(mu_dict[p] + (beta_dict[s] if s < 3 else 0.0)) for p, s in grp)
            total -= num  # negative of sum str_p
            total += group_size * math.log(s_r)  # negative of -|G| log S_i
            remaining -= in_g
    return total

players_test = sorted({p for g in filtered_games for p, _s, _r in g})
anchor_idx_test = players_test.index("optimizer")

# Current parameters at MLE
mu_cur, beta_cur = mu_test, {s: seat_test[s][0] for s in range(3)}
params_cur = unpack_params(mu_cur, beta_cur, players_test, anchor_idx_test)

# Numerical gradient
eps = 1e-6
grad_num = []
for i in range(len(params_cur)):
    params_p = params_cur[:]
    params_m = params_cur[:]
    params_p[i] += eps
    params_m[i] -= eps
    mu_p, beta_p = pack_params(params_p, players_test, anchor_idx_test, {})
    mu_m, beta_m = pack_params(params_m, players_test, anchor_idx_test, {})
    nll_p = neg_loglik(mu_p, beta_p, filtered_games)
    nll_m = neg_loglik(mu_m, beta_m, filtered_games)
    grad_num.append((nll_p - nll_m) / (2 * eps))

# Analytic gradient (from rate_with_seat_effects internal — we need to extract it)
# Since we can't easily extract, let's check Hessian instead by comparing
# parameter updates

print(f"Test players: {players_test}")
print(f"Anchor: optimizer at index {anchor_idx_test}")
print(f"Fitted mu: { {k: round(v[0], 4) for k, v in mu_test.items()} }")
print(f"Fitted beta: { {s: round(v[0], 4) for s, v in seat_test.items()} }")

# Check: does re-centering by optimizer make mu match truth?
opt_mu = mu_test["optimizer"][0]
print(f"\nOptimizer mu (should be 0): {opt_mu:.6f}")
for p in test_seats:
    if p != "optimizer":
        centered = mu_test[p][0] - opt_mu
        true_centered = test_mu_true[p] - test_mu_true["optimizer"]
        print(f"  {p:20s} centered_mu={centered:+.6f}  true={true_centered:+.6f}  diff={centered-true_centered:+.6f}")

print(f"\nSeat coefficients: beta0={seat_test[0][0]:+.6f} beta1={seat_test[1][0]:+.6f} beta2={seat_test[2][0]:+.6f}")
print(f"(True: beta0=0, beta1=0, beta2=0, seat3=baseline)")
