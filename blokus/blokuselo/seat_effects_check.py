"""Standalone test of rate_with_seat_effects with correct math.

Plackett-Luce with ties (groups) and seat effects:
strength(p, s) = exp(mu[p] + beta[s]), seat 3 = baseline

log P = sum_i sum_{p in G_i} str_p - sum_i |G_i| * log(S_i)
where S_i = sum_{j in R_i} str_j

Gradient w.r.t. str_k (k in group m):
dL/d(str_k) = 1 - sum_{i=1}^{m} |G_i| * str_k / S_i

Hessian w.r.t. str:
d²L/d(str_k)² = -sum_{i=1}^{m} |G_i| * str_k / S_i + sum_{i=1}^{m} |G_i| * str_k² / S_i²
d²L/d(str_k)d(str_l) = sum_{i=1}^{min(m_k,m_l)} |G_i| * str_k * str_l / S_i²

Transform to (mu, beta) via chain rule.
"""
import math
import random


def solve(A, b):
    m = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, m):
            f = M[r][col] / pv
            for c in range(col, m + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * m
    for r in range(m - 1, -1, -1):
        s = M[r][m]
        for c in range(r + 1, m):
            s -= M[r][c] * x[c]
        x[r] = s / M[r][r]
    return x


def invert(A):
    m = len(A)
    M = [list(row) + [1.0 if i == j else 0.0 for j in range(m)]
         for i, row in enumerate(A)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for c in range(m + m):
            M[col][c] /= pv
        for r in range(m):
            if r != col:
                f = M[r][col]
                for c in range(m + m):
                    M[r][c] -= f * M[col][c]
    return [row[m:] for row in M]


def rate_with_seat_effects(games, anchor=None, max_iter=20000, tol=1e-9):
    """Plackett-Luce MLE with seat-position effects.

    strength(player p at seat s) = exp(mu[p] + beta[s]), seat 3 = baseline (beta=0).

    Input: list of games, each a list of (player_key, seat_position, rank)
           seat_position in {0,1,2,3}, lower rank = better.
    Output: ({player_key: (mu, sigma)}, {seat_position: (effect, se)}).
    """
    players = sorted({p for g in games for p, _s, _r in g})
    n = len(players)
    if n == 0:
        return {}, {}
    index = {p: i for i, p in enumerate(players)}

    anchor_idx = index[anchor] if anchor and anchor in index else 0
    free_mu = [i for i in range(n) if i != anchor_idx]
    fpos = {i: k for k, i in enumerate(free_mu)}
    nf = len(free_mu)
    n_free = nf + 3

    mu = [0.0] * n
    beta = [0.0, 0.0, 0.0]

    # Pre-process games
    processed_games = []
    for g in games:
        by_rank = {}
        for p, s, r in g:
            by_rank.setdefault(r, []).append((index[p], s))
        groups = [by_rank[r] for r in sorted(by_rank)]
        processed_games.append(groups)

    def str_val(p_idx, seat):
        b = beta[seat] if 0 <= seat < 3 else 0.0
        return math.exp(mu[p_idx] + b)

    for iteration in range(max_iter):
        grad = [0.0] * n_free
        hess = [[0.0] * n_free for _ in range(n_free)]

        for groups in processed_games:
            # First pass: compute S_i for each group
            remaining = set((p, s) for grp in groups for p, s in grp)
            S_values = []
            group_sizes = []
            for grp in groups:
                group_sizes.append(len(grp))
                s_r = sum(str_val(p, s) for p, s in remaining)
                S_values.append(s_r)
                remaining -= set((p, s) for p, s in grp)

            # Second pass: gradient and Hessian
            remaining = set((p, s) for grp in groups for p, s in grp)

            for gi, grp in enumerate(groups):
                group_size = group_sizes[gi]
                S_i = S_values[gi]
                in_g = set(p for p, _s in grp)

                if S_i <= 0.0:
                    remaining -= set((p, s) for p, s in grp)
                    continue

                inv_sr = 1.0 / S_i
                inv_sr2 = inv_sr * inv_sr

                # str values for remaining players in this game
                remaining_list = sorted(remaining)
                str_vals = {p: str_val(p, s) for p, s in remaining_list}

                # Gradient: accumulate over groups where each player is in remaining
                for p, s in remaining_list:
                    e = str_vals[p]
                    if p in in_g:
                        grad_str = 1.0 - group_size * e * inv_sr
                    else:
                        grad_str = -group_size * e * inv_sr

                    if p != anchor_idx:
                        grad[fpos[p]] += grad_str * e
                    if s < 3:
                        grad[nf + s] += grad_str * e

                # Hessian
                for ai in range(len(remaining_list)):
                    p_a, s_a = remaining_list[ai]
                    e_a = str_vals[p_a]
                    a_in = p_a in in_g

                    d2_aa = -group_size * e_a * inv_sr + group_size * e_a * e_a * inv_sr2
                    d1_a = (1.0 if a_in else 0.0) - group_size * e_a * inv_sr

                    if p_a != anchor_idx:
                        ka = fpos[p_a]
                        hess[ka][ka] += d2_aa * e_a * e_a + d1_a * e_a

                    for bj in range(ai, len(remaining_list)):
                        p_b, s_b = remaining_list[bj]
                        e_b = str_vals[p_b]

                        d2_ab = group_size * e_a * e_b * inv_sr2

                        if p_a != anchor_idx and p_b != anchor_idx:
                            ka = fpos[p_a]
                            kb = fpos[p_b]
                            val = d2_ab * e_a * e_b
                            hess[ka][kb] += val
                            if ka != kb:
                                hess[kb][ka] += val

                        if p_a != anchor_idx and s_b < 3:
                            ka = fpos[p_a]
                            kb = nf + s_b
                            val = d2_ab * e_a * e_b
                            hess[ka][kb] += val
                            hess[kb][ka] += val

                        if p_b != anchor_idx and s_a < 3:
                            kb = fpos[p_b]
                            ka = nf + s_a
                            val = d2_ab * e_b * e_a
                            hess[kb][ka] += val
                            hess[ka][kb] += val

                        if s_a < 3 and s_b < 3:
                            ka = nf + s_a
                            kb = nf + s_b
                            val = d2_ab * e_a * e_b
                            hess[ka][kb] += val
                            if ka != kb:
                                hess[kb][ka] += val

                        if p_a == p_b and s_a == s_b and s_a < 3:
                            hess[nf + s_a][nf + s_a] += d1_a * e_a

                remaining -= set((p, s) for p, s in grp)

        # Convergence check
        gnorm = math.sqrt(sum(g * g for g in grad))
        if iteration > 0 and gnorm < tol:
            break

        # Newton step
        neg_h = [[-v for v in row] for row in hess]
        delta = solve(neg_h, grad)
        if delta is None:
            break

        step = 1.0
        mu_old = mu[:]
        beta_old = beta[:]
        improved = False
        for _ls in range(50):
            for k, i in enumerate(free_mu):
                mu[i] = mu_old[i] + step * delta[k]
            for s in range(3):
                beta[s] = beta_old[s] + step * delta[nf + s]

            # Recompute gradient to check improvement
            grad2 = [0.0] * n_free
            for groups in processed_games:
                remaining = set((p, s) for grp in groups for p, s in grp)
                S_values = []
                group_sizes = []
                for grp in groups:
                    group_sizes.append(len(grp))
                    s_r = sum(str_val(p, s) for p, s in remaining)
                    S_values.append(s_r)
                    remaining -= set((p, s) for p, s in grp)

            remaining = set((p, s) for grp in groups for p, s in grp)
            for gi, grp in enumerate(groups):
                group_size = group_sizes[gi]
                S_i = S_values[gi]
                in_g = set(p for p, _s in grp)
                if S_i <= 0.0:
                    remaining -= set((p, s) for p, s in grp)
                    continue
                inv_sr = 1.0 / S_i
                remaining_list = sorted(remaining)
                str_vals = {p: str_val(p, s) for p, s in remaining_list}
                for p, s in remaining_list:
                    e = str_vals[p]
                    if p in in_g:
                        grad_str = 1.0 - group_size * e * inv_sr
                    else:
                        grad_str = -group_size * e * inv_sr
                    if p != anchor_idx:
                        grad2[fpos[p]] += grad_str * e
                    if s < 3:
                        grad2[nf + s] += grad_str * e
                remaining -= set((p, s) for p, s in grp)

            gnorm2 = math.sqrt(sum(g * g for g in grad2))
            if gnorm2 < gnorm:
                improved = True
                break
            step *= 0.5

        if not improved:
            mu = mu_old
            beta = beta_old
            break

    # Final covariance
    grad = [0.0] * n_free
    hess = [[0.0] * n_free for _ in range(n_free)]
    for groups in processed_games:
        remaining = set((p, s) for grp in groups for p, s in grp)
        S_values = []
        group_sizes = []
        for grp in groups:
            group_sizes.append(len(grp))
            s_r = sum(str_val(p, s) for p, s in remaining)
            S_values.append(s_r)
            remaining -= set((p, s) for p, s in grp)

        remaining = set((p, s) for grp in groups for p, s in grp)
        for gi, grp in enumerate(groups):
            group_size = group_sizes[gi]
            S_i = S_values[gi]
            in_g = set(p for p, _s in grp)
            if S_i <= 0.0:
                remaining -= set((p, s) for p, s in grp)
                continue
            inv_sr = 1.0 / S_i
            inv_sr2 = inv_sr * inv_sr
            remaining_list = sorted(remaining)
            str_vals = {p: str_val(p, s) for p, s in remaining_list}
            for p, s in remaining_list:
                e = str_vals[p]
                if p in in_g:
                    grad_str = 1.0 - group_size * e * inv_sr
                else:
                    grad_str = -group_size * e * inv_sr
                if p != anchor_idx:
                    grad[fpos[p]] += grad_str * e
                if s < 3:
                    grad[nf + s] += grad_str * e

            for ai in range(len(remaining_list)):
                p_a, s_a = remaining_list[ai]
                e_a = str_vals[p_a]
                a_in = p_a in in_g
                d2_aa = -group_size * e_a * inv_sr + group_size * e_a * e_a * inv_sr2
                d1_a = (1.0 if a_in else 0.0) - group_size * e_a * inv_sr

                if p_a != anchor_idx:
                    ka = fpos[p_a]
                    hess[ka][ka] += d2_aa * e_a * e_a + d1_a * e_a

                for bj in range(ai, len(remaining_list)):
                    p_b, s_b = remaining_list[bj]
                    e_b = str_vals[p_b]
                    d2_ab = group_size * e_a * e_b * inv_sr2

                    if p_a != anchor_idx and p_b != anchor_idx:
                        ka = fpos[p_a]
                        kb = fpos[p_b]
                        val = d2_ab * e_a * e_b
                        hess[ka][kb] += val
                        if ka != kb:
                            hess[kb][ka] += val

                    if p_a != anchor_idx and s_b < 3:
                        ka = fpos[p_a]
                        kb = nf + s_b
                        val = d2_ab * e_a * e_b
                        hess[ka][kb] += val
                        hess[kb][ka] += val

                    if p_b != anchor_idx and s_a < 3:
                        kb = fpos[p_b]
                        ka = nf + s_a
                        val = d2_ab * e_b * e_a
                        hess[kb][ka] += val
                        hess[ka][kb] += val

                    if s_a < 3 and s_b < 3:
                        ka = nf + s_a
                        kb = nf + s_b
                        val = d2_ab * e_a * e_b
                        hess[ka][kb] += val
                        if ka != kb:
                            hess[kb][ka] += val

                    if p_a == p_b and s_a == s_b and s_a < 3:
                        hess[nf + s_a][nf + s_a] += d1_a * e_a

            remaining -= in_g

    neg_h = [[-v for v in row] for row in hess]
    inv = invert(neg_h)

    mu_result = {}
    for p in players:
        i = index[p]
        if i == anchor_idx:
            mu_result[p] = (0.0, 0.0)
        else:
            k = fpos[i]
            var = inv[k][k] if inv else 1.0
            mu_result[p] = (mu[i], math.sqrt(max(var, 1e-9)))

    seat_result = {}
    for s in range(3):
        k = nf + s
        var = inv[k][k] if inv else 1.0
        seat_result[s] = (beta[s], math.sqrt(max(var, 1e-9)))

    return mu_result, seat_result


# ---------- Test ----------
if __name__ == "__main__":
    import statistics

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
        scores = {}
        seat_of = {}
        for seat_idx, key in enumerate(keys):
            seat_of[key] = seat_idx
            scores[key] = KNOWN_MU[key] + seat_effect[seat_idx] + gumbel(rng)
        order = sorted(keys, key=lambda k: -scores[key])
        return [(k, seat_of[k], i + 1) for i, k in enumerate(order)]

    # Test 1: seat effects = 0, compare with rate()
    print("Test 1: seat effects = 0")
    from blokuselo import rate as rate_old

    rng = random.Random(12345)
    games = []
    for _ in range(200):
        combo = rng.sample(SEATS, 4)
        games.append(synth_game(combo, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, rng))

    rate_result = rate_old([[(p, r) for p, _s, r in g] for g in games], anchor="optimizer")
    mu_seat, seat_result = rate_with_seat_effects(games, anchor="optimizer")

    print(f"{'Seat':<22} {'rate() mu':>12} {'rwse mu':>12} {'diff':>10}")
    max_diff = 0.0
    for seat in SEATS:
        r_mu = rate_result[seat][0]
        s_mu = mu_seat[seat][0]
        diff = abs(r_mu - s_mu)
        max_diff = max(max_diff, diff)
        print(f"{seat:<22} {r_mu:+.6f}   {s_mu:+.6f}   {diff:+.6f}")
    print(f"Max |diff| = {max_diff:.2e} {'PASS' if max_diff < 1e-6 else 'FAIL'}")

    # Test 2: seat effects = 0.1 for seat 0
    print("\nTest 2: seat effects = seat0 +0.1, others 0")
    rng2 = random.Random(999)
    games2 = []
    for _ in range(300):
        combo = rng2.sample(SEATS, 4)
        games2.append(synth_game(combo, {0: 0.1, 1: 0.0, 2: 0.0, 3: 0.0}, rng2))

    mu_seat2, seat_result2 = rate_with_seat_effects(games2, anchor="optimizer")
    opt_mu = mu_seat2["optimizer"][0]
    print(f"Optimizer mu (should be 0): {opt_mu:.6f}")
    print(f"Seat effects: beta0={seat_result2[0][0]:+.6f} beta1={seat_result2[1][0]:+.6f} beta2={seat_result2[2][0]:+.6f}")
    print(f"(True: beta0=+0.1, beta1=0, beta2=0)")
    for p in SEATS[:5]:
        centered = mu_seat2[p][0] - opt_mu
        true_centered = KNOWN_MU[p] - KNOWN_MU["optimizer"]
        print(f"  {p:20s} centered_mu={centered:+.6f}  true={true_centered:+.6f}  diff={centered-true_centered:+.6f}")
