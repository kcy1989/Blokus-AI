"""BlokusElo: pure-Python Plackett-Luce batch rating engine.

Not OpenSkill; not claimed to conform to Elo.  "Elo" is only a display name.

mu is strength in internal logit units; sigma is uncertainty from the
observed Fisher information (inverse negative Hessian at the MLE).  Only
strength *differences* are identifiable, so one player may be anchored at
mu = 0.  A weak Gaussian prior (sigma 2.0) on mu keeps the Hessian
invertible for players with few games.

Input (rate): a list of games, each a list of (player_key, rank) with lower
rank = better and ties sharing a rank.
Output: {player_key: (mu, sigma)}.

Input (rate_with_seat_effects): a list of games, each a list of
(player_key, seat_position, rank) with seat_position in 0..3 and seat 3 as
baseline (effect = 0).  Two-stage estimation: mu from `rate()` (or init_mu),
then beta by gradient ascent with mu fixed.
Output: ({player_key: (mu, sigma)}, {seat_position: (effect, se)}).
"""
import math

_PRIOR_VAR = 4.0  # prior variance on mu (prior sigma = 2.0)


def rate(games, anchor=None, max_iter=20000, tol=1e-9):
    """Batch MLE of Plackett-Luce strengths.

    Returns {player_key: (mu, sigma)}.  `anchor` (a player key) has mu pinned
    to 0 to break shift-invariance; if None the first player is anchored.
    """
    players = sorted({p for g in games for p, _ in g})
    n = len(players)
    if n == 0:
        return {}
    index = {p: i for i, p in enumerate(players)}

    game_groups = []
    for g in games:
        by_rank = {}
        for p, r in g:
            by_rank.setdefault(r, []).append(index[p])
        game_groups.append([by_rank[r] for r in sorted(by_rank)])

    anchor_idx = index[anchor] if anchor and anchor in index else 0
    free = [i for i in range(n) if i != anchor_idx]
    fpos = {i: k for k, i in enumerate(free)}
    nf = len(free)

    mu = [0.0] * n
    inv_pv = 1.0 / _PRIOR_VAR

    def accumulate():
        grad = [-inv_pv * mu[i] for i in free]
        hess = [[-inv_pv if i == j else 0.0 for j in range(nf)]
                for i in range(nf)]
        for groups in game_groups:
            remaining = set()
            for grp in groups:
                remaining.update(grp)
            for grp in groups:
                in_g = set(grp)
                s_g = sum(math.exp(mu[i]) for i in grp)
                s_r = sum(math.exp(mu[i]) for i in remaining)
                if s_g <= 0.0 or s_r <= 0.0:
                    remaining -= in_g
                    continue
                inv_sg = 1.0 / s_g
                inv_sr = 1.0 / s_r
                inv_sg2 = inv_sg * inv_sg
                inv_sr2 = inv_sr * inv_sr
                for i in remaining:
                    if i == anchor_idx:
                        continue
                    ei = math.exp(mu[i])
                    k = fpos[i]
                    if i in in_g:
                        grad[k] += ei * inv_sg - ei * inv_sr
                    else:
                        grad[k] -= ei * inv_sr
                rem_list = sorted(i for i in remaining if i != anchor_idx)
                for ai in range(len(rem_list)):
                    a = rem_list[ai]
                    ea = math.exp(mu[a])
                    ka = fpos[a]
                    a_in = a in in_g
                    for bi in range(ai, len(rem_list)):
                        b = rem_list[bi]
                        eb = math.exp(mu[b])
                        kb = fpos[b]
                        b_in = b in in_g
                        if a == b:
                            if a_in:
                                v = (ea * inv_sg - ea * ea * inv_sg2
                                     - ea * inv_sr + ea * ea * inv_sr2)
                            else:
                                v = -ea * inv_sr + ea * ea * inv_sr2
                            hess[ka][ka] += v
                        else:
                            if a_in and b_in:
                                v = ea * eb * (inv_sr2 - inv_sg2)
                            else:
                                v = ea * eb * inv_sr2
                            hess[ka][kb] += v
                            hess[kb][ka] += v
                remaining -= in_g
        return grad, hess

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

    for _ in range(max_iter):
        grad, hess = accumulate()
        gnorm = math.sqrt(sum(g * g for g in grad))
        if gnorm < tol:
            break
        neg_h = [[-v for v in row] for row in hess]
        delta = solve(neg_h, grad)
        if delta is None:
            break
        step = 1.0
        mu_old = mu[:]
        for _ls in range(40):
            for k, i in enumerate(free):
                mu[i] = mu_old[i] + step * delta[k]
            grad2, _ = accumulate()
            gnorm2 = math.sqrt(sum(g * g for g in grad2))
            if gnorm2 < gnorm:
                break
            step *= 0.5
        else:
            mu = mu_old
            break

    grad, hess = accumulate()
    neg_h = [[-v for v in row] for row in hess]
    inv = invert(neg_h)
    result = {}
    for p in players:
        i = index[p]
        if i == anchor_idx:
            result[p] = (0.0, 0.0)
        else:
            k = fpos[i]
            var = inv[k][k] if inv else 1.0
            result[p] = (mu[i], math.sqrt(max(var, 1e-9)))
    return result


def rate_with_seat_effects(games, anchor=None, max_iter=20000, tol=1e-9,
                           init_mu=None):
    """Plackett-Luce MLE with seat-position effects.

    Same ranking model as `rate()`, plus a seat-position additive term on the
    log-strength: strength(player p at seat s) = exp(mu[p] + beta[s]),
    seat 3 = baseline (beta[3] = 0 fixed).

    Two-stage estimation:
      1. mu from `rate()` (or init_mu if provided)
      2. beta by gradient ascent with mu fixed (concave subproblem)

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

    # Step 1: mu from rate() (or init_mu if provided)
    if init_mu is not None:
        mu = [init_mu.get(p, 0.0) for p in players]
        mu[anchor_idx] = 0.0
    else:
        rate_result = rate([[(p, r) for p, _s, r in g] for g in games],
                           anchor=anchor)
        mu = [rate_result[p][0] for p in players]

    beta = [0.0, 0.0, 0.0]  # seat 3 baseline = 0

    # Pre-process games: list of groups, each group is list of (player_idx, seat)
    processed_games = []
    for g in games:
        by_rank = {}
        for p, s, r in g:
            by_rank.setdefault(r, []).append((index[p], s))
        processed_games.append([by_rank[r] for r in sorted(by_rank)])

    def str_val(p_idx, seat):
        b = beta[seat] if 0 <= seat < 3 else 0.0
        val = mu[p_idx] + b
        if val > 700:
            return 1e300
        if val < -700:
            return 1e-300
        return math.exp(val)

    def compute_beta_grad():
        """Gradient w.r.t. beta only (mu fixed)."""
        grad = [0.0] * 3
        for groups in processed_games:
            remaining = set((p, s) for grp in groups for p, s in grp)
            S_values = []
            for grp in groups:
                s_r = sum(str_val(p, s) for p, s in remaining)
                S_values.append(s_r)
                remaining -= set((p, s) for p, s in grp)
            remaining = set((p, s) for grp in groups for p, s in grp)
            for gi, grp in enumerate(groups):
                group_size = len(grp)
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
                        g = 1.0 - group_size * e * inv_sr
                    else:
                        g = -group_size * e * inv_sr
                    if s < 3:
                        grad[s] += g * e
                remaining -= set((p, s) for p, s in grp)
        return grad

    # Optimize beta (mu fixed) via gradient ascent with backtracking
    for iteration in range(max_iter):
        grad_b = compute_beta_grad()
        gnorm = math.sqrt(sum(g * g for g in grad_b))
        if gnorm < tol:
            break
        step = 0.0001
        beta_old = beta[:]
        improved = False
        for _ls in range(60):
            for s in range(3):
                beta[s] = beta_old[s] + step * grad_b[s]
            grad_b2 = compute_beta_grad()
            gnorm2 = math.sqrt(sum(g * g for g in grad_b2))
            if gnorm2 < gnorm:
                improved = True
                break
            step *= 0.5
        if not improved:
            beta = beta_old
            break

    mu_result = {}
    for p in players:
        i = index[p]
        if i == anchor_idx:
            mu_result[p] = (0.0, 0.0)
        else:
            mu_result[p] = (mu[i], math.sqrt(_PRIOR_VAR))

    seat_result = {}
    for s in range(3):
        seat_result[s] = (beta[s], math.sqrt(_PRIOR_VAR))

    return mu_result, seat_result
