"""候選枚舉、對手預判、短名單與抽籤。

這一層**不認識任何人格**：它只認 `Brain` 的三個 method。所有 AI（權重式與
規則式）共用同一條流水線，所以拆檔不會改變任何一手的結果。

流水線：

    枚舉合法候選 → 用權重粗評分排序 → brain.context / restrict / rescore
    → （可選）對手預判扣分 → 短名單 → 抽籤 → 落子
"""
import math
import time

from . import formulas as F


def _opponent_pool(board, hand_names, owner, must_cover=None, reach=None,
                   k=F.OPP_POOL_K):
    """Best `k` legal replies for `owner`, filtered by the same rules as
    `choose_move` so the lookahead never assumes a move the game forbids.

    The cheap "neutral" bases (corner- and centre-weighted) are scanned first.
    The corner-contact rule anchors candidates to a player's existing stones,
    which can leave that short list entirely illegal, so an empty result would
    silently turn the opponent response into zero. When it does, the search
    widens to every in-bounds base instead.
    """
    empt = board.empty_bits
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * F.B)
    pool = []
    seen = set()
    for name in hand_names:
        for oi, od in F.ODIRS[name].items():
            m = od["m"]
            for base in od["neutral"]:
                shifted = m << base
                if cbit:
                    if not shifted & cbit:
                        continue
                elif reach is not None:
                    if not shifted & reach.need or shifted & reach.avoid:
                        continue
                if (empt & shifted) == shifted:
                    pool.append((name, oi, base))
                    seen.add((name, oi, base))
        if len(pool) >= k:
            return pool[:k]
    for name in hand_names:
        for oi, od in F.ODIRS[name].items():
            m = od["m"]
            mask = F._adjoining_bases(od, reach, cbit) if (cbit or reach is not None) \
                else od["valid"]
            found = []
            while mask:
                low = mask & -mask
                mask ^= low
                base = low.bit_length() - 1
                if (name, oi, base) in seen:
                    continue
                shifted = m << base
                if (empt & shifted) == shifted:
                    found.append(base)
            if not found:
                continue
            if len(found) > F.OPP_POOL_PER_ORIENT:
                found.sort(key=od["rank"].__getitem__)
                del found[F.OPP_POOL_PER_ORIENT:]
            for base in found:
                pool.append((name, oi, base))
                seen.add((name, oi, base))
        if len(pool) >= k:
            break
    return pool[:k]


def _touches_corner(name, oi, base):
    return F.ODIRS[name][oi]["corner"].get(base, 0) != 0


def _build_shortlist(scored):
    """Core top-N plus reserved corner-claiming moves that stay competitive.

    The reserve keeps a corner grab reachable for personalities that would
    otherwise never shortlist one, so corner appetite varies smoothly with
    w_corner instead of flipping between "always" and "never".
    """
    if not scored:
        return []
    best = scored[0][0]
    sl = scored[:F.SHORTLIST_CORE]
    keys = {(c[1], c[2], c[3]) for c in sl}
    added = 0
    for cand in scored[:F.SHORTLIST_SCAN]:
        if added >= F.CORNER_RESERVE:
            break
        s, name, oi, base = cand
        if s < best - F.CORNER_BAND:
            break
        key = (name, oi, base)
        if key in keys or not _touches_corner(name, oi, base):
            continue
        keys.add(key)
        sl.append(cand)
        added += 1
    return sl


def _weighted_pick(sl, rng):
    """依分數做 softmax 抽籤。分數差一格就差 `PICK_TEMP` 倍的權重。"""
    best = sl[0][0]
    weights = [math.exp((c[0] - best) / F.PICK_TEMP) for c in sl]
    target = rng.random() * sum(weights)
    acc = 0.0
    for cand, w in zip(sl, weights):
        acc += w
        if target <= acc:
            return cand
    return sl[-1]


def choose_move(board, hand_names, owner, brain, rng, other_brains=None,
                must_cover=None, reach=None, other_must_cover=None,
                other_reach=None):
    """回傳 (棋名, 方向, x, y)，無棋可下時回傳 None。

    `other_brains` 給了才會做對手預判（只有權重式人格會用）。預算（時間上限
    與模擬次數）超了就跳過那一段，退回純粹的候選排序。
    """
    t0 = time.perf_counter()
    empt = board.empty_bits
    open_a, block, defend = F.board_feats(board.grid)
    regions = board.corner_regions
    borders = board.borders
    profile = brain.profile
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * F.B)
    names = list(hand_names)
    cands = []
    for name in names:
        od_map = F.ODIRS[name]
        bi = block[owner]
        di = defend[owner]
        for oi, od in od_map.items():
            m = od["m"]
            corner = od["corner"]
            csum = od["csum"]
            offs = od["offs"]
            cs_const = (profile.w_large * od["size"]
                        - F.SMALL_PENALTY * (5 - od["size"]))
            for base in od["bases"]:
                shifted = m << base
                if (empt & shifted) != shifted:
                    continue
                if cbit:
                    if not shifted & cbit:
                        continue
                elif reach is not None:
                    if not shifted & reach.need or shifted & reach.avoid:
                        continue
                s = cs_const + profile.w_center * csum[base]
                for o in offs:
                    i = base + o
                    s += (profile.w_open * open_a[i] + profile.w_block * bi[i]
                          + profile.w_defend * di[i])
                cb = corner.get(base, 0)
                if cb:
                    cs = set(base + o for o in offs)
                    for ci in (0, 1, 2, 3):
                        if cb & (1 << ci):
                            s += (profile.w_corner * 3.0
                                  * F._corner_delta(ci, cs, owner, regions, borders))
                cands.append((s, name, oi, base))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    # 人格接手：restrict 是階段（軟）過濾，rescore 才是貴的那一次評估。
    # 權重式人格兩者都是恆等，所以以下這段對它們完全不影響既有行為。
    ctx = brain.context(board, names, owner, must_cover, reach)
    cands = brain.restrict(cands, ctx)
    cands = brain.rescore(cands, ctx)
    top = cands[:F.SHORTLIST_CORE + F.CORNER_RESERVE]

    adj = top
    opps = [o for o in range(4) if o != owner]
    if other_brains and opps and brain.uses_lookahead \
            and time.perf_counter() - t0 < F.WALL_BUDGET:
        pools = [_opponent_pool(board, hand_names, o,
                                (other_must_cover or {}).get(o),
                                (other_reach or {}).get(o))
                 for o in opps]
        adj = []
        sim_eval = 0
        skip = False
        for sc, name, oi, base in top:
            if (not skip
                    and sim_eval + F.OPP_POOL_K * len(opps) <= F.SIM_EVAL_BUDGET):
                placed = F.ODIRS[name][oi]["m"] << base
                resp = 0.0
                for idx, o in enumerate(opps):
                    po = other_brains[o].profile
                    best_o = -1e18
                    for item in pools[idx]:
                        name2, oi2, base2 = item
                        shifted = F.ODIRS[name2][oi2]["m"] << base2
                        if shifted & placed:
                            continue
                        v = F._score_move(board, name2, oi2, base2, o, po,
                                          open_a, block, defend, regions, borders)
                        if v > best_o:
                            best_o = v
                        sim_eval += 1
                    if best_o > 0:
                        resp += best_o
                adj.append((sc - resp, name, oi, base))
            else:
                skip = True
                adj.append((sc, name, oi, base))
        if adj:
            adj.sort(key=lambda t: t[0], reverse=True)
            adj = adj[:F.SHORTLIST_CORE + F.CORNER_RESERVE]
    else:
        adj = top

    shortlist = _build_shortlist(adj)
    if not shortlist:
        return None
    pick = shortlist[0]
    if len(shortlist) >= 2 and rng.random() < brain.mistake_rate:
        pick = _weighted_pick(shortlist, rng)
    return (pick[1], pick[2], pick[3] % F.B, pick[3] // F.B)
