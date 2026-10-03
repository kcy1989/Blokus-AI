"""Candidate enumeration, opponent lookahead, shortlist and weighted pick.

This layer **knows no personality**: it only knows three methods of `Brain`.
Every AI (weighted and rule-based) shares the same pipeline, so splitting the
file does not change the result of any single move.

Pipeline:

    enumerate legal candidates -> sort by a rough weighted score
    -> brain.context / restrict / rescore
    -> (optional) opponent-lookahead penalty -> shortlist -> pick -> place
"""
import dataclasses
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
    """Softmax draw over the scores. A one-point score gap becomes a factor of
    `PICK_TEMP` in the weights."""
    best = sl[0][0]
    weights = [math.exp((c[0] - best) / F.PICK_TEMP) for c in sl]
    target = rng.random() * sum(weights)
    acc = 0.0
    for cand, w in zip(sl, weights):
        acc += w
        if target <= acc:
            return cand
    return sl[-1]


def _candidates(board, names, owner, profile, must_cover, reach,
                open_a, block, defend, regions, borders, empt):
    """Every legal (score, name, oi, base) for `owner`, in enumeration order.

    C3's replacement. The old loop visited every base of every orientation and
    tested the two constraints by hand. `_legal_bases` computes the same set as
    one bitmask per orientation, and walking it lowest bit first visits bases in
    ascending order - which is exactly the order `od["bases"]` is built in, so
    the candidate list comes out identical element for element.

    Extracted into its own function so the verification tool can compare the
    candidate list directly, not just the final move.
    """
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * F.B)
    cands = []
    bi = block[owner]
    di = defend[owner]
    for name in names:
        for oi, od in F.ODIRS[name].items():
            mask = F._legal_bases(od, empt, reach, cbit)
            if not mask:
                continue
            corner = od["corner"]
            csum = od["csum"]
            offs = od["offs"]
            cs_const = (profile.w_large * od["size"]
                        - F.SMALL_PENALTY * (5 - od["size"]))
            while mask:
                low = mask & -mask
                mask ^= low
                base = low.bit_length() - 1
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
    return cands


def choose_move(board, hand_names, owner, brain, rng, other_brains=None,
                must_cover=None, reach=None, other_must_cover=None,
                other_reach=None, trace=None):
    """Returns (piece name, orientation, x, y), or None when no move is
    possible.

    `other_brains` is what enables the opponent lookahead (only weighted
    personalities pass it). If the budget (wall-clock cap and simulation
    count) is exceeded, that stage is skipped and it falls back to plain
    candidate ordering.

    `trace` is a D1 opt-in. Pass a list and one dict per decision is appended,
    recording what the engine saw and why it chose what it chose - the raw
    material for a dataset. It never changes the return value, and with
    `trace=None` (the default) every extra branch below is skipped, so the
    decision path and its cost are bit-for-bit what they were before.
    """
    t0 = time.perf_counter()
    empt = board.empty_bits
    open_a, block, defend = F.board_feats(board.grid)
    regions = board.corner_regions
    borders = board.borders
    profile = brain.profile
    names = list(hand_names)
    cands = _candidates(board, names, owner, profile, must_cover, reach,
                        open_a, block, defend, regions, borders, empt)
    if not cands:
        if trace is not None:
            trace.append(_empty_trace(owner, brain))
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    if trace is not None:
        n_candidates = len(cands)
    # Personality takes over: restrict is the stage (soft) filter, rescore is
    # the expensive evaluation. For weighted personalities both are the
    # identity, so everything below leaves their existing behaviour untouched.
    ctx = brain.context(board, names, owner, must_cover, reach)
    cands = brain.restrict(cands, ctx)
    cands = brain.rescore(cands, ctx)
    top = cands[:F.SHORTLIST_CORE + F.CORNER_RESERVE]

    adj = top
    # Seats with no brain are people, not contestants. They are not opponents to
    # model a reply for, and the `other_brains[o].profile` below would raise on
    # them - which used to be impossible, because the one seat a person played
    # still carried a chess brain it never used.
    opps = [o for o in range(4)
            if o != owner and (other_brains or {}).get(o) is not None]
    lookahead_used = False
    if other_brains and opps and brain.uses_lookahead \
            and (not F.USE_WALL_BUDGET
                 or time.perf_counter() - t0 < F.WALL_BUDGET):
        lookahead_used = True
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
        if trace is not None:
            trace.append(_empty_trace(owner, brain))
        return None
    pick = shortlist[0]
    was_mistake = False
    if len(shortlist) >= 2 and rng.random() < brain.mistake_rate:
        pick = _weighted_pick(shortlist, rng)
        was_mistake = True
    if trace is not None:
        trace.append({
            "owner": owner,
            "brain_key": brain.key,
            "profile": dataclasses.asdict(brain.profile),
            "n_candidates": n_candidates,
            "shortlist": [(sc, n, oi, b) for sc, n, oi, b in shortlist],
            "picked": (pick[1], pick[2], pick[3]),
            "was_mistake": was_mistake,
            "lookahead_used": lookahead_used,
        })
    return (pick[1], pick[2], pick[3] % F.B, pick[3] // F.B)


def _empty_trace(owner, brain):
    """Trace record for a decision with no move: there was nothing to pick."""
    return {
        "owner": owner,
        "brain_key": brain.key,
        "profile": dataclasses.asdict(brain.profile),
        "n_candidates": 0,
        "shortlist": [],
        "picked": None,
        "was_mistake": False,
        "lookahead_used": False,
    }
