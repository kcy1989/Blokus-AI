"""Same-seed, move-by-move comparison between two implementations.

The hard rule for every behaviour-preserving change in this project: an old
build and a new build must produce the *same move* on *every turn* of many
games. "Close enough" is a bug.

    python3 tools/verify_same_moves.py --impl current --games 1000
    python3 tools/verify_same_moves.py --impl current --games 1000 \\
        --baseline record.json

`frozen_choose_move` below is a verbatim copy of `ai/chooser.choose_move` as it
stood *before* stage C rewrote the candidate enumeration. It is the oracle: the
optimised engine is checked against obviously-correct code that is still
available after the original is gone from the engine. It is never edited.

C2 and C3 also compare intermediate state, not just final moves, so the tool
checks that too:

  * `has_legal_new(owner) == has_legal_old(owner)` at every step
  * the candidate list is element-for-element equal at every decision point
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
import ai.formulas as F  # noqa: E402
from ai.chooser import _build_shortlist, _opponent_pool, _weighted_pick  # noqa: E402
from game import Game  # noqa: E402
from pieces import MASTER  # noqa: E402

_real_choose = ai.choose_move


# --------------------------------------------------------------- reference

def frozen_choose_move(board, hand_names, owner, brain, rng, other_brains=None,
                       must_cover=None, reach=None, other_must_cover=None,
                       other_reach=None):
    """Verbatim copy of chooser.choose_move before C3.

    The candidate enumeration here is the naive one: visit every base of every
    orientation and test the two constraints directly. C3 replaces it with a
    bitmask walk; this copy is what proves the replacement is equivalent.
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


def brute_candidates(board, hand_names, owner, brain, must_cover, reach):
    """The pre-C3 candidate list, in the pre-C3 order."""
    from ai.chooser import choose_move  # noqa: F401  (documented reference)
    empt = board.empty_bits
    open_a, block, defend = F.board_feats(board.grid)
    regions = board.corner_regions
    borders = board.borders
    profile = brain.profile
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * F.B)
    cands = []
    bi = block[owner]
    di = defend[owner]
    for name in list(hand_names):
        for oi, od in F.ODIRS[name].items():
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
    return cands


# --------------------------------------------------------------- recording

def make_game(seed):
    keys = list(ai.personality_keys())
    g = Game(random.Random(seed))
    g.setup_match(random.Random(seed).sample(keys, 4))
    g.start()
    return g


def play_recorded(seed, choose=None):
    """Play one game from `seed`; return [(turn_count, owner, move_or_None)]."""
    choose = choose or _real_choose
    g = make_game(seed)
    record = []
    guard = 0
    while g.state == "PLAYING" and guard < 600:
        guard += 1
        o = g.current_owner()
        mv = choose(g.board, g.hands[o].names, o, g.brains[o], g.rng,
                    other_brains=g.brains,
                    must_cover=g.must_cover(o), reach=g.reach(o),
                    other_must_cover={x: g.must_cover(x) for x in range(4)},
                    other_reach={x: g.reach(x) for x in range(4)})
        record.append((g.turn_count, o, mv))
        if mv is None:
            g.act_pass()
        else:
            g.act(*mv)
    return record


def record_all(impl, games):
    choose = {"current": _real_choose, "frozen": frozen_choose_move}[impl]
    return [play_recorded(s, choose=choose) for s in range(games)]


def compare(a, b, limit=3):
    diffs = []
    if len(a) != len(b):
        diffs.append("game count %d vs %d" % (len(a), len(b)))
    for gi, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            diffs.append("game %d: %d turns vs %d" % (gi, len(ra), len(rb)))
            if len(diffs) >= limit:
                return diffs
        for ti, (ta, tb) in enumerate(zip(ra, rb)):
            if ta != tb:
                diffs.append("game %d turn %d (t=%s): %r vs %r"
                             % (gi, ti, ta[0], ta, tb))
                if len(diffs) >= limit:
                    return diffs
    return diffs


# --------------------------------------------- C2 / C3 intermediate checks

def check_has_legal(games, rounds=1):
    """C2: the new has_legal must equal the old one at every step."""
    bad = 0
    checked = 0
    for seed in range(games):
        g = make_game(seed)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            o = g.current_owner()
            for who in range(4):
                cells_iter = [MASTER[n]["orientations"]
                              for n in g.hands[who].names]
                old = g.board.has_legal_move(who, cells_iter,
                                             g.must_cover(who), g.reach(who))
                new = g.has_legal(who)
                checked += 1
                if old != new:
                    bad += 1
                    if bad <= 3:
                        print("  MISMATCH seed=%d turn=%d owner=%d old=%s new=%s"
                              % (seed, g.turn_count, who, old, new))
            mv = _real_choose(g.board, g.hands[o].names, o, g.brains[o], g.rng,
                              other_brains=g.brains,
                              must_cover=g.must_cover(o), reach=g.reach(o),
                              other_must_cover={x: g.must_cover(x)
                                                for x in range(4)},
                              other_reach={x: g.reach(x) for x in range(4)})
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)
    return checked, bad


def check_candidates(games):
    """C3: the candidate list must be equal element for element, in order."""
    bad = 0
    checked = 0
    total = 0
    for seed in range(games):
        g = make_game(seed)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            o = g.current_owner()
            old = brute_candidates(g.board, g.hands[o].names, o, g.brains[o],
                                   g.must_cover(o), g.reach(o))
            new = new_candidates(g.board, g.hands[o].names, o, g.brains[o],
                                 g.must_cover(o), g.reach(o))
            checked += 1
            total += len(old)
            if old != new:
                bad += 1
                if bad <= 2:
                    print("  CANDIDATE MISMATCH seed=%d turn=%d owner=%d "
                          "old=%d new=%d" % (seed, g.turn_count, o, len(old), len(new)))
                    for i, (x, y) in enumerate(zip(old, new)):
                        if x != y:
                            print("    first diff at %d: %r vs %r" % (i, x, y))
                            break
            mv = _real_choose(g.board, g.hands[o].names, o, g.brains[o], g.rng,
                              other_brains=g.brains,
                              must_cover=g.must_cover(o), reach=g.reach(o),
                              other_must_cover={x: g.must_cover(x)
                                                for x in range(4)},
                              other_reach={x: g.reach(x) for x in range(4)})
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)
    return checked, total, bad


def new_candidates(board, hand_names, owner, brain, must_cover, reach):
    """The candidate list the *current* engine builds (C3's replacement)."""
    import ai.chooser as CH
    empt = board.empty_bits
    open_a, block, defend = F.board_feats(board.grid)
    return CH._candidates(board, list(hand_names), owner, brain.profile,
                          must_cover, reach, open_a, block, defend,
                          board.corner_regions, board.borders, empt)


def main():
    ap = argparse.ArgumentParser(description="same-seed move comparison")
    ap.add_argument("--impl", default="current", choices=["current", "frozen"])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--record", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--check-has-legal", type=int, default=0,
                    help="C2: compare has_legal over N games")
    ap.add_argument("--check-candidates", type=int, default=0,
                    help="C3: compare candidate lists over N games")
    args = ap.parse_args()

    if args.check_has_legal:
        checked, bad = check_has_legal(args.check_has_legal)
        print("has_legal: %d checks over %d games, %d mismatches"
              % (checked, args.check_has_legal, bad))
        return 1 if bad else 0

    if args.check_candidates:
        checked, total, bad = check_candidates(args.check_candidates)
        print("candidates: %d decision points, %d candidate rows, %d mismatches"
              % (checked, total, bad))
        return 1 if bad else 0

    t0 = time.perf_counter()
    rec = record_all(args.impl, args.games)
    print("recorded %d games with impl=%r in %.1fs"
          % (args.games, args.impl, time.perf_counter() - t0))
    print("total turns: %d" % sum(len(g) for g in rec))

    if args.baseline:
        with open(args.baseline) as fh:
            base = [[(t, o, tuple(m) if m else None) for t, o, m in g]
                    for g in json.load(fh)]
        diffs = compare(base, rec)
        if diffs:
            print("DIFFERENCES (%d shown):" % len(diffs))
            for d in diffs:
                print("  " + d)
            return 1
        print("IDENTICAL: every move matches on every turn")
        return 0

    if args.record:
        payload = [[(t, o, list(m) if m else None) for t, o, m in g] for g in rec]
        with open(args.record, "w") as fh:
            json.dump(payload, fh)
        print("wrote %s" % args.record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
