"""Stage F-prime: measure and verify, change nothing.

Every number in here is produced by *reading* the engine, the action table and
the AI pipeline. No existing file is modified and no slow path is optimised:
plan2.md is explicit that this stage exists to obtain facts, and a fact that
arrived with a fix attached is no longer a fact.

The report is split in two, because the two halves have different honesty
properties:

    results      legal-move counts, decision points, pass patterns, trace
                 fields. Reproducible bit for bit from a fixed seed.
    measurement  wall-clock timings. Not reproducible and not claimed to be.
                 `--no-timing` omits them, so the rest compares byte for byte.

    python3 bench_engine.py                 # everything
    python3 bench_engine.py --no-timing     # results only, byte-reproducible
    python3 bench_engine.py --sections f2 f3
"""
import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai  # noqa: E402
import engine  # noqa: E402
from board import Board  # noqa: E402
from config import B, CLOCKWISE_OWNERS, N, OWNER_CORNER  # noqa: E402
from game import Game  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")

# Stage boundaries are defined by the number of *real* pieces played, per
# plan2.md section 1. Passes never advance a stage: a player who cannot move
# has not made progress.
STAGE_OPENING = "opening"   # 0  <= n < 16
STAGE_MIDDLE = "middle"     # 16 <= n < 48
STAGE_END = "end"           # 48 <= n
STAGES = (STAGE_OPENING, STAGE_MIDDLE, STAGE_END)
STAGE_LABELS = {"opening": "開局", "middle": "中盤", "end": "殘局"}


def stage_of(n_real):
    if n_real < 16:
        return STAGE_OPENING
    if n_real < 48:
        return STAGE_MIDDLE
    return STAGE_END


# --------------------------------------------------------------------------
# section 1: the shared playout
# --------------------------------------------------------------------------

def random_playout(seed, hook=None, max_passes=1000, tape=None):
    """A whole game, every legal move equally likely.

    `random.Random(seed)` only, and the legal moves are *sorted* before the
    draw, so the same seed plays the same game on any machine. The engine
    returns sets and frozensets, whose iteration order depends on the hash
    seed, so drawing without sorting would make the games unreproducible.

    `hook(state, mask, n_real_moves, n_passes)` runs *before* every decision,
    including the ones where the mask is empty, which is what lets F3 look at
    the position `to_move` is about to be handed.

    `tape` is the (state, mask, n_real_moves, move) record of every decision,
    appended after the move has been drawn. It exists because F1B has to time
    `apply_move` on the move this game really played, and that is not knowable
    inside the hook.
    """
    rng = random.Random(seed)
    s = engine.initial_state()
    n_real = 0
    n_passes = 0
    while not engine.is_over(s):
        mask = engine.legal_move_mask(s)
        if hook is not None:
            hook(s, mask, n_real, n_passes)
        if mask == 0:
            n_passes += 1
            if n_passes > max_passes:
                raise RuntimeError(
                    "seed %d passed %d times without finishing" % (seed, n_passes))
            if tape is not None:
                tape.append((s, mask, n_real, None))
            s = engine.pass_turn(s)
            continue
        moves = sorted(engine.unpack_move_mask(mask))
        m = moves[rng.randrange(len(moves))]
        if tape is not None:
            tape.append((s, mask, n_real, m))
        s = engine.apply_move(s, m)
        n_real += 1
    return s, n_real, n_passes


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def pct(values, q):
    """Nearest-rank percentile: the smallest value at or above rank ceil(q*n)."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(q * len(s)))
    return s[k - 1]


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "max": None, "min": None}
    s = sorted(values)
    return {"n": len(s),
            "mean": sum(s) / len(s),
            "median": statistics.median(s),
            "p95": pct(s, 0.95),
            "max": s[-1],
            "min": s[0]}


PIECE_STRIDE = engine.MAX_ORIENTS * engine.N
assert PIECE_STRIDE == engine._MOVE_STRIDE, "move packing stride moved"


def pieces_in_mask(mask):
    """How many distinct piece kinds have at least one legal placement."""
    n = 0
    for p in range(engine.N_PIECES):
        if (mask >> (p * PIECE_STRIDE)) & ((1 << PIECE_STRIDE) - 1):
            n += 1
    return n


# --------------------------------------------------------------------------
# F1: engine speed
# --------------------------------------------------------------------------

def f1_game_timing(n_games):
    """A: one wall-clock number per whole game."""
    per_game, reals, passes, final = [], [], [], []
    for seed in range(n_games):
        t0 = time.perf_counter()
        s, n_real, n_passes = random_playout(seed)
        dt = time.perf_counter() - t0
        per_game.append(dt)
        reals.append(n_real)
        passes.append(n_passes)
        final.append(engine.result(s))
    total_time = sum(per_game)
    total_real = sum(reals)
    return {
        "games": n_games,
        "seconds_per_game": stats(per_game),
        "real_moves": stats(reals),
        "passes": stats(passes),
        "total_seconds": total_time,
        "total_real_moves": total_real,
        "real_moves_per_second": total_real / total_time,
        "mean_result_cells": [sum(r[i] for r in final) / len(final)
                              for i in range(4)],
    }


F1B_FUNCS = ("legal_move_mask", "legal_moves", "has_any_legal", "apply_move",
             "pass_turn", "to_plane", "hand_vectors", "normalize",
             "serialize", "deserialize")


def f1_function_timing(n_games, warmup=50):
    """B: every function, once per position, timed on its own.

    Positions come from real playouts rather than from a synthetic board, so
    the numbers describe the states the engine actually meets. The first
    `warmup` positions are called and thrown away: the first call into numpy
    and the first big-int allocations are not what a steady state looks like.
    """
    tape = []
    for seed in range(n_games):
        random_playout(seed, tape=tape)
    for _s, _m, _n, _mv in tape[:warmup]:
        _warm_one(_s, _mv)

    buckets = {k: {st: [] for st in STAGES} for k in F1B_FUNCS}
    for s, _m, n_real, mv in tape[warmup:]:
        stage = stage_of(n_real)
        blob = engine.serialize(s)
        for fn in F1B_FUNCS:
            # `apply_move` and `pass_turn` are two halves of the same decision,
            # so each is timed only where it is the call the game actually made.
            # Timing either one on the wrong kind of position would measure a
            # call the engine never performs.
            if fn == "pass_turn" and mv is not None:
                continue
            if fn == "apply_move" and mv is None:
                continue
            t0 = time.perf_counter()
            _call(fn, s, mv, blob)
            dt = (time.perf_counter() - t0) * 1000.0
            buckets[fn][stage].append(dt)
    out = {"games": n_games, "positions_collected": len(tape),
           "positions_timed": len(tape) - warmup, "warmup_skipped": warmup,
           "functions": {}}
    for fn in F1B_FUNCS:
        row = {st: stats(buckets[fn][st]) for st in STAGES}
        row["all"] = stats([dt for st in STAGES for dt in buckets[fn][st]])
        out["functions"][fn] = row
    to_plane_mean = out["functions"]["to_plane"]["all"]["mean"]
    norm_mean = out["functions"]["normalize"]["all"]["mean"]
    out["to_plane_normalize_share"] = (norm_mean / to_plane_mean
                                       if to_plane_mean else None)
    out["normalize_means_by_stage"] = {
        st: stats(buckets["normalize"][st])["mean"] for st in STAGES}
    out["to_plane_means_by_stage"] = {
        st: stats(buckets["to_plane"][st])["mean"] for st in STAGES}
    return out


def _call(fn, s, mv, blob):
    if fn == "legal_move_mask":
        return engine.legal_move_mask(s)
    if fn == "legal_moves":
        return engine.legal_moves(s)
    if fn == "has_any_legal":
        return engine.has_any_legal(s, s.to_move)
    if fn == "apply_move":
        return engine.apply_move(s, mv)
    if fn == "pass_turn":
        return engine.pass_turn(s)
    if fn == "to_plane":
        return engine.to_plane(s)
    if fn == "hand_vectors":
        return engine.hand_vectors(s)
    if fn == "normalize":
        return engine.normalize(s)
    if fn == "serialize":
        return engine.serialize(s)
    if fn == "deserialize":
        return engine.deserialize(blob)
    raise KeyError(fn)


def _warm_one(s, mv):
    blob = engine.serialize(s)
    for fn in F1B_FUNCS:
        if fn == "pass_turn" and mv is not None:
            continue
        if fn == "apply_move" and mv is None:
            continue
        _call(fn, s, mv, blob)


# --------------------------------------------------------------------------
# F1b: how many legal moves there are
# --------------------------------------------------------------------------

def f1b_distribution(n_games):
    """A results-region section: the shape of the search space."""
    counts = {st: [] for st in STAGES}
    by_bucket = {}
    kinds, all_counts = [], []
    pass_points = 0
    real_points = 0
    empty_at = {st: 0 for st in STAGES}
    for seed in range(n_games):
        tape = []
        random_playout(seed, tape=tape)
        for _s, mask, n_real, _mv in tape:
            stage = stage_of(n_real)
            if mask == 0:
                pass_points += 1
                empty_at[stage] += 1
                continue
            real_points += 1
            c = mask.bit_count()
            counts[stage].append(c)
            all_counts.append(c)
            kinds.append(pieces_in_mask(mask))
            b = n_real // 8
            by_bucket.setdefault(b, []).append(c)
    total_points = pass_points + real_points
    return {
        "games": n_games,
        "decision_points": total_points,
        "real_decision_points": real_points,
        "pass_decision_points": pass_points,
        "pass_share": pass_points / total_points,
        "pass_points_by_stage": empty_at,
        "legal_counts": {"all": stats(all_counts),
                         "opening": stats(counts[STAGE_OPENING]),
                         "middle": stats(counts[STAGE_MIDDLE]),
                         "end": stats(counts[STAGE_END])},
        "legal_counts_by_move_bucket": {
            str(b * 8) + "-" + str(b * 8 + 7): stats(v)
            for b, v in sorted(by_bucket.items())},
        "piece_kinds_available": stats(kinds),
    }


# --------------------------------------------------------------------------
# F2: the action table against the engine
# --------------------------------------------------------------------------

def f2_action_table():
    with open(os.path.join(HERE, "action_table.json"), encoding="utf-8") as fh:
        table = json.load(fh)
    shape = {}
    for k, v in table.items():
        shape[k] = {"type": type(v).__name__,
                    "len": len(v) if hasattr(v, "__len__") else None,
                    "sample": (v[:2] if isinstance(v, list) else v)}

    n_orients = sum(len(engine.ORIENTS[p]) for p in range(engine.N_PIECES))
    n_valid = sum(od["valid"].bit_count()
                  for p in range(engine.N_PIECES)
                  for od in engine.ORIENTS[p])
    n_bases = sum(od["valid"].bit_count()
                  for p in range(engine.N_PIECES)
                  for od in engine.ORIENTS[p])

    s0 = engine.initial_state()
    opening = {}
    for owner in range(4):
        opening[str(owner)] = engine.legal_move_mask(s0, owner).bit_count()

    # Brute force, written the slow obvious way: every orientation, every base
    # the orientation can sit on, and an explicit test for the corner square.
    corner_hits = {}
    for owner in range(4):
        cx, cy = OWNER_CORNER[owner]
        cbit = 1 << (cx + cy * B)
        n = 0
        for p in range(engine.N_PIECES):
            for od in engine.ORIENTS[p]:
                m, valid = od["m"], od["valid"]
                while valid:
                    low = valid & -valid
                    valid ^= low
                    if (m << (low.bit_length() - 1)) & cbit:
                        n += 1
        corner_hits[str(owner)] = n

    return {
        "action_table_shape": shape,
        "action_table_hash": table.get("hash"),
        "orientation_checks": {
            "engine_orientations": n_orients,
            "expected": 91,
            "pass": n_orients == 91,
        },
        "valid_base_checks": {
            "engine_valid_bases": n_valid,
            "expected": 30433,
            "pass": n_valid == 30433,
        },
        "action_table_placements": {
            "declared_total_placements": table.get("total_placements"),
            "declared_total_cells": table.get("total_cells"),
            "actions_entries": len(table.get("actions", [])),
            "actions_entry_shape": "list of [piece_name, orientation_index]",
            "pass": table.get("total_placements") == n_valid,
            "note": ("the table stores orientations, not per-base placements, so "
                     "30433 can only be checked against the declared field and "
                     "against ORIENTS recomputed from pieces.py"),
        },
        "opening_legal_moves": opening,
        "opening_all_equal": len(set(opening.values())) == 1,
        "opening_brute_force_corner_hits": corner_hits,
        "opening_matches_brute_force": opening == corner_hits,
        "config": {"B": B, "N": N,
                   "CLOCKWISE_OWNERS": list(CLOCKWISE_OWNERS),
                   "OWNER_CORNER": {str(k): list(v) for k, v in OWNER_CORNER.items()}},
        "engine_derived": {
            "N_PIECES": engine.N_PIECES,
            "MAX_ORIENTS": engine.MAX_ORIENTS,
            "HAND_CELLS_TOTAL": engine.HAND_CELLS_TOTAL,
            "PIECE_ORDER": list(engine.PIECE_ORDER),
            "total_valid_bases_recomputed": n_bases,
        },
    }


# --------------------------------------------------------------------------
# F3: the deferred stuck latch
# --------------------------------------------------------------------------

def f3_advance_latch(n_games, sample_seed=12345):
    """Situation X is a *documentation* fact, not a bug: `_advance` stops at the
    first owner who can still move, so an owner further round the table can be
    handed the turn with an empty mask and a stuck flag that is not yet set.
    The question is only how often it happens and what it costs."""
    total_points = 0
    x_hits = []
    y_hits = []
    games_with_x = 0
    max_pass_run = 0
    games_not_over = []
    games_final_legal = []
    end_mismatch = []

    for seed in range(n_games):
        tape = []
        final, n_real, n_passes = random_playout(seed, tape=tape)
        run = 0
        game_x = 0
        for s, mask, n_at, _mv in tape:
            total_points += 1
            if mask == 0:
                run += 1
                max_pass_run = max(max_pass_run, run)
            else:
                run = 0
            if mask == 0 and not s.stuck[s.to_move]:
                x_hits.append((seed, n_at, s))
                game_x += 1
            for o in range(4):
                if s.stuck[o] and engine.has_any_legal(s, o):
                    y_hits.append((seed, n_at, o, s))
        if game_x:
            games_with_x += 1
        if not engine.is_over(final):
            games_not_over.append(seed)
        if any(engine.has_any_legal(final, o) for o in range(4)):
            games_final_legal.append(seed)
        end_mismatch.append(_replay_result(seed, tape, final, n_real, n_passes))

    mismatches = [m for m in end_mismatch if m is not None]
    example = None
    if x_hits:
        pick = random.Random(sample_seed).randrange(len(x_hits))
        seed, n_at, s = x_hits[pick]
        example = {"seed": seed, "move_index": n_at,
                   "to_move": s.to_move, "stuck": list(s.stuck),
                   "serialize": engine.serialize(s)}
    return {
        "games": n_games,
        "decision_points_including_passes": total_points,
        "situation_x": {
            "definition": ("legal_move_mask(state) == 0 and "
                           "state.stuck[state.to_move] is False"),
            "count": len(x_hits),
            "share_of_all_decision_points": len(x_hits) / total_points,
            "games_with_at_least_one": games_with_x,
            "example": example,
        },
        "situation_y": {
            "definition": ("state.stuck[o] is True and "
                           "has_any_legal(state, o) is True"),
            "count": len(y_hits),
            "examples": [{"seed": a, "move_index": b, "owner": c,
                          "serialize": engine.serialize(d)}
                         for a, b, c, d in y_hits[:3]],
        },
        "situation_z": {"longest_pass_run": max_pass_run},
        "termination": {
            "games_not_over": games_not_over,
            "games_with_a_legal_move_at_the_end": games_final_legal,
            "all_games_over": not games_not_over,
            "all_final_positions_immobile": not games_final_legal,
        },
        "result_vs_game_replay": {
            "games_compared": n_games,
            "mismatches": len(mismatches),
            "detail": mismatches[:5],
            "source": "bench_engine._replay_result drives Game.act/act_pass "
                      "with the same move list",
        },
        "existing_test_coverage": {
            "file": "tests/test_engine_cross.py",
            "seeds": 40,
            "lines": "301-314, 343-367",
            "note": "test_engine_tracks_a_live_game_exactly already asserts "
                    "engine.result(s) == tuple(g.remaining_cells(o)) at the end "
                    "of every game and engine.stuck == g.stuck at every step",
        },
    }


def _replay_result(seed, tape, final, n_real, n_passes):
    """The engine's final score, recomputed by replaying the same moves into the
    human-facing Game. Two implementations, one answer."""
    s0 = engine.initial_state()
    g = Game(random.Random(seed))
    g.turn_order = list(s0.turn_order)
    g.turn_pos = 0
    g.start()
    for s, mask, _n, mv in tape:
        owner = g.current_owner()
        if owner != s.to_move:
            return {"seed": seed, "error": "turn order diverged at %d" % _n}
        if mask == 0:
            g.act_pass()
        else:
            # The tape holds engine moves, which number the pieces; Game
            # spells them out.
            piece_idx, oi, base = mv
            name = engine.PIECE_ORDER[piece_idx]
            x, y = base % B, base // B
            if not g.can_act(name, oi, x, y, owner):
                return {"seed": seed,
                        "error": "Game rejected %s at (%d,%d)" % (name, x, y)}
            g.act(name, oi, x, y)
    engine_cells = engine.result(final)
    game_cells = tuple(g.remaining_cells(o) for o in range(4))
    if engine_cells != game_cells:
        return {"seed": seed, "engine": list(engine_cells),
                "game": list(game_cells)}
    if g.state != "GAME_OVER":
        return {"seed": seed, "error": "Game did not finish: %s" % g.state}
    return None


# --------------------------------------------------------------------------
# F4: the AI pipeline, read only
# --------------------------------------------------------------------------

def _board_from_state(s):
    """A `Board` that shows the same position as an engine `State`.

    `choose_move` takes a Board, not a State, so F4 has to cross the two
    representations. The fields are filled in directly and the existing
    `_update_regions` is called once, because placing 60-odd pieces one at a
    time would recompute the corner regions on every call for no gain.
    """
    b = Board()
    b.grid = [-1] * N
    for o in range(4):
        bits = s.own_bits[o]
        while bits:
            low = bits & -bits
            bits ^= low
            b.grid[low.bit_length() - 1] = o
    b.owner_bits = list(s.own_bits)
    b.empty_bits = engine.ALL & ~(s.own_bits[0] | s.own_bits[1]
                                  | s.own_bits[2] | s.own_bits[3])
    b._update_regions()
    return b


def _hand_names(s, owner):
    bits = s.hand_bits[owner]
    return [engine.PIECE_ORDER[i] for i in range(engine.N_PIECES)
            if bits >> i & 1]


def _playout_key(key, seed):
    return key, seed


F4A_KEYS = ("optimizer", "wolf", "chess", "fox", "intruder", "builder")


def _run_pipeline_games(key, n_games):
    """One full game per seed with all four seats running the same personality.

    A personality has to be allowed to open, so a personality whose opening
    piece cannot reach its corner would never be measured at all. The seeds
    and the seat order are fixed, and the seat order is the plain clockwise
    one, so a rerun sees the same positions.
    """
    out = {"picked_total": 0, "picked_in_legal_mask": 0, "misses": [],
           "k_values": [], "n_cand_values": [], "empty_records": 0,
           "mistake_choices": 0, "lookahead_used": 0, "real_moves": 0}
    for seed in range(n_games):
        g = Game(random.Random(seed))
        g.turn_order = [CLOCKWISE_OWNERS[i] for i in range(4)]
        g.turn_pos = 0
        g.brains = {o: ai.make_brain(key, random.Random(seed * 4 + o))
                    for o in range(4)}
        g.start()
        s = engine.initial_state(g.turn_order)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            owner = g.current_owner()
            tr = []
            ai.choose_move(g.board, g.hands[owner].names, owner,
                           g.brains[owner], g.rng, other_brains=g.brains,
                           must_cover=g.must_cover(owner),
                           reach=g.reach(owner),
                           other_must_cover={x: g.must_cover(x) for x in range(4)},
                           other_reach={x: g.reach(x) for x in range(4)},
                           trace=tr)
            rec = tr[-1]
            if rec["picked"] is None:
                out["empty_records"] += 1
                g.act_pass()
                s = engine.pass_turn(s)
                continue
            out["k_values"].append(len(rec["shortlist"]))
            out["n_cand_values"].append(rec["n_candidates"])
            out["mistake_choices"] += 1 if rec["was_mistake"] else 0
            out["lookahead_used"] += 1 if rec["lookahead_used"] else 0
            name, oi, base = rec["picked"]
            out["picked_total"] += 1
            conv = (engine.PIECE_IDX[name], oi, base)
            if engine.move_bit(*conv) & engine.legal_move_mask(s, owner):
                out["picked_in_legal_mask"] += 1
            elif len(out["misses"]) < 5:
                out["misses"].append({"seed": seed, "picked": list(rec["picked"]),
                                      "converted": list(conv)})
            s = engine.apply_move(s, conv)
            g.act(name, oi, base % B, base // B)
            out["real_moves"] += 1
    return out


def f4a_trace_fields(n_games=200, keys=F4A_KEYS, k_games=40):
    """A: what a trace record actually holds, checked against the engine.

    The conversion is the interesting half. `picked` is `(name, oi, base)` in
    the AI's own spelling and the engine wants `(piece_idx, oi, base)`; the only
    difference is that the piece is named rather than numbered, and the
    numbering is `action_table.json`'s piece order. If that mapping were wrong
    the picked move would not be in the legal set, so membership over a long
    run is a real check rather than a tautology.

    The shortlist length K is measured for all six personalities on a smaller
    sample, because K is a property of the personality's `restrict`/`rescore`
    path rather than of the position.
    """
    main = _run_pipeline_games("optimizer", n_games)
    per_key = {}
    for key in keys:
        r = _run_pipeline_games(key, k_games)
        per_key[key] = {
            "games": k_games,
            "shortlist_k": stats(r["k_values"]),
            "n_candidates": stats(r["n_cand_values"]),
            "mistake_choices": r["mistake_choices"],
            "real_moves": r["real_moves"],
            "picked_legal_share": (r["picked_in_legal_mask"] / r["picked_total"]
                                   if r["picked_total"] else None),
        }
    return {
        "personality": "optimizer",
        "games": n_games,
        "real_moves": main["real_moves"],
        "trace_records_with_no_move": main["empty_records"],
        "conversion": "engine move = (PIECE_IDX[picked[0]], picked[1], picked[2])",
        "picked_in_legal_mask": main["picked_in_legal_mask"],
        "picked_total": main["picked_total"],
        "picked_legal_share": main["picked_in_legal_mask"] / main["picked_total"],
        "misses": main["misses"],
        "shortlist_k": stats(main["k_values"]),
        "n_candidates": stats(main["n_cand_values"]),
        "mistake_choices": main["mistake_choices"],
        "lookahead_used": main["lookahead_used"],
        "shortlist_core": ai.formulas.SHORTLIST_CORE,
        "corner_reserve": ai.formulas.CORNER_RESERVE,
        "per_personality": per_key,
        "per_personality_games": k_games,
    }


F4B_KEYS = ("optimizer", "wolf", "chess", "fox")
F4B_SAMPLE_SEED = 12345


def _midgame_pool(n_games):
    pool = []
    for seed in range(n_games):
        tape = []
        random_playout(seed, tape=tape)
        for s, mask, n_real, mv in tape:
            if mask == 0 or not (16 <= n_real < 48):
                continue
            pool.append(s)
    return pool


def f4b_randomness(n_games=100, n_sample=500, sample_seed=F4B_SAMPLE_SEED,
                   keys=F4B_KEYS, control=False):
    """B: how much of a personality's choice is its own randomness.

    The same position, the same brains, the same everything - only the `rng`
    differs. The fraction that changes is the ceiling on how well a model can
    imitate that personality's top-1 move: a personality that answers 30% of
    the time differently cannot be imitated better than 70% even by a perfect
    imitator.

    `USE_WALL_BUDGET` is switched off for the measurement, and the `control`
    run is what proves it matters. With the wall budget on, `choose_move`
    drops the opponent lookahead once a decision has taken 0.9s, so the same
    position and the same seed can give different answers purely because the
    machine was busy. That is a measurement obstacle rather than a defect -
    but it means "run the pipeline twice and compare" is not a validity check
    unless the budget is off, and the control row is the evidence.
    """
    pool = _midgame_pool(n_games)
    if len(pool) < n_sample:
        raise RuntimeError("only %d midgame positions available" % len(pool))
    sample = random.Random(sample_seed).sample(pool, n_sample)

    # The pipeline draws from its rng exactly once per decision, and only when
    # the shortlist has at least two entries, so the *first* draw of a fresh
    # Random(seed) is the whole story. Two fixed seeds would therefore test two
    # fixed numbers against the mistake rate instead of sampling it: seed 1
    # always opens with 0.134364, so a personality whose mistake rate sits
    # below that could never be seen to make a mistake. The two seeds are drawn
    # per position from a fixed generator instead, which keeps the whole thing
    # reproducible without freezing the draw.
    seed_rng = random.Random(sample_seed ^ 0x5EED)
    seed_pairs = [(seed_rng.randrange(1 << 30), seed_rng.randrange(1 << 30))
                  for _ in sample]

    out = {"playout_games": n_games, "midgame_positions_available": len(pool),
           "sampled_positions": n_sample, "sample_seed": sample_seed,
           "rng_seeds": "two per position, drawn from Random(sample_seed ^ 0x5EED)",
           "first_draw_of_seed_1": random.Random(1).random(),
           "default_wall_budget": ai.formulas.WALL_BUDGET,
           "default_use_wall_budget": ai.formulas.USE_WALL_BUDGET,
           "personalities": {}}
    for key in keys:
        brains = {o: ai.make_brain(key, random.Random(9000 + o)) for o in range(4)}
        differ = same = 0
        ctl_differ = ctl_same = 0
        wall_differ = wall_same = 0
        none_moves = 0
        entered = 0
        wall_ms = []
        for s, (sa, sb) in zip(sample, seed_pairs):
            a, a_entered = _choose_on(s, brains, sa, wall_budget=False)
            b, b_entered = _choose_on(s, brains, sb, wall_budget=False)
            if a is None:
                none_moves += 1
            if a_entered or b_entered:
                entered += 1
            if a == b:
                same += 1
            else:
                differ += 1
            if control:
                c1, _e1 = _choose_on(s, brains, sa, wall_budget=False)
                c2, _e2 = _choose_on(s, brains, sa, wall_budget=False)
                if c1 == c2:
                    ctl_same += 1
                else:
                    ctl_differ += 1
            w1, _we1 = _choose_on(s, brains, sa, wall_budget=True)
            w2, _we2 = _choose_on(s, brains, sa, wall_budget=True)
            if w1 == w2:
                wall_same += 1
            else:
                wall_differ += 1
            if not key.startswith("optimizer"):
                # How long one decision takes, because that is what decides
                # whether the wall budget can fire at all.
                t0 = time.perf_counter()
                _choose_on(s, brains, sa, wall_budget=True)
                wall_ms.append((time.perf_counter() - t0) * 1000.0)
        row = {"rng_a_vs_rng_b": {"differ": differ, "same": same,
                                  "differ_share": differ / n_sample},
               "mistake_rate": brains[0].mistake_rate,
               "mistake_rates": [brains[o].mistake_rate for o in range(4)],
               "entered_weighted_draw": entered,
               "entered_share": entered / n_sample,
               "changed_given_entered": (differ / entered if entered else None),
               "uses_lookahead": brains[0].uses_lookahead,
               "no_move_found": none_moves}
        if control:
            row["control_same_seed_budget_off"] = {
                "differ": ctl_differ, "same": ctl_same,
                "differ_share": ctl_differ / n_sample}
        # Kept out of the results section on purpose: with the wall budget on,
        # this row is a function of machine load, so it is a measurement.
        row["_measurement_wall_budget_on_same_seed"] = {
            "differ": wall_differ, "same": wall_same,
            "differ_share": wall_differ / n_sample}
        if wall_ms:
            row["_measurement_decision_ms"] = stats(wall_ms)
        out["personalities"][key] = row
    return out


def _choose_on(state, brains, rng_seed, wall_budget=None):
    """Call the real pipeline on an engine position, with a fresh board each
    time so one call cannot leave anything behind for the next.

    The other seats' `must_cover` / `reach` are derived from the position rather
    than left empty: the opponent lookahead filters its candidate replies with
    them, and passing `{}` would hand the lookahead a rule-free opponent and
    quietly change what is being measured.
    """
    board = _board_from_state(state)
    owner = state.to_move
    other_must_cover = {o: (OWNER_CORNER[o] if state.own_bits[o] == 0 else None)
                        for o in range(4)}
    other_reach = {o: (board.reach(o) if state.own_bits[o] != 0 else None)
                   for o in range(4)}
    names = _hand_names(state, owner)
    saved = ai.formulas.USE_WALL_BUDGET
    if wall_budget is not None:
        ai.formulas.USE_WALL_BUDGET = wall_budget
    record = []
    try:
        mv = ai.choose_move(
            board, names, owner, brains[owner], random.Random(rng_seed),
            other_brains=brains,
            must_cover=other_must_cover[owner], reach=other_reach[owner],
            other_must_cover=other_must_cover, other_reach=other_reach,
            trace=record)
    finally:
        ai.formulas.USE_WALL_BUDGET = saved
    entered = bool(record and record[-1]["was_mistake"])
    return mv, entered


def f4c_scoring():
    """C: the scoring rules, quoted from the source rather than summarised."""
    def excerpt(path, first, last):
        out = []
        with open(os.path.join(HERE, path), encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if first <= i <= last:
                    out.append("%4d| %s" % (i, line.rstrip("\n")))
        return out

    return {
        "remaining_cells": excerpt("game.py", 188, 189),
        "standings": excerpt("game.py", 191, 200),
        "winner": excerpt("game.py", 202, 207),
        "average_ranks": excerpt("tools/benchmark.py", 48, 69),
        "engine_result": excerpt("engine.py", 194, 207),
        "engine_is_over": excerpt("engine.py", 210, 212),
        "make_profile_perturbation": excerpt("ai/base.py", 40, 55),
        "callers_of_standings": _grep("standings()", ("game.py", "ui.py",
                                                      "match.py", "tools/benchmark.py")),
        "callers_of_average_ranks": _grep("average_ranks", ("tools/benchmark.py",
                                                             "tests/test_benchmark.py")),
    }


def _grep(needle, files):
    hits = []
    for rel in files:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if needle in line:
                    hits.append("%s:%d" % (rel, i))
    return hits


# --------------------------------------------------------------------------
# F5: hardware and environment, read only
# --------------------------------------------------------------------------

def f5_environment():
    def read(path, key):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith(key):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return None

    mem_kb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    break
    except OSError:
        pass

    numpy_version = None
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None

    torch = {"installed": False}
    try:
        import torch
        torch = {"installed": True, "version": torch.__version__,
                 "cuda_available": bool(torch.cuda.is_available()),
                 "cuda_version": getattr(torch.version, "cuda", None),
                 "gpus": [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]}
    except ImportError as exc:
        torch["error"] = str(exc)

    # `nvidia-smi` with no arguments prints a clock, the current temperature and
    # the current utilisation, none of which are properties of the machine. Only
    # the fixed fields are asked for, so this section stays reproducible.
    nvidia = None
    try:
        import subprocess
        nvidia = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=driver_version,name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        nvidia = nvidia.stdout.strip() if nvidia.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        nvidia = None

    return {
        "cpu_model": read("/proc/cpuinfo", "model name"),
        "logical_cores": os.cpu_count(),
        "physical_cores": read("/proc/cpuinfo", "cpu cores"),
        "siblings": read("/proc/cpuinfo", "siblings"),
        "platform_processor": platform.processor(),
        "platform_machine": platform.machine(),
        "mem_total_gib": (mem_kb / 1024 / 1024) if mem_kb else None,
        "python_version": sys.version.split()[0],
        "python_build": platform.python_build()[0],
        "numpy_version": numpy_version,
        "torch": torch,
        "nvidia_smi": nvidia,
        "process_model": "single process, single thread, no parallelism in "
                         "bench_engine.py",
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%%.%df" % digits) % v
    return str(v)


def _row(name, st, digits=3):
    return ("%-16s %5d %10s %10s %10s %10s %10s"
            % (name, st["n"], _fmt(st["mean"], digits), _fmt(st["median"], digits),
               _fmt(st["p95"], digits), _fmt(st["max"], digits),
               _fmt(st["min"], digits)))


STATS_HEAD = ("%-16s %5s %10s %10s %10s %10s %10s"
              % ("function/段", "n", "mean", "median", "p95", "max", "min"))


def render(raw, no_timing):
    L = []
    add = L.append
    add("# 階段 F′ 原始輸出 (bench_engine.py)")
    add("")
    add("results = 可逐位重現；measurement = 量測非結果。")
    add("")

    env = raw["environment"]
    add("## 環境")
    add("")
    add("| 項目 | 值 |")
    add("| --- | --- |")
    for k in ("cpu_model", "platform_processor", "platform_machine",
              "logical_cores", "physical_cores", "siblings", "mem_total_gib",
              "python_version", "numpy_version", "process_model"):
        add("| %s | %s |" % (k, env[k]))
    add("| torch | %s |" % ("未安裝" if not env["torch"]["installed"]
                            else env["torch"]))
    add("| nvidia-smi | %s |" % (env["nvidia_smi"] or "無 nvidia-smi"))
    add("")

    d = raw["f1b"]
    add("## 結果區：合法步數分佈 (F1b)")
    add("")
    add("種子 0..%d 共 %d 局；決策點 %d（其中需 pass 者 %d，佔 %.6f）"
        % (d["games"] - 1, d["games"], d["decision_points"],
           d["pass_decision_points"], d["pass_share"]))
    add("")
    add(STATS_HEAD)
    for key, label in (("all", "全體"), ("opening", "開局"),
                       ("middle", "中盤"), ("end", "殘局")):
        add(_row(label, d["legal_counts"][key], 2))
    add("")
    add("每 8 手一段（平均 / 最大）：")
    add("")
    add("| 段 | n | 平均 | 最大 |")
    add("| --- | --- | --- | --- |")
    for k, v in d["legal_counts_by_move_bucket"].items():
        add("| %s | %d | %s | %s |" % (k, v["n"], _fmt(v["mean"], 2), v["max"]))
    add("")
    add("可落子之棋子種類數：%s"
        % " ".join("%s=%s" % (k, _fmt(v, 2)) for k, v in
                   d["piece_kinds_available"].items()))
    add("")

    a = raw["f2"]
    add("## 結果區：動作表一致性 (F2)")
    add("")
    add("| 檢查 | 實測 | 預期 | 結論 |")
    add("| --- | --- | --- | --- |")
    add("| sum(len(ORIENTS[p])) | %d | 91 | %s |"
        % (a["orientation_checks"]["engine_orientations"],
           "PASS" if a["orientation_checks"]["pass"] else "FAIL"))
    add("| sum(valid.bit_count()) | %d | 30433 | %s |"
        % (a["valid_base_checks"]["engine_valid_bases"],
           "PASS" if a["valid_base_checks"]["pass"] else "FAIL"))
    add("| action_table.total_placements | %s | 30433 | %s |"
        % (a["action_table_placements"]["declared_total_placements"],
           "PASS" if a["action_table_placements"]["pass"] else "FAIL"))
    add("")
    add("| owner | 開局合法步 (engine) | 暴力法蓋到角格 | 一致 |")
    add("| --- | --- | --- | --- |")
    for owner in ("0", "1", "2", "3"):
        n = a["opening_legal_moves"][owner]
        c = a["opening_brute_force_corner_hits"][owner]
        add("| %s | %d | %d | %s |" % (owner, n, c, "是" if n == c else "**否**"))
    add("")
    add("四個數字相等：%s（四角對稱）"
        % ("是" if a["opening_all_equal"] else "**否（旋轉對稱反例）**"))
    add("")
    add("config：B=%s N=%s CLOCKWISE_OWNERS=%s OWNER_CORNER=%s"
        % (a["config"]["B"], a["config"]["N"], a["config"]["CLOCKWISE_OWNERS"],
           a["config"]["OWNER_CORNER"]))
    add("action_table hash：%s" % a["action_table_hash"])
    add("")

    f3 = raw["f3"]
    add("## 結果區：_advance 延遲鎖存 (F3)")
    add("")
    add("| 情況 | 次數 | 佔全部決策點 |")
    add("| --- | --- | --- |")
    add("| X: mask==0 且 stuck[to_move]==False | %d | %.6f |"
        % (f3["situation_x"]["count"],
           f3["situation_x"]["share_of_all_decision_points"]))
    add("| Y: stuck[o] 為 True 但 has_any_legal 為 True | %d | %.6f |"
        % (f3["situation_y"]["count"],
           f3["situation_y"]["count"] / f3["decision_points_including_passes"]))
    add("")
    add("- 至少出現一次 X 的局數：%d / %d" % (f3["situation_x"]["games_with_at_least_one"],
                                              f3["games"]))
    add("- 最長連續 pass：%d" % f3["situation_z"]["longest_pass_run"])
    add("- 全部以 is_over 結束：%s" % ("是" if f3["termination"]["all_games_over"] else "否"))
    add("- 終局四人皆無合法步：%s"
        % ("是" if f3["termination"]["all_final_positions_immobile"] else "否"))
    add("- engine.result 與 Game 重播餘格不一致：%d / %d 局"
        % (f3["result_vs_game_replay"]["mismatches"], f3["result_vs_game_replay"]["games_compared"]))
    ex = f3["situation_x"]["example"]
    if ex:
        add("")
        add("X 的抽樣例子（抽樣種子 %d）：seed=%d 第 %d 手，to_move=%d"
            % (12345, ex["seed"], ex["move_index"], ex["to_move"]))
        add("")
        add("```")
        add(ex["serialize"])
        add("```")
    add("")

    fa = raw["f4a"]
    add("## 結果區：trace 欄位與 picked 轉換 (F4A)")
    add("")
    add("- 轉換規則：`%s`" % fa["conversion"])
    add("- %d 局、%d 次實手，picked 落在 legal_move_mask 內：%d / %d = %.6f"
        % (fa["games"], fa["picked_total"], fa["picked_in_legal_mask"],
           fa["picked_total"], fa["picked_legal_share"]))
    add("- 短名單 K：n=%d 平均=%s 中位數=%s p95=%s 最大=%s 最小=%s"
        % (fa["shortlist_k"]["n"], _fmt(fa["shortlist_k"]["mean"], 2),
           _fmt(fa["shortlist_k"]["median"], 2), _fmt(fa["shortlist_k"]["p95"], 2),
           fa["shortlist_k"]["max"], fa["shortlist_k"]["min"]))
    add("- 候選總數 n_candidates：n=%d 平均=%s 中位數=%s p95=%s 最大=%s"
        % (fa["n_candidates"]["n"], _fmt(fa["n_candidates"]["mean"], 2),
           _fmt(fa["n_candidates"]["median"], 2), _fmt(fa["n_candidates"]["p95"], 2),
           fa["n_candidates"]["max"]))
    add("- K 的上界 = SHORTLIST_CORE %d + CORNER_RESERVE %d = %d"
        % (fa["shortlist_core"], fa["corner_reserve"],
           fa["shortlist_core"] + fa["corner_reserve"]))
    add("- 走過加權抽籤（was_mistake）的手數：%d / %d；啟用對手預判的手數：%d"
        % (fa["mistake_choices"], fa["real_moves"], fa["lookahead_used"]))
    add("")

    fb = raw["f4b"]
    add("## 結果區：人格選擇的隨機性 (F4B)")
    add("")
    add("母體：種子 0..%d 的隨機對局，其中盤（16 ≤ n < 48）決策點 %d 個，"
        "以種子 %d 抽樣 %d 個。量測時 `USE_WALL_BUDGET` 關閉，"
        "只讓 rng 成為唯一變因。"
        % (fb["playout_games"] - 1, fb["midgame_positions_available"],
           fb["sample_seed"], fb["sampled_positions"]))
    add("")
    add("隨機性只來自 `choose_move` 每手一次的 `rng.random() < mistake_rate` 判斷，"
        "之後由短名單做加權抽籤。「進入加權抽籤」是兩次呼叫中至少一次"
        "進入該分支的局面數，所以約為 mistake_rate 的兩倍；"
        "「抽籤後改變率」是抽中後真的換了一步的比例，"
        "剩下的機會是抽籤又抽回第一名。「兩次不同」除以兩次呼叫，"
        "就約等於 `mistake_rate × 抽籤後改變率`。")
    add("")
    add("| personality | mistake_rate (4 座) | uses_lookahead | 進入加權抽籤 | 兩次 RNG 選擇不同 | 佔比 | 抽籤後改變率 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for key, row in fb["personalities"].items():
        r = row["rng_a_vs_rng_b"]
        rates = "/".join("%.4f" % v for v in row["mistake_rates"])
        add("| %s | %s | %s | %d / %d = %.4f | %d / %d | **%.4f** | %s |"
            % (key, rates, row["uses_lookahead"], row["entered_weighted_draw"],
               fb["sampled_positions"], row["entered_share"], r["differ"],
               r["differ"] + r["same"], r["differ_share"],
               _fmt(row["changed_given_entered"], 4)))
    add("")
    add("對照組：同一個 RNG 種子呼叫兩次、wall budget 關閉。"
        "差異不可能來自抽籤，所以剩下的差異只能是機器負載。")
    add("")
    add("| personality | 同種子×2 不同 | 佔比 | 無棋可下 |")
    add("| --- | --- | --- | --- |")
    for key, row in fb["personalities"].items():
        c = row.get("control_same_seed_budget_off")
        ctxt = ("%d / %d = %.4f" % (c["differ"], c["differ"] + c["same"],
                                    c["differ_share"])) if c else "（未量）"
        add("| %s | %s | %d |" % (key, ctxt, row["no_move_found"]))
    add("")

    add("### 短名單長度 K 與候選數（各人格）")
    add("")
    add("| personality | K n | K 平均 | K 中位數 | K p95 | K 最大 | K 最小 | 候選平均 | 候選最大 | 犯錯抽籤 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, row in fa["per_personality"].items():
        k = row["shortlist_k"]
        c = row["n_candidates"]
        add("| %s | %d | %s | %s | %s | %d | %d | %s | %d | %d |"
            % (key, k["n"], _fmt(k["mean"], 2), _fmt(k["median"], 2),
               _fmt(k["p95"], 2), k["max"], k["min"], _fmt(c["mean"], 1),
               c["max"], row["mistake_choices"]))
    add("")

    fc = raw["f4c"]
    add("## 結果區：終局評分規則 (F4C)")
    add("")
    for title in ("remaining_cells", "standings", "winner", "average_ranks",
                  "engine_result", "engine_is_over", "make_profile_perturbation"):
        add("`%s`" % title)
        add("")
        add("```python")
        for line in fc[title]:
            add(line)
        add("```")
        add("")
    add("`standings()` 呼叫點：%s" % ", ".join(fc["callers_of_standings"]))
    add("")
    add("`average_ranks` 呼叫點：%s" % ", ".join(fc["callers_of_average_ranks"]))
    add("")

    if not no_timing:
        t = raw["timing"]
        add("## 量測區：整局耗時 (F1A) — 量測非結果")
        add("")
        g = t["game_timing"]
        add("| 項目 | n | 平均 | 中位數 | p95 | 最大 | 最小 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for label, key, digits in (("每局秒數", "seconds_per_game", 4),
                                   ("真實手數", "real_moves", 2),
                                   ("pass 次數", "passes", 2)):
            st = g[key]
            add("| %s | %d | %s | %s | %s | %s | %s |"
                % (label, st["n"], _fmt(st["mean"], digits),
                   _fmt(st["median"], digits), _fmt(st["p95"], digits),
                   st["max"], st["min"]))
        add("")
        add("- %d 局合計 %.4f 秒、%d 手 → **%.1f 真實手/秒**"
            % (g["games"], g["total_seconds"], g["total_real_moves"],
               g["real_moves_per_second"]))
        add("- 終局餘格平均（owner 0..3）：%s"
            % " ".join("%.2f" % v for v in g["mean_result_cells"]))
        add("")
        f = t["function_timing"]
        add("## 量測區：單函數耗時 (F1B) — 量測非結果")
        add("")
        add("量測局面：種子 0..%d 共 %d 局收集 %d 個決策點，"
            "前 %d 個預熱不計，實際計時 %d 個。"
            % (f["games"] - 1, f["games"], f["positions_collected"],
               f["warmup_skipped"], f["positions_timed"]))
        add("")
        add("| 函數 | 段 | n | 平均 ms | 中位數 ms | p95 ms | 最大 ms | 最小 ms |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for fn in F1B_FUNCS:
            row = f["functions"][fn]
            cells = []
            for st in list(STAGES) + ["all"]:
                s = row[st]
                cells.append("| %s | %s | %d | %s | %s | %s | %s | %s |"
                             % (fn, STAGE_LABELS.get(st, "全部"), s["n"],
                                _fmt(s["mean"], 4), _fmt(s["median"], 4),
                                _fmt(s["p95"], 4), _fmt(s["max"], 4),
                                _fmt(s["min"], 4)))
            L.extend(cells)
        add("")
        add("- to_plane 平均耗時中 normalize 佔 **%.4f**（normalize 平均 %.4f ms / to_plane 平均 %.4f ms）"
            % (f["to_plane_normalize_share"], stats_of(raw, "normalize"),
               stats_of(raw, "to_plane")))
        add("- normalize 平均 ms（開局/中盤/殘局）：%s / %s / %s"
            % tuple(_fmt(f["normalize_means_by_stage"][st], 4) for st in STAGES))
        add("- to_plane 平均 ms（開局/中盤/殘局）：%s / %s / %s"
            % tuple(_fmt(f["to_plane_means_by_stage"][st], 4) for st in STAGES))
        add("")
        add("### 量測區：wall budget 對可重現性的影響")
        add("")
        add("同一個 RNG 種子呼叫兩次，但保留 `USE_WALL_BUDGET = True`"
            "（預設值，`WALL_BUDGET = %s` 秒）。決策一旦超過這個時間，"
            "對手預判整段被跳過，答案就會變 —— 也就是說，"
            "`choose_move` 本身不是純函式，機器負載可以改變它的輸出。"
            % fb.get("default_wall_budget"))
        add("")
        add("| personality | 同種子×2 不同 | 佔比 | 單次決策耗時 ms（平均/中位/p95/最大） |")
        add("| --- | --- | --- | --- |")
        for key, row in fb["personalities"].items():
            w = row["_measurement_wall_budget_on_same_seed"]
            d = row.get("_measurement_decision_ms")
            dtxt = ("%s / %s / %s / %s"
                    % (_fmt(d["mean"], 1), _fmt(d["median"], 1),
                       _fmt(d["p95"], 1), _fmt(d["max"], 1))) if d else "n/a（規則式人格不用預判）"
            add("| %s | %d / %d | %.4f | %s |"
                % (key, w["differ"], w["differ"] + w["same"], w["differ_share"],
                   dtxt))
        add("")
        add("中盤局面的單次決策遠低於 %.1f 秒上限，所以這個開關在本次量測中"
            "從未觸發。開局與殘局會更慢，而滿載時更容易越線；"
            "這是量測事實，不是結論。" % fb["default_wall_budget"])
        add("")
    else:
        add("## 量測區")
        add("")
        add("（`--no-timing`：整局耗時、單函數耗時與 wall budget 量測已省略，"
            "此區留白。輸出因此可逐位元組重現。）")
        add("")
    return "\n".join(L) + "\n"


def stats_of(raw, fn):
    return raw["timing"]["function_timing"]["functions"][fn]["all"]["mean"]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="stage F-prime measurement")
    ap.add_argument("--no-timing", action="store_true",
                    help="omit every wall-clock number so the report is "
                         "byte-reproducible")
    ap.add_argument("--sections", nargs="+", default=["all"],
                    help="subset of f1 f1b f2 f3 f4a f4b f4c f5 env")
    ap.add_argument("--games", type=int, default=200,
                    help="games for F1A, F1b and F3 (seeds 0..N-1)")
    ap.add_argument("--func-games", type=int, default=100,
                    help="games for the per-function timings (F1B)")
    ap.add_argument("--f4a-games", type=int, default=200,
                    help="optimizer games for the trace/picked check (F4A)")
    ap.add_argument("--f4b-games", type=int, default=100,
                    help="playouts sampled for the personality randomness (F4B)")
    ap.add_argument("--f4b-samples", type=int, default=500)
    ap.add_argument("--f4b-control", action="store_true",
                    help="also call choose_move twice with the same seed, to "
                         "separate the draw from everything else")
    ap.add_argument("--json", default=os.path.join(REPORT_DIR, "f_prime_raw.json"))
    ap.add_argument("--report", default=None,
                    help="also write the rendered report to this path "
                         "(the committed report is written by hand)")
    args = ap.parse_args()

    want = set(args.sections)
    all_sections = want == {"all"}
    def on(name):
        return all_sections or name in want

    raw = {}
    if on("env") or on("f5"):
        raw["environment"] = f5_environment()
    if on("f1b"):
        raw["f1b"] = f1b_distribution(args.games)
    if on("f2"):
        raw["f2"] = f2_action_table()
    if on("f3"):
        raw["f3"] = f3_advance_latch(args.games)
    if on("f4a"):
        raw["f4a"] = f4a_trace_fields(args.f4a_games)
    if on("f4b"):
        raw["f4b"] = f4b_randomness(args.f4b_games, args.f4b_samples,
                                     control=args.f4b_control)
    if on("f4c"):
        raw["f4c"] = f4c_scoring()
    if on("f1") and not args.no_timing:
        raw["timing"] = {
            "game_timing": f1_game_timing(args.games),
            "function_timing": f1_function_timing(args.func_games),
        }

    if "environment" not in raw or "f1b" not in raw or "f2" not in raw \
            or "f3" not in raw or "f4a" not in raw or "f4b" not in raw \
            or "f4c" not in raw:
        print(json.dumps(raw, indent=1, sort_keys=True, default=str))
        return 0

    if args.no_timing:
        # Wall-clock numbers are kept out of the machine-readable copy as well
        # as out of the printed report: a JSON file whose bytes depend on how
        # busy the machine was cannot be diffed between two runs, and the whole
        # point of the flag is that it can be.
        _strip_measurements(raw)
    report = render(raw, args.no_timing)
    if args.report:
        # Off by default: `reports/f_prime_report.md` is written by hand, with
        # the reading of the numbers in it, and a re-run of this script must not
        # silently replace that with a table dump. The rendered form is still
        # what goes to stdout, so the numbers are always checkable.
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(report)
    sys.stdout.write(report)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote %s" % args.json, file=sys.stderr)
    return 0


def _strip_measurements(node):
    if isinstance(node, dict):
        for key in [k for k in node if k.startswith("_measurement")]:
            del node[key]
        for value in node.values():
            _strip_measurements(value)
    elif isinstance(node, list):
        for value in node:
            _strip_measurements(value)


if __name__ == "__main__":
    sys.exit(main())
