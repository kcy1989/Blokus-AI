"""This layer answers one question: what does a dataset of teacher choices look
like, and how is it written down?

The teacher is the optimizer personality. It is the right choice for one reason
above the others: stage F' measured its two calls with the same seed differing
0 times out of 500, so its top-1 move is a function of the position alone. A
learner trained on a deterministic label has something it can actually reach;
one trained on a personality that answers differently a quarter of the time is
being asked to guess.

That determinism is also the problem. Four optimizers playing each other
produce the same game every time, and a dataset of one game is a dataset of one
line through the tree. So every game here gets two sources of variety, both
drawn from the game seed:

    a random opening prefix, up to 12 real moves, played uniformly at random,
    which also contributes labels - the position is real even though the move
    that reached it was not chosen;
    and one controller personality per seat, drawn so that at least one seat is
    always the teacher. The others play the game out, so the prefix keeps
    extending itself.

Every sample records the teacher's move, the shortlist it came from and the
outcome of the game it was played in. Nothing here trains anything; the shards
are the raw material and `reports/g_report.md` section G4 says what the trial
run actually contains.

Two properties are enforced rather than assumed. The wall-clock budget in
`choose_move` is switched off for the whole run and restored in a `finally`,
because with it on the same position and the same seed can give different
answers and the dataset would not be reproducible. And `Game` is advanced
alongside the engine state with the stuck flags compared at every step, so the
board handed to the AI and the position the sample describes cannot drift
apart.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
import ai.formulas  # noqa: E402
import engine  # noqa: E402
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER, N  # noqa: E402
from game import Game  # noqa: E402
from rl.actions import index_to_move, legal_indices, move_to_index  # noqa: E402
from rl.env import utilities_from_remaining  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")

# Seed ranges. Train and validation are disjoint from each other, and both are
# disjoint from everything already used to produce a published number, so a
# dataset can be regenerated and compared without reusing a position a result
# was measured on.
TRAIN_SEED_BASE = 1_000_000
VALID_SEED_BASE = 2_000_000

# `bench_engine.py` used 0..199 for F1/F1b/F3, 0..99 for the per-function
# timings, 0..199 for the optimizer trace check and 0..99 for the randomness
# pool. `tools/benchmark.py`'s fixed-seat run draws from 20240101 upwards, one
# per game. Both are asserted against at run time rather than trusted.
BENCH_SEED_MAX = 199
BENCHMARK_SEED_BASE = 20240101
BENCHMARK_SEED_STRIDE = 1009
BENCHMARK_MAX_GAMES = 96          # 4 games x 24 permutations, its defaults

TEACHER = "optimizer"

# Opening prefix lengths. Even numbers only: a prefix of 6 real moves leaves the
# board half open, and an odd length would put the same number of players ahead
# as an even one while changing which of them moves next.
PREFIX_CHOICES = (0, 2, 4, 6, 8, 10, 12)

# Seat controllers. The teacher is half of all seats, on purpose: it is the only
# label this dataset carries, so a game in which the teacher sits out
# contributes nothing. The rest spread over the other five so the positions the
# teacher is asked about are reached by opponents that behave differently.
CONTROLLER_WEIGHTS = (
    (TEACHER, 0.50),
    ("wolf", 0.125),
    ("chess", 0.125),
    ("fox", 0.125),
    ("intruder", 0.0625),
    ("builder", 0.0625),
)
CONTROLLER_KEYS = tuple(k for k, _w in CONTROLLER_WEIGHTS)
CONTROLLER_TOTAL = sum(w for _k, w in CONTROLLER_WEIGHTS)

SHORTLIST_PAD = 40
MAX_PASSES = 200

# Samples per shard. A shard is a unit of loading and of reproducibility, so it
# is kept small enough to hold in memory and to diff. The split is by games
# rather than by rows, because a shard has to describe whole games for the
# `game_id` grouping to mean anything.
MAX_SAMPLES_PER_SHARD = 20_000


# --------------------------------------------------------------------------
# the game setup
# --------------------------------------------------------------------------

def draw_setup(seed):
    """The prefix length and the four seat controllers for one game.

    A single `random.Random(seed)` per game, and the whole setup is redrawn if
    it came out with no teacher in it. Redrawing the *whole* setup rather than
    only the seats keeps the draw simple: the alternative - keeping the prefix
    and re-rolling seats until one is the teacher - would bias the prefix
    towards the games that needed fewer attempts.
    """
    rng = random.Random(seed)
    for _attempt in range(1000):
        prefix = rng.choice(PREFIX_CHOICES)
        controllers = []
        for _seat in range(4):
            r = rng.random() * CONTROLLER_TOTAL
            acc = 0.0
            for key, weight in CONTROLLER_WEIGHTS:
                acc += weight
                if r < acc:
                    controllers.append(key)
                    break
            else:                                   # pragma: no cover
                controllers.append(CONTROLLER_KEYS[-1])
        if TEACHER in controllers:
            return prefix, tuple(controllers)
    raise RuntimeError("no setup with a teacher in it after 1000 draws")


# --------------------------------------------------------------------------
# the engine and the board, kept in step
# --------------------------------------------------------------------------

def board_from_state(state):
    """A `Board` showing the same position as an engine `State`.

    `choose_move` takes a Board, not a State. The fields are filled in directly
    and `_update_regions` is called once, because placing sixty-odd pieces one at
    a time would recompute the corner regions on every call for nothing.
    """
    from board import Board
    b = Board()
    b.grid = [-1] * N
    for o in range(4):
        bits = state.own_bits[o]
        while bits:
            low = bits & -bits
            bits ^= low
            b.grid[low.bit_length() - 1] = o
    b.owner_bits = list(state.own_bits)
    b.empty_bits = engine.ALL & ~(state.own_bits[0] | state.own_bits[1]
                                  | state.own_bits[2] | state.own_bits[3])
    b._update_regions()
    return b


def hand_names(state, owner):
    bits = state.hand_bits[owner]
    return [engine.PIECE_ORDER[i] for i in range(engine.N_PIECES)
            if bits >> i & 1]


class _Budget:
    """Turn the wall-clock budget off for the duration of a block.

    `choose_move` reads `time.perf_counter()` when `USE_WALL_BUDGET` is set, and
    drops the opponent lookahead once a decision has taken 0.9 seconds. That
    makes it a function of machine load as well as of its inputs, so a dataset
    collected with it on would differ between two runs on the same code. The
    attribute is module-global inside `ai.formulas`; there is no other handle on
    it, and it is restored on the way out even if something raises.
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.saved = None

    def __enter__(self):
        self.saved = ai.formulas.USE_WALL_BUDGET
        ai.formulas.USE_WALL_BUDGET = self.enabled
        return self

    def __exit__(self, *exc):
        ai.formulas.USE_WALL_BUDGET = self.saved
        return False


def teacher_move(state, brain):
    """The teacher's choice on this position, as an engine move and its trace.

    Returns `(index_to_move(move), record)` or `(None, record)` if the teacher
    somehow finds nothing where the engine found moves. The record is the trace
    dict, which carries the shortlist and the candidate count.
    """
    owner = state.to_move
    board = board_from_state(state)
    other_must_cover = {o: (OWNER_CORNER[o] if state.own_bits[o] == 0 else None)
                        for o in range(4)}
    other_reach = {o: (board.reach(o) if state.own_bits[o] != 0 else None)
                   for o in range(4)}
    trace = []
    mv = ai.choose_move(board, hand_names(state, owner), owner, brain,
                        random.Random(0), other_brains={o: brain for o in range(4)},
                        must_cover=other_must_cover[owner],
                        reach=other_reach[owner],
                        other_must_cover=other_must_cover,
                        other_reach=other_reach,
                        trace=trace)
    record = trace[-1]
    if mv is None:
        return None, record
    name, oi, base = record["picked"]
    return (engine.PIECE_IDX[name], oi, base), record


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------

def collect_game(seed, prefix_len, controllers):
    """Play one game, returning its samples and its own summary.

    Samples come back sorted by `(game_id, ply)` and the game is a pure
    function of its seed, so two workers given the same seed produce the same
    rows in the same order.
    """
    rng = random.Random(seed)
    s = engine.initial_state(CLOCKWISE_OWNERS)

    # The teacher of each seat, and the brain each seat actually plays with.
    # Both come from the same seed and never from a shared generator, so two
    # workers cannot interfere with each other.
    teacher_brains = {o: ai.make_brain(TEACHER, random.Random(seed * 8 + o + 1))
                      for o in range(4)}
    play_brains = {o: ai.make_brain(controllers[o], random.Random(seed * 8 + o + 101))
                   for o in range(4)}
    play_rngs = {o: random.Random(seed * 8 + o + 202) for o in range(4)}

    # `Game` is advanced in lockstep so the stuck flags can be compared. It is
    # the human-facing implementation; if the two ever disagree the dataset
    # would be describing a game that the engine does not think is happening.
    g = Game(random.Random(seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    g.brains = {o: play_brains[o] for o in range(4)}
    g.start()

    rows = []
    ply = 0
    passes = 0
    guard = 0
    with _Budget(False):
        while not engine.is_over(s):
            guard += 1
            if guard > 600:
                raise RuntimeError("seed %d did not finish" % seed)
            owner = s.to_move
            if owner != g.current_owner():
                raise RuntimeError("seed %d: engine and Game disagree on whose "
                                   "turn it is at ply %d" % (seed, ply))
            mask = engine.legal_move_mask(s)
            if mask == 0:
                # A pass is not a decision, so it produces no sample.
                s = engine.pass_turn(s)
                g.act_pass()
                passes += 1
                if passes > MAX_PASSES:
                    raise RuntimeError("seed %d passed %d times" % (seed, passes))
                continue
            if tuple(s.stuck) != tuple(g.stuck):
                raise RuntimeError("seed %d: stuck flags diverged at ply %d: %r "
                                   "vs %r" % (seed, ply, s.stuck, g.stuck))

            legal = legal_indices(s)
            n_legal = int(legal.size)
            in_prefix = ply < prefix_len

            if controllers[owner] == TEACHER:
                chosen, record = teacher_move(s, teacher_brains[owner])
                if chosen is None:
                    raise RuntimeError("seed %d: the teacher found no move where "
                                       "the engine found %d" % (seed, n_legal))
                index = move_to_index(chosen)
                if index not in legal:
                    raise RuntimeError("seed %d: the teacher's move %r is not "
                                       "legal at ply %d"
                                       % (seed, chosen, ply))
                rows.append(_row(s, index, record, n_legal, seed, ply, in_prefix))
            else:
                index = None

            if in_prefix:
                index = int(legal[rng.randrange(n_legal)])
            else:
                if controllers[owner] != TEACHER:
                    mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                        play_brains[owner], play_rngs[owner],
                                        other_brains=g.brains,
                                        must_cover=g.must_cover(owner),
                                        reach=g.reach(owner),
                                        other_must_cover={x: g.must_cover(x)
                                                          for x in range(4)},
                                        other_reach={x: g.reach(x)
                                                     for x in range(4)})
                    if mv is None:
                        raise RuntimeError("seed %d: seat %d found no move where "
                                           "the engine found %d"
                                           % (seed, owner, n_legal))
                    index = move_to_index((engine.PIECE_IDX[mv[0]], mv[1],
                                           mv[2] + mv[3] * B))
                if index not in legal:
                    raise RuntimeError("seed %d: seat %d produced an illegal "
                                       "move at ply %d" % (seed, owner, ply))
            move = index_to_move(index)
            name = engine.PIECE_ORDER[move[0]]
            s = engine.apply_move(s, move)
            g.act(name, move[1], move[2] % B, move[2] // B)
            ply += 1
    if tuple(s.stuck) != tuple(g.stuck):
        raise RuntimeError("seed %d: stuck flags diverged at the end" % seed)

    remaining = engine.result(s)
    util = utilities_from_remaining(remaining)
    for row in rows:
        row["outcome_util"] = util
        row["outcome_remaining"] = [r / 89.0 for r in remaining]
    rows.sort(key=lambda r: r["ply"])
    return rows, {"game_id": seed, "prefix_len": prefix_len,
                  "controllers": list(controllers), "ply": ply,
                  "passes": passes, "remaining": list(remaining),
                  "utilities": util}


def _row(state, index, record, n_legal, seed, ply, in_prefix):
    shortlist = record["shortlist"][:SHORTLIST_PAD]
    idx = np.full(SHORTLIST_PAD, -1, dtype=np.int32)
    score = np.full(SHORTLIST_PAD, np.nan, dtype=np.float32)
    for i, (sc, name, oi, base) in enumerate(shortlist):
        idx[i] = move_to_index((engine.PIECE_IDX[name], oi, base))
        score[i] = sc
    return {
        # 400 bits per player, so 50 bytes each. `frombuffer` rather than
        # `array`: the bytes are already the right dtype and copying them
        # through a list of byte strings first would make numpy try to parse
        # each one as a number.
        "own_bits": np.frombuffer(
            b"".join(m.to_bytes(50, "little") for m in state.own_bits),
            dtype=np.uint8).reshape(4, 50),
        "hand_bits": np.array(state.hand_bits, dtype=np.uint32),
        "stuck": np.array(state.stuck, dtype=bool),
        "to_move": np.int8(state.to_move),
        "action": np.int32(index),
        "n_legal": np.int16(n_legal),
        "n_candidates": np.int16(min(record["n_candidates"], 32767)),
        "shortlist_idx": idx,
        "shortlist_score": score,
        "ply": np.int16(ply),
        "game_id": np.int32(seed),
        "in_prefix": np.bool_(in_prefix),
        # Filled in once the game ends.
        "outcome_util": None,
        "outcome_remaining": None,
    }


# --------------------------------------------------------------------------
# shard writing
# --------------------------------------------------------------------------

FIELD_SPEC = {
    "own_bits": ("uint8", (4, 50)),
    "hand_bits": ("uint32", (4,)),
    "stuck": ("bool", (4,)),
    "to_move": ("int8", ()),
    "action": ("int32", ()),
    "n_legal": ("int16", ()),
    "n_candidates": ("int16", ()),
    "shortlist_idx": ("int32", (SHORTLIST_PAD,)),
    "shortlist_score": ("float32", (SHORTLIST_PAD,)),
    "ply": ("int16", ()),
    "game_id": ("int32", ()),
    "in_prefix": ("bool", ()),
    "outcome_util": ("float32", (4,)),
    "outcome_remaining": ("float32", (4,)),
}
FIELD_ORDER = tuple(FIELD_SPEC)


def stack(rows, index):
    """Turn a list of row dicts into one column array.

    Cast to the declared dtype rather than letting numpy promote: the utilities
    and the remaining-cell fractions are Python floats, and `np.stack` on a list
    of lists would quietly hand back `float64` and double the size of the two
    payoff columns in every shard.
    """
    name = FIELD_ORDER[index]
    dtype = np.dtype(FIELD_SPEC[name][0])
    if not rows:
        return np.empty(0, dtype=dtype)
    return np.stack([np.asarray(r[name], dtype=dtype) for r in rows])


def write_shard(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {name: stack(rows, i) for i, name in enumerate(FIELD_ORDER)}
    np.savez_compressed(path, **arrays)
    return {name: {"dtype": str(arr.dtype), "shape": list(arr.shape)}
            for name, arr in arrays.items()}


def load_shard(path):
    """Read a shard back as a dict of arrays."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def shard_name(prefix, lo, hi):
    return "%s_%d_%d.npz" % (prefix, lo, hi)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def check_seed_ranges(trains, valids):
    """Refuse to generate into a range another published number came from.

    Cheap, and the failure it prevents is not cheap to detect later: reusing a
    seed means the same game, and a validation set drawn from the same games as
    the training set reports a number that is not about generalisation.
    """
    problems = []
    if set(trains) & set(valids):
        problems.append("training and validation seeds overlap: %r"
                        % sorted(set(trains) & set(valids))[:5])
    for label, seeds in (("train", trains), ("validation", valids)):
        if not seeds:
            continue
        if min(seeds) <= BENCH_SEED_MAX:
            problems.append("%s seeds start at %d, inside bench_engine.py's "
                            "0..%d" % (label, min(seeds), BENCH_SEED_MAX))
        overlap = [s for s in seeds
                   if BENCHMARK_SEED_BASE
                   <= s <= BENCHMARK_SEED_BASE
                   + BENCHMARK_SEED_STRIDE * BENCHMARK_MAX_GAMES]
        if overlap:
            problems.append("%s seeds overlap tools/benchmark.py's range: %r"
                            % (label, overlap[:5]))
    if problems:
        raise ValueError("; ".join(problems))


def code_hash():
    """The commit the data was generated from, or a marker for a dirty tree."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=HERE,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):     # pragma: no cover
        return {"commit": None, "dirty": None, "note": "git was not available"}
    return {"commit": head or None, "dirty": bool(dirty),
            "note": ("the tree had uncommitted changes, so the commit alone does "
                     "not identify this data" if dirty else None)}


def action_table_hash():
    with open(os.path.join(HERE, "action_table.json"), encoding="utf-8") as fh:
        return json.load(fh)["hash"]


def collect_range(seeds, prefix="imit"):
    """Collect one contiguous seed range into one shard."""
    seeds = list(seeds)
    rows = []
    games = []
    t0 = time.perf_counter()
    for seed in seeds:
        prefix_len, controllers = draw_setup(seed)
        game_rows, summary = collect_game(seed, prefix_len, controllers)
        rows.extend(game_rows)
        games.append(summary)
    return rows, games, time.perf_counter() - t0


def split_seeds(lo, hi):
    return list(range(lo, hi + 1))


def _worker(args):
    """Collect one seed range, write it, and report what went in.

    Games are grouped back together before writing, so a shard never splits a
    game across two files even when the sample cap falls in the middle of one.
    """
    lo, hi, prefix, out_dir = args
    seeds = split_seeds(lo, hi)
    rows, games, seconds = collect_range(seeds, prefix)

    by_game = {}
    for row in rows:
        by_game.setdefault(int(row["game_id"]), []).append(row)
    written = []
    pending = []
    pending_lo = None
    pending_hi = None

    def flush():
        if not pending:
            return
        path = os.path.join(out_dir, shard_name(prefix, pending_lo, pending_hi))
        fields = write_shard(path, pending)
        written.append({"path": path, "seed_lo": pending_lo,
                        "seed_hi": pending_hi, "prefix": prefix,
                        "samples": len(pending), "fields": fields})

    for game in games:
        game_rows = by_game.get(game["game_id"], [])
        if pending and len(pending) + len(game_rows) > MAX_SAMPLES_PER_SHARD:
            flush()
            pending = []
            pending_lo = None
            pending_hi = None
        if not pending:
            pending_lo = game["game_id"]
        pending.extend(game_rows)
        pending_hi = game["game_id"]
    flush()

    return {"shards": written, "seed_lo": lo, "seed_hi": hi, "prefix": prefix,
            "games": len(games), "samples": len(rows), "seconds": seconds,
            "per_game": games}


def run(trains, valids, out_dir=DATA_DIR, workers=12, prefix="imit"):
    """Collect the training and validation sets in parallel.

    Each worker takes a contiguous run of seeds. No RNG is shared between
    workers - every draw comes from `random.Random(game_seed)` - so the result
    does not depend on how the seeds were divided, only on the seeds
    themselves. `tests/test_rl_collect.py` checks that by running the same range
    with one worker and with four and comparing the samples.
    """
    import multiprocessing as mp

    jobs = []
    for group, seeds in (("train", trains), ("valid", valids)):
        per = max(1, -(-len(seeds) // workers))
        for i in range(0, len(seeds), per):
            chunk = seeds[i:i + per]
            jobs.append((group, chunk))
    check_seed_ranges(trains, valids)
    os.makedirs(out_dir, exist_ok=True)

    payload = [(chunk[0], chunk[-1],
                "%s_%s" % (prefix, group), out_dir)
               for group, chunk in jobs]
    t0 = time.perf_counter()
    if workers == 1:
        results = [_worker(p) for p in payload]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_worker, payload)
    seconds = time.perf_counter() - t0

    shards = []
    for r in results:
        for s in r["shards"]:
            shards.append({"path": os.path.relpath(s["path"], out_dir),
                           "seed_lo": s["seed_lo"], "seed_hi": s["seed_hi"],
                           "prefix": s["prefix"], "samples": s["samples"],
                           "fields": s["fields"]})
    manifest = {
        "created_by": "rl/collect.py",
        "code": code_hash(),
        "action_table_hash": action_table_hash(),
        "teacher": TEACHER,
        "prefix_choices": list(PREFIX_CHOICES),
        "controller_weights": {k: w for k, w in CONTROLLER_WEIGHTS},
        "shortlist_pad": SHORTLIST_PAD,
        "max_samples_per_shard": MAX_SAMPLES_PER_SHARD,
        "field_spec": {k: {"dtype": v[0], "shape": list(v[1])}
                       for k, v in FIELD_SPEC.items()},
        "field_order": list(FIELD_ORDER),
        "shards": sorted(shards, key=lambda s: s["seed_lo"]),
        "excluded_seed_ranges": {
            "bench_engine": [0, BENCH_SEED_MAX],
            "tools_benchmark": [BENCHMARK_SEED_BASE,
                                BENCHMARK_SEED_BASE
                                + BENCHMARK_SEED_STRIDE * BENCHMARK_MAX_GAMES],
        },
        "wall_seconds": seconds,
        "workers": workers,
        "total_samples": sum(r["samples"] for r in results),
        "total_games": sum(r["games"] for r in results),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return manifest, results


# --------------------------------------------------------------------------
# the statistics the plan asks for
# --------------------------------------------------------------------------

def summarise(rows, games):
    """The numbers section G4 of the report has to carry.

    Every one is computed from the rows and the per-game summaries the run
    actually produced, not read off the sampling design, because the design is
    what the run is supposed to be checking.
    """
    rows = list(rows)
    if not rows:
        return {"samples": 0}
    n = len(rows)
    controllers = {}
    for g in games:
        for owner, key in enumerate(g["controllers"]):
            controllers.setdefault(key, [0, 0])
            controllers[key][0] += 1
            controllers[key][1] += g["ply"]
    seat_share = {k: v[0] / (4 * len(games)) for k, v in controllers.items()}
    n_legal = np.array([int(r["n_legal"]) for r in rows])
    shortlist_len = np.array([int((r["shortlist_idx"] >= 0).sum()) for r in rows])
    n_cand = np.array([int(r["n_candidates"]) for r in rows])
    in_prefix = np.array([bool(r["in_prefix"]) for r in rows])
    teacher_first = 0
    teacher_in_shortlist = 0
    for r in rows:
        first = int(r["shortlist_idx"][0])
        if first >= 0:
            teacher_first += 1
            if first == int(r["action"]):
                teacher_in_shortlist += 1
    ply = np.array([int(r["ply"]) for r in rows])
    by_bucket = {}
    for p in ply:
        b = int(p) // 8
        by_bucket[b] = by_bucket.get(b, 0) + 1
    # Position diversity: identical (own_bits, hand_bits, to_move) triples are
    # the same position reached twice, and the opening prefix is the only thing
    # in this dataset that can make that happen.
    keys = {}
    for r in rows:
        key = (bytes(b"".join(r["own_bits"])),
               bytes(b"".join(int(h).to_bytes(4, "little")
                              for h in r["hand_bits"])),
               int(r["to_move"]))
        keys[key] = keys.get(key, 0) + 1
    repeated = sum(v for v in keys.values() if v > 1)
    teacher_util = []
    teacher_remaining = []
    for g in games:
        for owner, key in enumerate(g["controllers"]):
            if key == TEACHER:
                teacher_util.append(g["utilities"][owner])
                teacher_remaining.append(g["remaining"][owner])
    return {
        "samples": n,
        "games": len(games),
        "distinct_positions": len(keys),
        "repeated_position_share": repeated / n,
        "seat_controller_share": seat_share,
        "seat_controller_counts": {k: v[0] for k, v in controllers.items()},
        "in_prefix_share": float(in_prefix.mean()),
        "in_prefix_samples": int(in_prefix.sum()),
        "n_legal": {"mean": float(n_legal.mean()), "max": int(n_legal.max()),
                    "min": int(n_legal.min()),
                    "p95": int(np.percentile(n_legal, 95))},
        "n_candidates": {"mean": float(n_cand.mean()), "max": int(n_cand.max())},
        "shortlist_len": {"mean": float(shortlist_len.mean()),
                          "max": int(shortlist_len.max()),
                          "p95": int(np.percentile(shortlist_len, 95))},
        "teacher_equals_shortlist_first": teacher_in_shortlist,
        "teacher_equals_shortlist_first_share": teacher_in_shortlist / n,
        "samples_with_empty_shortlist": n - teacher_first,
        "samples_per_ply_bucket": {str(b * 8): c
                                   for b, c in sorted(by_bucket.items())},
        "teacher_mean_utility": float(np.mean(teacher_util)) if teacher_util else None,
        "teacher_mean_remaining": (float(np.mean(teacher_remaining))
                                   if teacher_remaining else None),
        "teacher_seats": len(teacher_util),
    }


def main():
    ap = argparse.ArgumentParser(description="collect imitation data")
    ap.add_argument("--train-games", type=int, default=2000)
    ap.add_argument("--valid-games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=DATA_DIR)
    ap.add_argument("--prefix", default="imit")
    ap.add_argument("--json", default=None,
                    help="write the summary statistics here")
    args = ap.parse_args()

    trains = split_seeds(TRAIN_SEED_BASE,
                         TRAIN_SEED_BASE + args.train_games - 1)
    valids = split_seeds(VALID_SEED_BASE,
                         VALID_SEED_BASE + args.valid_games - 1)
    check_seed_ranges(trains, valids)
    manifest, results = run(trains, valids, out_dir=args.out,
                            workers=args.workers, prefix=args.prefix)

    # Read the shards back rather than reusing the rows the workers held. That
    # makes the statistics a statement about what was *written*, which is the
    # thing a later run will load, and it exercises `load_shard` on the way.
    rows = []
    games = []
    for r in results:
        for shard_info in r["shards"]:
            shard = load_shard(shard_info["path"])
            rows.extend(_rows_from_shard(shard))
        games.extend(r["per_game"])
    rows.sort(key=lambda row: (int(row["game_id"]), int(row["ply"])))
    stats = summarise(rows, games)
    disk = _disk_usage(args.out)
    out = {"manifest": manifest, "stats": stats, "disk": disk}
    print(json.dumps(stats, indent=1, sort_keys=True))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
        print("wrote %s" % args.json, file=sys.stderr)
    return 0


def _disk_usage(path):
    """Bytes on disk for the shards and the manifest, so the report can say how
    large the dataset is without someone shelling out to `du`."""
    total = 0
    files = {}
    if not os.path.isdir(path):
        return {"bytes": 0, "files": {}}
    for name in sorted(os.listdir(path)):
        size = os.path.getsize(os.path.join(path, name))
        total += size
        files[name] = size
    return {"bytes": total, "mib": total / 1024 / 1024, "files": files}


def _rows_from_shard(shard):
    """Rows back out of a written shard, in the shape `summarise` expects."""
    n = shard["action"].size
    out = []
    for i in range(n):
        row = {}
        for name in FIELD_ORDER:
            arr = shard[name]
            row[name] = arr[i] if arr.ndim else arr[i]
        out.append(row)
    return out


if __name__ == "__main__":
    sys.exit(main())