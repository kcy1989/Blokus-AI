"""This layer answers one question: is the opening book a better teacher than
the optimizer it wraps?

The design is paired. A *pair* is one configuration (which seat is under test,
and which three personalities sit in the other three seats) played twice with
the same game seed: once with the tested seat playing v1 and once with it
playing v2. The two games therefore differ in exactly one bit - whether the
tested seat's first three real moves came from the book - and the three
opponents are the same personalities in the same seats.

That pairing is the whole point. An unpaired experiment would compare v1's
average place against v2's average place across two independently drawn
opponent pools, and the noise from the pools is larger than any plausible
effect of an opening book. Pairing removes it: `d = rank_v2 - rank_v1` is a
per-game quantity, and the question becomes whether the mean of `d` is below
zero, with the pair as the unit of resampling.

Holding the opponents fixed across the two games of a pair needs care, because
the board they see is not the same board: v1 and v2 play different opening
moves, so from the first move on the positions differ and the opponents
legitimately answer differently. What is held fixed is the *randomness*, not
the moves:

  * every seat's brain is built from `random.Random(seat_seed(game_seed, seat))`
    fresh inside each game, so the weight perturbation is bit-identical in both;
  * every seat's choice draws come from `random.Random(seat_seed(game_seed, seat,
    CHOICE_SALT))`, a second stream that no amount of divergence on the board
    can shift.

Neither stream mentions whether the tested seat is v1 or v2, which is the whole
requirement. A design that seeded the opponents from one shared generator would
have the first opponent consume a different number of draws in each game and
every later opponent would inherit the difference - a confound that looks like a
result.

How the personalities actually get their randomness in `ai/`, which is what the
two streams have to cover:

  * `ai.make_brain(key, rng)` calls `make_profile(spec, rng)`, which multiplies
    the seven base numbers by `rng.uniform(0.7, 1.3)`. This is the weight
    perturbation of stage F' section 4, and it is the only use of `rng` at
    construction time.
  * `ai.choose_move(..., rng, ...)` draws exactly once, and only when
    `len(shortlist) >= 2 and rng.random() < brain.mistake_rate`. The rule-based
    personalities set `mistake_rate = 0`, so for the optimizer and the tested
    seat this stream is never touched at all.

So two streams per seat cover every source. The tested seat gets them too, for
uniformity: the optimizer does not read them, and a test asserts that.

Group A is the main judgement: three opponents drawn from builder, intruder,
wolf, chess and fox. Group B is the check against strong opposition: builder,
intruder, and a **v1** optimizer - not the version under test - so group B never
puts the tested personality in the same game as itself.

The turn order is the fixed clockwise order rather than a draw. Rotating it
would add a second random axis to every configuration, and fairness across seats
is already covered by putting the tested seat in each of the four positions in
turn. What must be identical between the two games of a pair is the turn order,
and a fixed order is the only way to guarantee that without seeding it twice.
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
import engine  # noqa: E402
from config import CLOCKWISE_OWNERS  # noqa: E402
from game import Game  # noqa: E402
from rl.actions import legal_indices, move_to_index  # noqa: E402
from rl.collect import _Budget, board_from_state  # noqa: E402
from rl.env import average_places  # noqa: E402
from rl.opening import (BOOK_STEPS, BookTracker, act_to_engine_move,  # noqa: E402
                        book_candidates, choose_v2, must_cover, reach,
                        to_act_move)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data", "h0")

TESTED_KEY = "optimizer"

# Group A's pool. Deliberately excludes the tested personality: the plan puts
# v1 and v2 in separate games, and an opponent that is a third copy of the thing
# under test would measure something else.
POOL_A = ("builder", "intruder", "wolf", "chess", "fox")
# Group B is fixed. The third opponent is v1 specifically, so that group B asks
# "does the book survive against the personality it is imitating" without ever
# putting two copies of the tested version on the same board.
GROUP_B_OPPONENTS = ("builder", "intruder", TESTED_KEY)

# The two random streams a seat gets. `PROFILE_SALT` and `CHOICE_SALT` differ by
# one bit, and both live below `BOOK_MARKER` in the same shifted layout
# `rl/opening.py` uses, so no seat stream and no book stream can collide.
PROFILE_SALT = 0x00
CHOICE_SALT = 0x01

# Seed ranges. Every pair's two games use one seed; the seed fixes the opponents'
# randomness and the book draw. Phase 1 uses 3,000,000 upwards and the expansion
# uses a further block, so the two phases never share a position.
SEED_BASE_A = 3_000_000
SEED_BASE_B = 3_100_000
SEED_BASE_A_EXPAND = 3_200_000
SEED_BASE_B_EXPAND = 3_300_000

# Ranges already used by a published number, refused at run time.
BENCH_SEED_MAX = 199
BENCHMARK_SEED_BASE = 20240101
BENCHMARK_SEED_STRIDE = 1009
BENCHMARK_MAX_GAMES = 96
G_TRAIN_BASE = 1_000_000
G_TRAIN_GAMES = 2_000
G_VALID_BASE = 2_000_000
G_VALID_GAMES = 200

PAIRS_A = 2_000
PAIRS_B = 1_000
PAIRS_A_EXPAND = 3_000        # brings A to 5,000 in total
PAIRS_B_EXPAND = 1_500        # brings B to 2,500 in total

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20_260_901
CI_LOW, CI_HIGH = 2.5, 97.5

MAX_PASSES = 200
MAX_PLIES = 600


def seat_seed(game_seed, seat, salt):
    """One seat's random stream, derived from the game seed and nothing else.

    The shifts rather than an addition, so `(game_seed, seat, salt)` maps
    injectively into the integers: with a game seed of 3,000,000 the two streams
    for a seat are eight apart and no seat's streams can reach another's.
    """
    return (game_seed << 8) | ((seat & 0x3) << 1) | (salt & 0x1)


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------

def configurations_a():
    """Every group A configuration: opponent trio, tested seat, and the order the
    trio fills the remaining seats.

    `C(5,3) = 10` trios, four tested seats and `3! = 6` orders, which is the
    240 the plan asks for. The order matters because the opponents are not
    interchangeable - a builder next to an intruder is a different board from an
    intruder next to a builder - and dropping it would quietly test 40 boards
    instead of 240.
    """
    out = []
    for trio in itertools.combinations(POOL_A, 3):
        for tested in range(4):
            others = [s for s in range(4) if s != tested]
            for order in itertools.permutations(trio):
                seats = {tested: TESTED_KEY}
                seats.update(zip(others, order))
                out.append((tested, tuple(seats[s] for s in range(4))))
    return out


def configurations_b():
    """Every group B configuration: the tested seat and the order the fixed trio
    fills the rest. Four seats times six orders is the plan's 24."""
    out = []
    for tested in range(4):
        others = [s for s in range(4) if s != tested]
        for order in itertools.permutations(GROUP_B_OPPONENTS):
            seats = {tested: TESTED_KEY}
            seats.update(zip(others, order))
            out.append((tested, tuple(seats[s] for s in range(4))))
    return out


def assign(pairs, configs):
    """Hand out configurations by cycling, so counts differ by at most one.

    Cycling is what makes the balance exact rather than statistical: pair `i`
    gets `configs[i % len(configs)]`, so every configuration is used
    `n // len` or `n // len + 1` times and the spread is one.
    """
    return [configs[i % len(configs)] for i in range(pairs)]


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------

def _brain_and_rng(game_seed, seat, key):
    """A seat's brain and its choice stream, both rebuilt from the same seed in
    every game this seat plays.

    Rebuilt rather than shared, which is the point: a shared object would carry
    the weight draw and the choice draws it had already consumed in the first
    game of the pair into the second, and the two games would no longer be
    playing the same opponent.
    """
    brain = ai.make_brain(key, random.Random(seat_seed(game_seed, seat,
                                                      PROFILE_SALT)))
    return brain, random.Random(seat_seed(game_seed, seat, CHOICE_SALT))


def play_game(game_seed, controllers, tested_seat, use_book):
    """One complete game. Returns a summary dict.

    `controllers` maps owner id to personality key; `controllers[tested_seat]` is
    the personality under test and is played with v1 logic or with the book
    wrapper, which is the only difference between the two games of a pair.

    The `BookTracker` is advanced for the tested seat in **both** games, so the
    two records carry the same "book abandoned" column and it means the same
    thing on either side. The two games get that column differently, on purpose:
    the v2 game asks `choose_v2`, which consumes the tracker, while the v1 game
    asks the position directly - "is the candidate set empty at this player's own
    step" - because v1 never touches the tracker. Reading it off the tracker in
    the v1 game would have produced a field that was always `None` rather than
    never wrong, which is the more dangerous of the two.
    """
    tested_brain, tested_rng = _brain_and_rng(game_seed, tested_seat,
                                              controllers[tested_seat])
    brains = {}
    rngs = {}
    for o in range(4):
        if o == tested_seat:
            continue
        brains[o], rngs[o] = _brain_and_rng(game_seed, o, controllers[o])

    state = engine.initial_state(CLOCKWISE_OWNERS)
    tracker = BookTracker(game_seed)

    # `Game` is advanced in lockstep with the engine so the stuck flags can be
    # compared at every step, exactly as `rl/collect.py` does. It is the
    # human-facing implementation; if the two ever disagree, the board handed to
    # the personalities is not the position the engine thinks is on the board.
    g = Game(random.Random(game_seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    g.brains = dict(brains)
    g.brains[tested_seat] = tested_brain
    g.start()

    ply = 0
    passes = 0
    book_moves = 0
    v1_abandoned = None
    with _Budget(False):
        while not engine.is_over(state):
            if ply + passes > MAX_PLIES:
                raise RuntimeError("seed %d did not finish" % game_seed)
            owner = state.to_move
            if owner != g.current_owner():
                raise RuntimeError("seed %d: engine and Game disagree on whose "
                                   "turn it is at ply %d"
                                   % (game_seed, ply))
            if engine.legal_move_mask(state) == 0:
                state = engine.pass_turn(state)
                g.act_pass()
                passes += 1
                if passes > MAX_PASSES:
                    raise RuntimeError("seed %d passed %d times"
                                       % (game_seed, passes))
                continue
            if tuple(state.stuck) != tuple(g.stuck):
                raise RuntimeError("seed %d: stuck flags diverged at ply %d: %r "
                                   "vs %r"
                                   % (game_seed, ply, state.stuck, g.stuck))

            legal = legal_indices(state)
            n_legal = int(legal.size)
            if owner == tested_seat:
                if use_book:
                    move, info = choose_v2(state, board_from_state(state), owner,
                                           tested_brain, tested_rng, tracker,
                                           other_brains=_tested_others(brains,
                                                                     tested_brain))
                    if info["source"] == "book":
                        book_moves += 1
                else:
                    # The v1 game never consults the book, so "would the book
                    # have been abandoned on this line" has to be asked directly:
                    # compute the candidate set the book would have faced at this
                    # player's own step and record whether it was empty. Without
                    # this the v1 column is structurally always None and the two
                    # columns of the record do not mean the same thing.
                    if (v1_abandoned is None
                            and tracker.own_moves[owner] < BOOK_STEPS
                            and not book_candidates(state, owner,
                                                    tracker.own_moves[owner])):
                        v1_abandoned = tracker.own_moves[owner]
                    move = _tested_move_v1(state, g, owner, tested_brain,
                                           tested_rng)
            else:
                move = _other_move(state, g, owner, brains, rngs)

            index = move_to_index(move)
            if index not in legal:
                raise RuntimeError("seed %d: seat %d produced an illegal move "
                                   "%r at ply %d" % (game_seed, owner, move, ply))
            act = to_act_move(move)
            state = engine.apply_move(state, move)
            g.act(*act)
            if owner == tested_seat:
                tracker.note_move(owner)
            ply += 1

    if tuple(state.stuck) != tuple(g.stuck):
        raise RuntimeError("seed %d: stuck flags diverged at the end" % game_seed)
    if [g.remaining_cells(o) for o in range(4)] != list(engine.result(state)):
        raise RuntimeError("seed %d: remaining cells diverged at the end"
                           % game_seed)

    remaining = list(engine.result(state))
    places = average_places(remaining)
    return {
        "game_seed": game_seed,
        "tested_seat": tested_seat,
        "use_book": use_book,
        "controllers": list(controllers),
        "ply": ply,
        "passes": passes,
        "remaining": remaining,
        "places": places,
        "tested_place": places[tested_seat],
        "tested_remaining": remaining[tested_seat],
        "book_moves": book_moves,
        "book_abandoned_at": (tracker.abandoned_at[tested_seat]
                              if use_book else v1_abandoned),
    }


def _tested_others(brains, tested_brain):
    """The opponent-lookahead argument, with the tested seat standing in for
    itself. The optimizer sets `uses_lookahead = False`, so the argument is never
    read here; it is filled in the same way `rl/collect.py` does so that the two
    modules cannot drift apart on this detail."""
    return {o: brains.get(o, tested_brain) for o in range(4)}


def _tested_move_v1(state, g, owner, brain, rng):
    """The unmodified v1 choice for the tested seat, on the `Game` board."""
    mv = ai.choose_move(g.board, g.hands[owner].names, owner, brain, rng,
                        other_brains={o: g.brains[o] for o in range(4)},
                        must_cover=must_cover(state, owner),
                        reach=reach(state, g.board, owner),
                        other_must_cover={o: must_cover(state, o)
                                          for o in range(4)},
                        other_reach={o: reach(state, g.board, o)
                                     for o in range(4)})
    if mv is None:
        raise RuntimeError("the tested player found no move where the engine "
                           "found some")
    return act_to_engine_move(mv[0], mv[1], mv[2], mv[3])


def _other_move(state, g, owner, brains, rngs):
    board = g.board
    mv = ai.choose_move(board, g.hands[owner].names, owner, brains[owner],
                        rngs[owner], other_brains=g.brains,
                        must_cover=must_cover(state, owner),
                        reach=reach(state, board, owner),
                        other_must_cover={o: must_cover(state, o)
                                          for o in range(4)},
                        other_reach={o: reach(state, board, o)
                                     for o in range(4)})
    if mv is None:
        raise RuntimeError("seat %d found no move where the engine found some"
                           % owner)
    return act_to_engine_move(mv[0], mv[1], mv[2], mv[3])


def play_pair(pair_id, group, game_seed, tested_seat, controllers):
    """Both games of one pair, plus the difference they measure."""
    v1 = play_game(game_seed, controllers, tested_seat, use_book=False)
    v2 = play_game(game_seed, controllers, tested_seat, use_book=True)
    return {
        "pair_id": pair_id,
        "group": group,
        "game_seed": game_seed,
        "tested_seat": tested_seat,
        "controllers": list(controllers),
        "opponents": [{"seat": o, "key": controllers[o]}
                      for o in range(4) if o != tested_seat],
        "rank_v1": v1["tested_place"],
        "rank_v2": v2["tested_place"],
        "remaining_v1": v1["tested_remaining"],
        "remaining_v2": v2["tested_remaining"],
        "d": v2["tested_place"] - v1["tested_place"],
        "d_remaining": v2["tested_remaining"] - v1["tested_remaining"],
        "book_abandoned_v1_at": v1["book_abandoned_at"],
        "book_abandoned_v2_at": v2["book_abandoned_at"],
        "book_moves_v2": v2["book_moves"],
        "ply_v1": v1["ply"],
        "ply_v2": v2["ply"],
    }


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def check_seed_ranges(seeds):
    """Refuse to run into a range a published number already came from.

    Membership rather than span. A list of seeds is a set of positions to be
    played, and two seeds either are or are not inside a used range; comparing
    `min` and `max` would reject a sparse list for covering a range it never
    touches. The planned runs are contiguous, so the two agree there - but the
    check should mean what it says.
    """
    problems = []
    if not seeds:
        return problems
    seeds = set(seeds)
    lo, hi = min(seeds), max(seeds)
    if lo < SEED_BASE_A:
        problems.append("seeds start at %d, below the plan's %d"
                        % (lo, SEED_BASE_A))
    bench = seeds & set(range(0, BENCH_SEED_MAX + 1))
    if bench:
        problems.append("seeds %r are inside bench_engine.py's 0..%d"
                        % (sorted(bench)[:5], BENCH_SEED_MAX))
    bench_lo = BENCHMARK_SEED_BASE
    bench_hi = BENCHMARK_SEED_BASE + BENCHMARK_SEED_STRIDE * BENCHMARK_MAX_GAMES
    over_bench = sorted(s for s in seeds if bench_lo <= s <= bench_hi)
    if over_bench:
        problems.append("seeds %r overlap tools/benchmark.py's %d..%d"
                        % (over_bench[:5], bench_lo, bench_hi))
    for label, base, count in (("stage G training", G_TRAIN_BASE, G_TRAIN_GAMES),
                               ("stage G validation", G_VALID_BASE,
                                G_VALID_GAMES)):
        block_lo, block_hi = base, base + count - 1
        hit = sorted(s for s in seeds if block_lo <= s <= block_hi)
        if hit:
            problems.append("seeds %r overlap %s's %d..%d"
                            % (hit[:5], label, block_lo, block_hi))
    if problems:
        raise ValueError("; ".join(problems))
    return problems


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def _worker(args):
    group, chunks = args
    out = []
    for pair_id, seed, tested, controllers in chunks:
        out.append(play_pair(pair_id, group, seed, tested, controllers))
    return out


def run_group(group, pairs, seed_base, workers=12):
    """Run one group of pairs, sharded over processes.

    Sharding is by pair, and a pair's result depends only on its own seed and
    configuration, so the output does not depend on how the pairs were divided.
    `tests/test_rl_paired.py` checks that by running a small set with one worker
    and with four and comparing every field.
    """
    import multiprocessing as mp

    configs = configurations_a() if group == "A" else configurations_b()
    assignment = assign(pairs, configs)
    seeds = [seed_base + i for i in range(pairs)]
    check_seed_ranges(seeds)

    jobs = []
    per = max(1, -(-pairs // workers))
    for i in range(0, pairs, per):
        chunk = [(i + j, seeds[i + j], assignment[i + j][0],
                  assignment[i + j][1])
                 for j in range(per) if i + j < pairs]
        jobs.append((group, chunk))

    t0 = time.perf_counter()
    if workers == 1:
        results = [_worker(job) for job in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    seconds = time.perf_counter() - t0

    rows = [row for chunk in results for row in chunk]
    rows.sort(key=lambda r: r["pair_id"])
    return rows, seconds


# --------------------------------------------------------------------------
# the statistics the plan asks for
# --------------------------------------------------------------------------

def bootstrap_ci(values, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """The mean and a percentile bootstrap interval, resampling the given values.

    The resampling unit is the pair, because a pair is the unit of independence:
    the two games inside one pair share a configuration, a seed and three
    opponents, so treating their two `d` values as independent would understate
    the interval. `plan4.md` fixes the iteration count and the seed here, so the
    interval is reproducible rather than a property of the run.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "iters": iters}
    rng = random.Random(seed)
    mean = sum(values) / n
    means = []
    for _ in range(iters):
        acc = 0.0
        for _ in range(n):
            acc += values[rng.randrange(n)]
        means.append(acc / n)
    means.sort()
    return {
        "n": n,
        "mean": mean,
        "ci_low": means[int(CI_LOW / 100.0 * iters)],
        "ci_high": means[int(CI_HIGH / 100.0 * iters)],
        "iters": iters,
        "seed": seed,
    }


def _group_ci(rows, key, iters, seed):
    return bootstrap_ci([r[key] for r in rows], iters, seed)


def _describe(values, iters, seed):
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    ci = bootstrap_ci(values, iters, seed)
    return {"n": n, "mean": mean, "sd": var ** 0.5, "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"]}


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def summarise(rows, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """Every number the plan lists, computed from the rows that were played.

    `primary` and the decision come first and are kept apart from the rest: the
    secondary numbers are descriptive and must not be allowed to move the
    verdict, so they are reported in a separate block that the decision function
    does not read.
    """
    a = [r for r in rows if r["group"] == "A"]
    b = [r for r in rows if r["group"] == "B"]

    def by_seat(group_rows):
        out = {}
        for seat in range(4):
            out[str(seat)] = _describe([r["d"] for r in group_rows
                                        if r["tested_seat"] == seat], iters, seed)
        return out

    def by_opponent(group_rows):
        out = {}
        for row in group_rows:
            for opp in row["opponents"]:
                out.setdefault(opp["key"], []).append(row["d"])
        return {k: _describe(v, iters, seed) for k, v in sorted(out.items())}

    abandoned_v2 = [r for r in rows if r["book_abandoned_v2_at"] is not None]
    kept_v2 = [r for r in rows if r["book_abandoned_v2_at"] is None]

    return {
        "primary": {
            "group_A_d": _group_ci(a, "d", iters, seed),
            "group_B_d": _group_ci(b, "d", iters, seed),
        },
        "counts": {
            "pairs": len(rows),
            "pairs_A": len(a),
            "pairs_B": len(b),
            "games": 2 * len(rows),
        },
        "secondary": {
            "group_A_d_remaining": _group_ci(a, "d_remaining", iters, seed),
            "group_B_d_remaining": _group_ci(b, "d_remaining", iters, seed),
            "group_A_d_sd": (sum((r["d"] - (sum(x["d"] for x in a) / len(a))) ** 2
                                 for r in a) / len(a)) ** 0.5 if a else None,
            "rank_correlation": _pearson([r["rank_v1"] for r in rows],
                                         [r["rank_v2"] for r in rows]),
            "share_d_zero": (sum(1 for r in rows if r["d"] == 0) / len(rows)
                             if rows else None),
            "share_d_negative": (sum(1 for r in rows if r["d"] < 0) / len(rows)
                                 if rows else None),
            "share_d_positive": (sum(1 for r in rows if r["d"] > 0) / len(rows)
                                 if rows else None),
            "group_A_d_by_tested_seat": by_seat(a),
            "group_B_d_by_tested_seat": by_seat(b),
            "group_A_d_by_opponent": by_opponent(a),
            "book_abandoned_v2_share": (len(abandoned_v2) / len(rows)
                                       if rows else None),
            "book_abandoned_v2_at_counts": _count(
                r["book_abandoned_v2_at"] for r in rows),
            "book_abandoned_v1_at_counts": _count(
                r["book_abandoned_v1_at"] for r in rows),
            "book_moves_v2_mean": (sum(r["book_moves_v2"] for r in rows)
                                   / len(rows) if rows else None),
            "d_when_book_abandoned": _describe(
                [r["d"] for r in abandoned_v2], iters, seed),
            "d_when_book_kept": _describe([r["d"] for r in kept_v2], iters, seed),
        },
    }


def _count(values):
    out = {}
    for v in values:
        key = "none" if v is None else str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def decide(stats):
    """The decision, from the rule `plan4.md` fixed before the experiment ran.

    Adopt v2 only if the group A interval lies entirely below zero **and** the
    group B interval reaches zero or below. Anything else is "keep v1", because
    the extra complexity of a book is not justified by a difference the data
    cannot see. The verdict string is returned together with the two interval
    endpoints it was read off, so the report cannot quote a decision that its own
    numbers do not support.
    """
    a = stats["primary"]["group_A_d"]
    b = stats["primary"]["group_B_d"]
    if a["n"] == 0 or b["n"] == 0:                  # pragma: no cover
        return {"verdict": "keep v1", "reason": "a group has no pairs",
                "A_ci_high": a["ci_high"], "B_ci_low": b["ci_low"]}
    a_better = a["ci_high"] < 0.0
    b_not_worse = b["ci_low"] <= 0.0
    if a_better and b_not_worse:
        verdict = "adopt v2"
        reason = "group A interval entirely below 0 and group B reaches 0"
    else:
        verdict = "keep v1"
        failed = []
        if not a_better:
            failed.append("group A 95%% CI high is %.6f, not below 0"
                          % a["ci_high"])
        if not b_not_worse:
            failed.append("group B 95%% CI low is %.6f, above 0" % b["ci_low"])
        reason = "; ".join(failed)
    return {"verdict": verdict, "reason": reason,
            "A_ci_high": a["ci_high"], "B_ci_low": b["ci_low"],
            "A_ci_low": a["ci_low"], "B_ci_high": b["ci_high"],
            "A_mean": a["mean"], "B_mean": b["mean"]}


def expansion_needed(stats):
    """Whether the primary interval still contains zero, which is the plan's only
    reason to spend a second phase."""
    a = stats["primary"]["group_A_d"]
    if a["n"] == 0:                                  # pragma: no cover
        return False
    return not (a["ci_high"] < 0.0 or a["ci_low"] > 0.0)


# --------------------------------------------------------------------------
# the opening diagnostic
# --------------------------------------------------------------------------

def diagnose_v1_opening(n_games, seed_base=SEED_BASE_A, controllers_pool=POOL_A):
    """How close is v1's own opening to the book, in the games it actually plays?

    `plan4.md` asks for this as a diagnostic rather than an assertion, because it
    decides how much of a difference v2 can possibly make: if v1 already plays
    book moves by itself, a book cannot help, and if it never does, the book is
    replacing a large part of the opening rather than a small part of it.

    Each game is a full game with the optimizer as the tested player, so the
    measured moves are the ones v1 would really play - against real opponents
    that have already moved, not against an empty board.
    """
    per_step = [0, 0, 0]
    in_book = [0, 0, 0]
    # Which piece v1 actually reaches for at each of its first three steps, keyed
    # by whatever it chose - not restricted to the book's three, because the
    # whole point is to see how often it picks one of them.
    piece_counts = [{} for _ in range(3)]
    abandoned_at = {}
    records = []
    for i in range(n_games):
        seed = seed_base + i
        rng = random.Random(seed)
        controllers = [TESTED_KEY]
        for _ in range(3):
            controllers.append(rng.choice(controllers_pool))
        tested_seat = i % 4
        opened, abandoned = _diagnose_game(seed, tuple(controllers), tested_seat)
        for row in opened:
            per_step[row["step"]] += 1
            in_book[row["step"]] += 1 if row["in_book"] else 0
            piece_counts[row["step"]][row["piece"]] = \
                piece_counts[row["step"]].get(row["piece"], 0) + 1
        key = "none" if abandoned is None else str(abandoned)
        abandoned_at[key] = abandoned_at.get(key, 0) + 1
        records.append({"game_seed": seed, "opened": opened,
                        "abandoned_at": abandoned})
    return {
        "games": n_games,
        "steps_observed": per_step,
        "steps_in_book": in_book,
        "in_book_share": [in_book[k] / per_step[k] if per_step[k] else None
                          for k in range(3)],
        "piece_counts": piece_counts,
        "abandoned_at_counts": dict(sorted(abandoned_at.items())),
        "share_all_three_in_book": (
            sum(1 for r in records
                if len(r["opened"]) == 3
                and all(o["in_book"] for o in r["opened"]))
            / len(records) if records else None),
        "records": records,
    }


def _diagnose_game(game_seed, controllers, tested_seat):
    """Play one game, recording the tested player's first three real moves.

    Returns `(opened, abandoned_at)`. The candidate set is computed *before*
    v1 answers, because the question is whether v1's own answer was in the set
    the book would have offered from the same position - and if the set was
    empty, the book would have abandoned here, which is a fact about the
    position and not about v1.
    """
    tested_brain, tested_rng = _brain_and_rng(game_seed, tested_seat,
                                              controllers[tested_seat])
    brains = {}
    rngs = {}
    for o in range(4):
        if o != tested_seat:
            brains[o], rngs[o] = _brain_and_rng(game_seed, o, controllers[o])

    state = engine.initial_state(CLOCKWISE_OWNERS)
    g = Game(random.Random(game_seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    g.brains = dict(brains)
    g.brains[tested_seat] = tested_brain
    g.start()

    own = 0
    opened = []
    abandoned = None
    with _Budget(False):
        while not engine.is_over(state) and own < 3:
            owner = state.to_move
            if engine.legal_move_mask(state) == 0:
                state = engine.pass_turn(state)
                g.act_pass()
                continue
            if owner == tested_seat:
                candidates = None
                if abandoned is None:
                    candidates = book_candidates(state, owner, own)
                board = g.board
                mv = ai.choose_move(board, g.hands[owner].names, owner,
                                    tested_brain, tested_rng,
                                    other_brains={o: g.brains[o]
                                                  for o in range(4)},
                                    must_cover=must_cover(state, owner),
                                    reach=reach(state, board, owner),
                                    other_must_cover={o: must_cover(state, o)
                                                      for o in range(4)},
                                    other_reach={o: reach(state, board, o)
                                                 for o in range(4)})
                name, oi, x, y = mv
                move = act_to_engine_move(name, oi, x, y)
                index = move_to_index(move)
                opened.append({
                    "step": own,
                    "piece": name,
                    "size": engine.PIECE_SIZE[engine.PIECE_IDX[name]],
                    "index": int(index),
                    "in_book": bool(candidates) and index in candidates,
                    "n_candidates": len(candidates) if candidates is not None
                                    else None,
                })
                if candidates is not None and not candidates and abandoned is None:
                    abandoned = own
                own += 1
            else:
                move = _other_move(state, g, owner, brains, rngs)
            state = engine.apply_move(state, move)
            g.act(*to_act_move(move))
    return opened, abandoned


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="stage H0 paired v1 vs v2")
    ap.add_argument("--pairs-a", type=int, default=PAIRS_A)
    ap.add_argument("--pairs-b", type=int, default=PAIRS_B)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--expand", action="store_true",
                    help="run the second phase and merge it with the first")
    ap.add_argument("--out", default=None, help="write the pairs as JSON lines")
    ap.add_argument("--json", default=None, help="write the raw numbers here")
    ap.add_argument("--diagnose-games", type=int, default=50)
    args = ap.parse_args()

    a_rows, a_seconds = run_group("A", args.pairs_a, SEED_BASE_A, args.workers)
    b_rows, b_seconds = run_group("B", args.pairs_b, SEED_BASE_B, args.workers)
    phase1_rows = a_rows + b_rows
    phase1_stats = summarise(phase1_rows)
    phase1_decision = decide(phase1_stats)
    phase1_expansion = expansion_needed(phase1_stats)
    phase1 = {"A": {"pairs": args.pairs_a, "seconds": a_seconds,
                    "seed_lo": SEED_BASE_A, "seed_hi": SEED_BASE_A + args.pairs_a - 1},
              "B": {"pairs": args.pairs_b, "seconds": b_seconds,
                    "seed_lo": SEED_BASE_B,
                    "seed_hi": SEED_BASE_B + args.pairs_b - 1}}

    rows = phase1_rows
    stats = phase1_stats
    decision = phase1_decision
    expanded = None
    if args.expand:
        ax, ax_seconds = run_group("A", PAIRS_A_EXPAND, SEED_BASE_A_EXPAND,
                                   args.workers)
        bx, bx_seconds = run_group("B", PAIRS_B_EXPAND, SEED_BASE_B_EXPAND,
                                   args.workers)
        rows = rows + ax + bx
        expanded = {"A": {"pairs": PAIRS_A_EXPAND, "seconds": ax_seconds,
                          "seed_lo": SEED_BASE_A_EXPAND,
                          "seed_hi": SEED_BASE_A_EXPAND + PAIRS_A_EXPAND - 1},
                    "B": {"pairs": PAIRS_B_EXPAND, "seconds": bx_seconds,
                          "seed_lo": SEED_BASE_B_EXPAND,
                          "seed_hi": SEED_BASE_B_EXPAND + PAIRS_B_EXPAND - 1}}
        stats = summarise(rows)
        decision = decide(stats)

    diagnosis = diagnose_v1_opening(args.diagnose_games)
    out = {
        "phase1": phase1,
        "phase1_stats": phase1_stats,
        "phase1_decision": phase1_decision,
        "expanded": expanded,
        "stats": stats,
        "decision": decision,
        "expansion_needed_after_phase1": phase1_expansion,
        "expanded_run": bool(args.expand),
        "diagnostic": diagnosis,
        "configurations": {"A": len(configurations_a()),
                           "B": len(configurations_b())},
    }
    print(json.dumps({"decision": decision, "primary": stats["primary"]},
                     indent=1, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        print("wrote %s" % args.out, file=sys.stderr)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out or args.json)),
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
        print("wrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
