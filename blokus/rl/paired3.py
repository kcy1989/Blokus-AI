"""This layer answers one question: with three arms played from the same
positions, does the differential-scoring optimizer (v3) beat the book version
(v2)?

A pair is one configuration and one game seed, and it plays **three** games: the
tested seat is v1, then v2, then v3. The other three personalities, the seating,
the game seed and every random stream are identical across the three games, so
`d32 = rank_v3 - rank_v2` isolates the scoring change and nothing else.

Two properties of the design are worth stating because they are what make the
comparison readable:

  * The pairing fixes the opponents' **randomness**, not their moves. v2 and v3
    open identically, but from the fourth move on their scoring differs, so the
    board diverges and the opponents legitimately answer different positions.
    Forcing the boards to stay identical would mean forcing the opponents to play
    badly.
  * No random opening prefix. H0 had to add one because the book's own moves are
    a deterministic function of the seed; here the book is used as-is in both v2
    and v3, so there is nothing left to randomise.

Each seat's two random streams come from `(game_seed, seat)` with no argument for
which arm is playing, exactly as in `rl/paired.py`, so an opponent's extra draws
in one game cannot leak into the next opponent's game in the next arm.

The bootstrap is vectorised with numpy. H0's version was a `10_000 x n` pure
Python loop and took longer than the 15,000 games it summarised; `plan5.md`
asks for numpy here and explicitly does not require the intervals to match H0's.
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
import engine  # noqa: E402
from ai.base import OPTIMIZER_KEY  # noqa: E402
from config import CLOCKWISE_OWNERS  # noqa: E402
from game import Game  # noqa: E402
from rl.actions import legal_indices, move_to_index  # noqa: E402
from rl.collect import _Budget  # noqa: E402
from rl.env import average_places  # noqa: E402
from rl.opening import (BookTracker, act_to_engine_move, choose_v2,  # noqa: E402
                        must_cover, reach, to_act_move)
from rl.v3 import choose_v3, make_brain  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data", "h1")

TESTED_KEY = OPTIMIZER_KEY

# Group A's pool, the same five as stage H0 and for the same reason: the tested
# personality is not among the opponents, so no arm is ever measured against a
# third copy of itself.
POOL_A = ("builder", "intruder", "wolf", "chess", "fox")
# Group B is fixed at the two strongest rule-based opponents plus v1 itself, so it
# asks whether v3 survives contact with the personality whose scoring it modifies.
GROUP_B_OPPONENTS = ("builder", "intruder", TESTED_KEY)

PROFILE_SALT = 0x00
CHOICE_SALT = 0x01

SEED_BASE_A = 4_000_000
SEED_BASE_B = 4_100_000
SEED_BASE_A_EXPAND = 4_200_000
SEED_BASE_B_EXPAND = 4_300_000

# Every range a published number already came from. `plan5.md` lists these and
# asks for an assertion, so they are written out rather than derived from the
# other modules: a guard that imported the ranges it is meant to police would
# agree with any change to them.
#
# plan8-B0b added the six blocks below, each read off a file rather than taken
# from the plan text:
#
#   * H-B1 and H-C1 are two runs over the *same* seeds - `data/hb1/manifest.json`
#     and `data/hc1/manifest.json` both report 60 train shards spanning
#     6,000,000..6,029,378 and 12 valid shards spanning 6,100,000..6,101,545 -
#     so one entry covers both and re-collecting one of them cannot silently
#     reuse the other.
#   * the evaluation seeds are single values, not spans. `match.run_league`
#     builds one `random.Random(seed)` and draws every seat, colour and opening
#     player from it, so `--seed 910001` *is* one run rather than the first of
#     910001 consecutive ones. 910001-910004 are the four H-C2 league seeds
#     (reports/hc2_report.md) and 920001 is the pool league; the span is written
#     for the four because all four were used.
#   * match.py's own default seed, 20260928, needs no entry: it already falls
#     inside the tools/benchmark.py range below.
#
# `OWN_RANGES` is deliberately *not* merged in here. `check_seed_ranges` below
# consults both tuples, so putting H1's own blocks into this one would make that
# guard reject H1's own seeds - which is exactly what
# `tests/test_rl_paired3.py::test_the_seed_guard_accepts_this_stages_four_blocks`
# asserts must not happen. `reject_reserved_seeds` reads both tuples instead, so
# the history is policed without the two lists being confused for each other.
RESERVED_RANGES = (
    ("bench_engine.py", 0, 199),
    ("tools/benchmark.py", 20_240_101, 20_240_101 + 1009 * 96),
    ("stage G training", 1_000_000, 1_001_999),
    ("stage G validation", 2_000_000, 2_000_199),
    ("stage H0 phase 1 A", 3_000_000, 3_001_999),
    ("stage H0 phase 1 B", 3_100_000, 3_100_999),
    ("stage H0 expansion A", 3_200_000, 3_202_999),
    ("stage H0 expansion B", 3_300_000, 3_301_499),
    ("H-B1/H-C1 imitation train", 6_000_000, 6_029_378),
    ("H-B1/H-C1 imitation valid", 6_100_000, 6_101_545),
    ("stage H-C2 evaluation", 910_001, 910_004),
    ("pool league 920001", 920_001, 920_001),
    ("stage RL train", 7_000_000, 7_019_999),
    ("stage RL validation", 7_100_000, 7_100_999),
    # plan9 step 3: one train block, shared by the three students o / b / i.
    # Reserved before their first pilot round ran; the label and range must stay
    # in step with `rl.rollout.RL_STUDENTS_BLOCK`, because the guard exempts a
    # claim by exact triple.
    ("stage RL students", 7_200_000, 7_219_999),
    # plan10 task 2: the four learners' same-table games (10 seeds a step),
    # their fixed-pool games (260 a step), and the fixed evaluation set every
    # progress curve is measured on. Three rows rather than one so the two
    # training streams cannot be confused for each other when a step's layout
    # is read off a log. Labels and ranges must stay in step with
    # `rl.multi.TABLE_BLOCK` / `FIXED_BLOCK` / `EVAL_BLOCK`.
    ("stage RL multiple table", 7_300_000, 7_300_999),
    ("stage RL multiple fixed", 7_310_000, 7_335_999),
    ("stage RL multiple validation", 7_400_000, 7_400_999),
    # plan10 extension (50k -> 100k): steps 101..200 continue the same two
    # formulas, and those seeds land in the *gap* between the two rows above -
    # 7_301_000..7_309_999 was never reserved and never used. A second row per
    # stream rather than widening the first, because the guard exempts a claim
    # by exact triple and a widened `TABLE_BLOCK` would no longer match the
    # registered one, making steps 1..100 refuse to start.
    ("stage RL multiple table 2", 7_301_000, 7_301_999),
    ("stage RL multiple fixed 2", 7_336_000, 7_361_999),
    # The large-sample evaluation of the 50k / 75k / 100k policies. Its own
    # row so it cannot be confused with the 100-game progress curve's seeds,
    # and so a bigger schedule cannot silently grow into the next stage's.
    ("stage RL multiple big eval", 7_410_000, 7_411_999),
    # plan10 extension, rollback rerun: steps 81..100 re-walked from the 40k
    # checkpoint on **new** seeds, to tell a one-off bad update apart from
    # something the 80->100 dynamics do every time. Three rows, sized to what
    # the rerun actually claims - 200 table seeds, 5200 fixed, 400 for its
    # evaluation - and they sit in the 38,000-wide gap between
    # `stage RL multiple fixed 2`'s top and `stage RL multiple validation`'s
    # floor. The `_SEED_BASE` constants beside them are the seed *step 1* would
    # have had, so the `(step - 1) * stride` formula is untouched; they are
    # arithmetic anchors and are never themselves claimed, which is why they
    # land inside an older row without colliding with it.
    ("stage RL rb80 table", 7_362_000, 7_362_199),
    ("stage RL rb80 fixed", 7_363_000, 7_368_199),
    ("stage RL rb80 big eval", 7_369_000, 7_369_399),
    # plan10 extension, h-only rerun (v2): the h learner walks steps 81..100
    # again from the 40k checkpoint, this time under `rl_train`'s single
    # seven-personality environment rather than `rl.multi`'s two. The training
    # row covers only the seeds those 20 rounds actually draw, and the fixed
    # evaluation set gets its own row - the same split as the rb80 rerun
    # above, for the same reason. Labels and ranges must stay in step with
    # `rl.rollout.RL_V2_H_BLOCK` / `RL_V2_H_EVAL_BLOCK`.
    ("stage RL v2 h train", 7_500_000, 7_509_999),
    ("stage RL v2 h eval", 7_520_000, 7_520_999),
    # plan10, the i learner: rounds 41..100 from its own 20k under a caller's
    # eleven-seat pool, 30,000 games, plus the acceptance evaluation. Two rows
    # for the same reason as the h rerun's two above. The training row sits in
    # the gap `stage RL validation` and `stage RL students` left - never
    # reserved, never used - so registering it moved nothing. Labels and ranges
    # must stay in step with `rl.rollout.RL_I50K_V2_BLOCK` /
    # `RL_I50K_V2_EVAL_BLOCK`.
    ("stage RL i50k v2 train", 7_170_000, 7_199_999),
    ("stage RL i50k v2 eval", 7_540_000, 7_540_999),
    # plan10, the o learner: rounds 41..100 from its own 20k under a caller's
    # eleven-seat pool - which includes the learner's own frozen starting
    # weights, a self-play exception the user declared - plus the acceptance
    # evaluation. Both sit above every row registered so far rather than in
    # the 7,1xx,xxx gap the i rerun filled, so registering them moved nothing.
    # Labels and ranges must stay in step with
    # `rl.rollout.RL_O50K_V2_BLOCK` / `RL_O50K_V2_EVAL_BLOCK`.
    ("stage RL o50k v2 train", 7_550_000, 7_579_999),
    ("stage RL o50k v2 eval", 7_590_000, 7_590_999),
    # plan10 v3 (2026-10-13 decree), the o learner again: a cold start on
    # `rl_o1000_20k` under a ten-seat pool that no longer holds the learner's
    # own start, dealt by the balanced schedule, rounds 1..60 = 30,000 games.
    # Unlike every row above, `seed_base` is where round 1 actually starts and
    # is drawn, so the base sits at the *floor* of the train row - which is
    # what the anchor test's "no anchor inside any other row" wording exists
    # for. Both ranges sit above every row registered so far (the highest of
    # which ends 7,590,999), so registering them moved nothing. Labels and
    # ranges must stay in step with `rl.rollout.RL_O50K_V3_BLOCK` /
    # `RL_O50K_V3_EVAL_BLOCK`.
    ("stage RL o50k v3 train", 7_620_000, 7_649_999),
    ("stage RL o50k v3 eval", 7_660_000, 7_660_999),
    # The pre-decreed re-verification (2026-10-13): the same 1,000-game
    # paired argmax vs `rl_o1000_20k`, but on a fresh segment - a second
    # measurement must not replay seeds the acceptance evaluation already
    # scored, or the two numbers would be the same look twice. Nine thousand
    # above the eval row, the gap style every pair above uses. Label must stay
    # in step with `rl.rollout.RL_O50K_V3_EVAL2_BLOCK`.
    ("stage RL o50k v3 eval2", 7_670_000, 7_670_999),
    # plan10 (2026-10-14 decree), the h learner: a cold start on
    # `rl_h1000_20k` under the balanced schedule, rounds 1..60 = 30,000 games,
    # under the caller's ten-seat pool which keeps the learner's own frozen
    # start - the self-play exception the user declared for this run. Round
    # numbering restarts at 1, so `seed_base` is where round 1 actually starts
    # and is drawn: the base sits at the floor of the row it claims, the shape
    # the anchor test's "no anchor inside any other row" wording exists for.
    # Both ranges sit above every row registered so far (the highest of which
    # ends 7,670,999), so registering them moved nothing. There is no second
    # evaluation row: the decree made the acceptance measurement a single
    # 1000-game paired argmax look. Labels and ranges must stay in step with
    # `rl.rollout.RL_H50K_V2_BLOCK` / `RL_H50K_V2_EVAL_BLOCK`.
    ("stage RL h50k v2 train", 7_680_000, 7_709_999),
    ("stage RL h50k v2 eval", 7_715_000, 7_715_999),
    # The pool-swapped re-measurement the user decreed on 2026-10-14, after
    # the criterion run on 7,715,000 came back at +0.006430, below its
    # 1.3 x se line. The opponent pool for it is the seven personalities plus
    # `rl_b1000_50k` and `rl_o1000_50k_v3`, so - unlike the training pool -
    # nothing in the set is the candidate itself, though both of those nets
    # sat in the pool it trained against. Fresh seeds are the point: a second
    # look on the seeds the first measurement scored would be the same 1,000
    # games twice, which is why this row sits 9,001 above that block's top
    # rather than inside it. Labels and ranges must stay in step with
    # `rl.rollout.RL_H50K_V2_EVAL2_BLOCK`.
    ("stage RL h50k v2 eval2", 7_725_000, 7_725_999),
    # plan10 (2026-10-14 decree), the i learner: a cold start on
    # `rl_i1000_20k` under the balanced schedule, rounds 1..60 = 30,000 games,
    # under the caller's ten-seat pool which - unlike the h run - keeps the
    # learner's own start out and holds no downstream version of the learner.
    # Round numbering restarts at 1, so `seed_base` is where round 1 actually
    # starts and is drawn: the base sits at the floor of the row it claims,
    # the shape the anchor test's "no anchor inside any other row" wording
    # exists for. Both ranges sit above every row registered so far (the
    # highest of which ends 7,725,999), so registering them moved nothing.
    # One evaluation row only - the decree made the acceptance measurement a
    # single 1000-game argmax look, so softmax is not run and no second
    # evaluation row is registered. Labels and ranges must stay in step with
    # `rl.rollout.RL_I50K_V3_BLOCK` / `RL_I50K_V3_EVAL_BLOCK`.
    ("stage RL i50k v3 train", 7_730_000, 7_759_999),
    ("stage RL i50k v3 eval", 7_765_000, 7_765_999),
)

PAIRS_A = 2_000
PAIRS_B = 1_000
PAIRS_A_EXPAND = 3_000        # brings A to 5,000 in total
PAIRS_B_EXPAND = 1_500        # brings B to 2,500 in total

# This stage's own four seed blocks. Kept here so the guard can check that they
# do not overlap each other, which is the mistake a copy of H0's block layout
# would actually make.
OWN_RANGES = (
    ("H1 phase 1 A", SEED_BASE_A, SEED_BASE_A + PAIRS_A - 1),
    ("H1 phase 1 B", SEED_BASE_B, SEED_BASE_B + PAIRS_B - 1),
    ("H1 expansion A", SEED_BASE_A_EXPAND,
     SEED_BASE_A_EXPAND + PAIRS_A_EXPAND - 1),
    ("H1 expansion B", SEED_BASE_B_EXPAND,
     SEED_BASE_B_EXPAND + PAIRS_B_EXPAND - 1),
)

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 26_010_101
CI_LOW, CI_HIGH = 2.5, 97.5

MAX_PASSES = 200
MAX_PLIES = 600


def seat_seed(game_seed, seat, salt):
    """One seat's random stream, derived from the game seed and nothing else.

    Shifts, not addition, so `(game_seed, seat, salt)` maps injectively into the
    integers and no seat's streams can reach another's.
    """
    return (game_seed << 8) | ((seat & 0x3) << 1) | (salt & 0x1)


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------

def configurations_a():
    """Group A: `C(5,3) = 10` opponent trios x 4 tested seats x `3! = 6` seatings
    = the 240 the plan asks for.

    The seating is part of the configuration and not folded away: the opponents
    are not interchangeable, so a builder next to an intruder is a different
    board from an intruder next to a builder.
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
    """Group B: the fixed trio, 4 tested seats x 6 seatings = 24."""
    out = []
    for tested in range(4):
        others = [s for s in range(4) if s != tested]
        for order in itertools.permutations(GROUP_B_OPPONENTS):
            seats = {tested: TESTED_KEY}
            seats.update(zip(others, order))
            out.append((tested, tuple(seats[s] for s in range(4))))
    return out


def configurations(group):
    return configurations_a() if group == "A" else configurations_b()


def assign(pairs, configs):
    """Hand out configurations by cycling, so counts differ by at most one."""
    return [configs[i % len(configs)] for i in range(pairs)]


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------

def _brain_and_rng(game_seed, seat, key):
    """A seat's brain and choice stream, rebuilt from the same seed in every
    game this seat plays.

    Rebuilt rather than shared: a shared brain would carry the weight draw it
    consumed in the first arm into the second, and the arms would no longer be
    facing the same opponent.
    """
    brain = ai.make_brain(key, random.Random(seat_seed(game_seed, seat,
                                                      PROFILE_SALT)))
    return brain, random.Random(seat_seed(game_seed, seat, CHOICE_SALT))


ARMS = ("v1", "v2", "v3")


def play_game(game_seed, controllers, tested_seat, arm):
    """One complete game with the tested seat on `arm`. Returns a summary dict.

    v1 is `rl.paired`'s path (its own `choose_move`, no book), v2 is
    `rl.opening.choose_v2`, v3 is `rl.v3.choose_v3`. The three share the
    `BookTracker` bookkeeping so their records carry the same columns, and the
    wall-clock budget is off inside a `finally` so it is restored even if a game
    raises.
    """
    tested_brain, tested_rng = _brain_and_rng(game_seed, tested_seat,
                                              controllers[tested_seat])
    brains = {}
    rngs = {}
    for o in range(4):
        if o == tested_seat:
            continue
        brains[o], rngs[o] = _brain_and_rng(game_seed, o, controllers[o])

    if arm == "v3":
        # same profile as v1, so v3 and v1 differ only in the objective function
        tested_brain = make_brain(tested_brain.profile)

    state = engine.initial_state(CLOCKWISE_OWNERS)
    tracker = BookTracker(game_seed)

    g = Game(random.Random(game_seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    g.brains = dict(brains)
    g.brains[tested_seat] = tested_brain
    g.start()

    ply = 0
    passes = 0
    book_moves = 0
    with _Budget(False):
        while not engine.is_over(state):
            if ply + passes > MAX_PLIES:
                raise RuntimeError("seed %d did not finish" % game_seed)
            owner = state.to_move
            if owner != g.current_owner():
                raise RuntimeError("seed %d: engine and Game disagree on whose "
                                   "turn it is at ply %d" % (game_seed, ply))
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
                                   "vs %r" % (game_seed, ply, state.stuck,
                                              g.stuck))

            if owner == tested_seat:
                move, info = _tested_move(arm, state, g, owner, tested_brain,
                                          tested_rng, tracker, brains)
                if info.get("source") == "book":
                    book_moves += 1
            else:
                move = _other_move(state, g, owner, brains, rngs)

            index = move_to_index(move)
            if index not in legal_indices(state, owner):
                raise RuntimeError("seed %d: seat %d produced an illegal move "
                                   "%r at ply %d" % (game_seed, owner, move, ply))
            state = engine.apply_move(state, move)
            g.act(*to_act_move(move))
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
    # the two stuck vectors are returned so a test can compare them rather than
    # take the agreement on trust from the two raises above
    return {"arm": arm, "ply": ply, "passes": passes, "remaining": remaining,
            "places": places, "tested_place": places[tested_seat],
            "tested_remaining": remaining[tested_seat],
            "book_moves": book_moves,
            "book_abandoned_at": tracker.abandoned_at[tested_seat],
            "stuck_engine": [bool(x) for x in state.stuck],
            "stuck_game": [bool(x) for x in g.stuck]}


def _tested_move(arm, state, g, owner, brain, rng, tracker, brains):
    others = {o: brains.get(o, brain) for o in range(4)}
    board = g.board
    if arm == "v1":
        mv = ai.choose_move(board, g.hands[owner].names, owner, brain, rng,
                            other_brains=others,
                            must_cover=must_cover(state, owner),
                            reach=reach(state, board, owner),
                            other_must_cover={o: must_cover(state, o)
                                              for o in range(4)},
                            other_reach={o: reach(state, board, o)
                                         for o in range(4)})
        if mv is None:                                  # pragma: no cover
            raise RuntimeError("the tested player found no move where the engine "
                               "found some")
        return act_to_engine_move(mv[0], mv[1], mv[2], mv[3]), {"source": "v1"}
    if arm == "v2":
        return choose_v2(state, board, owner, brain, rng, tracker,
                         other_brains=others)
    return choose_v3(state, board, owner, brain, rng, tracker,
                     other_brains=others)


def _other_move(state, g, owner, brains, rngs):
    board = g.board
    mv = ai.choose_move(board, g.hands[owner].names, owner, brains[owner],
                        rngs[owner], other_brains={o: brains.get(o, brains[owner])
                                                   for o in range(4)},
                        must_cover=must_cover(state, owner),
                        reach=reach(state, board, owner),
                        other_must_cover={o: must_cover(state, o)
                                          for o in range(4)},
                        other_reach={o: reach(state, board, o)
                                     for o in range(4)})
    if mv is None:                                      # pragma: no cover
        raise RuntimeError("seat %d found no move where the engine found some"
                           % owner)
    return act_to_engine_move(mv[0], mv[1], mv[2], mv[3])


def play_pair(pair_id, group, game_seed, tested_seat, controllers):
    """The three games of one pair, plus the differences they measure.

    `d32 = rank_v3 - rank_v2` is the only primary comparison: same opening, only
    the scoring from the fourth move on differs. `d31` and `d21` are reported for
    description only and take no part in the decision.
    """
    games = {arm: play_game(game_seed, controllers, tested_seat, arm)
             for arm in ARMS}
    v1, v2, v3 = (games[a] for a in ARMS)
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
        "rank_v3": v3["tested_place"],
        "remaining_v1": v1["tested_remaining"],
        "remaining_v2": v2["tested_remaining"],
        "remaining_v3": v3["tested_remaining"],
        "d32": v3["tested_place"] - v2["tested_place"],
        "d31": v3["tested_place"] - v1["tested_place"],
        "d21": v2["tested_place"] - v1["tested_place"],
        "d32_remaining": v3["tested_remaining"] - v2["tested_remaining"],
        "d31_remaining": v3["tested_remaining"] - v1["tested_remaining"],
        "d21_remaining": v2["tested_remaining"] - v1["tested_remaining"],
        "book_moves_v2": v2["book_moves"],
        "book_moves_v3": v3["book_moves"],
        "book_abandoned_v2_at": v2["book_abandoned_at"],
        "book_abandoned_v3_at": v3["book_abandoned_at"],
        "ply_v1": v1["ply"], "ply_v2": v2["ply"], "ply_v3": v3["ply"],
    }


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def check_seed_ranges(seeds, own_base=None):
    """Refuse to run into a range a published number already came from.

    Membership, not span: a list of seeds is a set of positions to be played and
    two seeds either are or are not inside a used range. Both the published
    ranges and this stage's own four blocks are checked, so a typo in a base or
    a count is caught before 15,000 games are played rather than after.
    """
    problems = []
    if not seeds:
        return problems
    unique = sorted(set(seeds))
    lo, hi = unique[0], unique[-1]
    if own_base is not None and lo < own_base:
        problems.append("seeds start at %d, below this stage's %d"
                        % (lo, own_base))
    for label, block_lo, block_hi in RESERVED_RANGES:
        hit = [s for s in unique if block_lo <= s <= block_hi]
        if hit:
            problems.append("seeds %r overlap %s's %d..%d"
                            % (hit[:5], label, block_lo, block_hi))
    for i, (la, alo, ahi) in enumerate(OWN_RANGES):
        for lb, blo, bhi in OWN_RANGES[i + 1:]:
            if not (ahi < blo or bhi < alo):
                problems.append("this stage's own seed ranges %s %d..%d and "
                                "%s %d..%d overlap"
                                % (la, alo, ahi, lb, blo, bhi))
    if own_base is not None:
        # the seeds must sit in their own block and in no other, which is the
        # mistake a mistyped base or count would make
        for label, block_lo, block_hi in OWN_RANGES:
            if block_lo <= own_base <= block_hi:
                continue
            hit = [s for s in unique if block_lo <= s <= block_hi]
            if hit:
                problems.append("seeds %r overlap this stage's own block %s "
                                "%d..%d" % (hit[:5], label, block_lo, block_hi))
    if problems:
        raise ValueError("; ".join(problems))
    return problems


def check_seeds_for_run(group, pairs, seed_base):
    """The seeds one `run_group` call would use, checked before it runs."""
    seeds = [seed_base + i for i in range(pairs)]
    return check_seed_ranges(seeds, own_base=seed_base)


def reject_reserved_seeds(lo, hi, own=None, extra_ranges=()):
    """Refuse a seed *block* that touches anything a published number used.

    `check_seed_ranges` takes the individual seeds a caller has already decided
    on, which is the right shape when the list is in hand. A stage that is
    choosing its own block knows only `lo` and `hi`, and asking it to
    materialise twenty thousand integers to find out that its base is taken is
    the wrong interface. This is that interface.

    Unlike `check_seed_ranges` it also reads `OWN_RANGES`. That is the whole
    reason it exists as a separate function: H1's four blocks are history like
    any other, but they cannot live in `RESERVED_RANGES` because the guard
    above consults both and would then reject H1's own seeds. Reading both
    tuples here keeps the history policed without the two lists colliding.

    `own` is `(label, lo, hi)`: the block the caller is claiming, exempt from
    the check. It is needed because plan8-B0b put RL's own train and validation
    blocks into `RESERVED_RANGES` - they are reserved *for* the coming run, and
    a guard that rejected them would stop the stage they were reserved for. The
    exemption is by exact triple rather than a blanket skip, so claiming RL's
    block while overlapping the imitation data still raises, and claiming a
    range that is not registered at all is allowed (that is a brand-new block).

    `extra_ranges` is for a block that is neither history nor registered yet -
    the caller polices itself, this way, without editing the table every other
    stage reads.

    Returns an empty list on success and raises `ValueError` naming every block
    it touched, so a caller can put the message straight in front of whoever
    picked the base.
    """
    lo, hi = int(lo), int(hi)
    if hi < lo:
        raise ValueError("seed range %d..%d ends before it starts" % (lo, hi))
    own_t = tuple(own) if own is not None else None
    problems = []
    blocks = tuple(RESERVED_RANGES) + tuple(OWN_RANGES) + tuple(extra_ranges)
    for block in blocks:
        # overlap, not containment: a block that starts inside the request or
        # ends inside it counts, and so does one that swallows it whole
        label, block_lo, block_hi = block
        if own_t is not None and tuple(block) == own_t:
            continue
        if block_lo <= hi and lo <= block_hi:
            problems.append("seeds %d..%d overlap %s's %d..%d"
                            % (lo, hi, label, block_lo, block_hi))
    if problems:
        raise ValueError("; ".join(problems))
    return []


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

    Sharding is by pair and a pair's result depends only on its own seed and
    configuration, so the output does not depend on how the pairs were divided.
    """
    import multiprocessing as mp

    configs = configurations(group)
    assignment = assign(pairs, configs)
    seeds = [seed_base + i for i in range(pairs)]
    check_seeds_for_run(group, pairs, seed_base)

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
# the statistics
# --------------------------------------------------------------------------

def bootstrap_ci(values, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """The mean and a percentile bootstrap interval, resampling the given values.

    The resampling unit is the pair, because a pair is the unit of independence:
    the three games inside one pair share a configuration, a seed and three
    opponents, so treating their two differences as independent would understate
    the interval.

    Vectorised: one `n x iters` matrix of uniform draws, so the cost is `n *
    iters` additions inside numpy instead of `iters` Python-level loops. The seed
    is fixed, so the interval is a property of the numbers and not of the run.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"n": 0, "mean": None, "sd": None,
                "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = arr[idx].mean(axis=1)
    return {"n": int(n), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "ci_low": float(np.percentile(means, CI_LOW)),
            "ci_high": float(np.percentile(means, CI_HIGH))}


def _describe(values, iters, seed):
    return bootstrap_ci(values, iters, seed)


def _group_ci(rows, field, iters, seed):
    return bootstrap_ci([r[field] for r in rows], iters, seed)


def _count(values):
    out = {}
    for v in values:
        key = "none" if v is None else str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def _grouped(rows, field, keyfn):
    buckets = {}
    for r in rows:
        buckets.setdefault(keyfn(r), []).append(r)
    return {k: _describe([r[field] for r in v], BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def summarise(rows, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """Every number the report quotes, in one dict."""
    a = [r for r in rows if r["group"] == "A"]
    b = [r for r in rows if r["group"] == "B"]
    return {
        "n_pairs": len(rows),
        "n_games": 3 * len(rows),
        "primary": {
            "group_A_d32": _group_ci(a, "d32", iters, seed),
            "group_B_d32": _group_ci(b, "d32", iters, seed),
        },
        "secondary": {
            "group_A_d31": _group_ci(a, "d31", iters, seed),
            "group_B_d31": _group_ci(b, "d31", iters, seed),
            "group_A_d21": _group_ci(a, "d21", iters, seed),
            "group_B_d21": _group_ci(b, "d21", iters, seed),
            "group_A_d32_remaining": _group_ci(a, "d32_remaining", iters, seed),
            "group_B_d32_remaining": _group_ci(b, "d32_remaining", iters, seed),
            "share_d32_zero": _share(a + b, "d32", lambda v: v == 0),
            "share_d32_negative": _share(a + b, "d32", lambda v: v < 0),
            "share_d32_positive": _share(a + b, "d32", lambda v: v > 0),
            "rank_corr_v2_v3": _pearson([r["rank_v2"] for r in rows],
                                       [r["rank_v3"] for r in rows]),
            "rank_corr_v1_v3": _pearson([r["rank_v1"] for r in rows],
                                       [r["rank_v3"] for r in rows]),
            "rank_corr_v1_v2": _pearson([r["rank_v1"] for r in rows],
                                       [r["rank_v2"] for r in rows]),
            "group_A_d32_by_tested_seat": _grouped(
                a, "d32", lambda r: r["tested_seat"]),
            "group_B_d32_by_tested_seat": _grouped(
                b, "d32", lambda r: r["tested_seat"]),
            "group_A_d32_by_opponent": _grouped(
                a, "d32", lambda r: "+".join(sorted(
                    o["key"] for o in r["opponents"]))),
            "group_B_d32_by_opponent": _grouped(
                b, "d32", lambda r: "+".join(sorted(
                    o["key"] for o in r["opponents"]))),
            "book_abandoned_v2_share": _share(a + b, "book_abandoned_v2_at",
                                              lambda v: v is not None),
            "book_abandoned_v3_share": _share(a + b, "book_abandoned_v3_at",
                                              lambda v: v is not None),
            "book_abandoned_v2_at_counts": _count(
                r["book_abandoned_v2_at"] for r in rows),
            "book_abandoned_v3_at_counts": _count(
                r["book_abandoned_v3_at"] for r in rows),
            "book_moves_v2_mean": _mean(a + b, "book_moves_v2"),
            "book_moves_v3_mean": _mean(a + b, "book_moves_v3"),
        },
    }


def _share(rows, field, pred):
    if not rows:
        return None
    return sum(1 for r in rows if pred(r[field])) / len(rows)


def _mean(rows, field):
    if not rows:
        return None
    return sum(r[field] for r in rows) / len(rows)


def decide(stats):
    """The decision, from the rule `plan5.md` fixed before the experiment ran.

    v3 becomes a candidate teacher only if group A's `d32` interval lies entirely
    below zero **and** group B's reaches zero or below. A boundary case of
    exactly zero is not "below zero": the rule says strictly less, and reading it
    as "at most" would adopt v3 on a tie.
    """
    a = stats["primary"]["group_A_d32"]
    b = stats["primary"]["group_B_d32"]
    if a["n"] == 0 or b["n"] == 0:                  # pragma: no cover
        return {"verdict": "keep v2", "reason": "a group has no pairs",
                "A_ci_high": a["ci_high"], "B_ci_low": b["ci_low"]}
    a_better = a["ci_high"] < 0.0
    b_not_worse = b["ci_low"] <= 0.0
    if a_better and b_not_worse:
        verdict = "adopt v3"
        reason = ("group A d32 95%% CI high is %.6f, below 0, and group B low "
                  "is %.6f, at or below 0" % (a["ci_high"], b["ci_low"]))
    else:
        verdict = "keep v2"
        failed = []
        if not a_better:
            failed.append("group A d32 95%% CI high is %.6f, not below 0"
                          % a["ci_high"])
        if not b_not_worse:
            failed.append("group B d32 95%% CI low is %.6f, above 0"
                          % b["ci_low"])
        reason = "; ".join(failed)
    return {"verdict": verdict, "reason": reason,
            "A_ci_high": a["ci_high"], "A_ci_low": a["ci_low"],
            "A_mean": a["mean"],
            "B_ci_low": b["ci_low"], "B_ci_high": b["ci_high"],
            "B_mean": b["mean"]}


def expansion_needed(stats):
    """Whether group A's primary interval still contains zero, which is the
    plan's only reason to spend a second phase."""
    a = stats["primary"]["group_A_d32"]
    if a["n"] == 0:                                  # pragma: no cover
        return False
    return not (a["ci_high"] < 0.0 or a["ci_low"] > 0.0)


def expanded(primary_rows, expansion_rows):
    """Pool the first phase and the expansion into one summary.

    The seeds are disjoint so the pairs are disjoint; pooling is what the plan
    asks for and it keeps every pair in the resampling pool rather than
    summarising the two phases separately and reporting one of them.
    """
    return summarise(list(primary_rows) + list(expansion_rows))


# --------------------------------------------------------------------------
# the diagnostic
# --------------------------------------------------------------------------

def diagnostic(rows, games=50, seed=77):
    """Play `games` three-arm games from the first rows and report the book and
    abandonment behaviour of the two book arms."""
    v2_books = 0
    v3_books = 0
    abandon_v2 = 0
    abandon_v3 = 0
    identical_opening = 0
    for r in rows[:games]:
        games_by_arm = {arm: play_game(r["game_seed"], r["controllers"],
                                       r["tested_seat"], arm)
                        for arm in ARMS}
        v2_books += games_by_arm["v2"]["book_moves"]
        v3_books += games_by_arm["v3"]["book_moves"]
        abandon_v2 += games_by_arm["v2"]["book_abandoned_at"] is not None
        abandon_v3 += games_by_arm["v3"]["book_abandoned_at"] is not None
        identical_opening += (games_by_arm["v2"]["book_moves"]
                              == games_by_arm["v3"]["book_moves"]
                              and games_by_arm["v2"]["book_abandoned_at"]
                              == games_by_arm["v3"]["book_abandoned_at"])
    return {"games": min(games, len(rows)),
            "book_moves_v2_total": v2_books,
            "book_moves_v3_total": v3_books,
            "abandoned_v2": abandon_v2,
            "abandoned_v3": abandon_v3,
            "identical_book_behaviour": identical_opening}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs-a", type=int, default=PAIRS_A)
    ap.add_argument("--pairs-b", type=int, default=PAIRS_B)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "pairs.jsonl"))
    ap.add_argument("--json", default=os.path.join(DATA_DIR, "raw.json"))
    ap.add_argument("--diagnose-games", type=int, default=50)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows_a, sec_a = run_group("A", args.pairs_a, SEED_BASE_A, args.workers)
    rows_b, sec_b = run_group("B", args.pairs_b, SEED_BASE_B, args.workers)
    rows = rows_a + rows_b

    payload = {"seed_bases": {"A": SEED_BASE_A, "B": SEED_BASE_B},
               "pairs_a": args.pairs_a, "pairs_b": args.pairs_b,
               "phase1": {"A": {"seconds": sec_a}, "B": {"seconds": sec_b}},
               "stats": summarise(rows),
               "decision": decide(summarise(rows)),
               "expansion_needed": expansion_needed(summarise(rows))}
    payload["phase1_stats"] = payload["stats"]
    payload["phase1_decision"] = payload["decision"]

    if args.expand:
        exp_a, exp_sec_a = run_group("A", PAIRS_A_EXPAND,
                                     SEED_BASE_A_EXPAND, args.workers)
        exp_b, exp_sec_b = run_group("B", PAIRS_B_EXPAND,
                                     SEED_BASE_B_EXPAND, args.workers)
        expansion = exp_a + exp_b
        rows = rows + expansion
        combined = summarise(rows)
        payload["expanded_stats"] = combined
        payload["decision"] = decide(combined)
        payload["expanded"] = {"A": {"seconds": exp_sec_a},
                               "B": {"seconds": exp_sec_b}}
        payload["expansion_pairs"] = {"A": PAIRS_A_EXPAND, "B": PAIRS_B_EXPAND}
        payload["expansion_needed"] = expansion_needed(combined)

    if args.diagnose_games:
        payload["diagnostic"] = diagnostic(rows, args.diagnose_games)

    write_jsonl(args.out, rows)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"decision": payload["decision"],
                      "pairs": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())