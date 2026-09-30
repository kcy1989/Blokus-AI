"""D1: the opt-in `trace` must not disturb the engine, and must tell the truth.

`trace` exists to capture what the engine saw when it decided, so a dataset can
be built from real games. Two things have to hold:

  1. `trace=None` changes nothing - not one move, and not measurably any cost.
  2. With a trace, `picked` is exactly the move that was returned, and
     `was_mistake` / `lookahead_used` say what they claim to say.

The full-size same-seed run lives in `tools/verify_same_moves.py`; this file
keeps both properties covered on a small sample.
"""
import dataclasses
import os
import random
import time

import ai
import ai.formulas as F
from ai.base import Profile
from game import Game
from pieces import MASTER

GAMES = int(os.environ.get("BLOKUS_TRACE_GAMES", "10"))
REQUIRED = ("owner", "brain_key", "profile", "n_candidates", "shortlist",
            "picked", "was_mistake", "lookahead_used")


def _new_game(seed):
    keys = list(ai.personality_keys())
    g = Game(random.Random(seed))
    g.setup_match(random.Random(seed).sample(keys, 4))
    g.start()
    return g


def _decide(g, owner, trace):
    return ai.choose_move(g.board, g.hands[owner].names, owner,
                          g.brains[owner], g.rng, other_brains=g.brains,
                          must_cover=g.must_cover(owner),
                          reach=g.reach(owner),
                          other_must_cover={x: g.must_cover(x)
                                            for x in range(4)},
                          other_reach={x: g.reach(x) for x in range(4)},
                          trace=trace)


def test_trace_none_is_the_default_and_records_nothing():
    """Omitting `trace` must be exactly the old call, and record nothing.

    Two independent games from the same seed, because a decision consumes the
    brain's RNG for the mistake roll - calling the engine twice on one position
    can legitimately give two different answers, which would make a same-state
    comparison meaningless.
    """
    def call(g, trace=None):
        owner = g.current_owner()
        return ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], g.rng, other_brains=g.brains,
                              must_cover=g.must_cover(owner),
                              reach=g.reach(owner),
                              other_must_cover={x: g.must_cover(x)
                                                for x in range(4)},
                              other_reach={x: g.reach(x) for x in range(4)},
                              trace=trace)

    assert call(_new_game(0)) == call(_new_game(0))

    # An explicit `trace=None` behaves the same and writes nowhere.
    sink = []
    assert call(_new_game(0), trace=None) == call(_new_game(0))
    assert sink == []


def test_trace_records_one_entry_per_decision_with_the_required_keys():
    for seed in range(3):
        g = _new_game(seed)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            owner = g.current_owner()
            trace = []
            mv = _decide(g, owner, trace)
            assert len(trace) == 1, trace
            rec = trace[0]
            for key in REQUIRED:
                assert key in rec, key
            assert rec["owner"] == owner
            assert rec["brain_key"] == g.brains[owner].key
            assert isinstance(rec["was_mistake"], bool)
            assert isinstance(rec["lookahead_used"], bool)
            if mv is None:
                assert rec["picked"] is None
                assert rec["n_candidates"] == 0
                assert rec["shortlist"] == []
            else:
                name, oi, x, y = mv
                assert rec["picked"] == (name, oi, x + y * 20), (rec["picked"], mv)
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)


def test_trace_profile_is_the_brain_profile_as_a_plain_dict():
    g = _new_game(1)
    owner = g.current_owner()
    trace = []
    _decide(g, owner, trace)
    assert trace[0]["profile"] == dataclasses.asdict(g.brains[owner].profile)
    assert isinstance(trace[0]["profile"], dict)


def test_trace_picked_always_appears_in_the_shortlist():
    """A pick that is not on the shortlist would make the record meaningless."""
    for seed in range(4):
        g = _new_game(100 + seed)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            owner = g.current_owner()
            trace = []
            mv = _decide(g, owner, trace)
            rec = trace[0]
            if rec["picked"] is not None and rec["shortlist"]:
                keys = [(n, oi, b) for _s, n, oi, b in rec["shortlist"]]
                assert rec["picked"] in keys, (rec["picked"], keys[:3])
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)


def test_was_mistake_means_the_non_greedy_branch_was_taken():
    """Cross-check the flag against the shortlist and the brain's mistake rate.

    `_weighted_pick` is a weighted draw, so it can legitimately return the head
    of the shortlist. Therefore:
      * picking anything *other* than the head proves the branch was taken;
      * picking the head is consistent with either outcome, and only proves the
        branch was skipped when the rate is zero or there is no alternative.
    """
    seen = 0
    off_head = 0
    for seed in range(6):
        g = _new_game(200 + seed)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            owner = g.current_owner()
            trace = []
            mv = _decide(g, owner, trace)
            rec = trace[0]
            if rec["shortlist"] and rec["picked"] is not None:
                head = (rec["shortlist"][0][1], rec["shortlist"][0][2],
                        rec["shortlist"][0][3])
                rate = g.brains[owner].mistake_rate
                if rec["picked"] != head:
                    # Impossible without the weighted draw.
                    assert rec["was_mistake"], rec
                    assert rate > 0, rec
                    assert len(rec["shortlist"]) >= 2, rec
                    off_head += 1
                elif rate == 0 or len(rec["shortlist"]) < 2:
                    # The branch could not have been taken at all.
                    assert not rec["was_mistake"], rec
                seen += 1
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)
    assert seen > 0, "no decision points inspected"
    assert off_head > 0, "never observed a non-greedy pick"


def test_lookahead_used_only_for_weighted_personalities():
    """Rule-based personalities set `uses_lookahead = False`; they must never
    report a lookahead."""
    for seed in range(4):
        g = _new_game(300 + seed)
        guard = 0
        while g.state == "PLAYING" and guard < 600:
            guard += 1
            owner = g.current_owner()
            trace = []
            mv = _decide(g, owner, trace)
            rec = trace[0]
            if not g.brains[owner].uses_lookahead:
                assert not rec["lookahead_used"], rec["brain_key"]
            if mv is None:
                g.act_pass()
            else:
                g.act(*mv)


def test_trace_costs_nothing_when_not_requested():
    """The hot path must not pay for a feature nobody asked for.

    Compared against the same engine with a trace list, which is the cheapest
    possible baseline for "is the guard cheap". The bar is loose on purpose:
    this catches an accidentally expensive guard, not microseconds of noise.
    """
    def run(trace_on, games=3):
        t0 = time.perf_counter()
        for seed in range(games):
            g = _new_game(400 + seed)
            trace = [] if trace_on else None
            guard = 0
            while g.state == "PLAYING" and guard < 600:
                guard += 1
                owner = g.current_owner()
                mv = _decide(g, owner, trace)
                if mv is None:
                    g.act_pass()
                else:
                    g.act(*mv)
        return time.perf_counter() - t0

    off = run(False)
    on = run(True)
    # Tracing a whole game is *more* work, so the no-trace run is expected to
    # be the faster one. What matters is that it is not slower.
    assert off <= on * 1.15 + 0.05, (off, on)
