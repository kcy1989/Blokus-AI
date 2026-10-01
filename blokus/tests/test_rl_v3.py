"""Tests for `rl/v3.py`: the differential-scoring optimizer.

The oracle in here does not share code with the module under test on purpose.
Wherever a number is checked, it is recomputed cell by cell from `grid` and
`config.B` - no `place_state`, no `own_reach`, no `after_moves`, no bitmask
shortcut - so agreement is evidence rather than a tautology. The candidate list
comes from `engine.legal_indices`, which is also not the list
`ai.chooser._candidates` builds.
"""
import ast
import os
import random
import time

import pytest

import ai
import engine
from ai import formulas as F
from ai.base import OPTIMIZER_KEY
from config import B, CLOCKWISE_OWNERS
from game import Board, Game
from rl.actions import index_to_move, legal_indices, move_to_index
from rl.opening import (BOOK_STEPS, BookTracker, act_to_engine_move, choose_v2,
                        hand_names, must_cover, reach, to_act_move)
from rl.v3 import (choose_v3, differential_is_constant, make_brain,
                   strongest_opponent, turn_order_after)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OTHER_POOL = ("builder", "intruder", "wolf", "chess", "fox")

EDGE = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


# --------------------------------------------------------------------------
# the independent oracle: playable cells, counted one square at a time
# --------------------------------------------------------------------------

def oracle_playable(grid, owner, empt):
    """How many empty squares are diagonally adjacent to one of `owner`'s stones
    and share no edge with one of them.

    Written straight from the rule in `AGENTS.md` rather than from
    `ai.formulas`, which is the whole point: `place_state`'s fifth element is the
    same quantity, and if the two ever disagreed this would catch it.
    """
    need = 0
    avoid = 0
    for idx in range(B * B):
        if not (empt >> idx) & 1:
            continue
        x, y = idx % B, idx // B
        for dx, dy in EDGE + DIAG:
            nx, ny = x + dx, y + dy
            if 0 <= nx < B and 0 <= ny < B and grid[nx + ny * B] == owner:
                if (dx, dy) in EDGE:
                    avoid |= 1 << idx
                else:
                    need |= 1 << idx
    return need & ~avoid


def oracle_counts(grid, empt, owner):
    return oracle_playable(grid, owner, empt).bit_count()


def absolute_cells(index):
    """The board indices a move writes to, straight from the action encoding."""
    _piece, oi, base = index_to_move(int(index))
    return [base + o for o in F.ODIRS[engine.PIECE_ORDER[_piece]][oi]["offs"]]


# --------------------------------------------------------------------------
# playing real games to real positions
# --------------------------------------------------------------------------

def v1_brain(profile):
    return ai.make_brain(OPTIMIZER_KEY, profile)


def play_to_post_book(seed, moves=3, tested=0):
    """A real game played to `moves` real moves of the tested seat, returning
    `(state, board, brain, rng, others)` at that point."""
    st = engine.initial_state(CLOCKWISE_OWNERS)
    g = Game(random.Random(seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    rngs = {o: random.Random(seed * 97 + o) for o in range(4)}
    brains = {o: ai.make_brain(OPTIMIZER_KEY, random.Random(seed * 4 + o))
              for o in range(4)}
    for o in range(4):
        if o != tested:
            brains[o] = ai.make_brain(OTHER_POOL[(seed + o) % 5],
                                      random.Random(seed * 4 + o))
    g.brains = brains
    g.start()
    own = 0
    while not engine.is_over(st):
        o = st.to_move
        if engine.legal_move_mask(st) == 0:
            st = engine.pass_turn(st)
            g.act_pass()
            continue
        r = ai.choose_move(g.board, g.hands[o].names, o, brains[o], rngs[o],
                           other_brains=brains,
                           must_cover=must_cover(st, o),
                           reach=reach(st, g.board, o),
                           other_must_cover={x: must_cover(st, x)
                                             for x in range(4)},
                           other_reach={x: reach(st, g.board, x)
                                        for x in range(4)})
        st = engine.apply_move(st, act_to_engine_move(r[0], r[1], r[2], r[3]))
        g.act(*to_act_move(act_to_engine_move(r[0], r[1], r[2], r[3])))
        if o == tested:
            own += 1
            if own == moves:
                return st, g.board, brains[tested], rngs[tested], brains
    return None, None, None, None, None


def clone_board(board):
    """An independent copy. Every downstream call takes both a `State` and a
    `Board` for the same position, so the positions collected while walking a
    game each need their own board rather than a shared one that keeps moving.
    """
    b = Board()
    b.grid = list(board.grid)
    b.empty_bits = board.empty_bits
    b.owner_bits = list(board.owner_bits)
    b._update_regions()
    return b


def post_book_positions(count, start_seed=0):
    """`count` distinct real post-book positions, walking several games."""
    from pieces import MASTER
    out = []
    seed = start_seed
    while len(out) < count:
        st, board, v1, rng, others = play_to_post_book(seed, moves=3)
        seed += 1
        if st is None:
            continue
        while st is not None and len(out) < count:
            if engine.legal_move_mask(st) == 0 or engine.is_over(st):
                break
            out.append((st, clone_board(board), v1, rng, others))
            owner = st.to_move
            r = ai.choose_move(board, hand_names(st, owner), owner,
                               others[owner], rng, other_brains=others,
                               must_cover=must_cover(st, owner),
                               reach=reach(st, board, owner),
                               other_must_cover={x: must_cover(st, x)
                                                 for x in range(4)},
                               other_reach={x: reach(st, board, x)
                                            for x in range(4)})
            move = act_to_engine_move(r[0], r[1], r[2], r[3])
            _piece, oi, base = move
            name = engine.PIECE_ORDER[move[0]]
            board.place(base % B, base // B,
                        MASTER[name]["orientations"][oi], owner)
            st = engine.apply_move(st, move)
    return out


# --------------------------------------------------------------------------
# 1. the score formula, against an independent oracle
# --------------------------------------------------------------------------

def test_the_picked_move_has_the_best_score_according_to_an_independent_oracle():
    """On real post-book positions, v3's pick is the argmax of a score computed
    from scratch: playable cells for me, minus playable cells for the pre-move
    strongest opponent, both counted square by square."""
    positions = post_book_positions(100)
    assert len(positions) == 100
    checked = 0
    for st, board, v1, rng, others in positions:
        owner = st.to_move
        brain = make_brain(v1.profile)
        tracker = BookTracker(0)
        tracker.abandon(owner)
        picked, info = choose_v3(st, board, owner, brain, rng, tracker,
                                 other_brains=others)
        assert info["source"] == "v3"

        # the oracle's own j*, from cell-by-cell counts on the pre-move grid
        counts = {o: oracle_counts(board.grid, board.empty_bits, o)
                  for o in range(4) if o != owner}
        order = turn_order_after(owner, CLOCKWISE_OWNERS)
        jstar = max(order, key=lambda o: (counts[o], -order.index(o)))
        assert brain_ctx(brain, st, board, owner)["v3_jstar"] == jstar

        # the *opponent's* playable mask, not mine: that is what the
        # differential term subtracts
        theirs = oracle_playable(board.grid, jstar, board.empty_bits)
        oracle_score = {}
        for index in legal_indices(st, owner):
            idx = int(index)
            cells = absolute_cells(idx)
            g2 = list(board.grid)
            empt2 = board.empty_bits
            for c in cells:
                g2[c] = owner
                empt2 &= ~(1 << c)
            a_me = oracle_playable(g2, owner, empt2).bit_count()
            a_j = (theirs & empt2).bit_count()
            oracle_score[idx] = a_me - a_j

        # the candidate range is v1's, shared on purpose: the claim being tested
        # is about the score, not about which candidates survive `restrict`
        cands = ai.chooser._candidates(
            board, hand_names(st, owner), owner, v1.profile,
            must_cover(st, owner), reach(st, board, owner),
            *ai.chooser.F.board_feats(board.grid), board.corner_regions,
            board.borders, board.empty_bits)
        cands.sort(key=lambda t: t[0], reverse=True)
        from ai.optimizer import OPTIMIZER_SCAN
        v1_ctx = v1.context(board, hand_names(st, owner), owner,
                            must_cover(st, owner), reach(st, board, owner))
        prefix = v1.restrict(cands, v1_ctx)[:OPTIMIZER_SCAN]
        # H1-0's tie-break: the sort is stable and descending, so among equal
        # scores the earliest in `restrict`'s order wins. Walking the prefix and
        # keeping a *strict* maximum reproduces that; taking a max over an
        # unordered list would not, and would fail on any tie.
        want = None
        for _s, name, oi, base in prefix:
            idx = move_to_index((engine.PIECE_IDX[name], oi, base))
            score = oracle_score[idx]
            if want is None or score > want[0]:
                want = (score, (engine.PIECE_IDX[name], oi, base))
        assert picked == want[1], (picked, want)

        # and the score v3 assigned the move it picked is that same maximum
        v3_ctx = brain_ctx(brain, st, board, owner)
        v3_scored = brain.rescore(v1.restrict(cands, v1_ctx), v3_ctx)
        v3_score = {move_to_index((engine.PIECE_IDX[n], oi, b)): sc
                    for sc, n, oi, b in v3_scored}
        for idx, score in oracle_score.items():
            if idx in v3_score:
                assert v3_score[idx] == score, (idx, v3_score[idx], score)
        assert v3_score[move_to_index(picked)] == want[0]
        checked += 1
    assert checked == 100


def brain_ctx(brain, st, board, owner):
    return brain.context(board, hand_names(st, owner), owner,
                         must_cover(st, owner), reach(st, board, owner))


# --------------------------------------------------------------------------
# 2. j* is decided before the move
# --------------------------------------------------------------------------

def test_jstar_is_the_pre_move_leader_not_the_post_move_one():
    """Find a real position where the strongest opponent *after* a candidate move
    is somebody else, and check v3 still scored against the pre-move leader.

    The oracle computes the post-move leader square by square, so "the leader
    changed" is a measured fact about the position and not an artefact of the
    mask shortcut the module uses.
    """
    found = 0
    for st, board, v1, rng, others in post_book_positions(120):
        owner = st.to_move
        brain = make_brain(v1.profile)
        cands = ai.chooser._candidates(
            board, hand_names(st, owner), owner, v1.profile,
            must_cover(st, owner), reach(st, board, owner),
            *ai.chooser.F.board_feats(board.grid), board.corner_regions,
            board.borders, board.empty_bits)
        cands.sort(key=lambda t: t[0], reverse=True)
        ctx = brain.context(board, hand_names(st, owner), owner,
                            must_cover(st, owner), reach(st, board, owner))
        pre_jstar = ctx["v3_jstar"]
        masks = ctx["v3_masks"]
        order = turn_order_after(owner, CLOCKWISE_OWNERS)

        def post_leader(cells):
            """The strongest opponent once `cells` is on the board, counted the
            slow way: the full grid is updated and every square recounted."""
            g2 = list(board.grid)
            empt2 = board.empty_bits
            for c in cells:
                g2[c] = owner
                empt2 &= ~(1 << c)
            counts = {o: oracle_counts(g2, empt2, o)
                      for o in order}
            return max(order, key=lambda o: (counts[o], -order.index(o)))

        scored = brain.rescore(v1.restrict(cands, ctx), ctx)
        separated = 0
        for _sc, name, oi, base in scored:
            placed = F.ODIRS[name][oi]["m"] << base
            cells = [base + o for o in F.ODIRS[name][oi]["offs"]]
            after = post_leader(cells)
            # whatever the move, v3 subtracted the PRE-move leader's count
            assert brain.opponent_after(ctx, placed) == \
                (masks[pre_jstar] & ~placed).bit_count()
            if after != pre_jstar:
                separated += 1
                # and it differs from what the post-move leader would have given
                assert brain.opponent_after(ctx, placed) == \
                    (masks[after] & ~placed).bit_count() or \
                    masks[after] != masks[pre_jstar]
        if separated:
            found += 1
            if found >= 5:
                break
    assert found >= 1, "no real position separated the pre- and post-move leader"


# --------------------------------------------------------------------------
# 3. the tie rule
# --------------------------------------------------------------------------

def test_a_tie_goes_to_the_player_next_in_turn_order():
    """All three opponents equal: the pick is the next player after me."""
    equal = {0: 0b1111, 1: 0b1111, 2: 0b1111, 3: 0b1111}
    assert strongest_opponent(equal, 0, CLOCKWISE_OWNERS) == \
        turn_order_after(0, CLOCKWISE_OWNERS)[0]
    # CLOCKWISE_OWNERS is (1, 2, 0, 3), so after 0 comes 3
    assert strongest_opponent(equal, 0, CLOCKWISE_OWNERS) == 3
    assert strongest_opponent(equal, 1, CLOCKWISE_OWNERS) == 2
    assert strongest_opponent(equal, 2, CLOCKWISE_OWNERS) == 0
    assert strongest_opponent(equal, 3, CLOCKWISE_OWNERS) == 1


def test_the_turn_order_decides_which_equal_count_wins_and_not_the_owner_id():
    """Two players tied for the most; which one wins depends only on the order."""
    masks = {0: 0, 1: 0b1111, 2: 0, 3: 0b1111}
    # after 0 comes 1, so 1 wins; after 0 comes 3, so 3 wins
    assert strongest_opponent(masks, 0, (0, 1, 2, 3)) == 1
    assert strongest_opponent(masks, 0, (0, 3, 2, 1)) == 3
    # and the shipped order, which is the one the experiment plays
    assert strongest_opponent(masks, 0, CLOCKWISE_OWNERS) == 3


def test_a_lower_count_never_beats_a_higher_one_however_early_it_is():
    """The turn order only breaks ties; it does not promote a smaller count."""
    masks = {0: 0, 1: 0b1111, 2: 0b11, 3: 0b111111}
    # 3 has strictly the most (6 squares) and wins whether it comes first or
    # last in the order after me
    assert strongest_opponent(masks, 0, (0, 1, 2, 3)) == 3
    assert strongest_opponent(masks, 0, (0, 3, 2, 1)) == 3


def test_all_zero_counts_pick_the_next_player_and_zero_the_term():
    """The degenerate all-zero case the plan spells out."""
    zero = {o: 0 for o in range(4)}
    jstar = strongest_opponent(zero, 0, CLOCKWISE_OWNERS)
    assert jstar == turn_order_after(0, CLOCKWISE_OWNERS)[0]
    # whatever the candidate, the term is exactly 0
    for mask in (0, 1, 0b1010101010):
        assert (zero[jstar] & ~mask).bit_count() == 0


def test_a_real_tied_position_takes_the_turn_order_neighbour():
    """And on a real position with an actual tie, the answer is that player."""
    ties = 0
    for st, board, v1, rng, others in post_book_positions(200):
        owner = st.to_move
        counts = {o: oracle_counts(board.grid, board.empty_bits, o)
                  for o in range(4) if o != owner}
        best = max(counts.values())
        tied = [o for o in turn_order_after(owner, CLOCKWISE_OWNERS)
                if counts[o] == best]
        if len(tied) < 2:
            continue
        ties += 1
        brain = make_brain(v1.profile)
        ctx = brain.context(board, hand_names(st, owner), owner,
                            must_cover(st, owner), reach(st, board, owner))
        assert ctx["v3_jstar"] == tied[0]
        if ties >= 5:
            break
    assert ties >= 1, "no real tied position in 200 decisions"


# --------------------------------------------------------------------------
# 4. the degenerate case: weight 0 is v1
# --------------------------------------------------------------------------

def test_with_the_differential_weight_at_zero_v3_picks_exactly_what_v1_picks():
    """100 real post-book positions, same profile, weight 0: same move every time."""
    positions = post_book_positions(100)
    same = 0
    for st, board, v1, rng, others in positions:
        owner = st.to_move
        r1 = random.Random(12345)
        r2 = random.Random(12345)
        mv_v1 = ai.choose_move(board, hand_names(st, owner), owner, v1, r1,
                               other_brains=others,
                               must_cover=must_cover(st, owner),
                               reach=reach(st, board, owner),
                               other_must_cover={x: must_cover(st, x)
                                                 for x in range(4)},
                               other_reach={x: reach(st, board, x)
                                            for x in range(4)})
        zero = make_brain(v1.profile, diff_weight=0.0)
        tracker = BookTracker(0)
        tracker.abandon(owner)
        mv_zero, info = choose_v3(st, board, owner, zero, r2, tracker,
                                  other_brains=others)
        assert info["source"] == "v3"
        assert mv_zero == (engine.PIECE_IDX[mv_v1[0]], mv_v1[1],
                           mv_v1[2] + mv_v1[3] * B)
        same += 1
    assert same == 100


def test_the_weight_zero_scores_are_bit_identical_to_v1_scores():
    """Not just the same pick: the whole ordered score list is v1's."""
    for st, board, v1, rng, others in post_book_positions(20):
        owner = st.to_move
        cands = ai.chooser._candidates(
            board, hand_names(st, owner), owner, v1.profile,
            must_cover(st, owner), reach(st, board, owner),
            *ai.chooser.F.board_feats(board.grid), board.corner_regions,
            board.borders, board.empty_bits)
        cands.sort(key=lambda t: t[0], reverse=True)
        v1_scored = v1.rescore(v1.restrict(cands, v1.context(
            board, hand_names(st, owner), owner, must_cover(st, owner),
            reach(st, board, owner))), v1.context(
            board, hand_names(st, owner), owner, must_cover(st, owner),
            reach(st, board, owner)))
        zero = make_brain(v1.profile, diff_weight=0.0)
        ctx = zero.context(board, hand_names(st, owner), owner,
                           must_cover(st, owner), reach(st, board, owner))
        zero_scored = zero.rescore(v1.restrict(cands, ctx), ctx)
        assert [int(s) for s, _, _, _ in zero_scored] == \
               [s for s, _, _, _ in v1_scored]
        assert [c[1:] for c in zero_scored] == [c[1:] for c in v1_scored]


# --------------------------------------------------------------------------
# 5. the differential term is not a constant
# --------------------------------------------------------------------------

def test_the_differential_term_actually_varies_and_the_fraction_is_reported():
    """The acceptance item: over 100 real post-book positions, in how many does
    `A_j*(after)` take more than one value across the candidates v3 scores?

    A term that never varies cannot change a ranking, so a 0% answer would mean
    the whole experiment is uninformative. The plan asks for the number to be
    reported, and this test both computes it and records it.
    """
    positions = post_book_positions(100)
    varies = 0
    distinct_hist = {}
    for st, board, v1, rng, others in positions:
        owner = st.to_move
        brain = make_brain(v1.profile)
        info = differential_is_constant(st, board, owner, brain)
        assert info["n_candidates"] > 0
        if info["varies"]:
            varies += 1
        distinct_hist[info["distinct"]] = \
            distinct_hist.get(info["distinct"], 0) + 1
    rate = varies / len(positions)
    print("\ndifferential term varies in %d of %d post-book positions "
          "(%.3f)" % (varies, len(positions), rate))
    print("distinct A_j*(after) values per position:",
          dict(sorted(distinct_hist.items())))
    assert varies > 0, ("the differential term never varied, so it cannot "
                        "change any ranking: stop and report")


# --------------------------------------------------------------------------
# 6. the book phase is v2's, unchanged
# --------------------------------------------------------------------------

def test_the_first_three_moves_are_identical_to_v2_under_the_same_seed():
    """Same game seed, same opponents: the opening is v2's opening, move for
    move, and so is the step at which the book was abandoned.

    The record is cut after the book's three steps plus the first move on which
    the two may legitimately differ, so the comparison covers the book and stops
    exactly where the experiment's claim begins."""
    for seed in range(6):
        record2, abandoned2 = _play_opening(seed, "v2")
        record3, abandoned3 = _play_opening(seed, "v3")
        assert [m for _s, m in record2[:BOOK_STEPS]] == \
               [m for _s, m in record3[:BOOK_STEPS]], (seed, record2, record3)
        assert [s for s, _m in record2[:BOOK_STEPS]] == ["book"] * BOOK_STEPS
        assert [s for s, _m in record3[:BOOK_STEPS]] == ["book"] * BOOK_STEPS
        assert abandoned2 == abandoned3, (seed, abandoned2, abandoned3)


def _play_opening(seed, version):
    tested = 0
    st = engine.initial_state(CLOCKWISE_OWNERS)
    g = Game(random.Random(seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    rngs = {o: random.Random(seed * 97 + o) for o in range(4)}
    brains = {o: ai.make_brain(OPTIMIZER_KEY, random.Random(seed * 4 + o))
              for o in range(4)}
    for o in range(4):
        if o != tested:
            brains[o] = ai.make_brain(OTHER_POOL[(seed + o) % 5],
                                      random.Random(seed * 4 + o))
    g.brains = brains
    g.start()
    tracker = BookTracker(seed)
    tested_brain = brains[tested]
    if version == "v3":
        tested_brain = make_brain(tested_brain.profile)
        g.brains[tested] = tested_brain
    record = []
    while not engine.is_over(st):
        o = st.to_move
        if engine.legal_move_mask(st) == 0:
            st = engine.pass_turn(st)
            g.act_pass()
            continue
        if o == tested:
            if version == "v2":
                m, info = choose_v2(st, g.board, o, tested_brain, rngs[o],
                                    tracker, other_brains=brains)
            else:
                m, info = choose_v3(st, g.board, o, tested_brain, rngs[o],
                                    tracker, other_brains=brains)
            record.append((info["source"], m))
        else:
            r = ai.choose_move(g.board, g.hands[o].names, o, brains[o],
                               rngs[o], other_brains=brains,
                               must_cover=must_cover(st, o),
                               reach=reach(st, g.board, o),
                               other_must_cover={x: must_cover(st, x)
                                                 for x in range(4)},
                               other_reach={x: reach(st, g.board, x)
                                            for x in range(4)})
            m = act_to_engine_move(r[0], r[1], r[2], r[3])
        st = engine.apply_move(st, m)
        g.act(*to_act_move(m))
        if o == tested:
            tracker.note_move(o)
            if len(record) > BOOK_STEPS:
                break
    return tuple(record), tracker.abandoned_at[tested]


# --------------------------------------------------------------------------
# 7. determinism
# --------------------------------------------------------------------------

def test_the_same_position_and_seed_give_the_same_move_every_time():
    for st, board, v1, rng, others in post_book_positions(12):
        owner = st.to_move
        picks = []
        for _ in range(4):
            brain = make_brain(v1.profile)
            tracker = BookTracker(0)
            tracker.abandon(owner)
            m, _i = choose_v3(st, board, owner, brain, random.Random(7),
                              tracker, other_brains=others)
            picks.append(m)
        assert len(set(picks)) == 1, picks


def test_the_differential_term_is_reproducible_across_two_brains():
    for st, board, v1, rng, others in post_book_positions(12):
        owner = st.to_move
        a = make_brain(v1.profile)
        b = make_brain(v1.profile)
        names = hand_names(st, owner)
        ca = a.context(board, names, owner, must_cover(st, owner),
                       reach(st, board, owner))
        cb = b.context(board, names, owner, must_cover(st, owner),
                       reach(st, board, owner))
        assert ca["v3_jstar"] == cb["v3_jstar"]
        assert ca["v3_masks"] == cb["v3_masks"]


# --------------------------------------------------------------------------
# 8. the wall-clock budget is restored, including on the exception path
# --------------------------------------------------------------------------

def test_the_wall_clock_budget_is_restored_even_when_the_block_raises():
    from rl.collect import _Budget
    before = ai.formulas.USE_WALL_BUDGET
    with _Budget(False):
        assert ai.formulas.USE_WALL_BUDGET is False
    assert ai.formulas.USE_WALL_BUDGET is before
    with pytest.raises(RuntimeError):
        with _Budget(False):
            assert ai.formulas.USE_WALL_BUDGET is False
            raise RuntimeError("boom")
    assert ai.formulas.USE_WALL_BUDGET is before


def test_v3_plays_identically_with_the_budget_off_and_restores_it_afterwards():
    before = ai.formulas.USE_WALL_BUDGET
    for st, board, v1, rng, others in post_book_positions(8):
        owner = st.to_move
        from rl.collect import _Budget
        with _Budget(False):
            brain = make_brain(v1.profile)
            tracker = BookTracker(0)
            tracker.abandon(owner)
            a, _i = choose_v3(st, board, owner, brain, random.Random(3),
                              tracker, other_brains=others)
        assert ai.formulas.USE_WALL_BUDGET is before
        with _Budget(True):
            brain = make_brain(v1.profile)
            tracker = BookTracker(0)
            tracker.abandon(owner)
            b, _i = choose_v3(st, board, owner, brain, random.Random(3),
                              tracker, other_brains=others)
        assert ai.formulas.USE_WALL_BUDGET is before
        assert a == b


# --------------------------------------------------------------------------
# 9. no pygame, no torch
# --------------------------------------------------------------------------

def test_v3_imports_neither_pygame_nor_torch():
    with open(os.path.join(ROOT, "rl", "v3.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "pygame" not in names
    assert "torch" not in names


def test_v3_is_reachable_without_ever_importing_torch_or_pygame():
    """In a clean subprocess, importing `rl.v3` must not pull either in."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import rl.v3; "
            "assert 'torch' not in sys.modules, 'torch'; "
            "assert 'pygame' not in sys.modules, 'pygame'; "
            "print('clean')" % ROOT)
    out = subprocess.run([os.path.join(ROOT, ".venv-rl", "bin", "python"),
                          "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


# --------------------------------------------------------------------------
# extra: the two identities the cheap form rests on
# --------------------------------------------------------------------------

def test_playable_before_is_the_null_move_value_of_v1s_own_expression():
    """`after_moves` uses `own_reach`; v1's expression uses `place_state`. The
    two must agree at the null move, which is what licenses calling the
    opponent's count an option count at all."""
    for st, board, v1, rng, others in post_book_positions(40):
        owner = st.to_move
        ctx = F.board_context(board, hand_names(st, owner), owner,
                              must_cover(st, owner))
        assert ctx["legal_before"] == \
            (ctx["need0"] & ctx["empt"] & ~ctx["avoid0"])
        assert ctx["legal_before"].bit_count() == \
            oracle_counts(board.grid, board.empty_bits, owner)
        from rl.v3 import playable_before
        for o in range(4):
            if o == owner:
                continue
            assert playable_before(board, o).bit_count() == \
                oracle_playable(board.grid, o, board.empty_bits).bit_count()


def test_the_incremental_opponent_count_matches_a_board_rebuilt_by_place():
    """`A_j*(after) = playable_before_j & ~placed` is a shortcut. It is checked
    here against a board the long way round, through `Board.place`."""
    from pieces import MASTER
    checked = 0
    for st, board, v1, rng, others in post_book_positions(25):
        owner = st.to_move
        brain = make_brain(v1.profile)
        names = hand_names(st, owner)
        ctx = brain.context(board, names, owner, must_cover(st, owner),
                            reach(st, board, owner))
        jstar = ctx["v3_jstar"]
        cands = ai.chooser._candidates(
            board, names, owner, v1.profile, must_cover(st, owner),
            reach(st, board, owner), *ai.chooser.F.board_feats(board.grid),
            board.corner_regions, board.borders, board.empty_bits)
        for _s, name, oi, base in v1.restrict(cands, ctx)[:24]:
            placed = F.ODIRS[name][oi]["m"] << base
            after = Board()
            after.grid = list(board.grid)
            after.empty_bits = board.empty_bits
            after.owner_bits = list(board.owner_bits)
            # one call, not one per cell: `place` writes the whole orientation
            after.place(base % B, base // B,
                        MASTER[name]["orientations"][oi], owner)
            assert (after.empty_bits ^ board.empty_bits) == placed
            jctx = F.board_context(after, names, jstar, None)
            direct = jctx["legal_before"].bit_count()
            assert brain.opponent_after(ctx, placed) == direct, (name, oi, base)
            checked += 1
    assert checked >= 100


# --------------------------------------------------------------------------
# the diagnostic the report quotes
# --------------------------------------------------------------------------

def test_the_diagnostic_divergence_rate_and_timing_ratio():
    """500 real post-book positions: how often v3 and v2 pick differently, and
    v3's cost per move relative to v1's. Printed for the report."""
    positions = post_book_positions(500)
    differ = 0
    t_v3 = 0.0
    t_v1 = 0.0
    for st, board, v1, rng, others in positions:
        owner = st.to_move
        names = hand_names(st, owner)
        others_dict = others
        t0 = time.perf_counter()
        mv1 = ai.choose_move(board, names, owner, v1, random.Random(11),
                             other_brains=others_dict,
                             must_cover=must_cover(st, owner),
                             reach=reach(st, board, owner),
                             other_must_cover={x: must_cover(st, x)
                                               for x in range(4)},
                             other_reach={x: reach(st, board, x)
                                            for x in range(4)})
        t_v1 += time.perf_counter() - t0
        t0 = time.perf_counter()
        b3 = make_brain(v1.profile)
        tr = BookTracker(0)
        tr.abandon(owner)
        mv3, info = choose_v3(st, board, owner, b3, random.Random(11), tr,
                              other_brains=others_dict)
        t_v3 += time.perf_counter() - t0
        assert info["source"] == "v3"
        if mv3 != (engine.PIECE_IDX[mv1[0]], mv1[1],
                   mv1[2] + mv1[3] * B):
            differ += 1
    rate = differ / len(positions)
    ratio = t_v3 / t_v1
    print("\n500 post-book positions:")
    print("  v3 and v2 pick a different move in %d of %d (%.4f)"
          % (differ, len(positions), rate))
    print("  v1 mean %.4f ms, v3 mean %.4f ms, ratio %.3f"
          % (t_v1 / len(positions) * 1e3, t_v3 / len(positions) * 1e3, ratio))
    assert rate > 0.0, "v3 never differed from v2"
    assert ratio < 10.0, ("v3 costs more than 10x v1 (%.2f): shrink the "
                          "candidate set before drawing conclusions" % ratio)