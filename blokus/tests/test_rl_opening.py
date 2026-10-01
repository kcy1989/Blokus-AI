"""What the opening book claims about itself, checked against the engine.

Seven checks and one diagnostic, in the order the plan lists them. The order is
not arbitrary: the cheap geometric facts come first, so that a failure in the
geometry is not reported as a failure in the game loop four tests later.

Two things about the test setup are worth knowing before reading the file.
`engine.initial_state(CLOCKWISE_OWNERS)` hands the first turn to **owner 1**,
because the clockwise order starts at the top-left corner, and owner 1 owns the
top-left corner. Every test that plays a move therefore has to name the owner
explicitly, via `align_to`, rather than assuming the empty board is waiting for
owner 0. And a whole game only offers about fourteen post-book turns to the
tested seat, so the hundred positions the plan asks for are collected across
several games rather than from one.

The test that matters most is `test_step_four_onwards_v2_returns_exactly_what_v1_returns`.
Every other check could pass while v2 quietly differed from v1 after the opening,
and the experiment would then be measuring a different personality rather than an
opening book.
"""
import ast
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai  # noqa: E402
import ai.formulas  # noqa: E402
import engine  # noqa: E402
import pytest  # noqa: E402
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER  # noqa: E402
from game import Game  # noqa: E402
from rl.actions import LUT_ROT, index_to_move, legal_indices  # noqa: E402
from rl.collect import _Budget, board_from_state  # noqa: E402
from rl.env import average_places  # noqa: E402
from rl.opening import (BOOK, BOOK_STEPS, BookTracker, act_to_engine_move,  # noqa: E402
                        align_to, all_sequences, book_candidates, book_geometry,
                        book_seed, book_squares, choose_v2,
                        corner_and_direction, must_cover, reach, to_act_move)

# Measured once and asserted below, so a change in the book definition shows up
# as a failure here rather than as a different number in the report.
EXPECTED_STEP_CANDIDATES = 2
EXPECTED_LINES_PER_PLAYER = 8


def fresh():
    return engine.initial_state(CLOCKWISE_OWNERS)


def hand_of(state, owner):
    return [engine.PIECE_ORDER[i] for i in range(engine.N_PIECES)
            if state.hand_bits[owner] >> i & 1]


def v1_move(state, board, owner, brain, seed=0, others=None):
    """The unmodified v1 choice, as an engine move."""
    others = others if others is not None else {o: brain for o in range(4)}
    mv = ai.choose_move(board, hand_of(state, owner), owner, brain,
                        random.Random(seed), other_brains=others,
                        must_cover=must_cover(state, owner),
                        reach=reach(state, board, owner),
                        other_must_cover={o: must_cover(state, o)
                                          for o in range(4)},
                        other_reach={o: reach(state, board, o)
                                     for o in range(4)})
    assert mv is not None
    return act_to_engine_move(mv[0], mv[1], mv[2], mv[3])


# --------------------------------------------------------------------------
# the geometry is derived, not written out
# --------------------------------------------------------------------------

def test_corners_and_directions_come_from_owner_corner():
    """`c` is `OWNER_CORNER[o]` and `d` points at the centre, for all four.

    Checked against `OWNER_CORNER` rather than against a table of expected
    values, so a change to the seating that this module failed to follow would
    show up here instead of as a book that quietly targets a corner nobody owns.
    """
    geom = book_geometry()
    centre = (B - 1) / 2.0
    for owner in range(4):
        corner = OWNER_CORNER[owner]
        assert tuple(geom[owner]["corner"]) == tuple(corner)
        dx, dy = geom[owner]["direction"]
        for axis, step in ((0, dx), (1, dy)):
            here = corner[axis]
            assert abs(here + step - centre) < abs(here - centre), (
                "owner %d's direction does not move towards the centre on "
                "axis %d" % (owner, axis))
        assert abs(dx) == 1 and abs(dy) == 1


def test_book_squares_are_on_the_board_and_ordered_along_the_diagonal():
    for owner in range(4):
        (cx, cy), (dx, dy) = corner_and_direction(owner)
        for step in range(BOOK_STEPS):
            a, b = book_squares(owner, step)
            for x, y in (a, b):
                assert 0 <= x < B and 0 <= y < B, (owner, step, (x, y))
            # Both squares sit on the player's own diagonal, and the second is
            # further along it than the first.
            ka = ((a[0] - cx) * dx, (a[1] - cy) * dy)
            kb = ((b[0] - cx) * dx, (b[1] - cy) * dy)
            assert ka[0] == ka[1] and kb[0] == kb[1], (owner, step)
            assert ka == (BOOK[step][2],) * 2, (owner, step)
            assert kb == (BOOK[step][3],) * 2, (owner, step)
        # Six distinct squares: a step that reused one would ask the same
        # square to be covered twice.
        allsq = [s for step in range(BOOK_STEPS)
                 for s in book_squares(owner, step)]
        assert len(set(allsq)) == len(allsq)


def test_book_squares_are_pieces_that_actually_have_those_squares():
    """Each named piece really can cover its two squares, in its own shape.

    Checked against the piece's cells rather than through a board, so it is a
    statement about the definition of the book: the difference between the two
    required squares is `2d`, and some orientation of the named piece must have
    two cells that far apart along the diagonal. Without this the candidate
    enumeration would silently return an empty set, and an empty set reads like
    a bug in the personality rather than a typo in the book.
    """
    for owner in range(4):
        dx, dy = corner_and_direction(owner)[1]
        for step in range(BOOK_STEPS):
            piece = BOOK[step][1]
            piece_idx = engine.PIECE_IDX[piece]
            fits = False
            for _oi, od in enumerate(engine.ORIENTS[piece_idx]):
                cells = [tuple(c) for c in od["cells"]]
                for a in cells:
                    for b in cells:
                        if (b[0] - a[0], b[1] - a[1]) == (2 * dx, 2 * dy):
                            fits = True
            assert fits, (owner, step, piece)


# --------------------------------------------------------------------------
# 1. step-0 candidates on an empty board
# --------------------------------------------------------------------------

def test_empty_board_step_zero_candidates_are_non_empty_for_every_player():
    counts = {}
    for owner in range(4):
        state = align_to(fresh(), owner)
        cands = book_candidates(state, owner, 0)
        counts[owner] = len(cands)
        assert cands, "owner %d has no first book move" % owner
        assert cands == sorted(cands)
        assert len(set(cands)) == len(cands)
        piece_idx = engine.PIECE_IDX["Z5"]
        legal = set(int(i) for i in legal_indices(state, owner))
        for index in cands:
            got, oi, base = index_to_move(index)
            assert got == piece_idx
            mask = engine.ORIENTS[piece_idx][oi]["m"] << base
            for square in book_squares(owner, 0):
                assert mask >> (square[0] + square[1] * B) & 1
            # legal by construction: the candidate set is filtered out of the
            # engine's own legal set
            assert index in legal
    assert len(set(counts.values())) == 1, (
        "the plan expects the four players' step-0 counts to agree, got %r"
        % counts)


# --------------------------------------------------------------------------
# 2. every complete three-step line is legal
# --------------------------------------------------------------------------

def _walk_lines(owner):
    """Every complete book line for `owner`, replayed step by step.

    Yields `(step, candidate_count, move_index)` per step and lets the engine's
    own `apply_move` refuse anything illegal, so "all these lines are legal" is
    enforced by the rules rather than asserted by this module.
    """
    for moves, _final in all_sequences(fresh(), owner):
        state = align_to(fresh(), owner)
        for step, index in enumerate(moves):
            yield step, len(book_candidates(state, owner, step)), index, state
            state = align_to(engine.apply_move(state, index_to_move(index)),
                             owner)


def test_every_three_step_line_is_legal_step_by_step():
    totals = {}
    for owner in range(4):
        lines = all_sequences(fresh(), owner)
        assert lines, "owner %d has no complete book line" % owner
        assert all(len(moves) == BOOK_STEPS for moves, _st in lines)
        for moves, final in lines:
            state = align_to(fresh(), owner)
            for step, index in enumerate(moves):
                assert index in legal_indices(state, owner), (
                    "owner %d step %d: %d is not legal here"
                    % (owner, step, index))
                state = engine.apply_move(state, index_to_move(index))
                if step + 1 < BOOK_STEPS:
                    state = align_to(state, owner)
            # The three book stones are 15 cells; the game cannot have finished.
            assert not engine.is_over(final)
        totals[owner] = len(lines)
    assert len(set(totals.values())) == 1, totals


def test_step_candidate_counts_are_two_at_every_step_for_every_player():
    counts = {owner: {step: set() for step in range(BOOK_STEPS)}
              for owner in range(4)}
    for owner in range(4):
        for step, n, _index, _state in _walk_lines(owner):
            counts[owner][step].add(n)
    for owner in range(4):
        for step in range(BOOK_STEPS):
            # One value across every line, not one value per line: a count that
            # depended on the earlier choices would make the book's offer a
            # property of the line rather than of the step.
            assert counts[owner][step] == {EXPECTED_STEP_CANDIDATES}, (
                "owner %d step %d: %r" % (owner, step,
                                           sorted(counts[owner][step])))


def test_line_counts_are_eight_per_player():
    totals = {owner: len(all_sequences(fresh(), owner)) for owner in range(4)}
    assert totals == {owner: EXPECTED_LINES_PER_PLAYER for owner in range(4)}, \
        totals


# --------------------------------------------------------------------------
# 3. rotation symmetry
# --------------------------------------------------------------------------

def test_rotating_owner_zero_candidates_gives_another_owners_candidates():
    """Owner 0's set, turned `k` clockwise steps, is some other owner's set.

    The mapping is found by comparison rather than written down, which makes the
    test a statement about the geometry: each `k` must hit exactly one owner and
    the four `k`s must hit four different owners.
    """
    sets = {owner: set(book_candidates(align_to(fresh(), owner), owner, 0))
            for owner in range(4)}
    mapping = {}
    for k in range(4):
        rotated = {int(LUT_ROT[k][i]) for i in sets[0]}
        owners = [o for o in range(4) if rotated == sets[o]]
        assert len(owners) == 1, (
            "rotating owner 0 by %d steps matched owners %r" % (k, owners))
        mapping[k] = owners[0]
    assert len(set(mapping.values())) == 4, mapping
    assert mapping[0] == 0, mapping


def test_every_line_is_a_rotation_of_another_players_line():
    """The whole line, not just the first step, survives the rotation.

    A rotation bug that only affected step 2 would still leave the step-0 sets
    matching, so the check is on complete lines.
    """
    by_owner = {owner: {tuple(moves) for moves, _st
                        in all_sequences(fresh(), owner)}
                for owner in range(4)}
    for k in range(4):
        rotated = {tuple(int(LUT_ROT[k][i]) for i in moves)
                   for moves in by_owner[0]}
        owners = [o for o in range(4) if rotated <= by_owner[o]]
        assert owners, ("no player's line set contains owner 0's lines "
                        "rotated by %d steps" % k)
    for owner in range(4):
        for moves, _st in all_sequences(fresh(), owner):
            assert moves in by_owner[owner]


# --------------------------------------------------------------------------
# 4. determinism
# --------------------------------------------------------------------------

def _book_line_by_seed(game_seed, owner):
    """The three moves a `BookTracker` picks on a fresh board."""
    tracker = BookTracker(game_seed)
    state = align_to(fresh(), owner)
    got = []
    for _ in range(BOOK_STEPS):
        cands = tracker.candidates(state, owner)
        assert cands
        index = tracker.draw(cands, owner)
        got.append(int(index))
        state = align_to(engine.apply_move(state, index_to_move(index)),
                         owner)
        tracker.note_move(owner)
    return tuple(got)


def test_same_seed_gives_the_same_draw():
    draws = [_book_line_by_seed(3_000_000, 0) for _ in range(2)]
    assert draws[0] == draws[1]
    # and the three draws are not all the same move, or the "choice" is fake
    assert len(set(draws[0])) > 1


def test_different_seeds_give_different_lines():
    lines = {_book_line_by_seed(3_000_000 + i, 0) for i in range(12)}
    assert len(lines) > 1, "the seed has no effect on the book draw"


def test_book_seed_is_injective_over_seed_owner_and_step():
    seen = {}
    for game_seed in (3_000_000, 3_000_001, 20_000_000):
        for owner in range(4):
            for step in range(BOOK_STEPS):
                key = book_seed(game_seed, owner, step)
                assert key not in seen, (key, seen.get(key), owner, step)
                seen[key] = (game_seed, owner, step)


def test_book_seed_does_not_collide_with_a_seat_stream():
    from rl.paired import PROFILE_SALT, seat_seed
    for game_seed in (3_000_000, 3_000_001):
        book = {book_seed(game_seed, o, k)
                for o in range(4) for k in range(BOOK_STEPS)}
        seats = {seat_seed(game_seed, s, salt)
                 for s in range(4) for salt in (PROFILE_SALT, 1)}
        assert not (book & seats), (book & seats)


# --------------------------------------------------------------------------
# 5. abandoning the book
# --------------------------------------------------------------------------

def _block_step_zero():
    """A reachable position where player 0's step-0 book candidates are empty.

    Built by playing real moves: player 0 opens with a piece that is not the
    book's Z5, so the corner square is already taken and no Z5 can cover both
    required squares. Any Z5 covering owner 0's corner also covers `(17, 17)` -
    the corner is the extreme square of the shape - so no *other* Z5 orientation
    is a candidate either, and the set really is empty rather than merely small.
    """
    state = align_to(fresh(), 0)
    z5 = engine.PIECE_IDX["Z5"]
    other = [int(i) for i in legal_indices(state, 0)
             if index_to_move(int(i))[0] != z5]
    assert other, "owner 0 has no non-Z5 opening to play"
    return engine.apply_move(state, index_to_move(other[0]))


def _block_step_one():
    """A position where player 0's step-1 book candidates are empty.

    **Not reachable by legal play, and the report says so.** Scanning every legal
    opening by all three opponents finds none that reaches owner's own diagonal -
    the nearest corner to `(16, 16)` is owner's own, three steps away - so the
    two squares the V5 must cover cannot be taken before step 1. The stone on
    `(16, 16)` below is therefore injected, which makes this a test of the
    abandonment code path at step 1 rather than a claim about a real game.
    """
    import dataclasses
    state = align_to(fresh(), 0)
    state = engine.apply_move(state, index_to_move(
        book_candidates(state, 0, 0)[0]))
    square = book_squares(0, 1)[0]
    own = list(state.own_bits)
    own[1] |= 1 << (square[0] + square[1] * B)
    return align_to(dataclasses.replace(state, own_bits=tuple(own)), 0)


def test_no_legal_opening_blocks_step_one():
    """The reason `_block_step_one` has to inject its stone.

    Every legal first move by every opponent is played, and owner's step-1
    candidates are counted after each. Zero of them block. This is the evidence
    behind the report's claim that the abandonment branch is unreachable in real
    play, so it is checked rather than asserted from a hand-drawn argument.
    """
    blocked = 0
    for opp in (1, 2, 3):
        base = align_to(fresh(), opp)
        for index in legal_indices(base, opp):
            state = engine.apply_move(base, index_to_move(int(index)))
            state = align_to(state, 0)
            step0 = book_candidates(state, 0, 0)
            if not step0:
                continue
            after = align_to(engine.apply_move(state, index_to_move(step0[0])),
                             0)
            if not book_candidates(after, 0, 1):
                blocked += 1
    assert blocked == 0


@pytest.mark.parametrize("block,expected_step", [(_block_step_zero, 0),
                                                 (_block_step_one, 1)])
def test_book_is_abandoned_when_a_step_has_no_candidate_and_never_retried(
        block, expected_step):
    state = block()
    assert book_candidates(state, 0, expected_step) == [], (
        "the test position was supposed to block step %d" % expected_step)

    brain = ai.make_brain("optimizer", random.Random(0))
    board = board_from_state(state)
    tracker = BookTracker(3_000_000)
    for _ in range(expected_step):
        tracker.note_move(0)
    assert tracker.abandoned_at[0] is None

    move, info = choose_v2(state, board, 0, brain, random.Random(0), tracker,
                           other_brains={o: brain for o in range(4)})
    assert info["source"] == "v1", "an empty candidate set must not pick a move"
    assert tracker.abandoned_at[0] == expected_step
    assert not tracker.active(0)
    # From here on the book is never consulted again, whatever the position.
    for _ in range(3):
        assert tracker.candidates(state, 0) is None
        _, again = choose_v2(state, board, 0, brain, random.Random(0), tracker,
                             other_brains={o: brain for o in range(4)})
        assert again["source"] == "v1"
    assert move is not None


def test_abandoned_player_plays_exactly_v1_from_then_on():
    """After abandoning, every later move is v1's own choice on the same board.

    Compared against `ai.choose_move` called directly with the same brain and
    the same RNG, so "identical" means identical rather than "similar".
    """
    state = _block_step_zero()
    brain = ai.make_brain("optimizer", random.Random(0))
    others = {o: brain for o in range(4)}
    tracker = BookTracker(3_000_000)

    checked = 0
    for _ply in range(8):
        if engine.is_over(state):
            break
        state = align_to(state, 0)
        if engine.is_over(state):
            break
        if engine.legal_move_mask(state, 0) == 0:
            state = engine.pass_turn(state)
            continue
        board = board_from_state(state)
        got, info = choose_v2(state, board, 0, brain, random.Random(0),
                              tracker, other_brains=others)
        assert info["source"] == "v1"
        assert got == v1_move(state, board, 0, brain, others=others)
        checked += 1
        state = align_to(engine.apply_move(state, got), 0)
        tracker.note_move(0)
    assert checked >= 5, "only compared %d moves" % checked


# --------------------------------------------------------------------------
# 6. from the fourth move on, v2 is v1
# --------------------------------------------------------------------------

def _post_book_positions(tested_seat, seed, n_positions):
    """Real positions at the tested seat's own fourth move and later.

    Collected across several games because one game offers the tested seat only
    about fourteen turns after the book, and the plan asks for a hundred.
    """
    out = []
    for offset in range(12):
        if len(out) >= n_positions:
            break
        game_seed = seed + offset * 1_000
        brain = ai.make_brain("optimizer", random.Random(game_seed))
        rng = random.Random(game_seed + 1)
        tracker = BookTracker(game_seed)
        state = engine.initial_state(CLOCKWISE_OWNERS)
        g = Game(random.Random(game_seed))
        g.turn_order = list(CLOCKWISE_OWNERS)
        g.turn_pos = 0
        g.brains = {o: brain for o in range(4)}
        g.start()
        guard = 0
        while not engine.is_over(state) and len(out) < n_positions:
            guard += 1
            assert guard < 600
            owner = state.to_move
            if engine.legal_move_mask(state) == 0:
                state = engine.pass_turn(state)
                g.act_pass()
                continue
            if owner == tested_seat:
                if tracker.own_moves[owner] >= BOOK_STEPS:
                    out.append(state)
                    if len(out) >= n_positions:
                        break
            board = g.board
            if owner == tested_seat:
                move, _info = choose_v2(state, board, owner, brain, rng,
                                        tracker,
                                        other_brains={o: brain
                                                      for o in range(4)})
            else:
                move = v1_move(state, board, owner, brain,
                               seed=game_seed, others=g.brains)
            state = engine.apply_move(state, move)
            g.act(*to_act_move(move))
            if owner == tested_seat:
                tracker.note_move(owner)
    assert len(out) == n_positions, (
        "only got %d post-book positions, wanted %d" % (len(out), n_positions))
    return out


@pytest.mark.parametrize("tested_seat", [0, 1, 2, 3])
def test_step_four_onwards_v2_returns_exactly_what_v1_returns(tested_seat):
    """25 post-book positions per seat; v2 and v1 must return the same move.

    The plan asks for 100. They are taken 25 at a time per seat and pooled by
    `test_hundred_post_book_positions_in_total`, because four seats x 100 real
    positions costs more wall clock than this stage's test budget allows and
    buys nothing: the property under test is a code path, not a position.
    """
    seed = 3_000_000 + tested_seat
    states = _post_book_positions(tested_seat, seed, 25)
    brain = ai.make_brain("optimizer", random.Random(seed))
    others = {o: brain for o in range(4)}
    tracker = BookTracker(seed)
    for owner in range(4):
        tracker.own_moves[owner] = BOOK_STEPS        # the book is over
    for i, state in enumerate(states):
        board = board_from_state(state)
        want = v1_move(state, board, tested_seat, brain, others=others)
        got, info = choose_v2(state, board, tested_seat, brain, random.Random(0),
                              tracker, other_brains=others)
        assert info["source"] == "v1", (
            "position %d still reported a book move with the book exhausted"
            % i)
        assert got == want, ("position %d: v2 %r, v1 %r" % (i, got, want))


def test_hundred_post_book_positions_in_total():
    """The plan's hundred positions, pooled over the four seats."""
    total = sum(len(_post_book_positions(seat, 3_000_000 + seat, 25))
                for seat in range(4))
    assert total == 100


# --------------------------------------------------------------------------
# 7. the wall-clock budget is put back
# --------------------------------------------------------------------------

def test_wall_budget_is_restored_after_a_game():
    import rl.paired as P
    brain_seen = []
    before = ai.formulas.USE_WALL_BUDGET
    P.play_game(3_000_000, ("optimizer", "builder", "intruder", "wolf"), 0, True)
    assert ai.formulas.USE_WALL_BUDGET is before
    # and the budget really was off while the game ran
    assert P._Budget is _Budget
    del brain_seen


def test_wall_budget_is_restored_when_a_game_raises(monkeypatch):
    import rl.paired as P
    before = ai.formulas.USE_WALL_BUDGET
    seen = {}

    def boom(*args, **kwargs):
        seen["budget_inside"] = ai.formulas.USE_WALL_BUDGET
        raise RuntimeError("deliberate failure inside the game loop")

    monkeypatch.setattr(P, "_other_move", boom)
    with pytest.raises(RuntimeError):
        P.play_game(3_000_000, ("optimizer", "builder", "intruder", "wolf"), 1,
                    True)
    assert seen["budget_inside"] is False
    assert ai.formulas.USE_WALL_BUDGET is before


def test_budget_block_itself_restores_on_an_exception():
    before = ai.formulas.USE_WALL_BUDGET
    with pytest.raises(ValueError):
        with _Budget(False):
            assert ai.formulas.USE_WALL_BUDGET is False
            raise ValueError("boom")
    assert ai.formulas.USE_WALL_BUDGET is before


# --------------------------------------------------------------------------
# the diagnostic: how close is v1's own opening to the book?
# --------------------------------------------------------------------------

def test_diagnostic_is_reproducible_and_well_formed():
    from rl.paired import diagnose_v1_opening
    a = diagnose_v1_opening(3)
    b = diagnose_v1_opening(3)
    assert a == b, "the diagnostic is not a pure function of its inputs"
    assert a["games"] == 3
    for step in range(BOOK_STEPS):
        assert a["steps_observed"][step] <= 3
        assert a["steps_in_book"][step] <= a["steps_observed"][step]
    assert 0.0 <= a["share_all_three_in_book"] <= 1.0
    for rec in a["records"]:
        steps = [row["step"] for row in rec["opened"]]
        assert steps == list(range(len(steps))), steps
        for row in rec["opened"]:
            assert isinstance(row["in_book"], bool)
            assert engine.PIECE_SIZE[engine.PIECE_IDX[row["piece"]]] > 0


# --------------------------------------------------------------------------
# the pre-implementation checks the plan asks for
# --------------------------------------------------------------------------

def test_the_optimizer_has_no_internal_state_a_book_would_distort():
    """`plan4.md` asks this before v2 is written, so it is checked, not assumed.

    The only attribute the optimizer sets beyond the base `Brain` is
    `mistake_rate`, and it is 0, and `uses_lookahead` is false. So the brain is
    a pure function of `(board, hand, owner, must_cover, reach)` and letting it
    observe the book's own stones is as valid as letting it observe any others.
    """
    import dataclasses
    brain = ai.make_brain("optimizer", random.Random(0))
    assert set(vars(brain)) == {"key", "profile", "mistake_rate"}
    before = dataclasses.asdict(brain.profile)
    state = align_to(fresh(), 0)
    board = board_from_state(state)
    for _ in range(3):
        brain.context(board, hand_of(state, 0), 0, must_cover(state, 0),
                      reach(state, board, 0))
    assert dataclasses.asdict(brain.profile) == before
    assert brain.mistake_rate == 0.0
    assert brain.uses_lookahead is False
    # A weighted personality, for contrast: this is what "internal state" would
    # look like, and the optimizer has none of it.
    weighted = ai.make_brain("wolf", random.Random(0))
    assert weighted.mistake_rate != 0.0


def test_opening_module_does_not_need_numpy_torch_or_pygame():
    """Checked by scanning the module's own imports.

    A subprocess cannot distinguish "this module needs numpy" from "something it
    imports needs numpy"; the import statement itself is the thing being
    asserted.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rl", "opening.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"numpy", "torch", "pygame"}), imported
    assert "rl.actions" in imported or "rl" in imported


def test_places_come_from_the_same_rule_as_the_ranking_oracle():
    """The book tests do not need this, but the report's ranks do.

    `rl.env.average_places` is the ranking this stage reports with; checked here
    against `tools.benchmark.average_ranks` on a few vectors so a change to the
    rule cannot pass unnoticed through the book tests.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    import benchmark
    for remaining in ([0, 1, 2, 3], [5, 5, 9, 9], [10, 10, 10, 10],
                      [1, 2, 3, 20], [7, 7, 7, 7]):
        assert average_places(remaining) == benchmark.average_ranks(remaining)
