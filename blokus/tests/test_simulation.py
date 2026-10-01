"""Headless simulation: 100 all-AI games plus end/draw rules."""
import random

import pytest

import ai
from ai.hunter import HunterBrain
from game import Game
from board import Board, neighbors_of
from pieces import MASTER
from config import (CLOCKWISE_OWNERS, COLORS, OWNER_CORNER, PLAYER_OWNER)

# What Hunter's first three moves are supposed to be. Written out here rather than
# read back from `ai.hunter.BOOK`, because a check that compares the book against
# itself only proves the book is consistent with the book. This is the statement
# of what the opening is meant to be, and it is the thing a regression would break.
HUNTER_BOOK_PIECES = ("Z5", "V5", "W5")
HUNTER_BOOK_CANDIDATES = 2


class HunterBookTrace:
    """Counters for the Hunter-specific checks, so a test can assert that they
    actually ran rather than passing vacuously."""

    def __init__(self):
        self.moves = 0
        self.book_moves = 0
        self.in_book = 0
        self.abandoned = 0
        self.post_book = 0


def check_hunter_book(g, owner, move, trace):
    """Hunter's book checks, shared by `test_100_games` and
    `test_hunter_seat_games` so the two cannot drift apart.

    Called with the board as it stood **before** `g.act`, so `g.placed[owner]` is
    the step this move is about to play. Three things are checked for a book step:
    the piece is the one the book prescribes, the move is one of the book's
    candidates, and the candidate set really does hold the expected number of
    them. The identity is checked first so a wrong piece is reported as a wrong
    piece rather than as a membership failure.

    A book that has been **abandoned** is allowed to stop matching: the rule is
    that the player then scores by the differential rule instead, and
    `test_the_abandoned_book_falls_back_to_a_legal_move` covers that branch.
    `tests/test_hunter.py` reaches the same rule with a constructed position.

    Returns what it decided, which the tests use to check coverage.
    """
    trace.moves += 1
    brain = g.brains[owner]
    if brain.book_abandoned:
        trace.abandoned += 1
        return "abandoned"
    step = g.placed[owner]
    if step >= len(HUNTER_BOOK_PIECES):
        trace.post_book += 1
        return "post_book"
    trace.book_moves += 1
    name, oi, x, y = move
    where = ("hunter %s oi=%d at (%d,%d) for owner %d on book step %d"
             % (name, oi, x, y, owner, step))
    assert name == HUNTER_BOOK_PIECES[step], \
        where + " is not the book's piece"
    candidates = ai.book_candidates(g.board, owner, step)
    assert len(candidates) == HUNTER_BOOK_CANDIDATES, \
        (where, "the book offered %d candidates" % len(candidates), candidates)
    base = x + y * 20
    assert any(n == name and o == oi and b == base
               for n, o, b in candidates), \
        (where, "not one of the book's candidates", candidates)
    trace.in_book += 1
    return "book"


def run_game(rng, on_move=None):
    g = Game(rng)
    g.set_player_color(rng.choice(list(COLORS)))
    g.start()
    while g.state == "PLAYING":
        owner = g.current_owner()
        # snapshot before the move: a piece must never be judged against its
        # own cells, only against stones that were already on the board
        before = {i for i in range(400) if g.board.grid[i] == owner}
        move = ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], rng, other_brains=g.brains,
                              must_cover=g.must_cover(owner),
                              reach=g.reach(owner),
                              other_must_cover={o: g.must_cover(o) for o in range(4)},
                              other_reach={o: g.reach(o) for o in range(4)})
        if move:
            if on_move is not None:
                on_move(g, owner, move, before)
            g.act(*move)
        else:
            g.act_pass()
    return g


def test_turn_order_is_clockwise_with_a_random_start():
    """Rotation must go clockwise around the four corners: top-left ->
    top-right -> bottom-right -> bottom-left.

    Only the starting point is random (whoever wins the right to move first),
    not the whole arrangement; with a shuffle, two adjacent moves sometimes
    land on diagonally opposite corners.
    """
    assert CLOCKWISE_OWNERS == (1, 2, 0, 3), CLOCKWISE_OWNERS
    corners = [OWNER_CORNER[o] for o in CLOCKWISE_OWNERS]
    assert corners == [(0, 0), (19, 0), (19, 19), (0, 19)], corners
    first_seen = set()
    for seed in range(200):
        g = Game(random.Random(seed))
        g.set_player_color("blue")
        order = g.turn_order
        assert sorted(order) == [0, 1, 2, 3], order
        assert order.count(PLAYER_OWNER) == 1, order
        idx = [CLOCKWISE_OWNERS.index(o) for o in order]
        for i in range(3):
            assert idx[i + 1] == (idx[i] + 1) % 4, order
        first_seen.add(order.index(PLAYER_OWNER))
        # the rotation must really follow that order, wrapping from the last
        # slot back to the first
        g.start()
        walked = [g.current_owner()]
        for _ in range(3):
            g.act_pass()
            walked.append(g.current_owner())
        assert walked == order, (walked, order)
    # all four player positions are reachable, i.e. sometimes first, sometimes
    # second
    assert first_seen == {0, 1, 2, 3}, first_seen


def test_100_games():
    rng = random.Random(42)
    total_turns = 0
    winners = []
    checked = [0]
    intruder_moves = [0, 0, 0, 0]
    hunter_trace = HunterBookTrace()
    hunter_moves = []
    keys_seen = set()
    neigh = neighbors_of()

    def check(g, owner, move, before):
        """Every stone after the opening must meet one of that owner's earlier
        stones at a corner, and must not share an edge with any of them."""
        if g.owner_key[owner] == "intruder":
            n = intruder_moves[owner]
            intruder_moves[owner] += 1
            # rule 1: use up the crossing pieces before switching to others
            if n < 3:
                assert move[0] in ai.LEAPERS, \
                    "intruder move %d was %s, expected a leaper" % (n, move)
        if g.owner_key[owner] == "optimizer" and g.must_cover(owner) is None:
            # with no urgent piece it always plays the biggest piece that fits
            counts = ai.placement_counts(g.board, g.hands[owner].names, owner, None)
            if not [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES]:
                biggest = max(MASTER[n]["size"] for n, k in counts.items() if k > 0)
                assert MASTER[move[0]]["size"] == biggest, (move, biggest, counts)
        if g.owner_key[owner] == "hunter":
            # the generic checks below only say the move was legal and followed
            # the corner-contact rule; nothing ties the first three to the book
            hunter_moves.append(move)
            check_hunter_book(g, owner, move, hunter_trace)
        if g.placed[owner] == 0:
            checked[0] += 1
            return
        name, oi, x, y = move
        cells = {(x + dx) + (y + dy) * 20
                 for dx, dy in MASTER[name]["orientations"][oi]}
        where = "%s oi=%d at (%d,%d) for owner %d" % (name, oi, x, y, owner)
        for i in cells:
            for n in neigh[i]:
                assert n not in before, where + " shares an edge with its own stone"
        before_xy = {(i % 20, i // 20) for i in before}
        assert any((i % 20 + dx, i // 20 + dy) in before_xy
                   for i in cells for dx in (-1, 1) for dy in (-1, 1)), \
            where + " has no corner contact"

    for i in range(100):
        g = run_game(rng, check)
        keys_seen |= {k for k in g.owner_key.values() if k}
        assert g.state == "GAME_OVER"
        assert g.turn_count < 300, g.turn_count
        total_turns += g.turn_count
        occupied = sum(1 for cell in g.board.grid if cell != -1)
        expected = sum(89 - sum(MASTER[n]["size"] for n in g.hands[o].names)
                       for o in range(4))
        assert occupied == expected, (occupied, expected)
        assert bin(g.board.empty_bits).count("1") == 400 - occupied
        # every player opened on their own corner square
        for owner in range(4):
            cx, cy = OWNER_CORNER[owner]
            assert g.board.grid[cx + cy * 20] == owner, (owner, "missed own corner")
        # The no-shared-edge rule cannot be re-checked on the final grid: a
        # single multi-cell piece is edge-adjacent to itself, and pieces are
        # not distinguishable once placed. It is checked per move above, where
        # the board as it stood before the move is still available.
        standings = g.standings()
        rem = [g.remaining_cells(o) for o in range(4)]
        # the score is the squares left unplaced, so fewer is better
        assert standings[0][1] == min(rem)
        assert [r for _, r in standings] == sorted(rem)
        # and it must agree with the hand accounting
        for o in range(4):
            assert rem[o] == sum(MASTER[n]["size"] for n in g.hands[o].names)
        # the game may only end once every one of the four is stuck
        assert not g.has_legal(0)
        for o in (1, 2, 3):
            assert not g.has_legal(o)
        winner = g.winner()
        assert winner in ("player", "draw", "1", "2", "3")
        best = min(rem)
        top = [j for j in range(4) if rem[j] == best]
        if winner == "player":
            assert rem[0] == best and len(top) == 1
        elif winner == "draw":
            assert 0 in top and len(top) > 1
        else:
            assert rem[int(winner)] == best and 0 not in top
        winners.append(winner)
    assert checked[0] == 400, "expected 400 opening moves, saw %d" % checked[0]
    # 100 games of 3-of-4 sampling means all four personalities should have
    # appeared; the games counts for wolf / chess / fox naturally differ from
    # game to game, and the leaderboard still totals up fine.
    assert keys_seen == set(ai.personality_keys()), keys_seen
    assert any(intruder_moves), "no game drew the intruder"
    # Hunter is a 3-of-4 draw like everyone else, so it appears in roughly half
    # the games; without this the new checks could pass by never running.
    assert any(hunter_moves), "no game drew the hunter"
    # and every book step it took was in the book's set. Deliberately not an
    # equality against 3 x games: a game that abandoned the book would play fewer
    # book steps and that is allowed, so counting them exactly would turn a
    # legitimate behaviour into a failure.
    assert hunter_trace.book_moves == hunter_trace.in_book, hunter_trace.__dict__
    assert hunter_trace.book_moves, hunter_trace.__dict__
    counts = {}
    for w in winners:
        counts[w] = counts.get(w, 0) + 1
    print("avg turns: %.1f winners=%s" % (total_turns / 100.0, counts))


HUNTER_SEAT_GAMES = 8


def _play_hunter_game(rng, seat, on_move, game_seed=None):
    """One game with Hunter pinned to `seat`, played with the same rules as
    `run_game`.

    `setup_match` rather than `set_player_color`, and that is the whole point:
    `set_player_color` hard-codes seat 0 to chess, so a Hunter test built on it
    can never cover seat 0 at all. `game_seed` is passed to the Hunter
    constructor when given, which is the second of the two ways Hunter gets its
    opening seed; left as None, the constructor derives one from the rng, which
    is the first and is what a normal game does.
    """
    g = Game(rng)
    others = [k for k in ai.personality_keys() if k != "hunter"][:3]
    keys = list(others)
    keys.insert(seat, "hunter")
    g.setup_match(keys)
    g.start()
    if game_seed is not None:
        g.brains[seat] = HunterBrain("hunter", g.brains[seat].profile,
                                     game_seed=game_seed)
    while g.state == "PLAYING":
        owner = g.current_owner()
        before = {i for i in range(400) if g.board.grid[i] == owner}
        move = ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], rng, other_brains=g.brains,
                              must_cover=g.must_cover(owner),
                              reach=g.reach(owner),
                              other_must_cover={o: g.must_cover(o)
                                                for o in range(4)},
                              other_reach={o: g.reach(o) for o in range(4)})
        if move:
            if on_move is not None:
                on_move(g, owner, move, before)
            g.act(*move)
        else:
            g.act_pass()
    return g


@pytest.mark.parametrize("seat", [0, 1, 2, 3])
@pytest.mark.parametrize("seed_source", ["derived", "injected"])
def test_hunter_seat_games(seat, seed_source):
    """Hunter at a given seat, `HUNTER_SEAT_GAMES` games, both ways of getting a
    seed, with the same book check `test_100_games` uses.

    4 seats x 2 seed sources x 8 games = 64 games. The seat-0 cases are the ones
    `test_100_games` structurally cannot produce, and the two seed sources are the
    two branches of Hunter's constructor.
    """
    trace = HunterBookTrace()
    seen = []
    rng = random.Random(1000 + seat)
    for n in range(HUNTER_SEAT_GAMES):
        def check(g, owner, move, before, _trace=trace):
            if g.owner_key[owner] == "hunter":
                seen.append(move)
                check_hunter_book(g, owner, move, _trace)
        g = _play_hunter_game(rng, seat, check,
                              game_seed=4242 if seed_source == "injected"
                              else None)
        assert g.state == "GAME_OVER"
    assert g.owner_key[seat] == "hunter"
    assert len(seen) >= 3 * HUNTER_SEAT_GAMES, len(seen)
    assert trace.book_moves, (seat, seed_source, trace.__dict__)
    assert trace.book_moves == trace.in_book, (seat, seed_source,
                                              trace.__dict__)
    assert trace.post_book, (seat, seed_source, trace.__dict__)
    if seed_source == "injected":
        assert g.brains[seat].game_seed_source == "argument"
    else:
        assert g.brains[seat].game_seed_source == "rng"


def _pick_subset(names, target):
    """Pieces from `names` whose sizes sum to exactly `target`."""
    sizes = [MASTER[n]["size"] for n in names]
    # reach[i] = (prev_index, took) for hitting sum i, built back to front
    table = [None] * (target + 1)
    table[0] = (-1, False)
    for j in range(len(names) - 1, -1, -1):
        s = sizes[j]
        for t in range(target, s - 1, -1):
            if table[t] is None and table[t - s] is not None:
                table[t] = (j, True)
    if table[target] is None:
        raise AssertionError("cannot make %d from %r" % (target, sizes))
    out, t = [], target
    while t:
        j, took = table[t]
        assert took, "broken subset-sum chain"
        out.append(names[j])
        t -= sizes[j]
    return out


def _finished_game(remaining):
    """A GAME_OVER game with a prescribed number of unplaced squares each."""
    g = Game(random.Random(0))
    g.state = "GAME_OVER"
    for o in range(4):
        for n in _pick_subset(list(MASTER), 89 - remaining[o]):
            g.hands[o].names.remove(n)
        assert g.remaining_cells(o) == remaining[o], (o, g.remaining_cells(o))
    return g


def test_tie_declaration():
    g = _finished_game([10, 10, 10, 10])
    assert g.winner() == "draw", g.winner()


def test_player_wins_declaration():
    g = _finished_game([4, 20, 30, 40])
    assert g.winner() == "player", g.winner()


def test_ai_wins_declaration():
    g = _finished_game([40, 30, 20, 4])
    assert g.winner() == "3", g.winner()


def test_fewer_remaining_wins():
    """The score is the squares left unplaced, so more remaining is worse."""
    g = _finished_game([0, 5, 10, 15])
    assert g.standings() == [(0, 0), (1, 5), (2, 10), (3, 15)]
    assert g.winner() == "player"
    g = _finished_game([15, 10, 5, 0])
    assert g.standings() == [(3, 0), (2, 5), (1, 10), (0, 15)]
    assert g.winner() == "3"


def test_full_hand_scores_zero():
    g = _finished_game([0, 0, 0, 0])
    assert g.winner() == "draw"
    assert all(r == 0 for _, r in g.standings())


def test_game_ends_only_when_all_four_are_stuck():
    """The player being stuck must not end the game on its own."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    g = Game(random.Random(0))
    g.start()
    # wall the player into its own corner: the player has no legal move, but
    # the three AI corners are still wide open
    g.board.place(19, 19, i1, 0)
    for x, y in ((19, 18), (18, 19), (18, 18)):
        g.board.place(x, y, i1, 1)
    g.placed[0] = 1
    assert not g.has_legal(0)
    assert g.has_legal(1) or g.has_legal(2) or g.has_legal(3)
    g.act_pass()
    assert g.state == "PLAYING", "must not end while the AIs can still move"


if __name__ == "__main__":
    test_100_games()
    test_tie_declaration()
    test_player_wins_declaration()
    test_ai_wins_declaration()
    test_fewer_remaining_wins()
    test_full_hand_scores_zero()
    test_game_ends_only_when_all_four_are_stuck()
    print("all simulation tests passed")


# --------------------------------------------------------------------------
# the abandonment branch
# --------------------------------------------------------------------------

def test_the_abandoned_book_falls_back_to_a_legal_move():
    """`check_hunter_book` allows an abandoned book to stop matching. This is the
    branch that permission covers, reached with a constructed position.

    The construction: an opponent V5 sits on the **second** square the book's
    step 0 needs (owner 0's corner is (19,19) and the step wants (17,17) as
    well), so no Z5 can cover both and the candidate set is empty. The owner can
    still open, because the corner-contact rule only demands the corner. That is
    exactly the situation the abandonment rule exists for, and it cannot arise
    from play - H1's 7,500 pairs and this stage's 100-game sample never abandoned.

    What is asserted is the fallback, not the book: the move is legal according to
    the engine, it covers the corner, and the book is reported abandoned.
    """
    from rl.actions import legal_indices, move_to_index
    from rl.collect import board_from_state
    from rl.opening import align_to
    import engine

    owner = 0
    opponent = 1
    corner = OWNER_CORNER[owner]
    # a real V5 covering the second square the book step needs
    mask = None
    for oi, od in ai.ODIRS["V5"].items():
        for base in od["bases"]:
            if (od["m"] << base) >> (corner[0] - 2 + (corner[1] - 2) * 20) & 1:
                mask = od["m"] << base
                break
        if mask:
            break
    assert mask is not None and mask.bit_count() == 5

    own = [0, 0, 0, 0]
    own[opponent] = mask
    full = sum(1 << i for i in range(engine.N_PIECES))
    state = engine.State(own_bits=tuple(own), hand_bits=(full,) * 4,
                         turn_order=CLOCKWISE_OWNERS, to_move=owner,
                         stuck=(False, False, False, False))
    board = board_from_state(state)

    # the premise: the book offers nothing, the player is not stuck
    assert ai.book_candidates(board, owner, 0) == []
    assert len(list(legal_indices(state, owner))) > 0

    brain = HunterBrain("hunter", ai.make_profile("chess", random.Random(0)),
                        game_seed=99)
    trace = HunterBookTrace()
    move = ai.choose_move(board, [engine.PIECE_ORDER[i]
                                  for i in range(engine.N_PIECES)],
                          owner, brain, random.Random(1),
                          must_cover=corner)
    assert move is not None
    assert brain.book_abandoned, "the empty candidate set should abandon the book"
    assert brain.book_moves_used == 0
    # and the check reports the branch instead of asserting anything
    assert check_hunter_book(type("G", (), {
        "brains": {owner: brain}, "board": board, "placed": [0] * 4})(),
        owner, move, trace) == "abandoned"
    assert trace.abandoned == 1 and trace.book_moves == 0

    # the fallback is a real legal move
    index = move_to_index((engine.PIECE_IDX[move[0]], move[1],
                           move[2] + move[3] * 20))
    assert index in set(int(i) for i in legal_indices(state, owner))
    placed = ai.ODIRS[move[0]][move[1]]["m"] << (move[2] + move[3] * 20)
    assert placed >> (corner[0] + corner[1] * 20) & 1, "must cover the corner"
    assert placed & ~engine.ALL == 0 and placed.bit_count() == \
        MASTER[move[0]]["size"]


# --------------------------------------------------------------------------
# destructive checks: prove the new assertions can fail
# --------------------------------------------------------------------------

def _book_step_1_position():
    """The position after Hunter's real book step 0, with the board and the
    engine state that go with it."""
    from rl.collect import board_from_state
    from rl.opening import align_to
    import engine
    state = align_to(engine.initial_state(CLOCKWISE_OWNERS), 0)
    board0 = board_from_state(state)
    step0 = sorted(ai.book_candidates(board0, 0, 0))[0]
    state = align_to(engine.apply_move(
        state, (engine.PIECE_IDX[step0[0]], step0[1], step0[2])), 0)
    return state, board_from_state(state)


def _book_check_for(owner, replacement_board=None):
    """The same shape of check `test_100_games` uses - the book check plus the
    generic edge rule - so a destructive test exercises the real path rather than
    `check_hunter_book` in isolation."""
    trace = HunterBookTrace()
    neigh = neighbors_of()

    def check(g, o, move, before):
        if g.owner_key[o] != "hunter":
            return
        check_hunter_book(g, o, move, trace)
        name, oi, x, y = move
        cells = {(x + dx) + (y + dy) * 20
                 for dx, dy in MASTER[name]["orientations"][oi]}
        where = "%s oi=%d at (%d,%d) for owner %d" % (name, oi, x, y, o)
        for i in cells:
            for n in neigh[i]:
                assert n not in before, where + " shares an edge with its own stone"
    return check, trace


def test_a_legal_move_outside_the_candidate_set_fails_the_membership_check(
        monkeypatch):
    """Destructive check (a), driven through a real game.

    At book step 1 the candidate set holds 2 of the 7 legal V5 placements, so a
    V5 at a legal base the book did not offer is reachable and must be rejected.
    The identity check passes for this move - it is the right piece - so the
    failure has to come from the membership assertion.
    """
    from rl.actions import index_to_move, legal_indices, move_to_index
    import engine

    _state, board = _book_step_1_position()
    candidates = {move_to_index((engine.PIECE_IDX[n], o, b))
                  for n, o, b in ai.book_candidates(board, 0, 1)}
    outside = sorted(m for m in (index_to_move(int(i))
                                  for i in legal_indices(_state, 0))
                     if engine.PIECE_ORDER[m[0]] == "V5" and m not in candidates)
    assert outside, "expected a legal V5 the book did not offer"
    wrong = outside[0]
    forced = (engine.PIECE_ORDER[wrong[0]], wrong[1],
             wrong[2] % 20, wrong[2] // 20)
    assert forced[0] == HUNTER_BOOK_PIECES[1], "the identity check must pass"

    real = ai.choose_move
    calls = {"n": 0}

    def patched(board, hand_names, owner, brain, rng, **kw):
        # let the real book step 0 happen, then force the move at step 1
        if not isinstance(brain, HunterBrain):
            return real(board, hand_names, owner, brain, rng, **kw)
        if brain.book_step == 0:
            return real(board, hand_names, owner, brain, rng, **kw)
        return forced
    monkeypatch.setattr(ai, "choose_move", patched)

    check, trace = _book_check_for(0)
    with pytest.raises(AssertionError) as exc:
        _play_hunter_game(random.Random(1), 0, check, game_seed=4242)
    message = str(exc.value)
    assert "not one of the book's candidates" in message, message
    del calls
    print("\n(a) membership check rejected inside a real game: %s"
          % message[:200])


def test_an_illegal_edge_touching_move_fails_the_generic_check(monkeypatch):
    """Destructive check (b), the pre-existing edge rule rather than the new one.

    A legal move can never share an edge with one of its own stones - the
    corner-contact rule forbids it - so the engine rejects it too and the move has
    to be forced in. That is the point: the generic check is what stands between a
    patched brain and a broken position.

    Forced on a **post-book** move, on purpose. On a book step the identity and
    membership assertions would fire first, and this test is about the generic
    rule, so the book has to be out of the way.
    """
    import engine

    _state, board = _book_step_1_position()
    own = {i for i in range(400) if board.grid[i] == 0}
    neigh = neighbors_of()
    found = None
    for name, spec in MASTER.items():
        for oi, orient in enumerate(spec["orientations"]):
            h = max(dy for _dx, dy in orient) + 1
            w = max(dx for dx, _dy in orient) + 1
            for y in range(20 - h + 1):
                for x in range(20 - w + 1):
                    cells = {(x + dx) + (y + dy) * 20 for dx, dy in orient}
                    if cells & own:
                        continue
                    if any(n in own for i in cells for n in neigh[i]):
                        found = (name, oi, x, y)
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    assert found, "no empty placement edge-touches the player's own stone"
    forced = found

    real = ai.choose_move

    def patched(board_, hand_names, owner, brain, rng, **kw):
        if not isinstance(brain, HunterBrain):
            return real(board_, hand_names, owner, brain, rng, **kw)
        if brain.book_step < len(HUNTER_BOOK_PIECES):
            return real(board_, hand_names, owner, brain, rng, **kw)
        return forced
    monkeypatch.setattr(ai, "choose_move", patched)

    check, trace = _book_check_for(0)
    with pytest.raises(AssertionError) as exc:
        _play_hunter_game(random.Random(1), 0, check, game_seed=4242)
    message = str(exc.value)
    assert "shares an edge with its own stone" in message, message
    print("\n(b) generic edge check rejected inside a real game: %s"
          % message[:200])


def test_a_legal_non_book_piece_on_the_first_move_fails_the_identity_check(
        monkeypatch):
    """Destructive check (c), driven through a real game."""
    from rl.actions import index_to_move, legal_indices
    from rl.collect import board_from_state
    from rl.opening import align_to
    import engine

    state = align_to(engine.initial_state(CLOCKWISE_OWNERS), 0)
    board = board_from_state(state)
    legal = sorted(set(index_to_move(int(i)) for i in legal_indices(state, 0)))
    wrong = next(m for m in legal if engine.PIECE_ORDER[m[0]] != "Z5")
    forced = (engine.PIECE_ORDER[wrong[0]], wrong[1],
              wrong[2] % 20, wrong[2] // 20)

    real = ai.choose_move

    def patched(board_, hand_names, owner, brain, rng, **kw):
        if not isinstance(brain, HunterBrain):
            return real(board_, hand_names, owner, brain, rng, **kw)
        if brain.book_step == 0:
            return forced
        return real(board_, hand_names, owner, brain, rng, **kw)
    monkeypatch.setattr(ai, "choose_move", patched)

    check, trace = _book_check_for(0)
    with pytest.raises(AssertionError) as exc:
        _play_hunter_game(random.Random(1), 0, check, game_seed=4242)
    message = str(exc.value)
    assert "is not the book's piece" in message, message
    print("\n(c) identity check rejected inside a real game: %s" % message[:200])
