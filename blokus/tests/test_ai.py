"""Tests for AI personalities: legality, style ratios, speed."""
import random
import time

import ai
from board import Board, neighbors_of
from game import Game
from pieces import MASTER


def make_test_board(rounds=3, seed=5):
    """A legal mid-game position.

    Built by playing real moves through `Game.act` rather than by dropping
    pieces on a bare board: the edge-adjacency rule means owner 0 must have
    opened on its own corner and every later stone must touch it, or `reach`
    is wrong for every candidate the AI considers.
    """
    rng = random.Random(seed)
    g = Game(rng)
    g.set_player_color("blue")
    g.start()
    played = 0
    while played < 4 * rounds:
        owner = g.current_owner()
        mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                            g.brains[owner], rng,
                            must_cover=g.must_cover(owner),
                            reach=g.reach(owner))
        if mv is None:
            g.act_pass()
        else:
            g.act(*mv)
            played += 1
    return g.board


def assert_corner_contact(board, owner, name, oi, x, y, before=None):
    """The house rule: the piece must meet one of the owner's existing stones
    at a corner and must not share an edge with any of them. `before` is the
    cell set the owner held before the move, so a piece is never compared
    against its own cells."""
    if before is None:
        before = {i for i in range(400) if board.grid[i] == owner}
    cells = MASTER[name]["orientations"][oi]
    where = "%s oi=%d at (%d,%d) for owner %d" % (name, oi, x, y, owner)
    for dx, dy in cells:
        for n in neighbors_of()[x + dx + (y + dy) * 20]:
            assert n not in before, where + " shares an edge with its own stone"
    before_xy = {(i % 20, i // 20) for i in before}
    assert any((x + dx + sx, y + dy + sy) in before_xy
               for dx, dy in cells for sx in (-1, 1) for sy in (-1, 1)), \
        where + " has no corner contact"


def test_all_moves_legal():
    board = make_test_board()
    for owner in range(4):
        reach = board.reach(owner)
        assert reach.need or reach.avoid, "fixture must give every owner a stone"
        brain = ai.make_brain(["wolf", "chess", "fox"][owner % 3],
                              random.Random(1))
        for _ in range(60):
            before = {i for i in range(400) if board.grid[i] == owner}
            mv = ai.choose_move(board, list(MASTER.keys()), owner, brain,
                                random.Random(_ * 7 + owner),
                                other_brains=None, reach=reach)
            assert mv is not None
            name, oi, x, y = mv
            assert board.can_place(x, y, MASTER[name]["orientations"][oi], owner,
                                   None, reach), (name, oi, x, y)
            assert_corner_contact(board, owner, name, oi, x, y, before)


def test_ai_respects_opening_corner_rule():
    """Before any stone is placed the corner rule is the only one that applies."""
    g = Game(random.Random(3))
    g.set_player_color("blue")
    g.start()
    owner = 0
    assert g.reach(owner) is None
    brain = ai.make_brain("wolf", random.Random(2))
    for seed in range(20):
        mv = ai.choose_move(g.board, g.hands[owner].names, owner, brain,
                            random.Random(seed), other_brains=None,
                            must_cover=g.must_cover(owner), reach=g.reach(owner))
        assert mv is not None
        name, oi, x, y = mv
        assert g.can_act(name, oi, x, y, owner), (name, oi, x, y)
        covered = any((x + dx, y + dy) == g.must_cover(owner)
                      for dx, dy in MASTER[name]["orientations"][oi])
        assert covered, (name, oi, x, y)


def test_ai_returns_none_when_fully_blocked():
    """A player with no usable corner square left must get None so the caller
    can pass the turn on, rather than an arbitrary illegal move."""
    i1 = MASTER["I1"]["orientations"][0]
    b = Board()
    b.place(0, 0, i1, 2)
    b.place(1, 0, i1, 2)
    b.place(0, 1, i1, 2)
    # wall owner 2 in: fill every square it could corner-touch with another
    # owner's stones, so no piece can satisfy the rule
    r = b.reach(2)
    for i in range(400):
        if (r.need >> i) & 1 and b.grid[i] == -1:
            b.place(i % 20, i // 20, i1, 1)
    assert b.reach(2).need & b.empty_bits == 0
    brain = ai.make_brain("fox", random.Random(1))
    assert ai.choose_move(b, ["I1"], 2, brain, random.Random(0),
                          other_brains=None, reach=b.reach(2)) is None
    assert not b.has_legal_move(2, [MASTER["I1"]["orientations"]], None,
                                 b.reach(2))


def _profile_scores(board, pred, owner=0, seed=1):
    """Score every legal candidate matching `pred` under all three personalities.

    Asserting on the *same* move across personalities (rather than comparing
    each one's argmax) is what makes these tests deterministic: the argmax
    collapses onto one obvious move once the adjacency rule shrinks the
    candidate set, but the underlying trade-offs still differ.
    """
    open_a, block, defend = ai.board_feats(board.grid)
    prof = {k: ai.make_profile(k, random.Random(seed))
            for k in ("wolf", "chess", "fox")}
    out = []
    for name, p in MASTER.items():
        for oi, od in ai.ODIRS[name].items():
            m = od["m"]
            for base in od["bases"]:
                shifted = m << base
                if (board.empty_bits & shifted) != shifted:
                    continue
                cells = [base + o for o in od["offs"]]
                if not pred(name, oi, base, cells):
                    continue
                out.append({k: ai._score_move(board, name, oi, base, owner, prof[k],
                                             open_a, block, defend,
                                             board.corner_regions, board.borders)
                            for k in prof})
    assert out, "fixture board has no candidate matching the filter"
    return out


def _assert_all(rows, winner, losers):
    for s in rows:
        for other in losers:
            assert s[winner] > s[other], (winner, other, s)


def test_wolf_prefers_claiming_open_corners():
    """The wolf is the corner specialist, and the only weight that expresses
    that is w_corner.

    The old test measured this as a corner rate over a whole-board candidate
    set. The adjacency rule anchors candidates to a player's own stones and
    forces the opening onto the owner's own corner, so in a real game position
    no legal move can cover a corner square at all and the rate is dead. The
    property still holds wherever a corner is genuinely available, which is
    what the corner reserve in `_build_shortlist` actually depends on.
    """
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    b.place(0, 0, i1, 1)
    b.place(19, 0, i1, 2)
    b.place(0, 19, i1, 3)
    # owner 0 owns (19,19), which is the only corner still free
    rows = _profile_scores(b, lambda n, oi, base, c: (ai.ODIRS[n][oi]["corner"]
                                                      .get(base, 0) >> 3) & 1)
    _assert_all(rows, "wolf", ("chess", "fox"))


def test_fox_prefers_blocking_moves():
    """Restricted to pieces of 2 cells or more: for a lone I1 the flat
    SMALL_PENALTY dominates every per-cell term, so which side of the
    trade-off wins is decided by where the single square sits rather than by
    the personality weights."""
    board = make_test_board()
    rows = _profile_scores(board, lambda n, oi, base, c: MASTER[n]["size"] >= 2
                           and any(board.grid[q] not in (-1, 0)
                                   for i in c for q in neighbors_of()[i]))
    _assert_all(rows, "fox", ("wolf", "chess"))


def test_fox_prefers_open_space():
    """fox has the highest w_open (2.0 vs 0.5/1.0), so it values unclaimed
    space most. Restricted to pieces of 3 cells or more: for a lone I1/I2 the
    flat SMALL_PENALTY and the tiny w_large term dominate, and a piece wedged
    in a dead corner is a special case rather than a counterexample."""
    board = make_test_board()
    rows = _profile_scores(board, lambda n, oi, base, c: MASTER[n]["size"] >= 3
                           and all(board.grid[q] in (-1, 0)
                                   for i in c for q in neighbors_of()[i]))
    _assert_all(rows, "fox", ("wolf", "chess"))


def test_mistake_rate_influences_choices():

    board = make_test_board()
    rng = random.Random(7)
    brain = ai.make_brain("chess", rng)
    assert 0 <= brain.mistake_rate <= 0.4
    mv = ai.choose_move(board, list(MASTER.keys()), 0, brain,
                        random.Random(0), other_brains=None,
                        reach=board.reach(0))
    assert mv is not None


def test_opponent_pool_falls_back_when_neutral_bases_are_illegal():
    """The neutral shortlist is centre/corner weighted, so once the rule pins
    a player to their own corner region it can be entirely illegal. A silent
    empty pool would zero out the opponent response and quietly disable the
    lookahead, so the pool must widen to every base instead."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    b.place(0, 0, i1, 0)          # owner 0 owns only its own corner
    reach = b.reach(0)
    assert len(ai._opponent_pool(b, list(MASTER), 0, None, reach, k=0)) == 0
    pool = ai._opponent_pool(b, list(MASTER), 0, None, reach)
    assert len(pool) == ai.OPP_POOL_K, len(pool)
    for name, oi, base in pool:
        m = ai.ODIRS[name][oi]["m"] << base
        assert (b.empty_bits & m) == m
        assert m & reach.need, "pool contains a move with no corner contact"
        assert not (m & reach.avoid), "pool contains a move sharing an edge"


def test_move_speed():
    board = make_test_board()
    rng = random.Random(3)
    brain = ai.make_brain("wolf", rng)
    reach = board.reach(0)
    t0 = time.perf_counter()
    for _ in range(50):
        mv = ai.choose_move(board, list(MASTER.keys()), 0, brain,
                            random.Random(rng.getrandbits(16)),
                            other_brains=None, reach=reach)
        assert mv is not None
    dt = (time.perf_counter() - t0) / 50
    assert dt < 1.5, dt
