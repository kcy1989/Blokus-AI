"""The Intruder: extension, usable corners, crossing, key squares, phase limits
and performance.

The rules come from the strategy description and all of the implementation lives
in the geometry and scoring functions in `ai.py`; each of them is pinned down
here one by one, so that nobody later silently rewrites them in a "cheaper" way.
"""
import random
import time

import ai
from board import Board, V, cells_to_vertices
from config import PERSONALITY_ORDER
from game import Game
from pieces import MASTER
from test_ai import assert_corner_contact

V_SHAPE = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}   # #. / #. / ###
I1 = MASTER["I1"]["orientations"][0]


def oi_of(name, cells):
    """The orientation index whose cells are exactly `cells`."""
    for oi, c in enumerate(MASTER[name]["orientations"]):
        if set(c) == set(cells):
            return oi
    raise AssertionError("no orientation of %s matches %r" % (name, sorted(cells)))


def corner_geometry(board=None, name="V5", oi=None, x=0, y=0, owner=1):
    """A hypothetical opening placement on owner 1's own corner (0, 0)."""
    if board is None:
        board = Board()
    if oi is None:
        oi = oi_of(name, V_SHAPE)
    return ai.move_geometry(board, name, oi, x, y, owner, must_cover=(0, 0))


# ---------------------------------------------------------------- extension

def test_extension_matches_the_worked_example():
    """`#.`/`#.`/`###` attached at (0,0): the farthest usable corner (3,3) is
    3+3 = 6 away from the contact corner."""
    ext, usable, squares = corner_geometry()
    assert ext == 6, ext
    # the 4 corners of the two placeable squares (3,1) and (3,3)
    assert usable == 8, usable
    assert squares == 2, squares


def test_extension_is_dynamic_not_a_piece_constant():
    """After (3,3) is blocked the extension must drop - extension is the board
    state after the move, not a constant of the piece."""
    b = Board()
    b.place(3, 3, I1, 2)
    blocked, _usable, squares = corner_geometry(b)
    assert blocked == 4 and squares == 1, (blocked, squares)
    b.place(3, 1, I1, 2)
    gone, usable, squares = corner_geometry(b)
    assert gone == 0 and usable == 0 and squares == 0, (gone, usable, squares)


def test_usable_vertices_come_from_the_general_rule():
    """Usable corners = the union of "the 4 corners of every placeable square",
    not a hand-written list.

    (3,3) and (3,2) are corners of the placeable squares (3,1) and (3,3);
    (1,0) and (0,3) are also corners of the piece itself, but the empty squares
    at those two positions share an edge with the player's own piece, so they are
    not in the placeable set.
    """
    b = Board()
    placed = ai.ODIRS["V5"][oi_of("V5", V_SHAPE)]["m"]
    legal = (b.dilate_diag(placed) & b.empty_bits) & ~(b.dilate(placed) & b.empty_bits)
    assert sorted((i % 20, i // 20) for i in range(400)
                  if (legal >> i) & 1) == [(3, 1), (3, 3)]
    got = {(i % V, i // V) for i in range(441)
           if (cells_to_vertices(legal) >> i) & 1}
    assert (3, 3) in got and (3, 2) in got
    assert (1, 0) not in got and (0, 3) not in got
    # (2,3) is the corner of the piece's own (2,2), not a corner of any
    # placeable square, so the general rule makes it unusable. Counting it as
    # usable would not change the maximum extension (still 6), but the general
    # rule is what holds for every shape and boundary, so the implementation
    # follows the general rule.
    assert (2, 3) not in got


def test_stretchers_reach_further_than_y5():
    """The reason Y5 is excluded is that its extension distance is only 5."""
    reach = {}
    for name in ("L5", "N5", "I5", "Y5"):
        reach[name] = max(corner_geometry(Board(), name, oi)[0]
                          for oi in range(len(MASTER[name]["orientations"])))
    assert reach["L5"] == reach["N5"] == reach["I5"] == 6, reach
    assert reach["Y5"] == 5, reach
    assert "Y5" not in ai.STRETCHERS


# -------------------------------------------------------------- crossing pieces

def test_leaper_set_is_derived_geometrically():
    """The 6-cell pieces have a 3x3 bounding box, but only 3 of them occupy two
    "diagonally opposite" corners; T5's two corners are adjacent."""
    assert ai.LEAPERS == {"V5", "W5", "Z5"}
    assert not ({"T5", "X5", "F5"} & ai.LEAPERS)
    for name in ("T5", "X5", "F5"):
        cells = MASTER[name]["orientations"][0]
        assert max(x for x, _ in cells) == 2 and max(y for _, y in cells) == 2, name
    for cells in (MASTER[n]["orientations"][0] for n in ("V5", "W5", "Z5")):
        assert ai.is_leaper(cells)


def test_crossing_needs_one_diagonal_mine_and_one_theirs():
    """The user-supplied micro diagram `1 2 / 3 4`: the opponent holds 2 and 3,
    we hold 4, so placing 1 completes the crossing.

    It must hold for both diagonal directions, and placing 2 or 3 (the opponent's
    line) does not count.
    """
    b = Board()
    b.place(6, 5, I1, 2)                 # 2
    b.place(5, 6, I1, 2)                 # 3
    b.place(6, 6, I1, 1)                 # 4
    assert not ai.has_crossed(b.owner_bits[1], b.owner_bits[2])
    b.place(5, 5, I1, 1)                 # 1
    assert ai.has_crossed(b.owner_bits[1], b.owner_bits[2])
    # the other direction: the same holds for the line where the crossing
    # pieces occupy {top-left, bottom-left}
    c = Board()
    c.place(5, 5, I1, 2)
    c.place(6, 6, I1, 2)
    c.place(5, 6, I1, 1)
    assert not ai.has_crossed(c.owner_bits[1], c.owner_bits[2])
    c.place(6, 5, I1, 1)
    assert ai.has_crossed(c.owner_bits[1], c.owner_bits[2])


def test_has_crossed_needs_no_history():
    """The decision only needs the board: the same position built with a
    different move order must give the same answer."""
    b = crossing_board()
    key = ai.key_cells(b.owner_bits[1], b.owner_bits[2], b.empty_bits)
    assert key, "fixture must hold a key square"
    assert not ai.has_crossed(b.owner_bits[1], b.owner_bits[2])
    b.place(4, 6, I1, 1)
    assert ai.has_crossed(b.owner_bits[1], b.owner_bits[2])

    other = Board()
    other.place(5, 6, I1, 2)
    other.place(6, 7, I1, 2)
    for i in range(6):
        other.place(i, i, I1, 1)
    other.place(4, 6, I1, 1)       # same position as b, different order
    assert other.grid == b.grid
    assert ai.has_crossed(other.owner_bits[1], other.owner_bits[2])
    # computing it a second time changes nothing
    assert ai.has_crossed(b.owner_bits[1], b.owner_bits[2]) == \
        ai.has_crossed(b.owner_bits[1], b.owner_bits[2])


def test_key_cell_completes_a_crossing_when_filled():
    """Filling in a key square completes the crossing, and it always meets the
    player's own piece corner-to-corner, never sharing an edge."""
    b = crossing_board()
    key = ai.key_cells(b.owner_bits[1], b.owner_bits[2], b.empty_bits)
    assert sorted((i % 20, i // 20) for i in range(400) if (key >> i) & 1) \
        == [(4, 6), (6, 6)]
    reach = b.reach(1)
    for c in [i for i in range(400) if (key >> i) & 1]:
        assert ai.has_crossed(b.owner_bits[1] | (1 << c), b.owner_bits[2])
        assert b.can_place(c % 20, c // 20, I1, 1, None, reach), \
            "a key square must be reachable by the corner-contact rule"


# ---------------------------------------------------------- phase limits

def intruder_game(seed, key="intruder", moves=None):
    """Real moves only, so every position obeys the corner-contact rule."""
    rng = random.Random(seed)
    g = Game(rng)
    g.set_player_color("blue")
    g.brains[1] = ai.make_brain(key, rng)
    g.owner_key[1] = key
    g.start()
    log = []
    while g.state == "PLAYING" and (moves is None or len(log) < moves):
        owner = g.current_owner()
        mv = ai.choose_move(g.board, g.hands[owner].names, owner, g.brains[owner],
                            rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner), reach=g.reach(owner),
                            other_must_cover={o: g.must_cover(o) for o in range(4)},
                            other_reach={o: g.reach(o) for o in range(4)})
        if owner == 1:
            log.append(mv)
        if mv:
            g.act(*mv)
        else:
            g.act_pass()
    return g, log


def test_opening_moves_are_leapers_then_stretchers():
    """Rules 1 and 2: use up the crossing pieces first, then the pieces that can
    stretch 6 squares, and only afterwards return to a general position."""
    _g, log = intruder_game(1)
    used = [mv[0] for mv in log if mv]
    first_six = used[:6]
    assert set(first_six[:3]) <= ai.LEAPERS, first_six
    assert set(first_six[3:6]) <= ai.STRETCHERS, first_six
    assert len(used) > 6


def test_phase_degrades_instead_of_returning_none():
    """When not a single crossing piece fits, it must fall back to the other
    pieces.

    The only meaning of `choose_move` returning None is "truly no piece can be
    placed", and this position clearly still has I1 to play, so it must return a
    legal placement.
    """
    b = Board()
    b.place(0, 0, I1, 1)              # owner 1 is left with only its corner
    for y in range(4):
        for x in range(4):
            if (x, y) in ((0, 0), (1, 1)):
                continue
            b.place(x, y, I1, 2)      # wall off everything except (1,1)
    reach = b.reach(1)
    legal = reach.need & ~reach.avoid & b.empty_bits
    assert [i % 20 for i in range(400) if (legal >> i) & 1] == [1]
    for name in ("V5", "L5", "I5"):
        for oi, od in ai.ODIRS[name].items():
            assert not (ai._adjoining_bases(od, reach, 0)
                        & ai._free_bases(od, b.empty_bits)), (name, oi)
    brain = ai.make_brain("intruder", random.Random(0))
    mv = ai.choose_move(b, list(MASTER.keys()), 1, brain, random.Random(1),
                        other_brains=None, reach=reach)
    assert mv == ("I1", 0, 1, 1), mv
    assert b.can_place(1, 1, MASTER["I1"]["orientations"][0], 1, None, reach)


# -------------------------------------------------- rules 3 / 4 / 5

def crossing_board():
    """Owner 1 runs the diagonal (0,0)->(5,5); owner 2's (5,6)+(6,7) hug the
    side.

    So (5,7) is a key square: filling it completes the crossing, and (6,6) opens
    up that position as soon as it is one of my pieces.
    """
    b = Board()
    for i in range(6):
        b.place(i, i, I1, 1)
    b.place(5, 6, I1, 2)
    b.place(6, 7, I1, 2)
    return b


def sets_up(brain, ctx, board, name, oi, base):
    od = ai.ODIRS[name][oi]
    (own, empt, need, avoid, _legal, _fresh,
     _anchors) = ai.place_geometry(board, name, oi, base, ctx)
    return brain._sets_up(ctx, od, own, empt, need, avoid)


def test_rule3_prefers_a_move_that_sets_up_a_crossing():
    """Rule 3: when an uncrossed opponent is right next door, it picks a
    placement that still allows a crossing once it is filled."""
    b = crossing_board()
    # a general position, so the phase rules do not kick in
    hand = ["I3", "O4", "I2", "I1"]
    brain = ai.make_brain("intruder", random.Random(0))
    mv = ai.choose_move(b, hand, 1, brain, random.Random(0), other_brains=None,
                        reach=b.reach(1))
    assert mv is not None
    assert b.can_place(mv[2], mv[3], MASTER[mv[0]]["orientations"][mv[1]], 1,
                       None, b.reach(1))
    ctx = brain.context(b, hand, 1, None, b.reach(1))
    assert ctx["uncrossed"], "fixture must have an uncrossed opponent"
    assert sets_up(brain, ctx, b, mv[0], mv[1], mv[2] + mv[3] * 20), mv


def test_intruder_ignores_the_weighted_lookahead():
    """The opponent-prediction term subtracts a **weighted** score whose
    magnitude is far larger than the rule score; letting it take part in the
    ordering would bury the rules completely, so the Intruder skips that
    section."""
    intruder = ai.make_brain("intruder", random.Random(0))
    assert intruder.uses_lookahead is False
    assert ai.make_brain("wolf", random.Random(0)).uses_lookahead is True
    b = crossing_board()
    hand = ["I3", "O4", "I2", "I1"]
    others = {o: ai.make_brain("chess", random.Random(o)) for o in (0, 2, 3)}
    without = ai.choose_move(b, hand, 1, intruder, random.Random(0),
                             other_brains=None, reach=b.reach(1))
    with_pool = ai.choose_move(b, hand, 1, intruder, random.Random(0),
                               other_brains=others, reach=b.reach(1))
    assert without == with_pool, (without, with_pool)


def test_rule3_is_silent_when_no_opponent_can_be_reached():
    """When no opponent is hugging us there is no key square, so this rule must
    not fire out of thin air."""
    b = Board()
    for i in range(6):
        b.place(i, i, I1, 1)
    hand = ["I3", "O4", "I2", "I1"]
    brain = ai.make_brain("intruder", random.Random(0))
    mv = ai.choose_move(b, hand, 1, brain, random.Random(0), other_brains=None,
                        reach=b.reach(1))
    ctx = brain.context(b, hand, 1, None, b.reach(1))
    assert not ctx["uncrossed"]
    assert not sets_up(brain, ctx, b, mv[0], mv[1], mv[2] + mv[3] * 20), mv


def test_rule4_values_room_over_size():
    """Rule 4: the crossing step is dominated by maximising the number of
    placeable squares, and it penalises having too few usable corners."""
    assert ai.crossing_bonus(60, 6) > ai.crossing_bonus(10, 6)
    assert ai.crossing_bonus(60, ai.SEAL_MIN) > ai.crossing_bonus(60, 0)
    # the placeable-square count also earns its keep without a crossing, but
    # with only half the weight
    assert ai.crossing_bonus(60, 6) - ai.W_CROSS > 2.0 * (ai.W_SQUARES * 10)


def test_rule5_five_cells_unless_the_square_is_strategic():
    """Rule 5: in a general position 5 cells come first; on a key square the
    4-cell and 3-cell pieces win out."""
    assert ai.size_bonus(5, False) > ai.size_bonus(4, False)
    assert ai.size_bonus(4, False) > ai.size_bonus(3, False)
    assert ai.size_bonus(3, True) > ai.size_bonus(5, False)
    assert ai.size_bonus(4, True) > ai.size_bonus(5, False)
    assert ai.size_bonus(5, True) > ai.size_bonus(4, True)


# -------------------------------------------------------- legality / performance

def test_intruder_moves_are_legal_in_a_real_game():
    g, log = intruder_game(3)
    assert len(log) > 5
    # replay and check the corner-contact rule at the moment of each move: the
    # final board cannot be re-verified, because a multi-cell piece sharing an
    # edge with itself is normal.
    rng = random.Random(3)
    g = Game(rng)
    g.set_player_color("blue")
    g.brains[1] = ai.make_brain("intruder", rng)
    g.owner_key[1] = "intruder"
    g.start()
    seen = 0
    while g.state == "PLAYING" and seen < 12:
        owner = g.current_owner()
        before = {i for i in range(400) if g.board.grid[i] == owner}
        mv = ai.choose_move(g.board, g.hands[owner].names, owner, g.brains[owner],
                            rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner), reach=g.reach(owner))
        if mv is None:
            assert not g.has_legal(owner)
            g.act_pass()
            continue
        if owner == 1 and g.placed[1] > 0:
            assert_corner_contact(g.board, 1, mv[0], mv[1], mv[2], mv[3], before)
            seen += 1
        g.act(*mv)


def test_intruder_move_speed():
    board = intruder_game(1)[0].board
    brain = ai.make_brain("intruder", random.Random(3))
    reach = board.reach(1)
    t0 = time.perf_counter()
    for _ in range(20):
        mv = ai.choose_move(board, list(MASTER.keys()), 1, brain,
                            random.Random(0), other_brains=None, reach=reach)
        assert mv is not None
    dt = (time.perf_counter() - t0) / 20
    assert dt < 1.5, dt


# -------------------------------------------------------------- sampling

def test_draw_personalities_picks_three_from_the_pool():
    rng = random.Random(4)
    seen = set()
    for _ in range(200):
        keys = ai.draw_personalities(rng, 3)
        assert len(set(keys)) == 3
        assert set(keys) <= set(ai.personality_keys())
        seen |= set(keys)
    assert seen == set(ai.personality_keys()), seen


def test_intruder_reaches_the_leaderboard(tmp_path):
    """All 4 personalities are leaderboard entries keyed by personality; games
    where one is absent do not affect the others."""
    from records import Records
    rec = Records(str(tmp_path / "records.json"))
    for keys in (("player", "wolf", "chess", "intruder"),
                 ("player", "fox", "intruder", "chess")):
        rec.record(list(zip(keys, (10, 20, 30, 40))))
    rows = dict((k, (g, rem)) for k, g, _p, rem in rec.rows())
    assert rows["intruder"][0] == 2, rows
    assert rows["player"][0] == 2
    assert set(rows) == {"player", "wolf", "chess", "fox", "intruder"}, rows
    assert "intruder" in PERSONALITY_ORDER
