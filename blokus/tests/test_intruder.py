"""入侵者：延伸度、可用交點、跨越、關鍵格、階段限制與效能。

規則來自策略描述，實作全部落在 `ai.py` 的幾何與評分函式上；這裡逐一把它們
釘住，免得日後有人用「更省」的寫法把它們悄悄改掉。
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


# ---------------------------------------------------------------- 延伸度

def test_extension_matches_the_worked_example():
    """`#.`/`#.`/`###` 接在 (0,0)：最遠的可用交點 (3,3) 距接觸交點 3+3 = 6。"""
    ext, usable, squares = corner_geometry()
    assert ext == 6, ext
    assert usable == 8, usable          # 兩個可放空格 (3,1)、(3,3) 的 4 個角
    assert squares == 2, squares


def test_extension_is_dynamic_not_a_piece_constant():
    """封掉 (3,3) 之後延伸度必須下降——延伸度是落子後的棋盤狀態，不是棋塊
    的一個常數。"""
    b = Board()
    b.place(3, 3, I1, 2)
    blocked, _usable, squares = corner_geometry(b)
    assert blocked == 4 and squares == 1, (blocked, squares)
    b.place(3, 1, I1, 2)
    gone, usable, squares = corner_geometry(b)
    assert gone == 0 and usable == 0 and squares == 0, (gone, usable, squares)


def test_usable_vertices_come_from_the_general_rule():
    """可用交點＝「所有可放空格的 4 個角」的聯集，不是手寫的清單。

    (3,3)、(3,2) 是可放空格 (3,1)、(3,3) 的角；(1,0)、(0,3) 雖然也是棋塊
    自己的角，但那兩個位置的空格都與自己的棋共邊，所以不在可放集合裡。
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
    # (2,3) 是自己那顆 (2,2) 的角，不是任何可放空格的角，依通用規則不可用。
    # 把它算成可用不會改變最大延伸度（仍是 6），但通用規則才對所有形狀與
    # 邊界都成立，所以實作走通用規則。
    assert (2, 3) not in got


def test_stretchers_reach_further_than_y5():
    """Y5 被排除的理由就是延伸距離只有 5。"""
    reach = {}
    for name in ("L5", "N5", "I5", "Y5"):
        reach[name] = max(corner_geometry(Board(), name, oi)[0]
                          for oi in range(len(MASTER[name]["orientations"])))
    assert reach["L5"] == reach["N5"] == reach["I5"] == 6, reach
    assert reach["Y5"] == 5, reach
    assert "Y5" not in ai.STRETCHERS


# -------------------------------------------------------------- 跨越棋

def test_leaper_set_is_derived_geometrically():
    """6 塊棋有 3x3 外框，但只有 3 塊佔兩個「對角」；T5 的兩個角是相鄰的。"""
    assert ai.LEAPERS == {"V5", "W5", "Z5"}
    assert not ({"T5", "X5", "F5"} & ai.LEAPERS)
    for name in ("T5", "X5", "F5"):
        cells = MASTER[name]["orientations"][0]
        assert max(x for x, _ in cells) == 2 and max(y for _, y in cells) == 2, name
    for cells in (MASTER[n]["orientations"][0] for n in ("V5", "W5", "Z5")):
        assert ai.is_leaper(cells)


def test_crossing_needs_one_diagonal_mine_and_one_theirs():
    """使用者給的微觀圖 `1 2 / 3 4`：對手佔 2、3，我佔 4，落 1 即完成跨越。

    兩種對角線方向都要成立，而落 2 或落 3（對手那條）不算。
    """
    b = Board()
    b.place(6, 5, I1, 2)                 # 2
    b.place(5, 6, I1, 2)                 # 3
    b.place(6, 6, I1, 1)                 # 4
    assert not ai.has_crossed(b.owner_bits[1], b.owner_bits[2])
    b.place(5, 5, I1, 1)                 # 1
    assert ai.has_crossed(b.owner_bits[1], b.owner_bits[2])
    # 換個方向：跨越棋佔 {左上, 左下} 的那條也一樣
    c = Board()
    c.place(5, 5, I1, 2)
    c.place(6, 6, I1, 2)
    c.place(5, 6, I1, 1)
    assert not ai.has_crossed(c.owner_bits[1], c.owner_bits[2])
    c.place(6, 5, I1, 1)
    assert ai.has_crossed(c.owner_bits[1], c.owner_bits[2])


def test_has_crossed_needs_no_history():
    """判定只需要棋盤：同一個盤面用不同的落子順序排出來，答案必須一樣。"""
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
    other.place(4, 6, I1, 1)                    # 與 b 同一盤面，順序不同
    assert other.grid == b.grid
    assert ai.has_crossed(other.owner_bits[1], other.owner_bits[2])
    # 再算一次不變
    assert ai.has_crossed(b.owner_bits[1], b.owner_bits[2]) == \
        ai.has_crossed(b.owner_bits[1], b.owner_bits[2])


def test_key_cell_completes_a_crossing_when_filled():
    """關鍵格填上去就跨越，而且對自己的棋一定是角對角、不共邊。"""
    b = crossing_board()
    key = ai.key_cells(b.owner_bits[1], b.owner_bits[2], b.empty_bits)
    assert sorted((i % 20, i // 20) for i in range(400) if (key >> i) & 1) \
        == [(4, 6), (6, 6)]
    reach = b.reach(1)
    for c in [i for i in range(400) if (key >> i) & 1]:
        assert ai.has_crossed(b.owner_bits[1] | (1 << c), b.owner_bits[2])
        assert b.can_place(c % 20, c // 20, I1, 1, None, reach), \
            "a key square must be reachable by the corner-contact rule"


# ---------------------------------------------------------- 階段限制

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
    """規則 1、2：跨越棋先用完，再用能伸 6 格的樹，最後才回到一般局面。"""
    _g, log = intruder_game(1)
    used = [mv[0] for mv in log if mv]
    first_six = used[:6]
    assert set(first_six[:3]) <= ai.LEAPERS, first_six
    assert set(first_six[3:6]) <= ai.STRETCHERS, first_six
    assert len(used) > 6


def test_phase_degrades_instead_of_returning_none():
    """跨越棋一格都下不了時必須退回其他棋塊。

    `choose_move` 回傳 None 的唯一意義是「真的無棋可下」，所以這裡的局面
    明明還有 I1 可下，就一定要回一個合法落子。
    """
    b = Board()
    b.place(0, 0, I1, 1)                     # owner 1 只剩自己的角
    for y in range(4):
        for x in range(4):
            if (x, y) in ((0, 0), (1, 1)):
                continue
            b.place(x, y, I1, 2)            # 把 (1,1) 以外整塊封死
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


# -------------------------------------------------- 規則 3 / 4 / 5

def crossing_board():
    """owner 1 走 (0,0)→(5,5) 的斜線；owner 2 的 (5,6)+(6,7) 貼在旁邊。

    (5,7) 因此是關鍵格：填上去就完成跨越，而 (6,6) 一旦是我的棋就開出
    這個位置。
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
    """規則 3：尚未跨越的對手就在旁邊時，會挑一個佔完之後還能跨越的落子。"""
    b = crossing_board()
    hand = ["I3", "O4", "I2", "I1"]           # 一般局面，階段不介入
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
    """對手預判扣的是**權重**分，量級遠大於規則分；讓它參與排序等於把規則
    整個蓋掉，所以入侵者不走那一段。"""
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
    """沒有對手貼上來時關鍵格不存在，這條規則不該憑空發動。"""
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
    """規則 4：跨越那一步以可放空格數最大化為主，並懲罰可用交點太少。"""
    assert ai.crossing_bonus(60, 6) > ai.crossing_bonus(10, 6)
    assert ai.crossing_bonus(60, ai.SEAL_MIN) > ai.crossing_bonus(60, 0)
    # 不跨越時可放空格數同樣有份，但權重只有一半
    assert ai.crossing_bonus(60, 6) - ai.W_CROSS > 2.0 * (ai.W_SQUARES * 10)


def test_rule5_five_cells_unless_the_square_is_strategic():
    """規則 5：一般局面 5 格優先；關鍵格上 4 格、3 格才贏得過。"""
    assert ai.size_bonus(5, False) > ai.size_bonus(4, False)
    assert ai.size_bonus(4, False) > ai.size_bonus(3, False)
    assert ai.size_bonus(3, True) > ai.size_bonus(5, False)
    assert ai.size_bonus(4, True) > ai.size_bonus(5, False)
    assert ai.size_bonus(5, True) > ai.size_bonus(4, True)


# -------------------------------------------------------- 合法性 / 效能

def test_intruder_moves_are_legal_in_a_real_game():
    g, log = intruder_game(3)
    assert len(log) > 5
    # 重跑一次並在每一手當下檢查角對角規則：最終盤面無法重驗，因為多格棋
    # 跟自己共邊是正常的。
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


# -------------------------------------------------------------- 抽樣

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
    """4 種人格都是 personality 為 key 的排行榜條目；缺席的局不影響其他人。"""
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
