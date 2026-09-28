"""優化者：先放最大棋塊、再留住可放空間，緊急時先救只剩一個落點的棋。

這個人格的目標函式只有一個量——落子後自己還有幾個可放空格——因為積分就是
「餘格愈少愈好」。這裡把三件事釘住：棋塊大小的優先、可放空間的最大化、
以及緊急規則會蓋過前兩者（且不蓋過開局的角位規則）。
"""
import random
import time

import ai
from board import Board
from config import PERSONALITY_ORDER
from game import Game
from pieces import MASTER
from test_ai import assert_corner_contact

I1 = MASTER["I1"]["orientations"][0]


def playout(seed, moves, brains=None):
    """一個只由**權重式人格**走出來的局面。

    三個 AI 席位是明確釘死的，沒有走 `set_player_color` 的抽籤：抽到誰會隨著
    人格池變動而漂移，那樣一改池子，這裡的局面就跟著變，底下每個斷言測的就不是
    本來要測的那個局面了。停在這裡的局面與優化者無關，所以它也不會隨優化者的
    行為改變而漂移。
    """
    rng = random.Random(seed)
    g = Game(rng)
    g.set_player_color("blue")
    for o, key in enumerate(("chess", "fox", "wolf"), start=1):
        g.owner_key[o] = key
        g.brains[o] = ai.make_brain(key, rng)
    if brains is not None:
        g.brains = brains
    g.start()
    n = 0
    while g.state == "PLAYING" and n < moves:
        owner = g.current_owner()
        mv = ai.choose_move(g.board, g.hands[owner].names, owner, g.brains[owner],
                            rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner), reach=g.reach(owner))
        if mv:
            g.act(*mv)
        else:
            g.act_pass()
        n += 1
    return g



def optimizer_brain():
    return ai.make_brain("optimizer", random.Random(0))


def pick(board, hand, owner, must_cover=None, reach=None, seed=0):
    return ai.choose_move(board, hand, owner, optimizer_brain(), random.Random(seed),
                         other_brains=None, must_cover=must_cover, reach=reach)


def legal_count(board, name, oi, base, ctx):
    return ai.place_state(board, name, oi, base, ctx)[4].bit_count()


# ------------------------------------------------------------ 可放空間

def test_optimizer_maximises_placeable_squares():
    """同一種棋塊之間，挑「放完之後自己還能下最多位置」的那一格。

    獨立枚舉全部合法候選重算一次，不靠 `rescore` 自己的結果。
    """
    g = playout(0, 12)
    owner = 1
    hand = ["L5", "N5", "I1"]
    board, reach = g.board, g.reach(owner)
    mv = pick(board, hand, owner, reach=reach)
    assert mv is not None
    name, oi, x, y = mv
    assert MASTER[name]["size"] == 5
    ctx = ai.board_context(board, hand, owner, None)
    got = legal_count(board, name, oi, x + y * 20, ctx)
    best = got
    empt = board.empty_bits
    for cand in ("L5", "N5", "I1"):
        for o, od in ai.ODIRS[cand].items():
            for base in _bases(od, empt, reach):
                best = max(best, legal_count(board, cand, o, base, ctx))
    assert got == best, (got, best)


def _bases(od, empt, reach):
    mask = ai._legal_bases(od, empt, reach)
    out = []
    while mask:
        low = mask & -mask
        mask ^= low
        out.append(low.bit_length() - 1)
    return out


def test_optimizer_prefers_the_largest_piece():
    """沒有緊急棋塊時，先放能放下的最大棋塊。"""
    g = playout(0, 12)
    owner = 1
    board, reach = g.board, g.reach(owner)
    counts = ai.placement_counts(board, ["F5", "I5", "I1"], owner, None)
    assert counts["I1"] > 1, counts
    assert not [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES], counts
    mv = pick(board, ["F5", "I5", "I1"], owner, reach=reach)
    assert MASTER[mv[0]]["size"] == 5, mv
    # 手上只剩小棋時就改放小棋（階段是軟的，永遠不會卡住）
    mv = pick(board, ["I3", "I2", "I1"], owner, reach=reach)
    assert MASTER[mv[0]]["size"] == 3, mv


# ------------------------------------------------------------ 緊急規則

def test_optimizer_places_the_urgent_piece_first():
    """I1 只剩一個落點時先放它，即使 5 格棋還有十幾個落點。"""
    g = playout(0, 6)
    owner = 1
    board, reach = g.board, g.reach(owner)
    hand = list(g.hands[owner].names)
    counts = ai.placement_counts(board, hand, owner, g.must_cover(owner))
    # 局勢前提：I1 只剩一個落點，5 格棋還很寬鬆
    assert g.must_cover(owner) is None
    assert counts["I1"] == 1, counts
    five = {k: v for k, v in counts.items() if MASTER[k]["size"] == 5}
    assert max(five.values()) >= 5, five
    ctx = ai.board_context(board, hand, owner, g.must_cover(owner))
    brain = optimizer_brain()
    full = brain.context(board, hand, owner, g.must_cover(owner), reach)
    assert "I1" in full["urgent"], full["urgent"]
    mv = pick(board, hand, owner, reach=reach)
    assert mv[0] in full["urgent"], (mv, full["urgent"])
    assert MASTER[mv[0]]["size"] < 5, mv


def test_urgent_rescue_prefers_the_bigger_piece():
    """同樣只剩一個落點時先救大的：一樣救不回來，少留 5 格比較不痛。"""
    g = playout(0, 6)
    owner = 1
    board, reach = g.board, g.reach(owner)
    hand = list(g.hands[owner].names)
    brain = optimizer_brain()
    full = brain.context(board, hand, owner, g.must_cover(owner), reach)
    counts = ai.placement_counts(board, hand, owner, g.must_cover(owner))
    urgent = [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES]
    assert len(urgent) > 1, urgent
    assert full["urgent"] == sorted(urgent, key=lambda n: -MASTER[n]["size"]), full
    assert MASTER[full["urgent"][0]]["size"] == max(MASTER[n]["size"] for n in urgent)


def test_opening_ignores_the_urgent_rule():
    """開局第一手時落點數是角位規則壓出來的，不是棋盤變擠的訊號。

    那時 O4 這種棋只有一種放法，若把它當成「急需」就會把第一手花在 4 格棋
    上，直接違背「優先放 5 格」。
    """
    b = Board()
    brain = optimizer_brain()
    ctx = brain.context(b, list(MASTER.keys()), 1, (0, 0), None)
    assert ctx["urgent"] == [], ctx["urgent"]
    assert ai.placement_counts(b, list(MASTER.keys()), 1, (0, 0))["O4"] == 1
    mv = pick(b, list(MASTER.keys()), 1, must_cover=(0, 0), reach=None)
    assert MASTER[mv[0]]["size"] == 5, mv
    assert b.can_place(mv[2], mv[3], MASTER[mv[0]]["orientations"][mv[1]], 1,
                       (0, 0), None)


def test_a_dead_piece_is_not_urgent():
    """落點數 0 的棋塊救不回來，不算急需（而且它根本沒有候選落點）。"""
    b = Board()
    b.place(0, 0, I1, 1)
    for y in range(4):
        for x in range(4):
            if (x, y) in ((0, 0), (1, 1)):
                continue
            b.place(x, y, I1, 2)
    reach = b.reach(1)
    hand = list(MASTER.keys())
    counts = ai.placement_counts(b, hand, 1, None)
    assert counts["I1"] == 1 and counts["F5"] == 0, counts
    ctx = optimizer_brain().context(b, hand, 1, None, reach)
    assert ctx["urgent"] == ["I1"], ctx["urgent"]
    mv = pick(b, hand, 1, reach=reach)
    assert mv == ("I1", 0, 1, 1), mv
    assert b.can_place(1, 1, I1, 1, None, reach)


# -------------------------------------------------------- 合法性 / 效能

def test_optimizer_moves_are_legal_in_a_real_game():
    """每一手都要滿足角對角、不共邊。最終盤面無法重驗，所以逐手檢查。"""
    rng = random.Random(5)
    g = Game(rng)
    g.set_player_color("blue")
    g.brains[1] = ai.make_brain("optimizer", rng)
    g.owner_key[1] = "optimizer"
    g.start()
    checked = 0
    while g.state == "PLAYING" and checked < 10:
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
            checked += 1
        g.act(*mv)


def test_optimizer_move_speed():
    board = playout(0, 40).board
    brain = optimizer_brain()
    reach = board.reach(1)
    t0 = time.perf_counter()
    for i in range(20):
        mv = ai.choose_move(board, list(MASTER.keys()), 1, brain,
                            random.Random(i), other_brains=None, reach=reach)
        assert mv is not None
    dt = (time.perf_counter() - t0) / 20
    assert dt < 1.5, dt


def test_optimizer_beats_the_weighted_personalities():
    """這個人格就是照分數定義的，所以它應該在餘格上真的贏。"""
    keys = ["optimizer", "wolf", "fox"]
    left = dict.fromkeys(keys + ["player"], 0)
    games = 12
    for seed in range(games):
        rng = random.Random(1000 + seed)
        g = Game(rng)
        g.set_player_color("blue")
        g.owner_key[1], g.owner_key[2], g.owner_key[3] = keys
        g.brains = {0: ai.make_brain("chess", rng)}
        for i, k in enumerate(keys, start=1):
            g.brains[i] = ai.make_brain(k, rng)
        g.start()
        while g.state == "PLAYING":
            owner = g.current_owner()
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner), reach=g.reach(owner),
                                other_must_cover={o: g.must_cover(o) for o in range(4)},
                                other_reach={o: g.reach(o) for o in range(4)})
            if mv:
                g.act(*mv)
            else:
                g.act_pass()
        for owner, k in ((0, "player"), (1, keys[0]), (2, keys[1]), (3, keys[2])):
            left[k] += g.remaining_cells(owner)
    avg = {k: v / float(games) for k, v in left.items()}
    assert avg["optimizer"] == min(avg.values()), avg
    assert avg["optimizer"] < ai.SQ_CAP / 4, avg


def test_optimizer_is_part_of_the_personality_pool():
    assert "optimizer" in ai.personality_keys()
    assert "optimizer" in PERSONALITY_ORDER
    assert isinstance(optimizer_brain(), ai.OptimizerBrain)
    assert optimizer_brain().uses_lookahead is False
    assert optimizer_brain().mistake_rate == 0.0
