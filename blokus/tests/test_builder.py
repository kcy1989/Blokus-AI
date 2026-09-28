"""築城者：估計還有多少格塞不進活區塊。

這個人格補的是優化者的洞：優化者只量「還有幾個可放空格」的**數量**，兩個 4 格
活區塊與一個 8 格活區塊在它眼裡一樣好，但前者任何 5 格棋都塞不進去。所以主項
是 `pack_lost`——用 FFD 把剩餘手牌裝進活區塊，裝不進去的格數就是估計送掉的格
數，而記分正是「餘格愈少愈好」。

這個設計最可能死的地方是 `pack_lost` 沒有分化（早期閉包飽和、`lost` 恆為 0，
於是退化��「更差的優化者」），所以最後一個測試專門盯這件事。
"""
import random
import time

import ai
from board import dilate
from config import PERSONALITY_ORDER
from game import Game
from pieces import MASTER
from test_ai import assert_corner_contact


def playout(seed, moves):
    """一個只由**權重式人格**走出來的局面。

    三個 AI 席位明確釘死，不走 `set_player_color` 的抽籤——人格池一變，局面就跟
    著漂移，底下每個斷言測的就不是本來要測的局面了。停在這裡的局面與築城者無
    關，所以它也不會隨築城者的行為改變而漂移。
    """
    rng = random.Random(seed)
    g = Game(rng)
    g.set_player_color("blue")
    for o, key in enumerate(("chess", "fox", "wolf"), start=1):
        g.owner_key[o] = key
        g.brains[o] = ai.make_brain(key, rng)
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


def builder_brain():
    return ai.make_brain("builder", random.Random(0))


def builder_ctx(board, hand, owner, must_cover=None):
    return builder_brain().context(board, hand, owner, must_cover, None)


def legal_cands(board, hand, owner, must_cover=None, reach=None):
    """獨立枚舉全部合法候選 [(name, oi, base)]，不靠 `choose_move` 的中間結果。"""
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * 20)
    out = []
    for name in hand:
        for oi, od in ai.ODIRS[name].items():
            mask = ai._legal_bases(od, board.empty_bits, reach, cbit)
            while mask:
                low = mask & -mask
                mask ^= low
                out.append((name, oi, low.bit_length() - 1))
    return out


def pack_stats(board, hand, owner, must_cover=None, reach=None):
    """每一個合法候選的 (lost, 可放空格數, 棋塊大小, name, oi, base)。"""
    ctx = builder_ctx(board, hand, owner, must_cover)
    rows = []
    for name, oi, base in legal_cands(board, hand, owner, must_cover, reach):
        _own, empt_after, _need, avoid, legal = ai.place_state(board, name, oi,
                                                              base, ctx)
        bins = ai.fill_components(ai.fillable_closure(empt_after, avoid, legal))
        rows.append((ai.pack_lost(ctx["sizes_after"][name], bins),
                     legal.bit_count(), MASTER[name]["size"], name, oi, base))
    return rows


# ------------------------------------------------------------ 純函式

def test_fillable_closure_respects_the_rules():
    """閉包必須含住每一個合法落點，且只落在空格、不碰 `avoid`。"""
    g = playout(0, 12)
    owner = 1
    hand = ["F5", "N5", "I3", "I2", "I1"]
    ctx = builder_ctx(g.board, hand, owner)
    name, oi, base = legal_cands(g.board, hand, owner, None, g.reach(owner))[0]
    _own, empt_after, _need, avoid, legal = ai.place_state(g.board, name, oi, base, ctx)
    fill = ai.fillable_closure(empt_after, avoid, legal)
    assert fill & legal == legal, "legal 必須整個落在閉包裡"
    assert fill & ~empt_after == 0, "閉包不能越出空格"
    assert fill & avoid == 0, "閉包不能壓到自己的棋"
    # 每一格都沿 4 鄰接走得到某個 legal 格（closure 的定義）
    reached = legal
    while True:
        grown = reached | (dilate(reached) & fill)
        if grown == reached:
            break
        reached = grown
    assert reached & fill == fill, "有格子走不回 legal"
    assert fill.bit_count() > legal.bit_count(), "閉包應該比 legal 大（泛洪有做事）"


def test_fill_components_partition_fill():
    """活區塊兩兩不交、聯集等於 `fill`、每塊 4-連通、面積加總等於位元數。"""
    g = playout(0, 12)
    owner = 1
    hand = ["L5", "N5", "I3", "I2", "I1"]
    ctx = builder_ctx(g.board, hand, owner)
    board = g.board
    for name, oi, base in legal_cands(board, hand, owner, None, g.reach(owner))[:40]:
        _own, empt_after, _need, avoid, legal = ai.place_state(board, name, oi,
                                                              base, ctx)
        fill = ai.fillable_closure(empt_after, avoid, legal)
        assert fill, "候選至少要留下一個合法落點"
        bins = ai.fill_components(fill)
        assert sum(bins) == fill.bit_count(), (name, oi, base)
        assert bins == sorted(bins, reverse=True), bins
        # 逐塊驗證不交性與連通性（自己重建一次，不靠 `fill_components` 的結果）
        rest = seen = 0
        rest = fill
        while rest:
            comp = rest & -rest
            rest ^= comp
            while True:
                grown = comp | (dilate(comp) & rest)
                if grown == comp:
                    break
                comp = grown
            assert not comp & seen, "區塊重疊"
            assert comp & ~fill == 0, "區塊跑到 `fill` 外面"
            seen |= comp
            rest &= ~comp
        assert seen == fill


def test_pack_lost_prefers_one_big_region():
    """**整個設計的單點測試**：同樣的總量，整塊遠比兩塊碎片能裝。

    手牌 [5,3,2,1]：
      A  兩個 4 格活區塊 → F5 塞不進（送 5）、I3 進第一塊（剩 1）、I2 進第二塊
         （剩 2）、I1 進 → 送掉 5 格
      B  一個 8 格活區塊 → F5 進（剩 3）、I3 進（剩 0）、I2 送 2、I1 送 1
         → 送掉 3 格

    優化者的「可放空格總數」在 A 與 B 幾乎一樣，會選 A 然後永遠送掉那 5 格。
    這是 FFD 演算法的實際算結果——計畫裡那張圖原本寫 A 送 7 格，是手算沒把
    「I2 也塞得進第二個 4 格活區塊」算進去。
    """
    sizes = [5, 3, 2, 1]
    assert ai.pack_lost(sizes, [4, 4]) == 5
    assert ai.pack_lost(sizes, [8]) == 3
    assert ai.pack_lost(sizes, [4, 4]) > ai.pack_lost(sizes, [8])


def test_pack_lost_is_zero_when_everything_fits():
    """總容量夠、而且形狀對得起來時，送掉的格數是 0。"""
    hand = sorted((MASTER[n]["size"] for n in MASTER), reverse=True)
    assert len(hand) == 21
    assert sum(hand) == 89
    assert ai.pack_lost(hand, [89]) == 0
    assert ai.pack_lost(hand, [40, 49]) == 0
    assert ai.pack_lost(hand, [30, 59]) == 0
    # 容量不夠就一定有剩餘（兩個 1 格活區塊只接得住最小的那塊棋）
    assert ai.pack_lost([5, 5], [4]) == 10
    assert ai.pack_lost(hand, [1, 1]) == 89 - 1


def test_pack_lost_is_first_fit_decreasing():
    """FFD 要求大小由大到小：順序不對，結果會整個變掉。"""
    assert ai.pack_lost([5, 3, 3], [5, 6]) == 0
    assert ai.pack_lost([3, 3, 5], [5, 6]) == 5


# ------------------------------------------------------------ 目標函式

def test_builder_maximises_packability():
    """真實中盤局面：獨立枚舉全部候選重算一次，斷言被選中的那一手就是最高分。

    這裡刻意連次要項一起比（不只比 `lost`），因為被選中的就是完整評分最高的
    那一手；只比 `lost` 反而會放寬到容許回歸。
    """
    g = playout(0, 12)
    owner = 1
    hand = ["F5", "N5", "I3", "I2", "I1"]
    board, reach = g.board, g.reach(owner)
    cands = legal_cands(board, hand, owner, None, reach)
    assert len(cands) <= ai.BUILDER_SCAN, "這一局候選少於掃描上限，才談得上窮舉"
    mv = ai.choose_move(board, hand, owner, builder_brain(), random.Random(0),
                        other_brains=None, reach=reach)
    assert mv is not None
    name, oi, x, y = mv
    ctx = builder_ctx(board, hand, owner)
    brain = builder_brain()
    got = brain._score(name, oi, x + y * 20, ctx)[0]
    scores = [brain._score(n, o, b, ctx)[0] for n, o, b in cands]
    assert got == max(scores), (got, max(scores))


def test_builder_prefers_bigger_pieces_when_packability_ties():
    """`lost` 與可放空格數都一樣時，選 5 格那一手——證明 `BUILDER_W_SIZE` 有做事。"""
    g = playout(0, 12)
    owner = 1
    hand = ["F5", "I1"]
    board, reach = g.board, g.reach(owner)
    rows = pack_stats(board, hand, owner, None, reach)
    tie = {}
    for lost, total, size, name, _oi, _base in rows:
        tie.setdefault((lost, total), set()).add(size)
    assert (0, 8) in tie and {1, 5} <= tie[(0, 8)], sorted(tie[(0, 8)])
    mv = ai.choose_move(board, hand, owner, builder_brain(), random.Random(0),
                        other_brains=None, reach=reach)
    assert MASTER[mv[0]]["size"] == 5, mv


def test_builder_never_returns_none_and_stays_legal():
    """整局走到結束，逐手驗角對角規則。"""
    rng = random.Random(5)
    g = Game(rng)
    g.set_player_color("blue")
    g.brains[1] = ai.make_brain("builder", rng)
    g.owner_key[1] = "builder"
    g.start()
    checked = 0
    while g.state == "PLAYING":
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
    assert checked > 0


def test_builder_move_speed():
    board = playout(0, 40).board
    brain = builder_brain()
    reach = board.reach(1)
    t0 = time.perf_counter()
    for i in range(20):
        mv = ai.choose_move(board, list(MASTER.keys()), 1, brain,
                            random.Random(i), other_brains=None, reach=reach)
        assert mv is not None
    dt = (time.perf_counter() - t0) / 20
    print("\n築城者單步 %.4f 秒" % dt)
    assert dt < 1.5, dt


def test_builder_is_in_the_pool_and_renders():
    assert "builder" in ai.personality_keys()
    assert "builder" in PERSONALITY_ORDER
    assert PERSONALITY_ORDER[-1] == "builder"
    from config import I
    from ui import PERSONA_ZH
    assert I["builder"] and I["builder_desc"]
    assert PERSONA_ZH["builder"] == I["builder"]
    brain = builder_brain()
    assert isinstance(brain, ai.BuilderBrain)
    assert brain.uses_lookahead is False
    assert brain.mistake_rate == 0.0


# ------------------------------------------------------------ 防退化

def test_packing_term_actually_discriminates():
    """**這個設計最可能死的地方**：若 `lost` 在候選之間從不分化，築城者就退化
    成「更差的優化者」。所以跑 20 局真實對局，統計每一手裡 `pack_lost` 有分化的
    比例，門檻 20%。
    """
    moves = discriminated = 0
    for seed in range(20):
        rng = random.Random(300 + seed)
        g = Game(rng)
        g.set_player_color("blue")
        g.brains[1] = ai.make_brain("builder", rng)
        g.owner_key[1] = "builder"
        g.start()
        while g.state == "PLAYING":
            owner = g.current_owner()
            if owner == 1:
                hand = g.hands[owner].names
                losts = {r[0] for r in pack_stats(g.board, hand, owner,
                                                  g.must_cover(owner),
                                                  g.reach(owner))}
                moves += 1
                if len(losts) > 1:
                    discriminated += 1
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner),
                                reach=g.reach(owner))
            if mv:
                g.act(*mv)
            else:
                g.act_pass()
    ratio = discriminated / float(moves)
    print("\npack_lost 分化比例 %.1f%%（%d/%d 手）" % (ratio * 100, discriminated, moves))
    assert ratio >= 0.20, ratio
