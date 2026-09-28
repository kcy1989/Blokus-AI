"""築城者：估計剩下的手牌有多少格再也放不下去。

目標函式的主項是 `formulas.pack_lost`——把剩餘手牌用 FFD 裝進落子後的活區
塊，裝不進去的格數以負權重計。因為記分是「餘格愈少愈好」，而餘下的格子只
有一種來源（手上還有棋，但盤面沒有它的合法落點），所以這個量跟分數是同一
個尺度。

大小偏好（`BUILDER_W_SIZE`）與可放空間總數（`BUILDER_W_TOTAL`）只是次要項：
前者讓它在一般局面看起來一樣偏愛 5 格，後者讓它在同樣「都不會送掉」的落點
之間挑寬鬆的那個。它們都刻意小到讓不過主項——所以不設階段、不設緊急規則：
那些都是主項的自然結果，加了反而變成兩套規則打架。

同樣不抽籤、不做對手預判，理由與 `intruder` 相同。
"""
from . import formulas as F
from .base import BUILDER_KEY, Brain
from pieces import MASTER

# 築城者第二階段的人數上限。它沒有階段過濾，要評的是**完整**候選清單（實測
# 數千個），所以那個上限是真的會擋到東西的：先用便宜的「可放空格數」篩，再對
# 前 BUILDER_SCAN 個算閉包與裝箱。
BUILDER_SCAN = 800
# 築城者的權重。`BUILDER_W_LOST` 的單位是格（0～89），而後兩項最多只有 ±2 與
# 4，所以「送掉 5 格」一定蓋得過任何大小或總數的差別——這符合記分：餘格就是
# 分數。大小偏好只是讓它在一般局面看起來一樣偏愛 5 格。
BUILDER_W_LOST = 1.0
BUILDER_W_SIZE = 0.35
BUILDER_W_TOTAL = 0.05


class BuilderBrain(Brain):
    """築城者：估計剩下的手牌有多少格再也放不下去。

    目標函式的主項是 `pack_lost`——把剩餘手牌用 FFD 裝進落子後的活區塊，裝不
    進去的格數以負權重計。因為記分是「餘格愈少愈好」，而餘下的格子只有一種來
    源（手上還有棋，但盤面沒有它的合法落點），所以這個量跟分數是同一個尺度。

    大小偏好（`BUILDER_W_SIZE`）與可放空間總數（`BUILDER_W_TOTAL`）只是次要
    項：前者讓它在一般局面看起來一樣偏愛 5 格，後者讓它在同樣「都不會送掉」
    的落點之間挑寬鬆的那個。它們都刻意小到讓不過主項——所以不設階段、不設緊急
    規則：那些都是主項的自然結果，加了反而變成兩套規則打架。
    """

    key = BUILDER_KEY
    uses_lookahead = False

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        self.mistake_rate = 0.0

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = F.board_context(board, hand_names, owner, must_cover)
        # 每種棋塊放掉之後，剩下的手牌大小由大到小。整手棋只算一次（21 筆），
        # 免得每個候選都做一次 list.remove。手牌的棋塊名不重複，所以用名字當
        # key 是安全的。
        sizes = sorted((MASTER[n]["size"] for n in hand_names), reverse=True)
        ctx["sizes_after"] = {name: tuple(sizes[:i] + sizes[i + 1:])
                              for i, name in enumerate(hand_names)}
        return ctx

    def restrict(self, cands, ctx):
        """不設階段：見類別說明，`lost` 已經吸收了那些規則。"""
        return cands

    def rescore(self, cands, ctx):
        """兩段式。候選沒有經過階段過濾，所以第一段是必要的一層，不只是加速。

        第一段（全體，便宜）只算 `place_state` 的可放空格數——那是優化者那條
        已經被驗證過、方向正確的指標，用來粗篩。第二段才對前 `BUILDER_SCAN` 個
        算閉包、活區塊與裝箱，完整評分。
        """
        board = ctx["board"]
        wide = [(F.place_state(board, n, oi, b, ctx)[4].bit_count(), n, oi, b)
                for _s, n, oi, b in cands]
        wide.sort(key=lambda t: t[0], reverse=True)
        scored = [self._score(n, oi, b, ctx) for _total, n, oi, b
                  in wide[:BUILDER_SCAN]]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _score(self, name, oi, base, ctx):
        board = ctx["board"]
        _own, empt_after, need, avoid, legal = F.place_state(board, name, oi, base, ctx)
        total = legal.bit_count()
        bins = F.fill_components(F.fillable_closure(empt_after, avoid, legal))
        lost = F.pack_lost(ctx["sizes_after"][name], bins)
        od = F.ODIRS[name][oi]
        s = (-BUILDER_W_LOST * lost + BUILDER_W_SIZE * (od["size"] - 3)
             + BUILDER_W_TOTAL * min(total, F.SQ_CAP))
        return (s, name, oi, base)
