"""優化者：先放最大的棋塊，再挑最能留住可放空格的落點。

目標函式只有一個量——**落子後自己還有幾個可放空格**。積分是「餘格愈少愈
好」，所以能放掉的格數就是分數，而一塊 5 格棋放掉一次就賺 5 格；這是為什
麼先挑大的，而同一種棋塊之間就比誰留下的可放空間多。

緊急規則是它的安全閥：某一塊棋只剩唯一一個落點時先把它放掉。那個位置被封
掉之後，那塊棋就永遠留在手上了——在這個記分法下等於直接送掉幾格。

同樣不抽籤、不做對手預判，理由與 `intruder` 相同。
"""
from . import formulas as F
from .base import OPTIMIZER_KEY, Brain
from pieces import MASTER

# 優化者的緊急門檻：棋塊只剩這麼多落點就先放掉它。
URGENT_PLACES = 1
# 第二階段（昂貴評估）的安全上限。
OPTIMIZER_SCAN = 1200


class OptimizerBrain(Brain):
    """效率式人格：先放最大的棋塊，再挑最能留住可放空格的落點。

    目標函式只有一個量——**落子後自己還有幾個可放空格**。積分是「餘格愈少
    愈好」，所以能放掉的格數就是分數，而一塊 5 格棋放掉一次就賺 5 格；這是
    為什麼先挑大的，而同一種棋塊之間就比誰留下的可放空間多。

    緊急規則是它的安全閥：某一塊棋只剩唯一一個落點時先把它放掉。那個位置被
    封掉之後，那塊棋就永遠留在手上了——在這個記分法下等於直接送掉幾格。
    """

    key = OPTIMIZER_KEY
    uses_lookahead = False

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        self.mistake_rate = 0.0

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = F.board_context(board, hand_names, owner, must_cover)
        # 開局第一手時「只剩一個落點」是角位規則壓出來的，還不是棋盤變擠的
        # 訊號——那時棋塊的落點數天生就少（O4 更是只有一種放法）。所以緊急
        # 規則不適用於開局，否則優化者會把第一手花在 O4／I1 上而不是 5 格棋。
        if must_cover is not None:
            ctx["urgent"] = []
        else:
            counts = F.placement_counts(board, hand_names, owner, must_cover)
            # 同樣只剩一個落點時先救大的：一樣救不回來，少留 5 格比較不痛。
            ctx["urgent"] = sorted((n for n, k in counts.items()
                                    if 0 < k <= URGENT_PLACES),
                                   key=lambda n: -MASTER[n]["size"])
        return ctx

    def restrict(self, cands, ctx):
        """緊急規則優先，其次是「能下得下的最大棋塊」。

        兩者都是軟限制：篩不出東西就退回原來的候選清單，所以這裡永遠不會
        讓 `choose_move` 回傳 None。
        """
        if ctx["urgent"]:
            keep = set(ctx["urgent"])
            sub = [c for c in cands if c[1] in keep]
            if sub:
                return sub
        by_size = {}
        for c in cands:
            by_size.setdefault(F.ODIRS[c[1]][c[2]]["size"], []).append(c)
        return by_size[max(by_size)] if by_size else cands

    def rescore(self, cands, ctx):
        scored = [(F.place_state(ctx["board"], n, oi, b, ctx)[4].bit_count(),
                   n, oi, b)
                  for _s, n, oi, b in cands[:OPTIMIZER_SCAN]]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored
