"""入侵者：靠「跨越」鑽進對手領地。

跨越是 Blokus 的勝負關鍵——一旦 2x2 方塊裡一條對角全是自己的、另一條有對
手，這局就贏了。入侵者用三個階段把這個目標拆開：

  規則 1  手上還有跨越棋，就只用跨越棋
  規則 2  否則用長手臂棋（L5／N5／I5）
  規則 3  都沒有了，用能佔到「關鍵格」的棋
  規則 4  這一手本身完成跨越 → 大幅加分，並要求跨過去之後還有地方可放
  規則 5  一般局面：5 格優先，戰略點放寬到 4 格、3 格

階段全部是**軟**限制：指定棋塊一格都下不了就退回下一階段。做成硬過濾的話
候選清單會被清空，`choose_move` 回傳 None，玩家就會被誤判成無棋可下而自動
過。

這個人格不抽籤（`mistake_rate = 0`）：戰略獎勵一被隨機性蓋掉就沒有意義，
「前 3 手只用跨越棋」這種性質也才檢查得起來。它也不做對手預判，因為預判
用的是權重評分，量級跟這裡的目標函式完全不同。
"""
from . import formulas as F
from .base import INTRUDER_KEY, Brain

# 第二階段（昂貴評估）的安全上限。階段限制加上角對角規則之後，候選通常只剩
# 幾百個（實測多不超過 400），所以這個上限幾乎不會擋到東西；它的作用只是
# 讓「每個候選都要模擬落子」這件事有個確定的花費上限。
INTRUDER_SCAN = 1200


class IntruderBrain(Brain):
    """規則式人格：依延伸度與可放空間挑棋，不做權重加總。"""

    key = INTRUDER_KEY
    uses_lookahead = False

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        # 規則式人格不抽籤：戰略獎勵一被隨機性蓋掉就沒有意義，「前 3 手只用
        # 跨越棋」這種性質也才檢查得起來。
        self.mistake_rate = 0.0

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = F.board_context(board, hand_names, owner, must_cover)
        ctx.update(F.crossing_context(board, hand_names, owner))
        return ctx

    def restrict(self, cands, ctx):
        """規則 1／2／5 的階段：跨越棋 → 長手臂 → 一般局面。

        每一階段都是軟的：該類棋一格都下不了就往下退，退到最後就是全部候選。
        做成硬過濾的話候選清單會被清空，`choose_move` 回傳 None，玩家就會被
        誤判成無棋可下而自動過。
        """
        for cls in (F.LEAPERS, F.STRETCHERS):
            sub = [c for c in cands if c[1] in cls]
            if sub:
                return sub
        return cands

    def rescore(self, cands, ctx):
        scored = []
        for _s, name, oi, base in cands[:INTRUDER_SCAN]:
            scored.append(self._score(name, oi, base, ctx))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _score(self, name, oi, base, ctx):
        od = F.ODIRS[name][oi]
        (own_after, empt_after, need, avoid, legal, fresh,
         anchors) = F.place_geometry(ctx["board"], name, oi, base, ctx)

        ext = F.extension(fresh, anchors)
        usable = F.cells_to_vertices(fresh).bit_count()
        squares = legal.bit_count()

        # 規則 4：這一手本身完成跨越。`pre` 是落子前的跨越方塊，用來扣掉
        # 「本來就跨越過、只是被這一手碰到」的那一格。
        crossed = False
        for o, m in ctx["opps"]:
            if F.cross_anchors(own_after, m) & (od["touch"] << base) \
                    & ~ctx["pre"][o]:
                crossed = True
                break

        # 規則 3：尚未跨越時搶關鍵格
        strategic = crossed
        if not crossed and ctx["uncrossed"]:
            strategic = self._sets_up(ctx, od, own_after, empt_after,
                                      need, avoid) or crossed

        s = F.size_bonus(od["size"], strategic)
        s += (F.crossing_bonus(squares, usable) if crossed
              else F.W_SQUARES * min(squares, F.SQ_CAP))
        s += F.W_VERTICES * min(usable, F.SQ_CAP)
        s += F.W_EXT * min(ext, F.EXT_CAP)
        return (s, name, oi, base)

    def _sets_up(self, ctx, od, own_after, empt_after, need, avoid):
        """佔下這一手之後，是否存在一個合法落子能完成跨越（規則 3 的關鍵格）。

        只檢查手上這塊棋與跨越棋，且只在尚未跨越的對手身上算——這是這條規則
        最貴的部分，其餘對手直接跳過。
        """
        reach = F.Reach(need, avoid)
        for o, m in ctx["uncrossed"]:
            key = F.key_cells(own_after, m, empt_after)
            if not key:
                continue
            kb = 0
            for off in od["offs"]:
                kb |= key >> off
            if kb and F._adjoining_bases(od, reach, 0) & kb \
                    & F._free_bases(od, empt_after):
                return True
            sp = ctx["selfpair"][o]
            for n, oi, lod in ctx["leaps"]:
                lkb = 0
                for off in lod["offs"]:
                    lkb |= key >> off
                hit = lkb | sp[(n, oi)]
                if not hit:
                    continue
                if F._adjoining_bases(lod, reach, 0) & hit \
                        & F._free_bases(lod, empt_after):
                    return True
        return False
