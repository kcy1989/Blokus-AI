"""人格共用的基底型別與權重檔。

`Profile` 是權重式人格的六個權重加一個出錯率；`Brain` 是「怎麼看一盤棋」的
介面。候選枚舉、對手預判、抽籤都在 `chooser.choose_move`，人格本身只負責三
件事：

    context   一盤棋裡與候選無關的量，先算一次
    restrict  階段（軟）過濾
    rescore   重新排分
"""
from dataclasses import dataclass

# 人格總表：權重式的三種，加上三種規則式的。
INTRUDER_KEY = "intruder"
OPTIMIZER_KEY = "optimizer"
BUILDER_KEY = "builder"
RULE_KEYS = (INTRUDER_KEY, OPTIMIZER_KEY, BUILDER_KEY)


@dataclass(frozen=True)
class Profile:
    """權重式人格的權重檔。

    六個 `w_*` 是候選評分的線性項：`w_corner` 角位連通區塊、`w_center` 靠近
    盤心、`w_block` 貼著對手的棋、`w_defend` 貼著自己棋的對手、`w_open` 鄰接
    的空格數、`w_large` 棋塊大小。`mistake_rate` 是每一步從短名單裡故意選次
    佳解的機率。
    """
    w_corner: float
    w_center: float
    w_block: float
    w_defend: float
    w_open: float
    w_large: float
    mistake_rate: float


def make_profile(spec, rng):
    """把一組基準權重乘上 0.7～1.3 的隨機擾動。

    擾動讓同一個人格每局略有不同，否則排行榜會變成一組固定對局的重播。
    """
    wc, wc2, wb, wd, wo, wl, mr = spec
    u = rng.uniform
    return Profile(
        wc * u(0.7, 1.3), wc2 * u(0.7, 1.3), wb * u(0.7, 1.3), wd * u(0.7, 1.3),
        wo * u(0.7, 1.3), wl * u(0.7, 1.3), mr * u(0.7, 1.3),
    )


class Brain:
    """評估層。

    候選枚舉、對手預判、抽籤都留在 `choose_move`；人格只負責「怎麼看一盤棋」：
    `context` 收一盤的共享前置量，`restrict` 做階段（軟）過濾，
    `rescore` 重新排分。
    """

    key = "chess"
    # 對手預判用的是**權重**評分，量級跟規則式的目標函式完全不同。規則式人格
    # 不用它：那一項會直接蓋掉規則排序，等於白算。
    uses_lookahead = True

    def __init__(self, key, profile):
        self.key = key
        self.profile = profile
        self.mistake_rate = profile.mistake_rate

    def context(self, board, hand_names, owner, must_cover, reach):
        return None

    def restrict(self, cands, ctx):
        return cands

    def rescore(self, cands, ctx):
        return cands


class WeightedBrain(Brain):
    """狼／棋手／狐狸共用的基底：權重加總，`restrict` 與 `rescore` 都是恆等。

    真正的差別全在 `Profile` 的六個權重上，所以三者的評分路徑完全相同。
    """
