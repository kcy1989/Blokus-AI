"""AI personalities, scoring, lookahead."""
import math
import time
from dataclasses import dataclass

from board import (BLOCK_ANCHORS, COL0, COL_LAST, Reach, V, cells_to_vertices,
                   dilate, neighbors_of)
from config import CORNERS_IDX
from pieces import MASTER


SMALL_PENALTY = 0.25
SIM_EVAL_BUDGET = 3000
WALL_BUDGET = 0.9
REGION_CAP = 120
NEIGH = neighbors_of()

SHORTLIST_CORE = 34
SHORTLIST_SCAN = 400
CORNER_RESERVE = 6
CORNER_BAND = 12.0
PICK_TEMP = 5.0

# How many responses per opponent the lookahead is allowed to score. The pool
# is built from the pre-ranked "neutral" bases first; when the edge-adjacency
# rule empties that pool it falls back to an exhaustive search (see
# `_opponent_pool`), so a fixed K keeps the cost bounded either way.
OPP_POOL_K = 40
OPP_POOL_PER_ORIENT = 6

B = 20
N = B * B

ROW0 = (1 << B) - 1
ROWN = ROW0 << (B * (B - 1))

CENTER = []
for y in range(B):
    for x in range(B):
        d = min(abs(x - 9.5), abs(y - 9.5))
        CENTER.append(1.0 - d / 9.0)

PERSONALITY_SPECS = {
    "wolf": (4.0, 1.0, 2.0, 0.3, 0.5, 3.0, 0.15),
    "chess": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05),
    "fox": (0.5, 1.0, 3.0, 3.0, 2.0, 0.5, 0.25),
}

# 人格總表：權重式的三種，加上三種規則式的。
INTRUDER_KEY = "intruder"
OPTIMIZER_KEY = "optimizer"
BUILDER_KEY = "builder"
RULE_KEYS = (INTRUDER_KEY, OPTIMIZER_KEY, BUILDER_KEY)
# 優化者的緊急門檻：棋塊只剩這麼多落點就先放掉它。
URGENT_PLACES = 1
# 第二階段（昂貴評估）的安全上限。
OPTIMIZER_SCAN = 1200
# 築城者第二階段的人數上限。它沒有階段過濾，要評的是**完整**候選清單
# （實測數千個），所以那個上限是真的會擋到東西的：先用便宜的「可放空格數」篩，
# 再對前 BUILDER_SCAN 個算閉包與裝箱。
BUILDER_SCAN = 800
# 築城者的權重。`BUILDER_W_LOST` 的單位是格（0～89），而後兩項最多只有 ±2 與
# 4，所以「送掉 5 格」一定蓋得過任何大小或總數的差別——這符合記分：餘格就是
# 分數。大小偏好只是讓它在一般局面看起來一樣偏愛 5 格。
BUILDER_W_LOST = 1.0
BUILDER_W_SIZE = 0.35
BUILDER_W_TOTAL = 0.05


def personality_keys():
    return tuple(PERSONALITY_SPECS) + RULE_KEYS


@dataclass(frozen=True)
class Profile:
    w_corner: float
    w_center: float
    w_block: float
    w_defend: float
    w_open: float
    w_large: float
    mistake_rate: float


def make_profile(key, rng):
    wc, wc2, wb, wd, wo, wl, mr = PERSONALITY_SPECS[key]
    u = rng.uniform
    return Profile(
        wc * u(0.7, 1.3), wc2 * u(0.7, 1.3), wb * u(0.7, 1.3), wd * u(0.7, 1.3),
        wo * u(0.7, 1.3), wl * u(0.7, 1.3), mr * u(0.7, 1.3),
    )


def draw_personalities(rng, n=3):
    return rng.sample(list(personality_keys()), n)


def _block_touch(cells):
    """2x2 方塊的左上角位元，這一手會碰到的那些方塊。

    判斷「這一手有沒有完成跨越」時用它與跨越方塊位元求交，不必枚舉全盤。
    """
    touch = 0
    for dx, dy in cells:
        for ax, ay in ((dx - 1, dy - 1), (dx, dy - 1), (dx - 1, dy), (dx, dy)):
            if ax >= 0 and ay >= 0:
                touch |= 1 << (ax + ay * B)
    return touch


def _pair_shifts(offs, cells):
    """棋塊內部「角對角相鄰」的兩格，其 2x2 另兩格的位移量。

    這是跨越的另一種來源：跨越那條對角線的兩格都屬於同一手（而不是一格
    既有棋加一格新棋）。位移量若等於棋塊自己的 offset，代表「另一格」永遠
    被自己蓋住，該位元必須棄掉。
    """
    off_set = set(offs)
    out = set()
    for dx, dy in cells:
        for ex, ey in cells:
            sx, sy = ex - dx, ey - dy
            if abs(sx) == 1 and abs(sy) == 1:
                o1 = dy * B + dx
                out.add(o1 + (1 if sx > 0 else -1))
                out.add(o1 + (B if sy > 0 else -B))
    return tuple(sorted(s for s in out if s >= 0 and s not in off_set))


def _build_od(name, oi, cells):
    mx = max(x for x, _ in cells)
    my = max(y for _, y in cells)
    offs = [dy * B + dx for dx, dy in cells]
    m = 0
    for o in offs:
        m |= 1 << o
    bases = [x + y * B for y in range(B - my) for x in range(B - mx)]
    base_set = set(bases)
    valid = 0
    for b in bases:
        valid |= 1 << b
    corner = {}
    for ci, cidx in enumerate(CORNERS_IDX):
        for o in offs:
            b = cidx - o
            if b in base_set:
                corner[b] = corner.get(b, 0) | (1 << ci)
    csum = {}
    for b in bases:
        s = 0.0
        for o in offs:
            s += CENTER[b + o]
        csum[b] = s
    ranked = sorted(bases, key=lambda b: (corner.get(b, 0) * 3.0 + csum[b], b),
                    reverse=True)
    rank = {b: i for i, b in enumerate(ranked)}
    return {"m": m, "offs": offs, "bases": bases, "valid": valid, "corner": corner,
            "csum": csum, "rank": rank, "neutral": ranked[:20], "size": len(offs),
            "touch": _block_touch(cells), "pair_shifts": _pair_shifts(offs, cells)}


def _adjoining_bases(od, reach, cbit):
    """Bitmask of bases whose piece satisfies the corner-contact rule.

    Bit i of `reach.need >> o` is set exactly when the square base+o is in
    `need`, so a handful of big-int ORs replaces scanning every base. Bases
    that also land on `avoid` (edge contact with your own stones) are then
    subtracted.
    """
    if cbit:
        hit = 0
        for o in od["offs"]:
            hit |= od["m"] & (cbit >> o)
        return hit & od["valid"]
    need = bad = 0
    for o in od["offs"]:
        need |= reach.need >> o
        bad |= reach.avoid >> o
    return (need & ~bad) & od["valid"]


def _free_bases(od, empt):
    """位元 b = 這一手下在 b 時，每一格都是空的。

    每個 offset 各貢獻「該格為空」的 base 位元，取交集即全部為空。跨行的
    誤算會被 `od["valid"]` 濾掉。
    """
    out = od["valid"]
    for o in od["offs"]:
        out &= empt >> o
    return out


def _legal_bases(od, empt, reach=None, cbit=0):
    """這一手下在哪些 base 合法：角對角規則，且每一格都是空格。

    `cbit`（開局的角位）與 `reach` 二擇一，兩者都不給就是完全沒有約束。
    """
    if cbit or reach is not None:
        mask = _adjoining_bases(od, reach, cbit)
    else:
        mask = od["valid"]
    return mask & _free_bases(od, empt)


def own_reach(board, owner):
    """自己目前的角對角規則（`Reach` 對）。還沒放過棋時回傳 None。"""
    own = board.owner_bits[owner]
    if not own:
        return None
    empt = board.empty_bits
    return Reach(board.dilate_diag(own) & empt, board.dilate(own) & empt)


ODIRS = {}
for _name, _p in MASTER.items():
    _od = {}
    for _oi, _cells in enumerate(_p["orientations"]):
        _od[_oi] = _build_od(_name, _oi, _cells)
    ODIRS[_name] = _od


# --------------------------------------------------------------------------
# 入侵者：延伸度、可用交點、可放空格數、跨越、關鍵格
# --------------------------------------------------------------------------
#
# 這些量全部依「落子後的棋盤」算，沒有任何一項是棋塊的常數：同一塊棋放在
# 不同位置、或者周圍被封住時，延伸度就不同。

# 規則之間的換算尺度。延伸度最遠到 EXT_CAP 就飽和，因為 20x20 的盤上真正
# 有意義的「伸多遠」只在十幾格內，之後再長只是數字變大。
EXT_CAP = 14
W_EXT = 1.0
W_SQUARES = 0.05
W_VERTICES = 0.04
SQ_CAP = 80
# 規則 5：一般局面 5 格優先，每少 1 格扣 W_SIZE。戰略點再加 W_STRATEGIC，
# 剛好大到讓 4 格／3 格的關鍵格贏過 5 格的填色。
W_SIZE = 1.0
W_STRATEGIC = 2.2
# 規則 4：跨越那一步額外加的分，並且可放空格數的權重加倍。
W_CROSS = 2.5
W_SEAL = 0.4
SEAL_MIN = 6
# 第二階段（昂貴評估）的安全上限。階段限制加上角對角規則之後，候選通常只剩
# 幾百個（實測多不超過 400），所以這個上限幾乎不會擋到東西；它的作用只是
# 讓「每個候選都要模擬落子」這件事有個確定的花費上限。
INTRUDER_SCAN = 1200


def size_bonus(size, strategic):
    """規則 5：一般局面 5 格優先；戰略點（關鍵格／跨越）放寬到 4 格、3 格。

    戰略獎勵刻意略大於「少一格」的差距，所以 3 格的關鍵格贏得過 5 格的填色，
    4 格的更是穩贏，但 5 格在戰略點仍然最好——允許用小棋，不是要求用小棋。
    """
    s = W_SIZE * (size - 3.0)
    return s + (W_STRATEGIC if strategic else 0.0)


def crossing_bonus(squares, usable):
    """規則 4：跨越那一步以可放空格數最大化為主，並額外懲罰可用交點太少。

    「剛跨過去就被封」是這條規則要防的事：跨過去之後要還有地方可放，否則
    這一手只是把棋送掉。
    """
    s = W_CROSS + 2.0 * W_SQUARES * min(squares, SQ_CAP)
    if usable < SEAL_MIN:
        s -= W_SEAL * (SEAL_MIN - usable)
    return s


def is_leaper(cells):
    """跨越棋：3x3 外框裡佔據「兩個對角」的形狀。

    依幾何判定而非棋名，所以 T5（(0,0)+(2,0) 相鄰）、X5（不佔角）、
    F5（只佔 (2,0)）都不會被算進來。
    """
    if max(x for x, _ in cells) != 2 or max(y for _, y in cells) != 2:
        return False
    have = set(cells)
    return have >= {(0, 0), (2, 2)} or have >= {(2, 0), (0, 2)}


LEAPERS = frozenset(name for name, p in MASTER.items()
                    if any(is_leaper(cells) for cells in p["orientations"]))

# 「垂直走 2 格、橫向走 4 格」與「1 格直 + 5 格」這類長手臂形狀。Y5 刻意
# 不列入：同樣條件下它的延伸距離只有 5，比 L5 短。
STRETCHERS = frozenset(n for n in ("L5", "N5", "I5") if n in MASTER)


def cross_anchors(me, them):
    """2x2 方塊（左上角位元）裡，一條對角全是 me、另一條對角含 them。

    2x2 的兩條對角線是 {左上, 右下} 與 {右上, 左下}；後者的兩格都比錨點
    右移／下移一步，所以那一式不與 `me` 求交。錨點必須落在 BLOCK_ANCHORS
    內，否則右移／下移會跨行接到別的列。
    """
    main = me & (me >> (B + 1)) & ((them >> 1) | (them >> B))
    anti = (me >> 1) & (me >> B) & (them | (them >> (B + 1)))
    return (main | anti) & BLOCK_ANCHORS


def key_cells(me, them, empt):
    """關鍵格：填上去就完成一次跨越的空格。

    跨越成立前，構成跨越那條對角線的是「一個空格 p + 一顆自己的棋 q（角對角
    貼著 p）」，而 2x2 的另一條對角要已經有對手。所以 p 與 q 之間那兩個共邊
    的格子必須有一個是對手——也正因如此 p 對 q 是角對角、對自己的棋不共邊，
    p 一定是合法的落子（仍可能被自己其他棋擋掉，那是 `_adjoining_bases` 的事）。

    連續兩次的「正前方」存在對手就叫「把對手夾在兩格之間」，正好是跨越成立
    的樣子。
    """
    out = 0
    # 四個對角方向各一式；括號內是 2x2 的另兩格，任一格是對手就算數。
    out |= (me << (B + 1)) & ~COL0 & ~ROW0 & ((them << 1) | (them << B))
    out |= (me << (B - 1)) & ~COL_LAST & ~ROW0 & ((them >> 1) | (them << B))
    out |= (me >> (B - 1)) & ~COL0 & ~ROWN & ((them << 1) | (them >> B))
    out |= (me >> (B + 1)) & ~COL_LAST & ~ROWN & ((them >> 1) | (them >> B))
    return out & empt


def has_crossed(me, them):
    """只看棋盤就能判定，不需要歷史：2x2 裡一條對角全是我的、另一條有對手。"""
    return bool(cross_anchors(me, them))


def _vertex_list(mask):
    """交點位元 → [(X, Y)]。接觸交點最多幾個，逐一攤開即可。"""
    out = []
    while mask:
        low = mask & -mask
        mask ^= low
        i = low.bit_length() - 1
        out.append((i % V, i // V))
    return out


def contact_vertices(placed, anchor, anchor_v=None):
    """接觸交點：新棋與自己既有棋共享的交點（開局首手則是它覆蓋的角交點）。

    合法性保證了任何共享都只是角對角——共邊本來就不合法。`anchor_v` 可以帶
    進來預先投影好的 `anchor` 交點集（見 `board_context`），省掉每個候選一次
    20 列的換算。
    """
    if anchor_v is None:
        anchor_v = cells_to_vertices(anchor)
    return _vertex_list(cells_to_vertices(placed) & anchor_v)


def extension(legal_new, anchors, limit=EXT_CAP):
    """延伸度：從接觸交點算起，這一手新打開的區域最遠伸多遠（曼哈頓距離）。

    只看「這一手新產生的可放空格」：它們全都角對角貼著新下的棋，所以這個
    距離量的是這塊棋自己伸出去多遠，而不是既有領土有多大。
    """
    best = 0
    m = legal_new
    while m and best < limit:
        low = m & -m
        m ^= low
        c = low.bit_length() - 1
        cx, cy = c % B, c // B
        for ax, ay in anchors:
            nx = cx if ax < cx else (cx + 1 if ax > cx + 1 else ax)
            ny = cy if ay < cy else (cy + 1 if ay > cy + 1 else ay)
            d = abs(ax - nx) + abs(ay - ny)
            if d > best:
                best = d
                if best >= limit:
                    return limit
    return best


def board_context(board, hand_names, owner, must_cover):
    """一盤棋裡與候選無關的量，全部先算一次。

    落子前就可放的空格、接觸用的錨點（開局是角位格）。跨越那套只在
    `IntruderBrain` 需要，所以由 `crossing_context` 另外補上。
    """
    own = board.owner_bits[owner]
    empt = board.empty_bits
    reach = own_reach(board, owner)
    if reach is None:
        need0 = avoid0 = legal_before = 0
        anchor = (1 << (must_cover[0] + must_cover[1] * B)
                  if must_cover is not None else 0)
    else:
        need0, avoid0 = reach.need, reach.avoid
        legal_before = need0 & ~avoid0
        anchor = own
    return {"board": board, "owner": owner, "own": own, "empt": empt,
            "anchor": anchor, "anchor_v": cells_to_vertices(anchor),
            "need0": need0, "avoid0": avoid0, "legal_before": legal_before,
            "dilate": board.dilate, "dilate_diag": board.dilate_diag}


def crossing_context(board, hand_names, owner):
    """規則 3／4 用的對手資訊：跨越狀態、尚未跨越的對手、跨越棋。

    跨越棋集合只跟手上的棋有關，所以 `selfpair` 那些 base 位元整局都不會變，
    整手棋只算一次。
    """
    own = board.owner_bits[owner]
    opps = [(o, board.owner_bits[o]) for o in range(4)
            if o != owner and board.owner_bits[o]]
    pre = {o: cross_anchors(own, m) for o, m in opps}
    uncrossed = [(o, m) for o, m in opps if not pre[o]]
    leaps = [(n, oi, ODIRS[n][oi]) for n in hand_names
             if n in LEAPERS for oi in ODIRS[n]]
    selfpair = {}
    for o, m in uncrossed:
        sp = {}
        for n, oi, od in leaps:
            acc = 0
            for sh in od["pair_shifts"]:
                acc |= m >> sh
            sp[(n, oi)] = acc & od["valid"]
        selfpair[o] = sp
    return {"opps": opps, "pre": pre, "uncrossed": uncrossed, "leaps": leaps,
            "selfpair": selfpair}


def placement_counts(board, hand_names, owner, must_cover):
    """每個棋塊現在還有幾個合法落點（跨全部 orientation 加總）。

    「只餘一個地方可放」就是這裡等於 1。等於 0 的棋塊已經救不回來了——根本
    沒有落點可選，不算急需。
    """
    reach = own_reach(board, owner)
    cbit = 0
    if reach is None and must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * B)
    empt = board.empty_bits
    out = {}
    for name in hand_names:
        total = 0
        for od in ODIRS[name].values():
            total += _legal_bases(od, empt, reach, cbit).bit_count()
        out[name] = total
    return out


def place_state(board, name, oi, base, ctx):
    """落子後的 (自己, 落子後空格, need, avoid, 可放空格)。

    `可放空格`就是「自己下一手還能下的位置」，兩個規則式人格都要它。
    """
    placed = ODIRS[name][oi]["m"] << base
    empt_after = ctx["empt"] & ~placed
    need = ctx["need0"] | ctx["dilate_diag"](placed)
    avoid = ctx["avoid0"] | ctx["dilate"](placed)
    return ctx["own"] | placed, empt_after, need, avoid, need & empt_after & ~avoid


def place_geometry(board, name, oi, base, ctx):
    """落子後的幾何量：(自己, 落子後空格, need, avoid, 可放空格, 新打開的,
    接觸交點)。

    這是全部規則的唯一來源——AI 評分與測試都走這裡，兩邊不會各算各的。
    """
    own, empt_after, need, avoid, legal = place_state(board, name, oi, base, ctx)
    placed = ODIRS[name][oi]["m"] << base
    return (own, empt_after, need, avoid, legal, legal & ~ctx["legal_before"],
            contact_vertices(placed, ctx["anchor"], ctx["anchor_v"]))


def move_geometry(board, name, oi, x, y, owner, must_cover=None):
    """一次假想落子的 (延伸度, 可用交點數, 可放空格數)，給測試與除錯用。"""
    ctx = board_context(board, [name], owner, must_cover)
    _own, _empt, _need, _avoid, legal, fresh, anchors = \
        place_geometry(board, name, oi, x + y * B, ctx)
    return (extension(fresh, anchors), cells_to_vertices(fresh).bit_count(),
            legal.bit_count())


# --------------------------------------------------------------------------
# 築城者：可填充閉包、活區塊、裝箱
# --------------------------------------------------------------------------
#
# 優化者只量「還有幾個可放空格」的**數量**。可放空間是分散的時候總量再大也沒用，
# 因為一個棋塊必須是連續的空地：兩個 4 格活區塊塞不進任何 5 格棋，一個 8 格
# 活區塊可以。下面這三個純函式補的就是「形狀」。

def fillable_closure(empt, avoid, legal):
    """可填充閉包：「我還填得進去的格子」。

    從 `legal`（落子後自己下一手能下的位置）出發，沿 4 鄰接泛洪，只走空格
    `empt` 且避開 `avoid`（與自己棋共邊，棋塊不能壓上去）。一個棋塊的落子必須
    從某個 `legal` 格出發、每一格都是空的、且棋塊形狀連通，所以這就是可填的
    格子。`fill` 每輪只增不減，所以一定收斂（實測 2～4 輪）。

    已知近似，寫下來是因為它**只是上界**，當排序 proxy 夠了，不拿來預測分數：
      (a) 沒有要求每走一步都重新滿足角對角接觸，中途可能已經離開了自己的棋；
      (b) 沒算對手接下來會佔掉多少；
      (c) 含斜對角步的棋塊（Z5 等）能跨過 4 鄰接不連通的地方，閉包會少算一點。
    """
    fill = legal & empt & ~avoid
    while True:
        grown = fill | (dilate(fill) & empt & ~avoid)
        if grown == fill:
            return fill
        fill = grown


def fill_components(mask):
    """`mask` 的 4 鄰接連通區塊面積，由大到小。

    連通區塊就是「活區塊」：一個棋塊只能整塊落在同一個活區塊裡，所以這裡的
    每個數字是一塊地最多能再吃掉的格數。用 `dilate` 迭代出來，回傳面積而不要
    回傳位元——後面只拿來當容量用。
    """
    out = []
    rest = mask
    while rest:
        comp = rest & -rest
        rest ^= comp
        while True:
            grown = comp | (dilate(comp) & rest)
            if grown == comp:
                break
            comp = grown
        rest &= ~comp
        out.append(comp.bit_count())
    out.sort(reverse=True)
    return out


def pack_lost(sizes_desc, bins):
    """first-fit-decreasing：把手牌裝進活區塊，裝不進去的格數。

    `sizes_desc` 是剩餘手牌大小由大到小，`bins` 是活區塊容量。每一塊找第一個塞
    得下的活區塊塞進去，塞不下就記成一筆送掉的格數。回傳的是「估計送不掉的格
    數」，單位就是格，所以跟記分（餘格愈少愈好）同一個尺度。
    """
    room = list(bins)
    lost = 0
    for k in sizes_desc:
        for i, r in enumerate(room):
            if r >= k:
                room[i] = r - k
                break
        else:
            lost += k
    return lost


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
    """狼／棋手／狐狸：權重加總。重構前後評分完全相同。"""


class IntruderBrain(Brain):
    """規則式人格：依延伸度與可放空間挑棋，不做權重加總。

    階段（規則 1／2／5）是**軟**限制：指定棋塊若一格都下不了就退回全部
    候選，否則候選清單會被清空，`choose_move` 回傳 None，玩家就會被誤判
    成無棋可下而自動過。
    """

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        # 規則式人格不抽籤：戰略獎勵一被隨機性蓋掉就沒有意義，「前 3 手只用
        # 跨越棋」這種性質也才檢查得起來。
        self.mistake_rate = 0.0
        self.uses_lookahead = False

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = board_context(board, hand_names, owner, must_cover)
        ctx.update(crossing_context(board, hand_names, owner))
        return ctx

    def restrict(self, cands, ctx):
        """規則 1／2／5 的階段：跨越棋 → 長手臂 → 一般局面。

        每一階段都是軟的：該類棋一格都下不了就往下退，退到最後就是全部候選。
        做成硬過濾的話候選清單會被清空，`choose_move` 回傳 None，玩家就會被
        誤判成無棋可下而自動過。
        """
        for cls in (LEAPERS, STRETCHERS):
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
        od = ODIRS[name][oi]
        (own_after, empt_after, need, avoid, legal, fresh,
         anchors) = place_geometry(ctx["board"], name, oi, base, ctx)

        ext = extension(fresh, anchors)
        usable = cells_to_vertices(fresh).bit_count()
        squares = legal.bit_count()

        # 規則 4：這一手本身完成跨越。`pre` 是落子前的跨越方塊，用來扣掉
        # 「本來就跨越過、只是被這一手碰到」的那一格。
        crossed = False
        for o, m in ctx["opps"]:
            if cross_anchors(own_after, m) & (od["touch"] << base) & ~ctx["pre"][o]:
                crossed = True
                break

        # 規則 3：尚未跨越時搶關鍵格
        strategic = crossed
        if not crossed and ctx["uncrossed"]:
            strategic = self._sets_up(ctx, od, own_after, empt_after,
                                      need, avoid) or crossed

        s = size_bonus(od["size"], strategic)
        s += (crossing_bonus(squares, usable) if crossed
              else W_SQUARES * min(squares, SQ_CAP))
        s += W_VERTICES * min(usable, SQ_CAP)
        s += W_EXT * min(ext, EXT_CAP)
        return (s, name, oi, base)

    def _sets_up(self, ctx, od, own_after, empt_after, need, avoid):
        """佔下這一手之後，是否存在一個合法落子能完成跨越（規則 3 的關鍵格）。

        只檢查手上這塊棋與跨越棋，且只在尚未跨越的對手身上算——這是這條規則
        最貴的部分，其餘對手直接跳過。
        """
        reach = Reach(need, avoid)
        for o, m in ctx["uncrossed"]:
            key = key_cells(own_after, m, empt_after)
            if not key:
                continue
            kb = 0
            for off in od["offs"]:
                kb |= key >> off
            if kb and _adjoining_bases(od, reach, 0) & kb & _free_bases(od, empt_after):
                return True
            sp = ctx["selfpair"][o]
            for n, oi, lod in ctx["leaps"]:
                lkb = 0
                for off in lod["offs"]:
                    lkb |= key >> off
                hit = lkb | sp[(n, oi)]
                if not hit:
                    continue
                if _adjoining_bases(lod, reach, 0) & hit \
                        & _free_bases(lod, empt_after):
                    return True
        return False


class OptimizerBrain(Brain):
    """效率式人格：先放最大的棋塊，再挑最能留住可放空格的落點。

    目標函式只有一個量——**落子後自己還有幾個可放空格**。積分是「餘格愈少
    愈好」，所以能放掉的格數就是分數，而一塊 5 格棋放掉一次就賺 5 格；這是
    為什麼先挑大的，而同一種棋塊之間就比誰留下的可放空間多。

    緊急規則是它的安全閥：某一塊棋只剩唯一一個落點時先把它放掉。那個位置被
    封掉之後，那塊棋就永遠留在手上了——在這個記分法下等於直接送掉幾格。
    """

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        self.mistake_rate = 0.0
        self.uses_lookahead = False

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = board_context(board, hand_names, owner, must_cover)
        # 開局第一手時「只剩一個落點」是角位規則壓出來的，還不是棋盤變擠的
        # 訊號——那時棋塊的落點數天生就少（O4 更是只有一種放法）。所以緊急
        # 規則不適用於開局，否則優化者會把第一手花在 O4／I1 上而不是 5 格棋。
        if must_cover is not None:
            ctx["urgent"] = []
        else:
            counts = placement_counts(board, hand_names, owner, must_cover)
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
            by_size.setdefault(ODIRS[c[1]][c[2]]["size"], []).append(c)
        return by_size[max(by_size)] if by_size else cands

    def rescore(self, cands, ctx):
        scored = [(place_state(ctx["board"], n, oi, b, ctx)[4].bit_count(),
                   n, oi, b)
                  for _s, n, oi, b in cands[:OPTIMIZER_SCAN]]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored


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

    def __init__(self, key, profile):
        Brain.__init__(self, key, profile)
        self.mistake_rate = 0.0
        self.uses_lookahead = False

    def context(self, board, hand_names, owner, must_cover, reach):
        ctx = board_context(board, hand_names, owner, must_cover)
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
        wide = [(place_state(board, n, oi, b, ctx)[4].bit_count(), n, oi, b)
                for _s, n, oi, b in cands]
        wide.sort(key=lambda t: t[0], reverse=True)
        scored = [self._score(n, oi, b, ctx) for _total, n, oi, b
                  in wide[:BUILDER_SCAN]]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def _score(self, name, oi, base, ctx):
        board = ctx["board"]
        _own, empt_after, need, avoid, legal = place_state(board, name, oi, base, ctx)
        total = legal.bit_count()
        bins = fill_components(fillable_closure(empt_after, avoid, legal))
        lost = pack_lost(ctx["sizes_after"][name], bins)
        od = ODIRS[name][oi]
        s = (-BUILDER_W_LOST * lost + BUILDER_W_SIZE * (od["size"] - 3)
             + BUILDER_W_TOTAL * min(total, SQ_CAP))
        return (s, name, oi, base)


# 規則式人格 = 目標函式不是權重加總的那幾種。它們仍然帶一份權重檔，只用在第一
# 階段的粗篩與被預判時的評分。
RULE_BRAIN_CLASSES = {INTRUDER_KEY: IntruderBrain, OPTIMIZER_KEY: OptimizerBrain,
                      BUILDER_KEY: BuilderBrain}


def make_brain(key, rng):
    # 規則式人格仍然帶一份權重檔，只用在第一階段的粗篩與被預判時的評分；
    # 它們自己的目標函式完全在各自的 Brain 類別裡。
    cls = RULE_BRAIN_CLASSES.get(key)
    if cls is not None:
        return cls(key, make_profile("chess", rng))
    return WeightedBrain(key, make_profile(key, rng))


def board_feats(grid):
    open_a = [0] * N
    block = [[0] * N for _ in range(4)]
    for p in range(N):
        if grid[p] != -1:
            continue
        for n in NEIGH[p]:
            q = grid[n]
            if q == -1:
                open_a[p] += 1
            elif q >= 0:
                for o in range(4):
                    if o != q:
                        block[o][p] += 1
    nc = [[0] * 4 for _ in range(N)]
    for n in range(N):
        if grid[n] != -1:
            continue
        for m in NEIGH[n]:
            o = grid[m]
            if o >= 0:
                nc[n][o] += 1
    defend = [[0] * N for _ in range(4)]
    for p in range(N):
        if grid[p] != -1:
            continue
        for n in NEIGH[p]:
            if grid[n] == -1:
                total = nc[n][0] + nc[n][1] + nc[n][2] + nc[n][3]
                for o in range(4):
                    defend[o][p] += total - nc[n][o]
    return open_a, block, defend


def _bfs_count(cs, starts):
    seen = set()
    stack = list(starts)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        for n in NEIGH[p]:
            if n in cs and n not in seen:
                stack.append(n)
    return len(seen)


def _corner_delta(ci, cs, owner, regions, borders):
    region = regions[owner][ci]
    size = len(region)
    if size > REGION_CAP:
        return 0
    if size == 0:
        cidx = CORNERS_IDX[ci]
        if cidx not in cs:
            return 0
        return _bfs_count(cs, [cidx])
    if not (cs & borders[owner][ci]):
        return 0
    touches = [c for c in cs if any(n in region for n in NEIGH[c])]
    if not touches:
        return 0
    return _bfs_count(cs, touches)


def _score_move(board, name, oi, base, owner, profile, open_a, block, defend, regions, borders):
    od = ODIRS[name][oi]
    cells = [base + o for o in od["offs"]]
    cs = set(cells)
    s = profile.w_large * od["size"] - SMALL_PENALTY * (5 - od["size"])
    s += profile.w_center * od["csum"][base]
    bi = block[owner]
    di = defend[owner]
    for i in cells:
        s += profile.w_open * open_a[i] + profile.w_block * bi[i] + profile.w_defend * di[i]
    cb = od["corner"].get(base, 0)
    if cb:
        for ci in (0, 1, 2, 3):
            if cb & (1 << ci):
                s += profile.w_corner * 3.0 * _corner_delta(ci, cs, owner, regions, borders)
    return s


def _opponent_pool(board, hand_names, owner, must_cover=None, reach=None, k=OPP_POOL_K):
    """Best `k` legal replies for `owner`, filtered by the same rules as
    `choose_move` so the lookahead never assumes a move the game forbids.

    The cheap "neutral" bases (corner- and centre-weighted) are scanned first.
    The corner-contact rule anchors candidates to a player's existing stones,
    which can leave that short list entirely illegal, so an empty result would
    silently turn the opponent response into zero. When it does, the search
    widens to every in-bounds base instead.
    """
    empt = board.empty_bits
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * B)
    pool = []
    seen = set()
    for name in hand_names:
        for oi, od in ODIRS[name].items():
            m = od["m"]
            for base in od["neutral"]:
                shifted = m << base
                if cbit:
                    if not shifted & cbit:
                        continue
                elif reach is not None:
                    if not shifted & reach.need or shifted & reach.avoid:
                        continue
                if (empt & shifted) == shifted:
                    pool.append((name, oi, base))
                    seen.add((name, oi, base))
        if len(pool) >= k:
            return pool[:k]
    for name in hand_names:
        for oi, od in ODIRS[name].items():
            m = od["m"]
            mask = _adjoining_bases(od, reach, cbit) if (cbit or reach is not None) \
                else od["valid"]
            found = []
            while mask:
                low = mask & -mask
                mask ^= low
                base = low.bit_length() - 1
                if (name, oi, base) in seen:
                    continue
                shifted = m << base
                if (empt & shifted) == shifted:
                    found.append(base)
            if not found:
                continue
            if len(found) > OPP_POOL_PER_ORIENT:
                found.sort(key=od["rank"].__getitem__)
                del found[OPP_POOL_PER_ORIENT:]
            for base in found:
                pool.append((name, oi, base))
                seen.add((name, oi, base))
        if len(pool) >= k:
            break
    return pool[:k]


def _touches_corner(name, oi, base):
    return ODIRS[name][oi]["corner"].get(base, 0) != 0


def _build_shortlist(scored):
    """Core top-N plus reserved corner-claiming moves that stay competitive.

    The reserve keeps a corner grab reachable for personalities that would
    otherwise never shortlist one, so corner appetite varies smoothly with
    w_corner instead of flipping between "always" and "never".
    """
    if not scored:
        return []
    best = scored[0][0]
    sl = scored[:SHORTLIST_CORE]
    keys = {(c[1], c[2], c[3]) for c in sl}
    added = 0
    for cand in scored[:SHORTLIST_SCAN]:
        if added >= CORNER_RESERVE:
            break
        s, name, oi, base = cand
        if s < best - CORNER_BAND:
            break
        key = (name, oi, base)
        if key in keys or not _touches_corner(name, oi, base):
            continue
        keys.add(key)
        sl.append(cand)
        added += 1
    return sl


def _weighted_pick(sl, rng):
    best = sl[0][0]
    weights = [math.exp((c[0] - best) / PICK_TEMP) for c in sl]
    target = rng.random() * sum(weights)
    acc = 0.0
    for cand, w in zip(sl, weights):
        acc += w
        if target <= acc:
            return cand
    return sl[-1]


def choose_move(board, hand_names, owner, brain, rng, other_brains=None,
                must_cover=None, reach=None, other_must_cover=None, other_reach=None):
    t0 = time.perf_counter()
    empt = board.empty_bits
    open_a, block, defend = board_feats(board.grid)
    regions = board.corner_regions
    borders = board.borders
    profile = brain.profile
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * B)
    names = list(hand_names)
    cands = []
    for name in names:
        od_map = ODIRS[name]
        bi = block[owner]
        di = defend[owner]
        for oi, od in od_map.items():
            m = od["m"]
            corner = od["corner"]
            csum = od["csum"]
            offs = od["offs"]
            cs_const = profile.w_large * od["size"] - SMALL_PENALTY * (5 - od["size"])
            for base in od["bases"]:
                shifted = m << base
                if (empt & shifted) != shifted:
                    continue
                if cbit:
                    if not shifted & cbit:
                        continue
                elif reach is not None:
                    if not shifted & reach.need or shifted & reach.avoid:
                        continue
                s = cs_const + profile.w_center * csum[base]
                for o in offs:
                    i = base + o
                    s += profile.w_open * open_a[i] + profile.w_block * bi[i] + profile.w_defend * di[i]
                cb = corner.get(base, 0)
                if cb:
                    cs = set(base + o for o in offs)
                    for ci in (0, 1, 2, 3):
                        if cb & (1 << ci):
                            s += profile.w_corner * 3.0 * _corner_delta(ci, cs, owner, regions, borders)
                cands.append((s, name, oi, base))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    # 人格接手：restrict 是階段（軟）過濾，rescore 才是貴的那一次評估。
    # 權重式人格兩者都是恆等，所以以下這段對它們完全不影響既有行為。
    ctx = brain.context(board, names, owner, must_cover, reach)
    cands = brain.restrict(cands, ctx)
    cands = brain.rescore(cands, ctx)
    top = cands[:SHORTLIST_CORE + CORNER_RESERVE]

    adj = top
    opps = [o for o in range(4) if o != owner]
    if other_brains and opps and brain.uses_lookahead \
            and time.perf_counter() - t0 < WALL_BUDGET:
        pools = [_opponent_pool(board, hand_names, o,
                                (other_must_cover or {}).get(o),
                                (other_reach or {}).get(o))
                 for o in opps]
        adj = []
        sim_eval = 0
        skip = False
        for sc, name, oi, base in top:
            if not skip and sim_eval + OPP_POOL_K * len(opps) <= SIM_EVAL_BUDGET:
                placed = ODIRS[name][oi]["m"] << base
                resp = 0.0
                for idx, o in enumerate(opps):
                    po = other_brains[o].profile
                    best_o = -1e18
                    for item in pools[idx]:
                        name2, oi2, base2 = item
                        shifted = ODIRS[name2][oi2]["m"] << base2
                        if shifted & placed:
                            continue
                        v = _score_move(board, name2, oi2, base2, o, po, open_a, block, defend, regions, borders)
                        if v > best_o:
                            best_o = v
                        sim_eval += 1
                    if best_o > 0:
                        resp += best_o
                adj.append((sc - resp, name, oi, base))
            else:
                skip = True
                adj.append((sc, name, oi, base))
        if adj:
            adj.sort(key=lambda t: t[0], reverse=True)
            adj = adj[:SHORTLIST_CORE + CORNER_RESERVE]
    else:
        adj = top

    shortlist = _build_shortlist(adj)
    if not shortlist:
        return None
    pick = shortlist[0]
    if len(shortlist) >= 2 and rng.random() < brain.mistake_rate:
        pick = _weighted_pick(shortlist, rng)
    return (pick[1], pick[2], pick[3] % B, pick[3] // B)
