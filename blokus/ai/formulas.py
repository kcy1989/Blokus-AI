"""AI 共用的計算公式：棋盤幾何、落子推演、量化指標。

這裡只放「任何人格都用得到」的部分——把一個落子換算成數字的那一層。個人
格怎麼組權重、怎麼分階段，住在各自的模組裡。

分三段：

1. 位元幾何：`ODIRS`、`_legal_bases`、`place_geometry`。
   20x20 的棋盤用一顆 400-bit 的整數表示，所以「這一手合法嗎」「落子後長
   什麼樣」都是幾個大整數的位移與求交，不用枚舉格子。
2. 量化指標：延伸度、可放空格數、可用交點數、活區塊、裝箱。
   這些是把位元攤成數字的地方，記分只看「餘格」，所以每一個指標的單位都
   刻意跟格數對齊。
3. 權重表：所有公式共用的常數。數值之間的大小關係是有意義的，改之前先看
   各常數旁邊的註解。
"""
from board import (BLOCK_ANCHORS, COL0, COL_LAST, Reach, V, cells_to_vertices,
                   dilate, neighbors_of)
from config import B, CORNERS_IDX, N
from pieces import MASTER

# --------------------------------------------------------------------------
# 常數
# --------------------------------------------------------------------------

# 權重式人格評分時，少一格就扣的分。刻意是平的 0.25／格：一塊 1 格的棋相對
# 5 格棋先天少賺 1.0 分，任何「大小」相關的權重都蓋不過它。
SMALL_PENALTY = 0.25
# 角位連通區塊大於此值就不再算角位加成，避免一開局就為了一小塊地做昂貴的 BFS。
REGION_CAP = 120
# 每一手的推演預算，讓「每個候選都模擬落子」這件事有個確定的花費上限。
SIM_EVAL_BUDGET = 3000
# 單手思考的時間上限（秒），超過就不再做對手預判。
WALL_BUDGET = 0.9
# 4 鄰接表，只建一次。
NEIGH = neighbors_of()

# 短名單的組裝：前 N 個是核心，之後最多再補這麼多個搶角候選。
SHORTLIST_CORE = 34
SHORTLIST_SCAN = 400
CORNER_RESERVE = 6
# 補進短名單的搶角手，容許比最佳分低這麼多。
CORNER_BAND = 12.0
# 抽籤時的 softmax 溫度：分數差一格就變成 e 倍的權重差距。
PICK_TEMP = 5.0

# 對手預判要評多少個應手。候選池先從「中性」落點建；當角對角規則把那個池
# 清空時改用窮舉（見 `chooser._opponent_pool`），所以固定 K 不論走哪條路都
# 讓花費有上限。
OPP_POOL_K = 40
OPP_POOL_PER_ORIENT = 6

ROW0 = (1 << B) - 1
ROWN = ROW0 << (B * (B - 1))

# 每一格距棋盤中心的親近度，0（邊）到 1（正中）。權重式人格的 w_center
# 就是乘在這個上面的。
CENTER = []
for y in range(B):
    for x in range(B):
        d = min(abs(x - 9.5), abs(y - 9.5))
        CENTER.append(1.0 - d / 9.0)

# 入侵者的延伸度上限：20x20 的盤上「伸多遠」只有十幾格有意義，再長只是數字
# 變大。
EXT_CAP = 14
W_EXT = 1.0
W_SQUARES = 0.05
W_VERTICES = 0.04
SQ_CAP = 80
# 一般局面 5 格優先，每少 1 格扣 W_SIZE。戰略點再加 W_STRATEGIC，剛好大到
# 讓 4 格／3 格的關鍵格贏過 5 格的填色。
W_SIZE = 1.0
W_STRATEGIC = 2.2
# 跨越那一步額外加的分，並且可放空格數的權重加倍。
W_CROSS = 2.5
W_SEAL = 0.4
SEAL_MIN = 6


# --------------------------------------------------------------------------
# 位元幾何
# --------------------------------------------------------------------------

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
    """一種棋塊 × 一種方向 的所有預計算結果。

    落點(`bases`)、每個落點的中心分(`csum`)、占掉哪些角(`corner`)、跨越
    用到的兩組位元(`touch`、`pair_shifts`)都在這裡算一次，之後幾千個候選都
    只是查表。
    """
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
# 跨越：2x2 裡一條對角全是自己、另一條有對手
# --------------------------------------------------------------------------

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
    out |= (me >> (B + 1)) & ~COL_LAST & ~ROWN & ((them >> 1) | (them << B))
    return out & empt


def has_crossed(me, them):
    """只看棋盤就能判定，不需要歷史：2x2 裡一條對角全是我的、另一條有對手。"""
    return bool(cross_anchors(me, them))


# --------------------------------------------------------------------------
# 量化指標
# --------------------------------------------------------------------------

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
# 可填充閉包、活區塊、裝箱
# --------------------------------------------------------------------------
#
# 前面只量「還有幾個可放空格」的**數量**。可放空間是分散的時候總量再大也沒
# 用，因為一個棋塊必須是連續的空地：兩個 4 格活區塊塞不進任何 5 格棋，一
# 個 8 格活區塊可以。下面這三個純函式補的就是「形狀」。

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


# --------------------------------------------------------------------------
# 規則式人格用的分項公式
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# 權重式人格：候選的評分函式
# --------------------------------------------------------------------------

def board_feats(grid):
    """每一個空格的三組特徵：鄰接空格數、挨著各家的棋數、挨著對手的棋數。

    一次 O(400 × 4) 掃完，之後所有候選都只是查表加總。
    """
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


def _score_move(board, name, oi, base, owner, profile, open_a, block, defend,
                regions, borders):
    """權重式人格給單一候選的分數。

    六個權重項（角、中心、壓制、防守、開闊、大小）線性加總，所以每個權重
    的大小關係就是它的取捨方向。`chooser.choose_move` 內嵌了同一段算式，
    那裡是為了省掉函式呼叫而重寫的，兩邊必須同步。
    """
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
