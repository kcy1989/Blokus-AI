"""Formulas shared by the AI: board geometry, placement extrapolation, and
quantitative metrics.

Only the parts "every personality can use" live here -- the layer that turns
one placement into numbers. How a personality assembles weights and how it
stages its work lives in its own module.

Three parts:

1. Bit geometry: `ODIRS`, `_legal_bases`, `place_geometry`.
   The 20x20 board is one 400-bit integer, so "is this move legal" and "what
   does the board look like after it" are just a few shifts and ANDs of big
   integers -- no cell enumeration needed.
2. Quantitative metrics: extension, playable squares, usable vertices, live
   components, bin packing.
   This is where bits get flattened into numbers. Scoring only looks at
   "squares left", so every metric is deliberately measured in cells.
3. The weight table: constants shared by all formulas. The relative sizes of
   these values are meaningful -- read the note next to each constant before
   changing it.
"""
from board import (BLOCK_ANCHORS, COL0, COL_LAST, Reach, V, cells_to_vertices,
                   dilate, neighbors_of)
from config import B, CORNERS_IDX, N
from pieces import MASTER

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Points deducted per missing cell when scoring for a weighted personality.
# Deliberately flat at 0.25/cell: a 1-cell piece earns 1.0 less than a 5-cell
# piece inherently, and no "size"-related weight can outbid that.
SMALL_PENALTY = 0.25
# Corner-connected regions bigger than this stop counting toward the corner
# bonus, so the opening never pays for an expensive BFS over a tiny patch.
REGION_CAP = 120
# Per-move extrapolation budget, so "simulate a placement for every candidate"
# has a definite cost ceiling.
SIM_EVAL_BUDGET = 3000
# Wall-clock cap for thinking about one move (seconds); past it, no more
# opponent lookahead.
WALL_BUDGET = 0.9
# 4-neighbour table, built once.
NEIGH = neighbors_of()

# Shortlist assembly: the first N are the core, after that at most this many
# extra corner-claiming candidates may be added.
SHORTLIST_CORE = 34
SHORTLIST_SCAN = 400
CORNER_RESERVE = 6
# A corner move added to the shortlist may score this much below the best.
CORNER_BAND = 12.0
# Softmax temperature when picking: a one-point score gap becomes a factor of
# e in the weights.
PICK_TEMP = 5.0

# How many replies the opponent lookahead evaluates. The candidate pool is
# first built from "neutral" bases; when the corner-contact rule empties that
# pool, an exhaustive scan is used instead (see `chooser._opponent_pool`), so
# a fixed K caps the cost on either path.
OPP_POOL_K = 40
OPP_POOL_PER_ORIENT = 6

ROW0 = (1 << B) - 1
ROWN = ROW0 << (B * (B - 1))

# How close each cell is to the board centre, 0 (edge) to 1 (dead centre).
# A weighted personality's w_center multiplies this.
CENTER = []
for y in range(B):
    for x in range(B):
        d = min(abs(x - 9.5), abs(y - 9.5))
        CENTER.append(1.0 - d / 9.0)

# Extension cap for the intruder: on a 20x20 board "how far it reaches" only
# means something up to about a dozen cells; beyond that the number just grows.
EXT_CAP = 14
W_EXT = 1.0
W_SQUARES = 0.05
W_VERTICES = 0.04
SQ_CAP = 80
# In ordinary positions 5 cells win, and every missing cell costs W_SIZE. On
# strategic squares W_STRATEGIC is added, just large enough to let a 4- or
# 3-cell key square beat a 5-cell filler.
W_SIZE = 1.0
W_STRATEGIC = 2.2
# Extra points for the crossing step, and the weight on playable squares is
# doubled there.
W_CROSS = 2.5
W_SEAL = 0.4
SEAL_MIN = 6


# --------------------------------------------------------------------------
# Bit geometry
# --------------------------------------------------------------------------

def _block_touch(cells):
    """Top-left bits of the 2x2 blocks this move would touch.

    Used to intersect with the crossing-block bits when deciding "did this move
    complete a crossing", without enumerating the whole board.
    """
    touch = 0
    for dx, dy in cells:
        for ax, ay in ((dx - 1, dy - 1), (dx, dy - 1), (dx - 1, dy), (dx, dy)):
            if ax >= 0 and ay >= 0:
                touch |= 1 << (ax + ay * B)
    return touch


def _pair_shifts(offs, cells):
    """The offsets of the other two cells of the 2x2, for two cells inside a
    piece that are diagonally adjacent.

    This is the other source of a crossing: both cells along the crossing
    diagonal belong to the same move (rather than one old stone plus one new
    one). If the offset equals one of the piece's own offsets, the "other cell"
    is always covered by the piece itself, and that bit must be dropped.
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
    """Everything precomputed for one piece type x one orientation.

    Bases (`bases`), the centre score of each base (`csum`), which corners are
    covered (`corner`), and the two bit sets used for crossings (`touch`,
    `pair_shifts`) are all computed once here, so the thousands of candidates
    that follow are just table lookups.
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
    """Bit b is set when playing this move at b leaves every cell empty.

    Each offset contributes the base bits for which "that cell is empty"; the
    intersection is exactly "all of them are empty". Row-wrap mistakes are
    filtered out by `od["valid"]`.
    """
    out = od["valid"]
    for o in od["offs"]:
        out &= empt >> o
    return out


def _legal_bases(od, empt, reach=None, cbit=0):
    """The bases where this move is legal: corner-contact rule, and every cell
    is an empty square.

    `cbit` (the opening corner) and `reach` are mutually exclusive; given
    neither, there is no constraint at all.
    """
    if cbit or reach is not None:
        mask = _adjoining_bases(od, reach, cbit)
    else:
        mask = od["valid"]
    return mask & _free_bases(od, empt)


def own_reach(board, owner):
    """Our current corner-contact rule (a `Reach` pair). Returns None before we
    have placed any stone."""
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
# Crossing: one diagonal of a 2x2 is all ours, the other one holds an opponent
# --------------------------------------------------------------------------

def is_leaper(cells):
    """A crossing piece: a shape occupying "two opposite corners" of the 3x3
    frame.

    Decided geometrically rather than by piece name, so T5 ((0,0) and (2,0)
    adjacent), X5 (covers no corner) and F5 (covers only (2,0)) are all left
    out.
    """
    if max(x for x, _ in cells) != 2 or max(y for _, y in cells) != 2:
        return False
    have = set(cells)
    return have >= {(0, 0), (2, 2)} or have >= {(2, 0), (0, 2)}


LEAPERS = frozenset(name for name, p in MASTER.items()
                    if any(is_leaper(cells) for cells in p["orientations"]))

# Long-armed shapes like "2 cells vertically, 4 horizontally" or "1 straight
# plus 5 sideways". Y5 is deliberately excluded: under the same condition its
# reach is only 5, shorter than L5.
STRETCHERS = frozenset(n for n in ("L5", "N5", "I5") if n in MASTER)


def cross_anchors(me, them):
    """2x2 blocks (as top-left bits) where one diagonal is all me and the other
    diagonal holds them.

    The two diagonals of a 2x2 are {top-left, bottom-right} and {top-right,
    bottom-left}; both cells of the latter sit one step right/down of the
    anchor, so that form is not intersected with `me`. The anchor must fall
    inside BLOCK_ANCHORS, otherwise shifting right/down would wrap into
    another row.
    """
    main = me & (me >> (B + 1)) & ((them >> 1) | (them >> B))
    anti = (me >> 1) & (me >> B) & (them | (them >> (B + 1)))
    return (main | anti) & BLOCK_ANCHORS


def key_cells(me, them, empt):
    """Key squares: empty squares that, if filled, complete a crossing.

    Before a crossing holds, the diagonal that makes it up is "an empty square
    p plus one of our own stones q (diagonally touching p)", and the other
    diagonal of the 2x2 must already hold an opponent. So one of the two
    edge-adjacent cells between p and q has to be an opponent -- and precisely
    because p touches q diagonally and shares no edge with our own stones, p is
    always a legal placement (it may still be blocked by our other stones, and
    that is `_adjoining_bases`' job).

    An opponent sitting directly ahead twice in a row is what "trapping the
    opponent between two cells" means, which is exactly the shape of a
    completed crossing.
    """
    out = 0
    # One form per diagonal direction; in parentheses are the other two cells
    # of the 2x2 -- either one being an opponent counts.
    out |= (me << (B + 1)) & ~COL0 & ~ROW0 & ((them << 1) | (them << B))
    out |= (me << (B - 1)) & ~COL_LAST & ~ROW0 & ((them >> 1) | (them << B))
    out |= (me >> (B - 1)) & ~COL0 & ~ROWN & ((them << 1) | (them >> B))
    out |= (me >> (B + 1)) & ~COL_LAST & ~ROWN & ((them >> 1) | (them << B))
    return out & empt


def has_crossed(me, them):
    """Decided from the board alone, no history needed: one diagonal of a 2x2 is
    all mine and the other one holds an opponent."""
    return bool(cross_anchors(me, them))


# --------------------------------------------------------------------------
# Quantitative metrics
# --------------------------------------------------------------------------

def _vertex_list(mask):
    """Vertex bits -> [(X, Y)]. There are at most a few contact vertices, so
    just spread them out one by one."""
    out = []
    while mask:
        low = mask & -mask
        mask ^= low
        i = low.bit_length() - 1
        out.append((i % V, i // V))
    return out


def contact_vertices(placed, anchor, anchor_v=None):
    """Contact vertices: vertices shared by the new stone and our existing
    stones (for the opening move, the corner vertices it covers).

    Legality guarantees that any sharing is purely diagonal -- sharing an edge
    is illegal in the first place. `anchor_v` can carry in a pre-projected
    vertex set for `anchor` (see `board_context`), saving a 20-row conversion
    per candidate.
    """
    if anchor_v is None:
        anchor_v = cells_to_vertices(anchor)
    return _vertex_list(cells_to_vertices(placed) & anchor_v)


def extension(legal_new, anchors, limit=EXT_CAP):
    """Extension: measured from the contact vertices, how far the region newly
    opened by this move reaches (Manhattan distance).

    Only "playable squares newly created by this move" are considered: they all
    sit diagonally against the stone just played, so this distance measures how
    far this piece itself reaches out, not how big the existing territory is.
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
    """Everything in a position that is independent of the candidate, computed
    once up front.

    Squares playable before we move, and the anchors used for contact (the
    corner cell at the opening). The crossing machinery is only needed by
    `IntruderBrain`, so `crossing_context` supplies it separately.
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
    """Opponent information for rules 3/4: crossing state, opponents not yet
    crossed, and crossing pieces.

    The set of crossing pieces depends only on the pieces in hand, so those
    `selfpair` base bits never change during a game and are computed once per
    move.
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
    """How many legal placements each piece still has (summed over all
    orientations).

    "Only one place left to play" means 1 here. A piece at 0 is already beyond
    saving -- it has no placement at all, so it is not urgent.
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
    """After the placement: (ours, empty-after, need, avoid, playable squares).

    `playable squares` means "where we could still play next"; both
    rule-based personalities need it.
    """
    placed = ODIRS[name][oi]["m"] << base
    empt_after = ctx["empt"] & ~placed
    need = ctx["need0"] | ctx["dilate_diag"](placed)
    avoid = ctx["avoid0"] | ctx["dilate"](placed)
    return ctx["own"] | placed, empt_after, need, avoid, need & empt_after & ~avoid


def place_geometry(board, name, oi, base, ctx):
    """Geometry after the placement: (ours, empty-after, need, avoid, playable
    squares, newly opened, contact vertices).

    This is the single source for all the rules -- AI scoring and tests both go
    through here, so the two can never drift apart.
    """
    own, empt_after, need, avoid, legal = place_state(board, name, oi, base, ctx)
    placed = ODIRS[name][oi]["m"] << base
    return (own, empt_after, need, avoid, legal, legal & ~ctx["legal_before"],
            contact_vertices(placed, ctx["anchor"], ctx["anchor_v"]))


def move_geometry(board, name, oi, x, y, owner, must_cover=None):
    """(extension, usable vertex count, playable square count) for one
    hypothetical placement, for tests and debugging."""
    ctx = board_context(board, [name], owner, must_cover)
    _own, _empt, _need, _avoid, legal, fresh, anchors = \
        place_geometry(board, name, oi, x + y * B, ctx)
    return (extension(fresh, anchors), cells_to_vertices(fresh).bit_count(),
            legal.bit_count())


# --------------------------------------------------------------------------
# Fillable closure, live components, bin packing
# --------------------------------------------------------------------------
#
# So far we only measure the **quantity** of "how many playable squares are
# left". A large total is useless when the playable space is scattered, because
# a piece must land on contiguous empty ground: two 4-cell live components
# cannot hold any 5-cell piece, one 8-cell component can. These three pure
# functions supply the missing "shape".

def fillable_closure(empt, avoid, legal):
    """Fillable closure: "the cells I can still fill".

    Starting from `legal` (where we could play next after the placement), flood
    along 4-adjacency, walking only empty cells `empt` and avoiding `avoid`
    (edge-sharing our own stones -- a piece may not press on top of them). A
    piece's placement must start from some `legal` cell, cover only empty
    cells, and be connected as a shape, so this is exactly the set of
    fillable cells. `fill` only grows, never shrinks, so it always converges
    (2-4 rounds in practice).

    Known approximation, written down because it is **only an upper bound** --
    good enough as a ranking proxy, not used to predict scores:
      (a) it does not require re-satisfying corner contact on every step, so it
          may drift away from our own stones mid-flood;
      (b) it does not account for how much the opponents will take next;
      (c) pieces with diagonal steps (Z5 and friends) can cross gaps that are
          not 4-connected, so the closure undercounts a little.
    """
    fill = legal & empt & ~avoid
    while True:
        grown = fill | (dilate(fill) & empt & ~avoid)
        if grown == fill:
            return fill
        fill = grown


def fill_components(mask):
    """Areas of the 4-connected components of `mask`, largest first.

    A connected component is a "live component": a piece can only land entirely
    inside one live component, so each number here is the most cells that patch
    of land can still swallow. Iterated out with `dilate`; areas are returned,
    not bits -- they are only used as capacities afterwards.
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
    """first-fit-decreasing: pack the hand into the live components, and count
    the cells that do not fit.

    `sizes_desc` is the remaining hand sizes, largest first; `bins` holds the
    live component capacities. Each piece goes into the first component that
    can take it; if none can, it is booked as cells thrown away. What comes
    back is the "estimated number of cells that cannot be placed", measured in
    cells, so it shares a scale with the scoring (fewer leftover cells is
    better).
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
# Per-term formulas for the rule-based personalities
# --------------------------------------------------------------------------

def size_bonus(size, strategic):
    """Rule 5: 5 cells preferred in ordinary positions; relaxed to 4 and 3
    cells on strategic squares (key squares / crossings).

    The strategic reward is deliberately a bit larger than the "one cell
    missing" gap, so a 3-cell key square beats a 5-cell filler and a 4-cell one
    wins comfortably -- but 5 cells is still best on a strategic square. This
    allows small pieces; it does not demand them.
    """
    s = W_SIZE * (size - 3.0)
    return s + (W_STRATEGIC if strategic else 0.0)


def crossing_bonus(squares, usable):
    """Rule 4: the crossing step is mainly driven by maximising the number of
    playable squares, plus an extra penalty for too few usable vertices.

    "Crossing and immediately being sealed in" is what this rule guards
    against: after crossing there must still be somewhere to play, otherwise
    the move just throws the piece away.
    """
    s = W_CROSS + 2.0 * W_SQUARES * min(squares, SQ_CAP)
    if usable < SEAL_MIN:
        s -= W_SEAL * (SEAL_MIN - usable)
    return s


# --------------------------------------------------------------------------
# Weighted personalities: candidate scoring functions
# --------------------------------------------------------------------------

def board_feats(grid):
    """Three feature sets for every empty square: number of adjacent empty
    squares, number of stones of each player beside it, and number of opponent
    stones beside it.

    One O(400 x 4) scan does it all; every candidate afterwards is just a table
    lookup and a sum.
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
    """The score a weighted personality gives to a single candidate.

    Six weight terms (corner, centre, block, defend, open, size) summed
    linearly, so the relative sizes of the weights are exactly their trade-off
    directions. `chooser.choose_move` inlines the same expression, rewritten
    there to avoid a function call; the two must stay in sync.
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
