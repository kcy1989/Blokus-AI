"""20x20 board: grid, placement, connectivity, scoring."""
from typing import NamedTuple

from config import B, N, CORNERS

# Column guards for the bitmask dilations below. Without them `mask >> 1` on an
# x=0 square would land on (y-1, 19) and `mask << 1` on an x=19 square on
# (y+1, 0), i.e. a row wrap that invents neighbours which do not exist.
COL0 = 0
COL_LAST = 0
for _y in range(B):
    COL0 |= 1 << (_y * B)
    COL_LAST |= 1 << (_y * B + B - 1)
ALL = (1 << N) - 1

# The corner-contact rule reads far more naturally in vertex language: two
# squares share an edge iff they share two vertices, and they touch at a corner
# iff they share exactly one. The 20x20 squares therefore sit on a 21x21 grid
# of 交點, indexed Y*V + X.
V = B + 1
ROW_BITS = (1 << B) - 1

# Top-left corner of every 2x2 方塊, i.e. the anchors a crossing can sit on.
BLOCK_ANCHORS = 0
for _y in range(B - 1):
    BLOCK_ANCHORS |= ((1 << (B - 1)) - 1) << (_y * B)


def dilate(mask):
    """One step of 4-directional (edge-only) expansion of a bitmask.

    The column guards mask the *source* cells, not the results: an x=0
    square shifted right by one lands on (y-1, 19), so those source bits
    are dropped before the shift rather than after it.
    """
    left = (mask & ~COL0) >> 1
    right = (mask & ~COL_LAST) << 1
    return (left | right | (mask >> B) | (mask << B)) & ALL


def dilate_diag(mask):
    """One step of 8-directional expansion, i.e. corner contact only.

    Same source-side guarding as `dilate`: shifting by B-1 moves (x, y) to
    (x+1, y-1), so it must exclude x=B-1 sources, while B+1 moves to
    (x-1, y-1) and must exclude x=0 sources.
    """
    left = mask & ~COL0
    right = mask & ~COL_LAST
    return ((left >> (B + 1)) | (right >> (B - 1))
            | (left << (B - 1)) | (right << (B + 1))) & ALL


def cells_to_vertices(mask):
    """Project a 400-bit square mask onto the 441-bit 交點 grid.

    A square (x, y) owns the four vertices (x, y) (x+1, y) (x, y+1) (x+1, y+1),
    so the whole projection is a row-wise widening plus one final shift down by
    a vertex row.
    """
    out = 0
    for y in range(B):
        row = (mask >> (y * B)) & ROW_BITS
        if row:
            out |= (row | (row << 1)) << (y * V)
    return out | (out << V)

_NEIGHBORS_CACHE = None


class Reach(NamedTuple):
    """The corner-contact rule, as a single value.

    A piece is legal when it covers at least one square of `need` (corner
    contact with one of your own stones) and none of `avoid` (no edge contact
    with your own stones). Keeping both halves in one object means a caller
    cannot enforce only part of the rule.
    """

    need: int
    avoid: int


def neighbors_of():
    global _NEIGHBORS_CACHE
    if _NEIGHBORS_CACHE is not None:
        return _NEIGHBORS_CACHE
    out = [[] for _ in range(N)]
    for y in range(B):
        for x in range(B):
            i = y * B + x
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < B and 0 <= ny < B:
                    out[i].append(ny * B + nx)
    _NEIGHBORS_CACHE = out
    return out


class Board:
    def __init__(self):
        self.grid = [-1] * N
        self.empty_bits = ALL
        self.owner_bits = [0, 0, 0, 0]
        self.corner_regions = [[frozenset() for _ in range(4)] for _ in range(4)]
        self.borders = [[frozenset() for _ in range(4)] for _ in range(4)]

    def owner(self, x, y):
        if 0 <= x < B and 0 <= y < B:
            return self.grid[y * B + x]
        return -2

    def can_place(self, x, y, cells, owner, must_cover=None, reach=None):
        """Legality choke point.

        `must_cover` forces the opening piece onto a corner square. `reach` is a
        `Reach(need, avoid)` pair enforcing the corner-contact rule: the piece
        must share a *corner* with one of `owner`'s stones and must not share
        an edge with any of them. Both are None by default so rule-free tests
        (scoring fixtures) keep working; real games always pass them through
        `Game`.
        """
        placed = 0
        for dx, dy in cells:
            cx, cy = x + dx, y + dy
            if cx < 0 or cx >= B or cy < 0 or cy >= B or self.grid[cy * B + cx] != -1:
                return False
            placed |= 1 << (cy * B + cx)
        if must_cover is not None:
            mx, my = must_cover
            if not any(x + dx == mx and y + dy == my for dx, dy in cells):
                return False
        if reach is not None:
            if not (placed & reach.need):
                return False
            if placed & reach.avoid:
                return False
        return True


    def place(self, x, y, cells, owner):
        for dx, dy in cells:
            if not (0 <= x + dx < B and 0 <= y + dy < B):
                raise ValueError("piece out of bounds at (%d,%d)" % (x, y))
        base = x + y * B
        occ = 0
        for dx, dy in cells:
            idx = base + dy * B + dx
            self.grid[idx] = owner
            occ |= 1 << idx
        self.empty_bits &= ~occ
        self.owner_bits[owner] |= occ
        self._update_regions()

    def unplace(self, x, y, cells):
        base = x + y * B
        for dx, dy in cells:
            idx = base + dy * B + dx
            self.owner_bits[self.grid[idx]] &= ~(1 << idx)
            self.grid[idx] = -1
            self.empty_bits |= 1 << idx
        self._update_regions()

    def dilate(self, mask):
        """One step of 4-directional (edge-only) expansion of a bitmask.

        The column guards mask the *source* cells, not the results: an x=0
        square shifted right by one lands on (y-1, 19), so those source bits
        are dropped before the shift rather than after it.
        """
        return dilate(mask)

    def dilate_diag(self, mask):
        """One step of 8-directional expansion, i.e. corner contact only.

        Same source-side guarding as `dilate`: shifting by B-1 moves (x, y) to
        (x+1, y-1), so it must exclude x=B-1 sources, while B+1 moves to
        (x-1, y-1) and must exclude x=0 sources.
        """
        return dilate_diag(mask)

    def reach(self, owner):
        """Corner-contact rule: empty squares diagonally adjacent to `owner`'s
        stones, plus the empty squares edge-adjacent to them (which the piece
        may not cover)."""
        own = self.owner_bits[owner]
        return Reach(self.dilate_diag(own) & self.empty_bits,
                     self.dilate(own) & self.empty_bits)

    def connected_region(self, x, y, owner):
        start = x + y * B
        if self.grid[start] != owner:
            return frozenset()
        seen = {start}
        stack = [start]
        neigh = neighbors_of()
        while stack:
            p = stack.pop()
            for n in neigh[p]:
                if n not in seen and self.grid[n] == owner:
                    seen.add(n)
                    stack.append(n)
        return frozenset(seen)

    def _update_regions(self):
        """Corner blobs and their outlines.

        These feed the AI's corner heuristic in `_score_move`. They are not
        the player-facing score any more: a game is scored on the squares a
        player failed to place, tracked by `Game.remaining_cells`.
        """
        for o in range(4):
            for ci, (cx, cy) in enumerate(CORNERS):
                reg = self.connected_region(cx, cy, o)
                br = set(reg)
                for c in reg:
                    br.update(neighbors_of()[c])
                self.corner_regions[o][ci] = reg
                self.borders[o][ci] = frozenset(br)

    def any_legal(self, cells, must_cover=None, reach=None):
        mx = max(x for x, _ in cells)
        my = max(y for _, y in cells)
        empt = self.empty_bits
        offs = [dy * B + dx for dx, dy in cells]
        m = 0
        for o in offs:
            m |= 1 << o
        cbit = 0
        if must_cover is not None:
            cbit = 1 << (must_cover[0] + must_cover[1] * B)
        for y in range(B - my):
            ny = y * B
            for x in range(B - mx):
                shifted = m << (ny + x)
                if cbit:
                    if not shifted & cbit:
                        continue
                elif reach is not None:
                    if not shifted & reach.need or shifted & reach.avoid:
                        continue
                if (empt & shifted) == shifted:
                    return True
        return False

    def has_legal_move(self, owner, cells_iter, must_cover=None, reach=None):
        for orientations in cells_iter:
            for cells in orientations:
                if self.any_legal(cells, must_cover, reach):
                    return True
        return False

    def all_owner_cells(self, owner):
        grid = self.grid
        return [i for i, o in enumerate(grid) if o == owner]
