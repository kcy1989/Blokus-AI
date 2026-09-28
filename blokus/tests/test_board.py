"""Tests for the 20x20 board: placement, scoring, connectivity."""
import random
import time

from config import B, OWNER_CORNER
from board import Board, Reach, neighbors_of
from pieces import MASTER

QUADRANT_ORIGIN = {0: (10, 10), 1: (0, 0), 2: (10, 0), 3: (0, 10)}


def _pack_quadrant(names, w=B // 2, h=B // 2, limit=20.0):
    """Backtracking search that fits every piece in names inside a w x h box."""
    opts = []
    for n in sorted(names, key=lambda n: (-MASTER[n]["size"], n)):
        o = []
        for oi, cells in enumerate(MASTER[n]["orientations"]):
            mx = max(x for x, _ in cells)
            my = max(y for _, y in cells)
            if mx >= w or my >= h:
                continue
            m = 0
            for dx, dy in cells:
                m |= 1 << (dy * w + dx)
            o.append((m, mx, my, oi))
        opts.append((n, o))
    deadline = time.time() + limit

    def rec(i, used, acc):
        if i == len(opts):
            return acc
        if time.time() > deadline:
            raise TimeoutError("packing search expired")
        n, o = opts[i]
        for m, mx, my, oi in o:
            for y in range(h - my):
                for x in range(w - mx):
                    shifted = m << (y * w + x)
                    if used & shifted:
                        continue
                    got = rec(i + 1, used | shifted, acc + [(n, x, y, oi)])
                    if got is not None:
                        return got
        return None

    got = rec(0, 0, [])
    if got is None:
        raise AssertionError("could not pack pieces into %dx%d" % (w, h))
    return got


def pack_all(b, owner):
    """Place all 21 pieces of `owner` inside that owner's own 10x10 quadrant."""
    ox, oy = QUADRANT_ORIGIN[owner]
    for n, lx, ly, oi in _pack_quadrant(list(MASTER)):
        cells = MASTER[n]["orientations"][oi]
        assert b.can_place(ox + lx, oy + ly, cells, owner), "cannot place " + n
        b.place(ox + lx, oy + ly, cells, owner)


def test_can_place_boundary_overlap_corner():
    b = Board()
    assert b.can_place(0, 0, MASTER["I1"]["orientations"][0], 0)
    assert not b.can_place(19, 0, MASTER["I5"]["orientations"][1], 0)
    b.place(19, 0, MASTER["I1"]["orientations"][0], 1)
    assert not b.can_place(19, 0, MASTER["I1"]["orientations"][0], 1)
    assert b.can_place(0, 19, MASTER["I1"]["orientations"][0], 2)
    assert not b.can_place(-1, 0, MASTER["I1"]["orientations"][0], 2)
    assert not b.can_place(0, -1, MASTER["I1"]["orientations"][0], 2)


def test_can_place_requires_own_corner_for_opening():
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    l5 = MASTER["L5"]["orientations"][1]   # oi=1 has its bottom-right cell filled
    # without the rule anything goes
    assert b.can_place(5, 5, l5, 0)
    # with the rule only placements covering (19, 19) are legal
    assert not b.can_place(5, 5, l5, 0, must_cover=(19, 19))
    assert not b.can_place(0, 0, l5, 0, must_cover=(19, 19))
    assert b.can_place(18, 16, l5, 0, must_cover=(19, 19))
    assert b.can_place(19, 19, i1, 0, must_cover=(19, 19))
    # oi=0 cannot reach that corner at all, so nothing else is accepted
    l5a = MASTER["L5"]["orientations"][0]
    assert not b.can_place(19, 16, l5a, 0, must_cover=(19, 19))
    assert not b.can_place(18, 16, l5a, 0, must_cover=(19, 19))


def test_every_piece_can_cover_its_own_corner():
    """The opening rule must never make a player unable to move."""
    for owner, (cx, cy) in OWNER_CORNER.items():
        b = Board()
        cells_iter = [MASTER[n]["orientations"] for n in MASTER]
        assert b.has_legal_move(owner, cells_iter, must_cover=(cx, cy)), owner


def test_any_legal_respects_must_cover():
    b = Board()
    names = [MASTER[n]["orientations"] for n in MASTER]
    assert b.has_legal_move(0, names, must_cover=None)
    assert b.has_legal_move(0, names, must_cover=(19, 19))
    assert b.has_legal_move(0, names, must_cover=(0, 0))
    assert b.has_legal_move(0, names, must_cover=(19, 0))
    assert b.has_legal_move(0, names, must_cover=(0, 19))
    b.place(0, 0, MASTER["I1"]["orientations"][0], 1)
    b.place(19, 0, MASTER["I1"]["orientations"][0], 1)
    b.place(0, 19, MASTER["I1"]["orientations"][0], 1)
    b.place(19, 19, MASTER["I1"]["orientations"][0], 1)
    assert not b.has_legal_move(0, names, must_cover=(19, 19))
    assert b.has_legal_move(0, names, must_cover=None)


def test_full_hand_fills_each_quadrant():
    """All 21 pieces of all four players fit in the 10x10 quadrant each owns."""
    b = Board()
    for o in range(4):
        pack_all(b, o)
    assert sum(1 for i in range(400) if b.grid[i] != -1) == 356
    for o in range(4):
        assert b.all_owner_cells(o)


def _brute_has_legal(board, names):
    for n in names:
        for cells in MASTER[n]["orientations"]:
            mx = max(x for x, _ in cells)
            my = max(y for _, y in cells)
            for y in range(20 - my):
                for x in range(20 - mx):
                    if board.can_place(x, y, cells, 0):
                        return True
    return False


def test_has_legal_move_on_empty_board():
    b = Board()
    assert b.has_legal_move(0, [MASTER[n]["orientations"] for n in MASTER])


def test_has_legal_move_matches_brute_force():
    names = [MASTER[n]["orientations"] for n in MASTER]
    packed = Board()
    for o in range(4):
        pack_all(packed, o)
    rng = random.Random(5)
    partial = Board()
    for _ in range(150):
        n = rng.choice(list(MASTER))
        oc = rng.choice(MASTER[n]["orientations"])
        mx = max(x for x, _ in oc)
        my = max(y for _, y in oc)
        spots = [(x, y) for y in range(20 - my) for x in range(20 - mx)
                 if partial.can_place(x, y, oc, 0)]
        if spots:
            x, y = rng.choice(spots)
            partial.place(x, y, oc, rng.randrange(4))
    for bd in (Board(), partial, packed):
        assert bd.has_legal_move(0, names) == _brute_has_legal(bd, list(MASTER))


def test_no_legal_move_on_completely_full_board():
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    for i in range(400):
        b.place(i % 20, i // 20, i1, (i // 20) % 4)
    assert sum(1 for c in b.grid if c == -1) == 0
    assert not b.has_legal_move(0, [MASTER[n]["orientations"] for n in MASTER])


# ---------------------------------------------------------------- corner contact

def _xy(i):
    return (i % 20, i // 20)


def _mask_set(mask):
    return {_xy(i) for i in range(400) if (mask >> i) & 1}


def _brute_need(b, owner):
    out = set()
    for i in range(400):
        if b.grid[i] != owner:
            continue
        x, y = _xy(i)
        for dx in (-1, 1):
            for dy in (-1, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 20 and 0 <= ny < 20 and b.grid[ny * 20 + nx] == -1:
                    out.add((nx, ny))
    return out


def _brute_avoid(b, owner):
    out = set()
    for i in range(400):
        if b.grid[i] == owner:
            for n in neighbors_of()[i]:
                if b.grid[n] == -1:
                    out.add(_xy(n))
    return out


def test_corner_contact_only():
    """The house rule: a new piece must meet one of your own stones at a
    corner, and must not share an edge with any of them."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    b.place(10, 10, i1, 0)
    r = b.reach(0)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        assert b.can_place(10 + dx, 10 + dy, i1, 0, None, r), (dx, dy)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        assert not b.can_place(10 + dx, 10 + dy, i1, 0, None, r), (dx, dy)
    # far away: neither corner nor edge contact
    assert not b.can_place(0, 0, i1, 0, None, r)
    # reach=None keeps the rule-free behaviour scoring fixtures rely on
    assert b.can_place(0, 0, i1, 0)
    assert b.can_place(11, 10, i1, 0, None, Reach(0, 0)) is False


def test_same_colour_edge_contact_is_rejected():
    """Guards the half of the rule that is easy to forget: a piece that
    corner-touches one stone but edge-touches another is still illegal."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    b.place(10, 10, i1, 0)   # corner-contact source
    b.place(12, 10, i1, 0)   # edge-contact source
    r = b.reach(0)
    assert (11, 9) in _mask_set(r.need)
    assert (11, 10) in _mask_set(r.avoid)
    # (11,11) only touches diagonally, (11,9) only diagonally
    assert b.can_place(11, 11, i1, 0, None, r)
    assert b.can_place(11, 9, i1, 0, None, r)
    # (11,10) is edge-adjacent to both stones: corner contact is not enough
    assert not b.can_place(11, 10, i1, 0, None, r)
    # (13,11) only touches (12,10) diagonally
    assert b.can_place(13, 11, i1, 0, None, r)
    # (12,11) is edge-adjacent to (12,10) and corner-free
    assert not b.can_place(12, 11, i1, 0, None, r)


def test_squares_needing_and_avoiding_contact_are_unusable():
    """A square diagonally next to one of your stones but edge next to another
    lands in both halves of the rule, so no piece can ever be put there."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    b.place(10, 10, i1, 0)
    b.place(12, 9, i1, 0)
    r = b.reach(0)
    overlap = _mask_set(r.need & r.avoid)
    assert overlap == {(11, 9), (11, 10)}
    for x, y in overlap:
        assert not b.can_place(x, y, i1, 0, None, r)
    # ... and the rule is still satisfiable elsewhere
    assert b.can_place(9, 9, i1, 0, None, r)
    assert b.can_place(11, 11, i1, 0, None, r)


def test_opening_piece_is_exempt():
    """The two rules are mutually exclusive: `Game.reach` returns None (not an
    empty Reach) while placed==0, so the opening is decided by must_cover."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    assert b.reach(0) == Reach(0, 0), "no stones means no corner squares"
    assert b.can_place(19, 19, i1, 0, (19, 19), None)
    assert not b.can_place(10, 10, i1, 0, (19, 19), None)
    b.place(19, 19, i1, 0)
    r = b.reach(0)
    assert b.can_place(18, 18, i1, 0, None, r)
    assert not b.can_place(18, 19, i1, 0, None, r)
    assert not b.can_place(19, 19, i1, 0, None, r)


def test_reach_masks_do_not_wrap_rows():
    """The diagonal dilation shifts by B-1 and B+1, which straddle a row, so a
    single unguarded shift invents corner contacts across the board edge."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    b.place(0, 5, i1, 0)
    r = b.reach(0)
    assert _mask_set(r.need) == {(1, 4), (1, 6)}
    assert _mask_set(r.avoid) == {(0, 4), (0, 6), (1, 5)}
    b2 = Board()
    b2.place(19, 7, i1, 0)
    r2 = b2.reach(0)
    assert _mask_set(r2.need) == {(18, 6), (18, 8)}
    assert _mask_set(r2.avoid) == {(19, 6), (19, 8), (18, 7)}
    # all four edges at once
    b3 = Board()
    for x, y in ((0, 0), (19, 0), (0, 19), (19, 19)):
        b3.place(x, y, i1, 0)
    r3 = b3.reach(0)
    assert _mask_set(r3.need) == {(1, 1), (18, 1), (1, 18), (18, 18)}
    assert _mask_set(r3.avoid) == {(1, 0), (0, 1), (18, 0), (19, 1),
                                   (0, 18), (1, 19), (18, 19), (19, 18)}


def test_reach_masks_match_brute_force():
    rng = random.Random(11)
    for trial in range(10):
        b = Board()
        for _ in range(rng.randrange(1, 30)):
            n = rng.choice(list(MASTER))
            oc = rng.choice(MASTER[n]["orientations"])
            mx = max(x for x, _ in oc)
            my = max(y for _, y in oc)
            spots = [(x, y) for y in range(20 - my) for x in range(20 - mx)
                     if b.can_place(x, y, oc, 0)]
            if spots:
                x, y = rng.choice(spots)
                b.place(x, y, oc, rng.randrange(4))
        for o in range(4):
            r = b.reach(o)
            assert _mask_set(r.need) == _brute_need(b, o), (trial, o)
            assert _mask_set(r.avoid) == _brute_avoid(b, o), (trial, o)
            # both halves only ever contain squares the piece could occupy
            assert not ((r.need | r.avoid) & ~b.empty_bits), (trial, o)


def test_has_legal_move_respects_corner_rule():
    names = [MASTER[n]["orientations"] for n in MASTER]
    b = Board()
    b.place(10, 10, MASTER["I1"]["orientations"][0], 0)
    assert b.has_legal_move(0, names, None, b.reach(0))
    # owner 1 owns nothing, so the rule leaves it with no move at all
    assert not b.has_legal_move(1, names, None, b.reach(1))


def test_any_legal_agrees_with_can_place():
    b = Board()
    b.place(6, 6, MASTER["O4"]["orientations"][0], 0)
    r = b.reach(0)
    for name in ("I1", "L5", "T4", "O4", "V3"):
        cells = MASTER[name]["orientations"][0]
        mx = max(x for x, _ in cells)
        my = max(y for _, y in cells)
        brute = any(b.can_place(x, y, cells, 0, None, r)
                    for y in range(20 - my) for x in range(20 - mx))
        assert b.any_legal(cells, None, r) == brute, name


def test_unplace_restores_owner_bits():
    b = Board()
    b.place(4, 4, MASTER["O4"]["orientations"][0], 0)
    full = b.reach(0)
    b.unplace(4, 4, MASTER["O4"]["orientations"][0])
    assert b.reach(0) == Reach(0, 0)
    assert b.empty_bits == (1 << 400) - 1
    b.place(4, 4, MASTER["O4"]["orientations"][0], 2)
    assert b.reach(0) == Reach(0, 0) and b.reach(2) == full
