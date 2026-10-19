"""Window geometry: the scale/zoom constants, the hand-rack orientation
map, and the `Layout` class that derives every rectangle from them.

This module answers one question - what geometry does the window have - so
the drawing code in `ui.py` does not also own the arithmetic. It imports
only `pygame`, `pieces` and `config`, so nothing here can pull in the UI,
the game, or the AI.

`ui.py` re-exports every name here, and the same name in both modules is
bound to the same object; that only holds while nobody patches one side at
runtime.
"""
import pygame

from config import I
from pieces import MASTER

SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0)
DEFAULT_SCALE = 1.0
# Hand zoom is deliberately independent of board scale: a taller, more readable
# hand band costs vertical space the board would otherwise use, and only the
# user knows which side of that trade they want.
# Thumbnails fill the rack at the top of this range, so the default is the
# biggest size the rack can hold; the lower steps trade size for a calmer,
# easier-to-scan rack. See `Layout._max_fitting_hand_cell`.
HAND_ZOOMS = (1.0, 1.4, 1.7, 2.0)
DEFAULT_HAND_ZOOM = 2.0
DESKTOP_MARGIN = 48
# SDL reports the X11/Wayland *root* size, which on a scaled 4K desktop is twice
# the visible mode. Auto-fit therefore assumes a conservative usable area; the
# user can still push past it with '+'.
AUTO_FIT_CAP = (1920, 1080)


HAND_ITEMS = {}
for _n, _p in MASTER.items():
    best = min(range(len(_p["orientations"])),
               key=lambda i: (max(y for y in [c[1] for c in _p["orientations"][i]]),
                               max(x for x in [c[0] for c in _p["orientations"][i]]), i))
    HAND_ITEMS[_n] = best


class Layout:
    """Every piece of window geometry, derived from scale and hand zoom.

    The window is a fixed 16:9 frame at every scale: `scale` picks the height
    and the width follows from the aspect ratio. Inside it there are three
    columns - the player panel, the board, and the hand rack - and `MARGIN` is
    the single gutter between all of them, so the window reads as a grid rather
    than as regions that happen to touch.

    The hand rack is a right-hand column built by `layout_hand`: one captioned
    card per piece size, 1..5 ascending, each wrapping into as many rows as the
    column is narrow. Geometry always comes from the *full* 21-piece hand so the
    rack does not reflow as pieces are played, while the drawn slots re-pack
    from whatever is actually left.
    """

    def __init__(self, scale=DEFAULT_SCALE, hand_zoom=DEFAULT_HAND_ZOOM):
        s = float(scale)
        self.scale = s
        self.hand_zoom = float(hand_zoom)

        # One margin for the whole window: screen edge to panel content, panel
        # to board, board to buttons, and board to hand rack. Every gap you can
        # see is this number, which is what makes the window read as a grid
        # instead of a board crammed into a corner.
        self.MARGIN = max(8, int(round(18 * s)))
        self.DIVIDER_W = max(1, int(round(2 * s)))
        self.GUTTER = max(2, int(round(4 * s)))

        self.PANEL_W = int(round(330 * s))
        self.BTN_H = max(30, int(round(46 * s)))
        # Deliberately roomy: the three buttons sit close together, and a
        # fat-fingered confirm should not be one slip from cancel.
        self.BTN_GAP = int(round(20 * s))
        self.BTN_PAD = self.MARGIN

        # Hand thumbnails scale with the board, so the two knobs stay
        # independent: hand_zoom still picks the cell size, scale still picks
        # the board size.
        self.HAND_GAP = max(4, int(round(6 * s)))
        self.GROUP_GAP = max(6, int(round(9 * s)))
        # Each hand group is a captioned card: a label strip on top, then the
        # thumbnails, inside one rounded frame. Five cards stacked down a
        # narrow column is a lot of chrome, so the strip and padding stay tight
        # and their floors are low enough to fit the smallest scale.
        self.HAND_LABEL_H = max(10, int(round(15 * s)))
        self.HAND_PAD = max(3, int(round(8 * s)))

        # The window is a fixed 16:9 frame. `scale` picks its height and the
        # width follows from the aspect ratio, so the layout can never drift
        # out of proportion - not on a resize, not at any scale. The floor
        # keeps the smallest scale inside a 1000px-wide desktop once the
        # window margin is added.
        self.H = max(480, int(round(720 * s)))
        self.W = int(round(self.H * 16.0 / 9.0))

        # Three columns: the player panel, the board, and the hand rack.
        # Height is MARGIN + board + MARGIN + buttons + MARGIN, so the board is
        # as large as 16:9 allows and the columns all share one vertical rhythm.
        self.BO = (self.PANEL_W + self.MARGIN, self.MARGIN)
        self.board_px = self.H - 3 * self.MARGIN - self.BTN_H
        self.CELL = max(6, self.board_px // 20)
        self.board_px = 20 * self.CELL

        self.BTN_Y = self.BO[1] + self.board_px + self.MARGIN
        # The hand rack is a right-hand column spanning the full window height,
        # so picking a piece and reading the board are the same eye movement.
        self.HAND_X0 = self.BO[0] + self.board_px + self.MARGIN
        self.HAND_W = self.W - self.MARGIN - self.HAND_X0
        self.HAND_Y0 = self.MARGIN

        # The thumbnails are sized to *fill* the rack rather than to a fixed
        # multiple of the scale, otherwise the column ends in a large void
        # while the board runs the full window height and the window reads as
        # lopsided. `max_cell` is the largest cell the full hand fits at; the
        # zoom knob then scales that, so the top of the range fills the rack
        # and the lower steps give progressively smaller, easier-to-scan
        # thumbnails. Nothing is ever clipped: the result is capped at
        # `max_cell` by construction.
        self.hand_zoom = float(hand_zoom)
        self.HAND_CELL = 6  # provisional; the search below replaces it
        max_cell = self._max_fitting_hand_cell()
        cell = min(max_cell,
                   max(6, int(round(max_cell * self.hand_zoom
                                    / HAND_ZOOMS[-1]))))
        self.HAND_CELL = cell
        self.hand_h = self.layout_hand(sorted(MASTER))[2]

    def _max_fitting_hand_cell(self):
        """Largest thumbnail cell the full 21-piece hand fits in the rack."""
        lo, hi, best = 6, max(6, self.HAND_W // 5), 6
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._rack_fits(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _rack_fits(self, cell=None):
        """Does the full hand fit the rack at this thumbnail cell size?"""
        cell = self.HAND_CELL if cell is None else cell
        if cell * 5 > self.HAND_W - 2 * self.HAND_PAD:
            return False  # even the widest single piece is too wide
        keep = self.HAND_CELL
        self.HAND_CELL = cell
        try:
            return self.layout_hand(sorted(MASTER))[2] <= self.H - 2 * self.MARGIN
        finally:
            self.HAND_CELL = keep

    def layout_hand(self, names, width=None):
        """Pack `names` into the right-hand rack.

        One card per piece size, 1..5 cells ascending, so every size has its own
        caption and its own frame. The small groups hold only a handful of
        pieces, so the cards *flow*: each one is placed to the right of the
        previous and only wraps onto a new line when the column runs out. That
        keeps 1/2/3 sitting shoulder to shoulder instead of spending a whole
        screen height on three tiny stacks.

        Returns (slots, cards, height): screen rects per piece name, the
        per-group frame plus its caption, and the rack height in pixels.
        """
        width = self.HAND_W if width is None else width
        x0 = self.HAND_X0
        c = self.HAND_CELL
        label_h, pad = self.HAND_LABEL_H, self.HAND_PAD
        gap = self.HAND_GAP
        by_size = {}
        for n in names:
            by_size.setdefault(MASTER[n]["size"], []).append(n)

        # Phase 1: lay each card's own contents out and measure it. A card is
        # as wide as its widest row plus padding, and as tall as its caption
        # plus every row it wrapped into.
        drafts = []
        for size in sorted(by_size):
            items = []
            row_h = 0
            for n in sorted(by_size[size]):
                cells = MASTER[n]["orientations"][HAND_ITEMS[n]]
                w = (max(x for x, _ in cells) + 1) * c
                h = (max(yy for _, yy in cells) + 1) * c
                items.append((n, w, h))
                row_h = max(row_h, h)
            rows = [[]]
            row_w = 0
            for n, w, _h in items:
                if rows[-1] and row_w + w > width - 2 * pad:
                    rows.append([])
                    row_w = 0
                rows[-1].append((n, w))
                row_w += w + gap
            body_w = max(sum(w for _n, w in line) + gap * (len(line) - 1)
                         for line in rows)
            drafts.append((size, I["group_fmt"].format(size), rows, row_h,
                           body_w + 2 * pad,
                           label_h + 2 * pad + row_h * len(rows)))

        # Phase 2: flow the cards left to right, wrapping when the next one no
        # longer fits, then centre each resulting line in the column. Cards in
        # a line share a top edge so the captions line up.
        lines = []
        line = []
        line_w = 0
        for draft in drafts:
            card_w = draft[4]
            if line and line_w + card_w > width:
                # `line_w` carries a trailing gap for every card added, so take
                # one back off to get the real span of the line.
                lines.append((line, line_w - self.GROUP_GAP))
                line, line_w = [], 0
            line.append(draft)
            line_w += card_w + self.GROUP_GAP
        if line:
            lines.append((line, line_w - self.GROUP_GAP))

        slots = {}
        cards = []
        y = 0
        for line, line_w in lines:
            x = x0 + max(0, (width - line_w) // 2)
            line_h = 0
            for size, key, rows, row_h, card_w, card_h in line:
                rect = pygame.Rect(x, self.HAND_Y0 + y, card_w, card_h)
                cards.append((rect, key))
                body_top = rect.y + label_h + pad
                for r_line in rows:
                    total = sum(w for _n, w in r_line) + gap * (len(r_line) - 1)
                    # Centre the row inside the card, then centre each thumbnail
                    # in its row: bottom-aligning left a 1-row bar (I3) hanging
                    # off the floor of a 2-row group, which made the rack untidy.
                    xs = rect.x + pad + (card_w - 2 * pad - total) // 2
                    for n, w in r_line:
                        cells = MASTER[n]["orientations"][HAND_ITEMS[n]]
                        h = (max(yy for _, yy in cells) + 1) * c
                        slots[n] = pygame.Rect(xs, body_top + (row_h - h) // 2,
                                               w, h)
                        xs += w + gap
                    body_top += row_h
                x += card_w + self.GROUP_GAP
                line_h = max(line_h, card_h)
            y += line_h + self.GROUP_GAP
        height = (y - self.GROUP_GAP) if cards else 0
        return slots, cards, height

    def board_rect(self):
        return pygame.Rect(self.BO[0], self.BO[1], self.board_px, self.board_px)

    def cell_at(self, pos):
        """Board cell under a screen point, or None when outside the board."""
        x, y = pos
        if not (self.BO[0] <= x < self.BO[0] + self.board_px
                and self.BO[1] <= y < self.BO[1] + self.board_px):
            return None
        return ((x - self.BO[0]) // self.CELL, (y - self.BO[1]) // self.CELL)

    def fits(self, desktop):
        return (self.W <= desktop[0] - DESKTOP_MARGIN
                and self.H <= desktop[1] - DESKTOP_MARGIN)


# module-level defaults at scale 1.0 (kept for backwards compatibility / tests)
_L0 = Layout(DEFAULT_SCALE)
W, H, CELL, BO, PANEL_W = _L0.W, _L0.H, _L0.CELL, _L0.BO, _L0.PANEL_W
HAND_CELL, HAND_X0, HAND_Y0 = _L0.HAND_CELL, _L0.HAND_X0, _L0.HAND_Y0
BTN_Y, BTN_H = _L0.BTN_Y, _L0.BTN_H
