"""Pygame UI for single-player Blokus vs 3 AI."""
import json
import os
import threading
import time

import pygame

import ai
import records as records_mod
from config import COLORS, HAND_CELLS_TOTAL, I, OWNER_CORNER, PERSONALITY_ORDER
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
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

COLOR_LABELS = {"blue": "藍", "green": "綠", "red": "紅", "yellow": "黃"}
PERSONA_ZH = {k: I[k] for k in PERSONALITY_ORDER}

# One background colour for the whole window; the sidebar, the hand band and the
# board are deliberate blocks on top of it, so no accidental seams appear.
BG = (23, 28, 38)
BAND_BG = (28, 34, 46)
BOARD_BG = (12, 15, 20)
DIVIDER = (48, 56, 70)

CJK_CANDIDATES = ("Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans TC",
                  "Noto Sans SC", "Microsoft JhengHei", "PingFang TC", "Heiti TC")
CJK_FILE_HINTS = ("notosanscjk", "notosanstc", "notosanssc", "jhenghei", "pingfang",
                  "heiti", "wqy", "droidsansfallback", "sourcehansans", "msyh", "simhei")
TOFU = "\ue000"


def _has_cjk(font):
    try:
        glyph = font.render("永", True, (255, 255, 255))
        tofu = font.render(TOFU, True, (255, 255, 255))
    except Exception:
        return False
    return (pygame.image.tostring(glyph, "RGBA")
            != pygame.image.tostring(tofu, "RGBA"))


def find_fonts():
    out = []
    for root in ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]:
        if not os.path.isdir(root):
            continue
        for a, _, files in os.walk(root):
            for f in sorted(files):
                if f.lower().endswith((".ttf", ".ttc", ".otf")):
                    out.append(os.path.join(a, f))
    return out


def load_cjk_font_path():
    """Return a font file path that really carries CJK glyphs, else None."""
    if not pygame.font.get_init():
        pygame.font.init()
    for name in CJK_CANDIDATES:
        path = pygame.font.match_font(name)
        if not path:
            continue
        try:
            if _has_cjk(pygame.font.Font(path, 16)):
                return path
        except Exception:
            continue
    for path in find_fonts():
        low = path.lower().replace(" ", "").replace("-", "")
        if not any(h in low for h in CJK_FILE_HINTS):
            continue
        try:
            if _has_cjk(pygame.font.Font(path, 16)):
                return path
        except Exception:
            continue
    return None


def load_cjk_font(size=16):
    path = load_cjk_font_path()
    if path is None:
        print("warning: no CJK-capable font found; Chinese text may not render.")
        print("         install one with: sudo apt install fonts-noto-cjk")
        return pygame.font.Font(None, size)
    return pygame.font.Font(path, size)


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


HAND_ITEMS = {}
for _n, _p in MASTER.items():
    best = min(range(len(_p["orientations"])),
               key=lambda i: (max(y for y in [c[1] for c in _p["orientations"][i]]),
                               max(x for x in [c[0] for c in _p["orientations"][i]]), i))
    HAND_ITEMS[_n] = best
GROUP_BG = (36, 42, 55)



def corner_zh(owner):
    return (I["corner_br"] if owner == 0 else I["corner_tl"] if owner == 1
            else I["corner_tr"] if owner == 2 else I["corner_bl"])


def shade(color, factor):
    return tuple(max(0, min(255, int(v * factor))) for v in color)


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


def desktop_size():
    """True desktop resolution, independent of the current window size."""
    try:
        sizes = pygame.display.get_desktop_sizes()
        real = [s for s in sizes if s[0] >= 640 and s[1] >= 480]
        if real:
            return max(real, key=lambda s: s[0] * s[1])
    except Exception:
        pass
    return (1920, 1080)


def auto_fit_area():
    d = desktop_size()
    return (min(d[0], AUTO_FIT_CAP[0]), min(d[1], AUTO_FIT_CAP[1]))


def best_scale(desktop=None, hand_zoom=DEFAULT_HAND_ZOOM):
    area = desktop or auto_fit_area()
    for s in reversed(SCALES):
        if Layout(s, hand_zoom).fits(area):
            return s
    return SCALES[0]


# module-level defaults at scale 1.0 (kept for backwards compatibility / tests)
_L0 = Layout(DEFAULT_SCALE)
W, H, CELL, BO, PANEL_W = _L0.W, _L0.H, _L0.CELL, _L0.BO, _L0.PANEL_W
HAND_CELL, HAND_X0, HAND_Y0 = _L0.HAND_CELL, _L0.HAND_X0, _L0.HAND_Y0
BTN_Y, BTN_H = _L0.BTN_Y, _L0.BTN_H


class UI:
    def __init__(self, screen, font_path, scale=None, hand_zoom=None,
                 records=None):
        self.screen = screen
        self.font_path = font_path
        self._fonts = {}
        self.game = None
        self.state = "SETUP_COLOR"
        self.running = True
        self.drag = None
        self.anim = None
        self.ai_move = None
        self.ai_ready = False
        self.ai_pending = False
        self.ai_owner = -1
        self.ai_thread = None
        self.ai_token = 0
        self.toasts = []
        self.records = records if records is not None else records_mod.Records()
        self.result_recorded = False
        self._hand_cache = None
        self._hand_cache_key = None
        settings = load_settings()
        if hand_zoom is not None and float(hand_zoom) not in HAND_ZOOMS:
            hand_zoom = DEFAULT_HAND_ZOOM
        if scale is None:
            scale = settings.get("scale")
            if scale not in SCALES:
                scale = best_scale(hand_zoom=hand_zoom or DEFAULT_HAND_ZOOM)
            hand_zoom = settings.get("hand_zoom", hand_zoom or DEFAULT_HAND_ZOOM)
            if float(hand_zoom) not in HAND_ZOOMS:
                hand_zoom = DEFAULT_HAND_ZOOM
        self.scale = scale
        self.hand_zoom = float(hand_zoom if hand_zoom is not None else DEFAULT_HAND_ZOOM)
        self.L = Layout(self.scale, self.hand_zoom)
        self._sync_window()

    # ---------- window / scale ----------

    def _sync_window(self):
        self.screen = pygame.display.set_mode((self.L.W, self.L.H), pygame.RESIZABLE)
        self._fonts = {}
        self._hand_cache_key = None

    def _save(self):
        save_settings({"scale": self.scale, "hand_zoom": self.hand_zoom})

    def set_scale(self, scale, announce=True):
        if scale not in SCALES:
            return
        desktop = desktop_size()
        chosen = scale
        while chosen != SCALES[0] and not Layout(chosen, self.hand_zoom).fits(desktop):
            chosen = SCALES[SCALES.index(chosen) - 1]
        self.scale = chosen
        self.L = Layout(self.scale, self.hand_zoom)
        self._sync_window()
        self._save()
        if announce and chosen != scale:
            self.toast(I["scale_hint"].format(int(self.scale * 100)))
        return chosen

    def bump_scale(self, direction):
        i = SCALES.index(self.scale)
        i = max(0, min(len(SCALES) - 1, i + direction))
        return self.set_scale(SCALES[i])

    def set_hand_zoom(self, zoom, announce=True):
        zoom = min(HAND_ZOOMS, key=lambda z: abs(z - float(zoom)))
        chosen = zoom
        desktop = desktop_size()
        while (chosen > HAND_ZOOMS[0]
               and not Layout(self.scale, chosen).fits(desktop)):
            chosen = HAND_ZOOMS[HAND_ZOOMS.index(chosen) - 1]
        self.hand_zoom = chosen
        self.L = Layout(self.scale, self.hand_zoom)
        self._sync_window()
        self._save()
        if announce and chosen != zoom:
            self.toast(I["hand_zoom_hint"].format(int(self.L.hand_zoom * 100)))
        return chosen

    def bump_hand_zoom(self, direction):
        i = min(range(len(HAND_ZOOMS)),
                key=lambda j: abs(HAND_ZOOMS[j] - self.hand_zoom))
        i = max(0, min(len(HAND_ZOOMS) - 1, i + direction))
        return self.set_hand_zoom(HAND_ZOOMS[i])

    def fit_scale(self):
        """Largest board scale that fits; shrink the hand only if even the
        smallest board scale does not fit at the current hand zoom."""
        zoom = self.hand_zoom
        i = min(range(len(HAND_ZOOMS)), key=lambda j: abs(HAND_ZOOMS[j] - zoom))
        while i < len(HAND_ZOOMS) - 1 and not Layout(SCALES[0], HAND_ZOOMS[i]).fits(auto_fit_area()):
            i += 1
        self.hand_zoom = HAND_ZOOMS[i]
        return self.set_scale(best_scale(hand_zoom=self.hand_zoom))

    @property
    def font_scale(self):
        return max(0.85, min(self.L.scale, 1.45))

    # ---------- text ----------

    def font_of(self, size):
        size = max(9, int(round(size * self.font_scale)))
        f = self._fonts.get(size)
        if f is None:
            try:
                f = pygame.font.Font(self.font_path, size)
            except Exception:
                f = pygame.font.Font(None, size)
            self._fonts[size] = f
        return f

    def label(self, text, x, y, size, color, bg=None, center=False, right=False):
        img = self.font_of(size).render(text, True, color)
        if center:
            r = img.get_rect(center=(int(x), int(y)))
        elif right:
            r = img.get_rect(top=int(y), right=int(x))
        else:
            r = img.get_rect(x=int(x), y=int(y))
        if bg is not None:
            surf = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            surf.fill(bg + (220,))
            surf.blit(img, (0, 0))
            self.screen.blit(surf, r)
        else:
            self.screen.blit(img, r)
        return r

    def draw_alpha_rect(self, rect, color, alpha):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill(color + (alpha,))
        self.screen.blit(s, rect)

    # ---------- game flow ----------

    def new_game(self):
        self.game.reset()
        self.state = "SETUP_COLOR"
        self.drag = None
        self.anim = None
        self.ai_move = None
        self.ai_ready = False
        self.ai_pending = False
        self.ai_thread = None
        self.ai_token += 1
        self.toasts = []
        self.result_recorded = False

    def set_color(self, color):
        self.game.set_player_color(color)
        self.state = "SETUP_INFO"
        self.drag = None

    def toast(self, text):
        self.toasts.append((text, time.time() + 2.4))

    def remaining_cells(self, owner):
        return sum(MASTER[n]["size"] for n in self.game.hands[owner].names)

    def tick(self, events):
        for e in events:
            if e.type == pygame.QUIT:
                self.running = False
                return
            if e.type == pygame.VIDEORESIZE:
                # The window is locked to 16:9, so a manual drag is snapped
                # back to the framed size rather than allowed to skew the
                # layout. `scale` and the `+`/`-` keys are the real controls.
                self.L = Layout(self.scale, self.hand_zoom)
                self._sync_window()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                if e.key == pygame.K_r:
                    self.new_game()
                    return
                if e.key in ( pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.bump_scale(-1)
                    return
                if e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.bump_scale(1)
                    return
                if e.key == pygame.K_LEFTBRACKET:
                    self.bump_hand_zoom(-1)
                    return
                if e.key == pygame.K_RIGHTBRACKET:
                    self.bump_hand_zoom(1)
                    return
                if e.key == pygame.K_f:
                    self.fit_scale()
                    return
        if self.state == "SETUP_COLOR":
            self.handle_setup_color(events)
        elif self.state == "SETUP_INFO":
            self.handle_setup_info(events)
        elif self.state == "PLAYING":
            self.handle_playing(events)
        elif self.state == "ANIM":
            self.handle_anim()
        elif self.state == "GAME_OVER":
            self.handle_game_over(events)
        self.draw_toasts()
        pygame.display.update()

    def handle_setup_color(self, events):
        s = self.L.scale
        bw, bh = int(round(150 * s)), int(round(220 * s))
        gap = int(round(20 * s))
        total = 4 * bw + 3 * gap
        x0 = (self.L.W - total) // 2
        y0 = int(round(220 * s))
        self.draw_setup_color()
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for i, c in enumerate(["blue", "green", "red", "yellow"]):
                    r = pygame.Rect(x0 + i * (bw + gap), y0, bw, bh)
                    if r.collidepoint(e.pos):
                        self.set_color(c)
                        return

    def handle_setup_info(self, events):
        self.draw_setup_info()
        buttons = self.setup_button_rects()
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for bid, r in buttons.items():
                    if not r.collidepoint(e.pos):
                        continue
                    if bid == "redraw":
                        self.game.redraw_personalities()
                    else:
                        self.game.start()
                        self.state = "PLAYING"
                    return
            # Enter and space are the keyboard version of the start button;
            # clicking anywhere else deliberately does not start, or a stray
            # click next to "redraw AI" would launch the game.
            elif e.type == pygame.KEYDOWN and e.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.game.start()
                self.state = "PLAYING"
                return

    def handle_anim(self):
        self.draw_all()
        if self.anim and time.time() - self.anim["t"] >= 0.3:
            if self.game.state == "GAME_OVER":
                self.enter_game_over()
            else:
                self.state = "PLAYING"
            self.anim = None

    def handle_game_over(self, events):
        if self.game.state != "GAME_OVER":
            return
        self.draw_results()
        # The board stays on screen; only an explicit choice starts a new game.
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if self.new_game_rect().collidepoint(e.pos):
                    self.new_game()
                    return
            elif e.type == pygame.KEYDOWN and e.key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.new_game()
                return

    def enter_game_over(self):
        """Switch to the results view and bank this game's outcome once."""
        self.state = "GAME_OVER"
        if not self.result_recorded:
            self.result_recorded = True
            self.records.record(self.contestant_standings())
        self.announce_result()

    def announce_result(self):
        win = self.game.winner()
        if win == "player":
            self.label_color = (120, 220, 140)
        elif win == "draw":
            self.label_color = (235, 220, 120)
        else:
            self.label_color = COLORS[self.game.colors[int(win)]]

    def contestant_standings(self):
        """(key, remaining) per contestant.

        Keyed by role rather than seat: the AI seats get a different
        personality every game, so a record keyed by owner would blur three
        different contestants together.
        """
        out = []
        for owner, rem in self.game.standings():
            key = (records_mod.PLAYER_KEY if owner == 0
                   else self.game.owner_key[owner])
            out.append((key, rem))
        return out

    def contestant_name(self, key):
        if key == records_mod.PLAYER_KEY:
            return I["player"]
        return PERSONA_ZH.get(key, key)

    def contestant_color(self, key):
        if key == records_mod.PLAYER_KEY:
            return COLORS[self.game.colors[0]]
        for o in (1, 2, 3):
            if self.game.owner_key[o] == key:
                return COLORS[self.game.colors[o]]
        return (200, 200, 210)

    def new_game_rect(self):
        L = self.L
        w = L.PANEL_W - 2 * L.MARGIN
        h = int(round(48 * L.scale))
        return pygame.Rect(L.MARGIN, L.BTN_Y - L.MARGIN - h, w, h)

    # ---------- player turn ----------

    def handle_playing(self, events):
        owner = self.game.current_owner()
        if owner != 0:
            self.draw_all()
            self.handle_ai_turn(owner)
            return

        if not self.game.has_legal(0):
            self.toast(I["auto_pass"])
            self.game.act_pass()
            if self.game.state == "GAME_OVER":
                self.enter_game_over()
            self.draw_all()
            return

        self.draw_all()
        buttons = self.button_rects()
        for bid, r in buttons.items():
            self.draw_button(bid, r)
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                self.update_drag(e.pos)
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    self.try_confirm()
                elif e.key in (pygame.K_x, pygame.K_UP, pygame.K_RIGHT):
                    self.rotate_drag()
                elif e.key in (pygame.K_c, pygame.K_BACKSPACE, pygame.K_DELETE):
                    self.drag = None
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                pos = e.pos
                if e.button == 3:
                    # right click rotates the held piece
                    if self.drag:
                        self.rotate_drag()
                    continue
                hit = False
                for bid, r in buttons.items():
                    if r.collidepoint(pos):
                        hit = True
                        if bid == "rotate":
                            self.rotate_drag()
                        elif bid == "confirm":
                            self.try_confirm()
                        elif bid == "cancel":
                            self.drag = None
                        break
                if hit:
                    continue
                if self.pick_other_piece(pos):
                    # left-clicking another piece swaps to it straight away,
                    # so the held one never has to be cancelled first
                    continue
                if self.drag:
                    # clicking the board locks the placement where it was clicked
                    self.update_drag(pos, force=True)
                    cell = self.L.cell_at(pos)
                    if cell is not None:
                        self.drag["locked"] = True
                else:
                    for name in self.hand_order():
                        r = self.hand_slot(name)
                        if r is not None and r.collidepoint(pos):
                            self.start_drag(name, pos)
                            break

    def pick_other_piece(self, pos):
        """Select a different hand piece under `pos`, if there is one.

        Only switches to a piece other than the one already held, so clicking
        the held piece (or empty hand band) keeps its current preview instead
        of silently dropping it.
        """
        if not self.drag:
            return False
        for name in self.hand_order():
            if name == self.drag["name"]:
                continue
            r = self.hand_slot(name)
            if r is not None and r.collidepoint(pos):
                self.start_drag(name, pos)
                return True
        return False

    def handle_ai_turn(self, owner):
        if self.ai_ready and self.ai_owner == owner:
            mv = self.ai_move
            self.ai_ready = False
            self.ai_pending = False
            self.ai_move = None
            if mv is None:
                self.toast("%s %s" % (PERSONA_ZH[self.game.owner_key[owner]],
                                       I["auto_pass"]))
                self.game.act_pass()
            else:
                self.game.act(*mv)
            if self.game.state == "GAME_OVER":
                self.enter_game_over()
            elif mv is not None:
                self.anim = {"name": mv[0], "oi": mv[1], "x": mv[2], "y": mv[3],
                             "owner": owner, "t": time.time()}
                self.state = "ANIM"
        elif not self.ai_pending:
            self.ai_owner = owner
            self.ai_pending = True
            self.ai_ready = False
            self.ai_move = None
            self.ai_token += 1
            self.ai_thread = threading.Thread(target=self.ai_worker,
                                             args=(owner, self.ai_token), daemon=True)
            self.ai_thread.start()

    def button_rects(self):
        """Confirm / rotate / cancel, in the strip directly under the board.

        The strip is centred under the board rather than pushed to the window
        edge, and it sits MARGIN below the board and MARGIN above the hand
        band, so the vertical rhythm matches the margin everywhere else.
        """
        L = self.L
        bw = int(round(120 * L.scale))
        rw = int(round(168 * L.scale))
        gap = L.BTN_GAP
        total = 2 * bw + rw + 2 * gap
        left = L.BO[0] + (L.board_px - total) // 2
        return {
            "cancel": pygame.Rect(left + total - bw, L.BTN_Y, bw, L.BTN_H),
            "rotate": pygame.Rect(left + bw + gap, L.BTN_Y, rw, L.BTN_H),
            "confirm": pygame.Rect(left, L.BTN_Y, bw, L.BTN_H),
        }

    def setup_panel_rect(self):
        """The centred column on the setup screen, in the same scale the drawing
        uses, so hit boxes and labels can never drift apart."""
        s = self.L.scale
        w = int(round(620 * s))
        return pygame.Rect((self.L.W - w) // 2, 0, w, self.L.H)

    def setup_button_rects(self):
        """Redraw AI (left) and Start (right), side by side under the panel."""
        L = self.L
        panel = self.setup_panel_rect()
        y = self.setup_content_bottom()
        bh = L.BTN_H
        bw = int(round(112 * L.scale))
        sw = int(round(128 * L.scale))
        pad = int(round(18 * L.scale))
        return {
            "redraw": pygame.Rect(panel.x, y, bw, bh),
            "start_game": pygame.Rect(panel.right - sw, y, sw, bh),
        }

    def setup_content_bottom(self):
        """Y where the setup screen's text ends (the buttons sit right below)."""
        s = self.L.scale
        rows = int(round(132 * s)) + 4 * int(round(74 * s))
        return rows + int(round((30 + 20 * len(I["turn_note"])) * s)) + int(round(6 * s))

    # ---------- hand ----------

    def hand_order(self):
        """Pieces sorted by cell count then name, independent of Hand order."""
        return sorted(self.game.hands[0].names,
                      key=lambda n: (MASTER[n]["size"], n))

    def hand_layout(self):
        """Rack geometry, always measured from the *full* 21-piece hand.

        Playing a piece must not reflow the rack: re-packing from whatever is
        left makes every other thumbnail jump, which is both hard to follow and
        a trap to click. So the geometry is fixed and a used piece simply leaves
        an empty slot behind.
        """
        if self._hand_cache_key != "full":
            self._hand_cache = self.L.layout_hand(sorted(MASTER))[:2]
            self._hand_cache_key = "full"
        return self._hand_cache

    def hand_index(self, name):
        names = self.hand_order()
        return names.index(name) if name in names else -1

    def hand_slot(self, name):
        """Fixed slot for `name`, or None once the piece has been played."""
        slots, _cards = self.hand_layout()
        if name not in self.game.hands[0].names:
            return None
        return slots.get(name)

    def start_drag(self, name, pos):
        if name in self.game.hands[0].names:
            self.drag = {"name": name, "oi": HAND_ITEMS[name], "rot": 0,
                         "x": None, "y": None, "valid": False, "locked": False,
                         "mx": pos[0], "my": pos[1], "over": False}

    def rotate_drag(self):
        if not self.drag:
            return
        n = self.drag["name"]
        self.drag["oi"] = (self.drag["oi"] + 1) % len(MASTER[n]["orientations"])
        self.drag["rot"] += 1
        if self.drag["x"] is not None:
            self._refresh_preview()

    def try_confirm(self):
        d = self.drag
        if not d or d["x"] is None or not d["valid"]:
            if d and d["x"] is not None and not d["valid"]:
                if self.game.must_cover(0) is not None:
                    self.toast(I["open_rule_toast"].format(corner_zh(0)))
                elif self.game.placed[0] > 0:
                    self.toast(I["touch_rule_toast"])
            return False
        self.place_drag()
        self.drag = None
        return True

    def place_drag(self):
        n, oi = self.drag["name"], self.drag["oi"]
        x, y = self.drag["x"], self.drag["y"]
        self.game.act(n, oi, x, y)
        if self.game.state == "GAME_OVER":
            self.enter_game_over()
        else:
            self.anim = {"name": n, "oi": oi, "x": x, "y": y, "owner": 0,
                         "t": time.time()}
            self.state = "ANIM"

    def _refresh_preview(self):
        """Legality comes from Game itself, so the preview can never disagree
        with what `act` will accept."""
        d = self.drag
        d["valid"] = self.game.can_act(d["name"], d["oi"], d["x"], d["y"], 0)

    def update_drag(self, pos, force=False):
        """Track the cursor. The preview keeps its last board position when the
        cursor leaves the board, so the confirm button stays reachable."""
        d = self.drag
        if not d or pos is None:
            return
        d["mx"], d["my"] = pos
        cell = self.L.cell_at(pos)
        d["over"] = cell is not None
        if cell is None or (d["locked"] and not force):
            return
        d["x"] = max(0, min(19, cell[0]))
        d["y"] = max(0, min(19, cell[1]))
        self._refresh_preview()

    def draw_drag_ghost(self):
        d = self.drag
        if not d or d["x"] is not None or d["over"]:
            return
        cells = MASTER[d["name"]]["orientations"][d["oi"]]
        c = self.L.CELL
        pw = (max(x for x, _ in cells) + 1) * c
        ph = (max(y for _, y in cells) + 1) * c
        x0 = d["mx"] - pw // 2
        y0 = d["my"] - ph // 2
        col = COLORS[self.game.colors[0]]
        for dx, dy in cells:
            r = pygame.Rect(x0 + dx * c, y0 + dy * c, c, c)
            self.draw_alpha_rect(r, col, 190)
            self.draw_alpha_rect(pygame.Rect(r.x + 1, r.y + 1, c - 3, c - 3),
                                 shade(col, 0.7), 120)

    # ---------- geometry helpers ----------

    def draw_button(self, bid, r, size=18):
        """Action buttons in the strip under the board.

        `confirm` is the action the player reaches for after picking a cell, so
        it carries the only accent fill; rotate and cancel stay neutral. That
        keeps the strip calm while making the intended click obvious, and the
        disabled confirm greys out rather than inviting a click it will reject.
        """
        surf = self.screen
        s = self.L.scale
        enabled = True
        if bid == "confirm":
            enabled = bool(self.drag and self.drag["x"] is not None and self.drag["valid"])
        accent = bid == "confirm"
        if not enabled:
            fill, edge, ink = (48, 53, 63), (64, 70, 82), (128, 133, 143)
        elif accent:
            fill, edge, ink = (46, 104, 78), (86, 178, 132), (240, 250, 244)
        else:
            fill, edge, ink = (44, 56, 76), (86, 112, 150), (226, 233, 244)
        radius = max(4, int(round(8 * s)))
        pygame.draw.rect(surf, fill, r, border_radius=radius)
        pygame.draw.rect(surf, edge, r, max(1, int(round(2 * s))),
                         border_radius=radius)
        img = self.font_of(size).render(I[bid], True, ink)
        self.screen.blit(img, img.get_rect(center=r.center))

    # ---------- drawing ----------

    def draw_all(self):
        self.screen.fill(BG)
        self.draw_board()
        self.draw_left()
        self.draw_hand()
        self.draw_drag_ghost()

    def draw_left(self):
        L = self.L
        surf = self.screen
        s = L.scale
        # Everything in the sidebar hangs off the same content edge as the
        # board's margin, so the panel reads as a column rather than a stack
        # of independently placed labels.
        x0 = L.MARGIN
        inner = L.PANEL_W - 2 * L.MARGIN
        right = x0 + inner
        sw = int(round(26 * s))
        name_x = x0 + sw + int(round(12 * s))

        self.label(I["title"], x0, L.MARGIN, 28, (235, 235, 235))
        owner = self.game.current_owner()
        base_y = L.MARGIN + int(round(52 * s))
        row_h = int(round(76 * s))
        # Rows follow the turn order, not the seat index: whoever moves first
        # this game is row one. Seat order is an internal detail the player
        # never sees, and it looked arbitrary next to the turn indicator.
        for row, i in enumerate(self.game.turn_order):
            y = base_y + row * row_h
            color = COLORS[self.game.colors[i]]
            active = (i == owner)
            if active:
                hl = pygame.Surface((L.PANEL_W, row_h - int(round(6 * s))),
                                    pygame.SRCALPHA)
                hl.fill((255, 255, 255, 22))
                surf.blit(hl, (0, y - int(round(5 * s))))
            name = I["player"] if i == 0 else PERSONA_ZH[self.game.owner_key[i]]
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(x0, y + int(round(8 * s)), sw, sw))
            self.label(name, name_x, y, 22,
                       (235, 235, 240) if active else (185, 190, 200))
            self.label(corner_zh(i), right, y + int(round(4 * s)), 16,
                       (150, 155, 165), right=True)
            left_pieces = len(self.game.hands[i].names)
            left_cells = self.remaining_cells(i)
            self.label(I["left_fmt"].format(left_pieces, left_cells),
                       name_x, y + int(round(32 * s)), 15, (140, 145, 155))
            # live bar: how many of the 89 cells this player still holds.
            # Remaining cells are the score, and fewer is better, so the bar
            # drains as the player does well.
            frac = left_cells / float(HAND_CELLS_TOTAL)
            bar = pygame.Rect(name_x, y + int(round(56 * s)),
                              right - name_x, max(3, int(round(6 * s))))
            pygame.draw.rect(surf, (40, 45, 54), bar)
            fill = pygame.Rect(bar.x, bar.y, int(bar.w * frac), bar.h)
            pygame.draw.rect(surf, color, fill)
            pygame.draw.rect(surf, (70, 76, 88), bar, 1)

        ty = base_y + 4 * row_h + int(round(10 * s))
        self.label("%s：%s" % (I["turn"], I["player"] if owner == 0
                               else PERSONA_ZH[self.game.owner_key[owner]]),
                   x0, ty, 20, (225, 225, 232))
        if owner != 0 and self.ai_pending and not self.ai_ready:
            oc = self.game.colors[owner]
            self.label("%s %s %s…" % (PERSONA_ZH[self.game.owner_key[owner]],
                                      COLOR_LABELS[oc], I["thinking"]),
                       x0, ty + int(round(34 * s)), 17, COLORS[oc])
        # The sidebar is only as tall as the board, so secondary rows are
        # dropped from the bottom when there is not enough room for them.
        side_bottom = L.BO[1] + L.board_px + L.MARGIN
        y = ty + int(round(48 * s))
        step = int(round(24 * s))
        # A rule between the turn block and the hint block, so the lower half
        # reads as one footnote rather than drifting text.
        pygame.draw.rect(surf, (46, 53, 66),
                         pygame.Rect(x0, y - int(round(16 * s)), inner,
                                     max(1, int(round(1 * s)))))

        def hint(text, size=15, color=(130, 136, 146), dy=0):
            nonlocal y
            top = y + dy
            if top + int(round(size * s)) <= side_bottom:
                self.label(text, x0, top, size, color)
            y += step

        hint(I["scale_hint"].format(int(self.scale * 100)))
        # Show the zoom the rack actually got, which can be less than asked
        # for when the column is too narrow or short to hold it.
        hint(I["hand_zoom_hint"].format(int(L.hand_zoom * 100)))
        for line in I["keys_hint"]:
            hint(line, 14, (120, 126, 136))
        if self.drag and self.drag["locked"]:
            hint(I["locked"], 16, (120, 220, 150))
        if self.game.must_cover(0) is not None:
            hint(I["open_rule_hint"].format(corner_zh(0)), 15, (235, 200, 120))
        elif self.game.placed[0] > 0:
            hint(I["touch_rule_hint"], 15, (235, 200, 120))
        hy = y + int(round(6 * s))
        hstep = int(round(18 * s))
        for line in I["hand_hint"]:
            if hy + hstep <= side_bottom:
                self.label(line, x0, hy, 14, (110, 116, 126))
            hy += hstep
        # Separator between the panel and the board column. It runs the full
        # window height: the board is a square that stops short of the bottom,
        # and a rule that stopped with it left the panel looking unterminated
        # above the button strip.
        pygame.draw.rect(surf, DIVIDER,
                         pygame.Rect(L.PANEL_W - L.GUTTER, 0, L.GUTTER, L.H))

    def corner_marks(self):
        """Own corner per player; greyed out once that player has played."""
        marks = {}
        for owner, (cx, cy) in OWNER_CORNER.items():
            if owner == 0 or self.game.owner_key.get(owner):
                if self.game.placed[owner] == 0:
                    marks[(cx, cy)] = (COLORS[self.game.colors[owner]], owner)
        return marks

    def draw_board(self):
        L = self.L
        surf = self.screen
        board = self.game.board
        sx, sy = L.BO
        c = L.CELL
        pad = max(2, int(round(4 * L.scale)))
        surf.fill(BOARD_BG, pygame.Rect(sx - pad, sy - pad, L.board_px + 2 * pad,
                                        L.board_px + 2 * pad))
        marks = self.corner_marks()
        for x in range(20):
            for y in range(20):
                owner = board.grid[x + y * 20]
                r = pygame.Rect(sx + x * c, sy + y * c, c, c)
                if owner >= 0:
                    col = COLORS[self.game.colors[owner]]
                    pygame.draw.rect(surf, col, r)
                    inset = max(1, int(round(1 * L.scale)))
                    pygame.draw.rect(surf, shade(col, 0.75),
                                     pygame.Rect(r.x + inset, r.y + inset,
                                                 c - 2 * inset, c - 2 * inset), 1)
                else:
                    pygame.draw.rect(surf, (30, 36, 46) if (x + y) % 2 == 0 else (24, 29, 38), r)
                    pygame.draw.rect(surf, (44, 51, 63), r, 1)
                    if (x, y) in marks:
                        pygame.draw.rect(
                            surf, marks[(x, y)][0],
                            r.inflate(-max(1, c // 4), -max(1, c // 4)), 2)
        if self.anim:
            name, oi, x, y = self.anim["name"], self.anim["oi"], self.anim["x"], self.anim["y"]
            cells = MASTER[name]["orientations"][oi]
            t = min(1.0, (time.time() - self.anim["t"]) / 0.3)
            col = COLORS[self.game.colors[self.anim.get("owner", 0)]]
            for dx, dy in cells:
                r = pygame.Rect(sx + (x + dx) * c, sy + (y + dy) * c, c, c)
                self.draw_alpha_rect(r, col, int(255 * t))
        d = self.drag
        if d and d["x"] is not None:
            cells = MASTER[d["name"]]["orientations"][d["oi"]]
            col = (70, 220, 130) if d["valid"] else (220, 70, 90)
            for dx, dy in cells:
                r = pygame.Rect(sx + (d["x"] + dx) * c, sy + (d["y"] + dy) * c, c, c)
                self.draw_alpha_rect(r, col, 170 if d["valid"] else 110)
            w = (max(dx for dx, _ in cells) + 1) * c
            h = (max(dy for _, dy in cells) + 1) * c
            box = pygame.Rect(sx + d["x"] * c, sy + d["y"] * c, w, h)
            lw = max(2, int(round(2 * L.scale)))
            pygame.draw.rect(surf, (255, 255, 255) if d["valid"] else (255, 160, 160), box, lw)
        pygame.draw.rect(surf, (60, 66, 78),
                         pygame.Rect(sx - pad, sy - pad, L.board_px + 2 * pad,
                                     L.board_px + 2 * pad),
                         max(1, int(round(2 * L.scale))))

    def draw_hand(self):
        L = self.L
        surf = self.screen
        s = L.scale
        # The rack is a right-hand column for the full window height, so the
        # hand and the board are read in one glance instead of after a long
        # downward move.
        rack = pygame.Rect(L.HAND_X0 - L.MARGIN, 0,
                           L.W - L.HAND_X0 + L.MARGIN, L.H)
        pygame.draw.rect(surf, BAND_BG, rack)
        pygame.draw.rect(surf, DIVIDER,
                         pygame.Rect(rack.x, 0, max(1, int(round(2 * s))), L.H))
        slots, cards = self.hand_layout()
        # One captioned card per group: light fill, dark frame, caption strip
        # on top naming the group. The card owns the spacing, so the pieces
        # themselves can sit on a plain background.
        for rect, key in cards:
            radius = max(4, int(round(8 * s)))
            pygame.draw.rect(surf, (31, 37, 48), rect, border_radius=radius)
            pygame.draw.rect(surf, (18, 22, 30), rect,
                             max(1, int(round(2 * s))), border_radius=radius)
            strip = pygame.Rect(rect.x + max(1, int(round(2 * s))), rect.y,
                                rect.w - 2 * max(1, int(round(2 * s))),
                                L.HAND_LABEL_H)
            self.label(key, strip.centerx, strip.centery,
                       max(10, int(round(12 * s))), (168, 176, 190), center=True)
        pcolor = COLORS[self.game.colors[0]]
        c = L.HAND_CELL
        held = self.drag["name"] if self.drag else None
        # Geometry is fixed for the whole rack; only the pieces still in hand
        # are drawn, so a played piece leaves an empty slot instead of pulling
        # the rest of the rack up behind it.
        in_hand = set(self.game.hands[0].names)
        for name, r in slots.items():
            if name not in in_hand:
                continue
            dragging = name == held
            # Thumbnails always use the flattest orientation: a rotated piece
            # can be taller than its group's row (I5 upright is 1x5 in a 3-row),
            # and the board preview is the authoritative shape view anyway.
            for dx, dy in MASTER[name]["orientations"][HAND_ITEMS[name]]:
                pygame.draw.rect(surf, pcolor,
                                 pygame.Rect(r.x + dx * c, r.y + dy * c,
                                             max(1, c - 1), max(1, c - 1)))
            if dragging:
                grow = int(round(4 * s))
                pygame.draw.rect(surf, (235, 235, 235),
                                 r.inflate(grow, grow), 2)
                if self.drag["rot"]:
                    self.draw_rotate_badge(r, self.drag["rot"], rack.y)

    def draw_rotate_badge(self, r, rot, rack_top):
        """Rotation counter for the held piece, pinned inside the hand rack."""
        text = I["rotate_badge"].format(rot)
        img = self.font_of(11).render(text, True, (245, 247, 250))
        w = max(img.get_width() + 6, 14)
        h = max(img.get_height() + 2, 12)
        x = r.right - w
        y = max(rack_top + 1, r.top - h + 2)
        plate = pygame.Surface((w, h), pygame.SRCALPHA)
        plate.fill((20, 24, 32, 230))
        surf = self.screen
        surf.blit(plate, (x, y))
        surf.blit(img, img.get_rect(center=(x + w // 2, y + h // 2)))

    def draw_setup_color(self):
        s = self.L.scale
        surf = self.screen
        surf.fill((12, 15, 20))
        self.label(I["title"], self.L.W // 2, int(round(30 * s)), 30,
                   (235, 235, 235), center=True)
        self.label(I["choose_color"], self.L.W // 2, int(round(110 * s)), 22,
                   (200, 200, 210), center=True)
        bw, bh = int(round(150 * s)), int(round(220 * s))
        gap = int(round(20 * s))
        x0 = (self.L.W - (4 * bw + 3 * gap)) // 2
        y0 = int(round(220 * s))
        for i, col in enumerate(["blue", "green", "red", "yellow"]):
            r = pygame.Rect(x0 + i * (bw + gap), y0, bw, bh)
            surf.fill(COLORS[col], r)
            pygame.draw.rect(surf, (110, 118, 132), r, 2)
            self.label(COLOR_LABELS[col], r.centerx, r.bottom + int(round(12 * s)),
                       20, (235, 235, 235), center=True)
        self.label(I["scale_hint"].format(int(self.scale * s * 100)),
                   self.L.W // 2, y0 + bh + int(round(48 * s)), 15,
                   (130, 136, 146), center=True)

    def draw_setup_info(self):
        s = self.L.scale
        surf = self.screen
        surf.fill((12, 15, 20))
        panel = self.setup_panel_rect()
        cx, sw = panel.x, int(round(22 * s))
        self.label(I["title"], cx, int(round(26 * s)), 30, (235, 235, 235))
        pcol = COLORS[self.game.colors[0]]
        pygame.draw.rect(surf, pcol, pygame.Rect(cx, int(round(84 * s)), sw, sw))
        self.label("%s（%s）：%s" % (I["player"], I["corner_br"],
                                    COLOR_LABELS[self.game.colors[0]]),
                   cx + sw + int(round(12 * s)), int(round(84 * s)), 22, pcol)
        y = int(round(132 * s))
        for o in range(1, 4):
            key = self.game.owner_key[o]
            color = COLORS[self.game.colors[o]]
            pygame.draw.rect(surf, color, pygame.Rect(cx, y, sw, sw))
            self.label("%s（%s）：%s" % (PERSONA_ZH[key], corner_zh(o),
                                        COLOR_LABELS[self.game.colors[o]]),
                       cx + sw + int(round(12 * s)), y, 22, color)
            self.label(I[key + "_desc"], cx + sw + int(round(12 * s)),
                       y + int(round(30 * s)), 16, (165, 170, 180))
            y += int(round(74 * s))
        # The rotation is clockwise with a random starting point, so who moves
        # first differs every game; spelled out so the player does not assume
        # they always go first.
        order = " → ".join(
            (I["player"] if o == 0 else PERSONA_ZH[self.game.owner_key[o]])
            + (I["turn_first"] if i == 0 else "")
            for i, o in enumerate(self.game.turn_order))
        self.label(I["turn_order_fmt"].format(order), cx, y, 17, (185, 190, 200))
        for i, line in enumerate(I["turn_note"]):
            self.label(line, cx, y + int(round((30 + 20 * i) * s)), 16,
                       (150, 155, 165))
        for bid, r in self.setup_button_rects().items():
            self.draw_button(bid, r, size=20)

    def draw_results(self):
        """Draw the finished board, then a results panel in the sidebar.

        The board is never covered: the whole point of this screen is that the
        player can study how the game ended before choosing to start another.
        """
        self.draw_all()
        L = self.L
        s = L.scale
        surf = self.screen
        top = int(round(10 * s))
        bottom = L.BO[1] + L.board_px + L.MARGIN
        panel = pygame.Rect(0, top, L.PANEL_W, bottom - top)
        surf.fill((12, 15, 20), panel)
        pygame.draw.rect(surf, DIVIDER, panel, max(1, int(round(2 * s))))

        x = L.MARGIN
        inner = L.PANEL_W - 2 * x
        y = top + int(round(14 * s))
        y = self._row(I["game_over"], x, y, 24, (240, 240, 245), inner)
        win = self.game.winner()
        if win == "player":
            msg, wcol = I["win_player"], (120, 220, 140)
        elif win == "draw":
            msg, wcol = I["draw"], (235, 220, 120)
        else:
            o = int(win)
            msg = I["win_ai"].format(PERSONA_ZH[self.game.owner_key[o]])
            wcol = COLORS[self.game.colors[o]]
        y = self._row(msg, x, y, 20, wcol, inner)
        y = self._row(I["less_is_better"], x, y, 12, (120, 126, 136), inner)

        y = self._row(I["this_game"], x, y + int(round(6 * s)), 15, (200, 205, 215),
                      inner)
        ranks = {key: (rank, points) for key, rank, points in self.records.last}
        # contestant_standings() already comes best-first from game.standings()
        for key, rem in self.contestant_standings():
            col = self.contestant_color(key)
            rank, points = ranks.get(key, (0, 0))
            self.label(I["left_won_fmt"].format(rank, rem),
                       x + int(round(14 * s)), y, 14, col)
            self.label(I["points_fmt"].format(points), x + inner, y, 14, col,
                       right=True)
            sw = int(round(10 * s))
            pygame.draw.rect(surf, col, pygame.Rect(x, y + int(round(3 * s)), sw, sw))
            y += int(round(24 * s))

        y = self._row(I["leaderboard"], x, y + int(round(10 * s)), 15,
                      (200, 205, 215), inner)
        for key, games, avg_points, avg_remaining in self.records.rows():
            col = self.contestant_color(key)
            self.label(self.contestant_name(key), x + int(round(14 * s)), y, 14, col)
            self.label(I["games_fmt"].format(games), x + inner, y, 12,
                       (130, 136, 146), right=True)
            y += int(round(17 * s))
            y = self._row(I["avg_fmt"].format(avg_points, avg_remaining),
                          x + int(round(14 * s)), y, 12, (165, 170, 180), inner)

        r = self.new_game_rect()
        pygame.draw.rect(surf, (70, 110, 140), r)
        pygame.draw.rect(surf, (120, 170, 210), r, 2)
        img = self.font_of(20).render(I["new_game"], True, (240, 240, 245))
        surf.blit(img, img.get_rect(center=r.center))

    def _row(self, text, x, y, size, color, inner):
        self.label(text, x, y, size, color)
        return y + int(round((size + 8) * self.L.scale))

    def ai_worker(self, owner, token):
        if self.ai_token != token:
            return
        mv = ai.choose_move(self.game.board, self.game.hands[owner].names, owner,
                            self.game.brains[owner], self.game.rng,
                            other_brains=self.game.brains,
                            must_cover=self.game.must_cover(owner),
                            reach=self.game.reach(owner),
                            other_must_cover={o: self.game.must_cover(o)
                                              for o in range(4)},
                            other_reach={o: self.game.reach(o) for o in range(4)})
        if self.ai_token == token:
            self.ai_move = mv
            self.ai_ready = True

    def draw_toasts(self):
        self.toasts = [t for t in self.toasts if time.time() < t[1]]
        for text, _ in self.toasts:
            img = self.font_of(18).render(text, True, (235, 235, 235))
            surf = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 170))
            surf.blit(img, (0, 0))
            self.screen.blit(surf, (self.L.W // 2 - img.get_size()[0] // 2,
                                     int(round(20 * self.L.scale))))
