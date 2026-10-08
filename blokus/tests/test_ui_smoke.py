"""Headless UI smoke tests: setup flow, lock/confirm, scaling, remaining cells."""
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai
from config import HAND_CELLS_TOTAL
from game import Game
from pieces import MASTER
from records import Records
from config import COLORS, I
import seats as seats_mod
import ui
from ui import (DEFAULT_HAND_ZOOM, HAND_ITEMS, HAND_ZOOMS, PERSONA_ZH, SCALES,
                Layout, UI, best_scale, load_cjk_font_path)


def make_ui(seed=7, scale=1.0, hand_zoom=2.0):
    pygame.init()
    u = UI(None, load_cjk_font_path(), scale=scale, hand_zoom=hand_zoom)
    u.game = Game(random.Random(seed))
    return u


def click(u, pos, button=1):
    u.tick([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)])


def key(u, k=pygame.K_SPACE):
    u.tick([pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode=" ")])


def move_mouse(u, pos):
    u.tick([pygame.event.Event(pygame.MOUSEMOTION, pos=pos, buttons=(0, 0, 0), rel=(0, 0))])


def wheel(u, dy=1):
    """One wheel notch: `dy > 0` is the wheel being flicked upwards.

    The direction is the one pygame documents, and stage 7 pins which way the
    list moves as a result - see `test_the_wheel_scrolls_the_list_down`.
    """
    u.tick([pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=dy)])


def menu_windows(u, kind="option"):
    """Every distinct scroll position of the open picker, in order.

    Scrolls one row at a time with the wheel until `menu_rects` stops handing
    back something new, which is `max_scroll` clamping itself, so a caller can
    walk the whole list without knowing the page size.
    """
    out, seen = [], None
    for _ in range(64):
        rects = u.menu_rects()
        if rects == seen:
            break
        out.append(rects)
        seen = rects
        wheel(u, -1)
    return out


def open_fake_menu(monkeypatch, rows=24, scale=1.0):
    """An option picker over `rows` made-up choices.

    Stage 7 has to prove scrolling with more rows than any real pool produces,
    and the real pool is deliberately not the subject: stage 6 changes it from
    fourteen rows to seventeen, and a test built on that number would be a
    test of the roster. The fake keys also let `seat_label` be stubbed, so
    nothing here reads `config.I` for names that do not exist.
    """
    keys = ["fake_%02d" % i for i in range(rows)]
    monkeypatch.setattr(seats_mod, "seat_menu_options", lambda: tuple(keys))
    monkeypatch.setattr(ui, "seat_label", lambda k: k)
    u = at_seat_screen(make_ui(scale=scale))
    click(u, u.seat_rects()["0:option"].center)
    assert u.seat_pick == (0, "option")
    return u, keys


def cell_pos(u, x, y):
    L = u.L
    return (L.BO[0] + x * L.CELL + L.CELL // 2, L.BO[1] + y * L.CELL + L.CELL // 2)


def wait_for_player_turn(u, limit=400):
    for _ in range(limit):
        u.tick(pygame.event.get())
        if u.state == "PLAYING" and u.game.current_owner() == 0:
            return True
        if u.ai_pending and not u.ai_ready:
            pygame.time.wait(10)
        else:
            pygame.time.wait(5)
    return False


def start_game(u):
    """The seat screen with its defaults, then straight into play."""
    u.tick([])
    assert u.state == "SETUP_SEATS"
    key(u)
    assert u.state == "PLAYING"
    return u


def at_seat_screen(u):
    """The seat screen, with the four seats still at their defaults."""
    u.tick([])
    assert u.state == "SETUP_SEATS"
    return u


def start_game_human_at(u, seat):
    """A game whose human plays `seat`, not necessarily seat 0.

    The seat screen lets any of the four be a person, so the tests that drive a
    human's moves have to be able to say which one.
    """
    at_seat_screen(u)
    u.seat_keys = ["chess"] * 4
    u.seat_keys[seat] = "human"
    u.seat_colours = ["blue", "green", "red", "yellow"]
    key(u)
    assert u.state == "PLAYING"
    return u


def brain_for(u, owner):
    """Any brain, used only to find a legal move for a test to play.

    A human seat has no brain at all now - that is what makes "is this seat a
    person" answerable - so the helper borrows one from a seat that does.
    """
    for o in range(4):
        if u.game.brains.get(o) is not None:
            return u.game.brains[o]
    raise AssertionError("no AI seat to borrow a brain from")


def best_player_move(u, seed=0):
    owner = u.game.current_owner()
    return ai.choose_move(u.game.board, u.game.hands[owner].names, owner,
                          brain_for(u, owner),
                          random.Random(seed), other_brains=None,
                          must_cover=u.game.must_cover(owner),
                          reach=u.game.reach(owner))


# ---------------------------------------------------------------- fonts

def test_font_path_has_cjk():
    path = load_cjk_font_path()
    assert path is not None, "no CJK font resolved on this system"


def test_font_sizes_are_distinct():
    u = make_ui()
    assert u.font_of(30).get_height() > u.font_of(16).get_height()


# ---------------------------------------------------------------- layout / scaling

def test_layout_grows_with_scale():
    small = Layout(1.0)
    big = Layout(1.5)
    assert big.CELL > small.CELL
    assert big.board_px > small.board_px
    assert big.W > small.W and big.H > small.H


def test_window_is_locked_to_16_9_at_every_scale_and_zoom():
    """The frame ratio is a promise: nothing may make it drift."""
    for s in SCALES:
        for z in HAND_ZOOMS:
            L = Layout(s, z)
            assert abs(L.W / L.H - 16.0 / 9.0) < 0.002, (s, z, L.W, L.H)


def test_layout_keeps_everything_inside_window():
    for s in SCALES:
        for z in HAND_ZOOMS:
            L = Layout(s, z)
            assert L.BO[0] + L.board_px <= L.HAND_X0, (s, z, "board overlaps rack")
            assert L.BTN_Y + L.BTN_H <= L.H, (s, z, "buttons below window")
            slots, _cards, h = L.layout_hand(sorted(MASTER))
            assert h == L.hand_h
            for name, r in slots.items():
                assert r.left >= L.HAND_X0 - 1, (s, z, name, r)
                assert r.right <= L.W - L.MARGIN + 1, (s, z, name, r)
                assert r.top >= L.HAND_Y0, (s, z, name, r)
                assert r.bottom <= L.H, (s, z, name, r)
            # The rack is a right-hand column beside the board, so the whole
            # hand has to fit the window height.
            assert L.hand_h <= L.H, (s, z, L.hand_h, L.H)


def test_three_columns_share_one_margin():
    """Panel | board | hand rack, with the same gutter between all of them."""
    for s in SCALES:
        L = Layout(s, 2.0)
        assert L.BO[0] - L.PANEL_W == L.MARGIN, (s, L.BO[0], L.PANEL_W)
        assert L.HAND_X0 - (L.BO[0] + L.board_px) == L.MARGIN, s
        assert L.W - L.MARGIN - (L.HAND_X0 + L.HAND_W) == 0, s
        # Vertically: MARGIN above the board, MARGIN to the buttons. The board
        # is an exact multiple of its cell size, so on small scales integer
        # truncation leaves the bottom margin a little larger, never smaller.
        assert L.BO[1] == L.MARGIN, (s, L.BO[1])
        assert L.BTN_Y - (L.BO[1] + L.board_px) == L.MARGIN, s
        assert L.H - (L.BTN_Y + L.BTN_H) >= L.MARGIN, s


def test_buttons_sit_under_the_board_with_room_between_them():
    """Confirm has to be reachable right under the cell you just picked."""
    L = Layout(1.0, 2.0)
    u = make_ui(scale=1.0, hand_zoom=2.0)
    rects = u.button_rects()
    left, right = L.BO[0], L.BO[0] + L.board_px
    for r in rects.values():
        assert r.top == L.BTN_Y, r
        assert r.left >= left and r.right <= right, (r, left, right)
        assert r.bottom <= L.H, r
    # The strip is centred under the board.
    ordered = sorted(rects.values(), key=lambda r: r.left)
    board_mid = (left + right) // 2
    strip_mid = (ordered[0].left + ordered[-1].right) // 2
    assert abs(board_mid - strip_mid) <= 2, (board_mid, strip_mid)
    for a, b in zip(ordered, ordered[1:]):
        gap = b.left - a.right
        assert gap >= L.BTN_GAP, (a, b, gap)


def test_hand_has_one_captioned_card_per_piece_size():
    """Every size gets its own frame and its own "N cells" caption, 1..5."""
    L = Layout(1.0, 2.0)
    slots, cards, _h = L.layout_hand(sorted(MASTER))
    assert [k for _r, k in cards] == [I["group_fmt"].format(n) for n in (1, 2, 3, 4, 5)]
    # Cards are stacked 1..5 from the top, and each frame encloses its own
    # pieces plus the caption strip.
    tops = [r.top for r, _k in cards]
    assert tops == sorted(tops)
    # Every frame encloses at least one piece, and encloses only its own size.
    for i, (rect, _k) in enumerate(cards):
        inside = [n for n, r in slots.items() if rect.contains(r)]
        assert inside, rect
        assert {MASTER[n]["size"] for n in inside} == {i + 1}, (i, inside)
    # Within a group the thumbnails read top to bottom, left to right, and the
    # group stays inside its own frame. The exact row count follows from how
    # the group wraps in the narrow rack, so it is not pinned here.
    for size in (1, 2, 3, 4, 5):
        rs = [r for n, r in slots.items() if MASTER[n]["size"] == size]
        order = sorted(rs, key=lambda r: (r.top, r.left))
        assert order == sorted(rs, key=lambda r: (round(r.top, 3), r.left)), size
        card = cards[size - 1][0]
        for r in rs:
            assert card.contains(r), (size, r, tuple(card))


def test_hand_rows_are_centred_in_the_rack():
    """A short group must not hug one side of the rack while the other empties."""
    for s in (0.8, 1.0, 1.4):
        L = Layout(s, 2.0)
        slots, cards, _h = L.layout_hand(sorted(MASTER))
        mid = L.HAND_X0 + L.HAND_W // 2
        lines = {}
        for rect, _k in cards:
            assert any(rect.contains(r) for r in slots.values()), rect
            lines.setdefault(rect.top, []).append(rect)
        assert lines, "no cards laid out"
        # Cards flow left to right and wrap, so at least one line carries more
        # than one card - that is what stops three tiny groups from spending
        # three screen-heights between them.
        assert max(len(rs) for rs in lines.values()) > 1, \
            "cards did not sit side by side"
        for top, rs in lines.items():
            rs.sort(key=lambda r: r.left)
            # The frame hugs its rows, so its two side gutters are equal...
            left_gutter = rs[0].left - L.HAND_X0
            right_gutter = (L.HAND_X0 + L.HAND_W) - rs[-1].right
            assert abs(left_gutter - right_gutter) <= 2, (s, top, rs)
            # ...and the line as a whole is centred on the column.
            span = (rs[0].left + rs[-1].right) // 2
            assert abs(span - mid) <= 2, (s, top, span, mid)
            # Cards in a line share a top edge, and never touch each other.
            for a, b in zip(rs, rs[1:]):
                assert b.left - a.right >= 1, (s, a, b)


def test_hand_fills_the_rack_at_the_default_zoom():
    """The rack is sized to its column, not left with a void under the cards."""
    for s in SCALES:
        L = Layout(s, DEFAULT_HAND_ZOOM)
        avail = L.H - 2 * L.MARGIN
        # The top of the zoom range fills the column; the board next to it runs
        # the full window height, so a half-empty rack read as lopsided.
        assert L.hand_h >= avail * 0.85, (s, L.hand_h, avail)
        # Smaller steps trade size for a calmer rack, and still fill a decent
        # share rather than collapsing to thumbnails.
        small = Layout(s, HAND_ZOOMS[0])
        assert small.HAND_CELL < L.HAND_CELL, (s, small.HAND_CELL, L.HAND_CELL)
        assert small.hand_h >= avail * 0.3, (s, small.hand_h, avail)
    # The rack is a narrow column, so a group wraps into as many rows as it
    # needs; its exact height follows from that wrapping, not a fixed formula.
    for s in SCALES:
        for z in HAND_ZOOMS:
            L = Layout(s, z)
            assert L.hand_h <= L.H, (s, z, L.hand_h, L.H)
            assert L.HAND_CELL * 5 <= L.HAND_W - 2 * L.HAND_PAD, (s, z)


def test_no_two_hand_pieces_ever_overlap():
    """Pieces sit in their own slots; the rack must never stack them."""
    for s in SCALES:
        for z in HAND_ZOOMS:
            L = Layout(s, z)
            slots, cards, _h = L.layout_hand(sorted(MASTER))
            names = sorted(slots)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    assert not slots[a].colliderect(slots[b]), (s, z, a, b)
            # ...and every piece is inside one of the cards, never straddling
            # a border or floating loose.
            for n, r in slots.items():
                assert any(c.contains(r) for c, _k in cards), (s, z, n, r)
            for i, (a, _ka) in enumerate(cards):
                for b, _kb in cards[i + 1:]:
                    assert not a.colliderect(b), (s, z, tuple(a), tuple(b))


def test_playing_a_piece_never_moves_the_others():
    """Rack geometry is fixed; a used piece leaves an empty slot behind."""
    u = start_game(make_ui())
    before = {n: u.hand_slot(n) for n in u.hand_order()}
    assert len(before) == 21
    # play a few pieces of the hand directly
    for name in ("T5", "V5", "L5", "I5", "O4"):
        if name in u.game.hands[0].names:
            u.game.hands[0].names.remove(name)
    for n in u.hand_order():
        assert u.hand_slot(n) == before[n], (n, u.hand_slot(n), before[n])
    # a played piece has no live slot at all
    for name in ("T5", "V5", "L5", "I5", "O4"):
        assert u.hand_slot(name) is None, name


def test_hand_is_never_clipped_at_any_zoom():
    """Whatever zoom is asked for, the whole hand stays on screen."""
    for s in SCALES:
        for z in HAND_ZOOMS:
            L = Layout(s, z)
            slots, _cards, h = L.layout_hand(sorted(MASTER))
            assert h <= L.H, (s, z, h, L.H)
            for n, r in slots.items():
                assert r.bottom <= L.H, (s, z, n, r)
                assert r.right <= L.W, (s, z, n, r)


def test_five_cell_group_wraps_instead_of_being_clipped():
    """The 5-cell group is 41 cells wide, so it must wrap in the narrow rack."""
    for z in HAND_ZOOMS:
        wide = Layout(1.0, z)
        slots, _cards, _h = wide.layout_hand(sorted(MASTER))
        rows = [r for n, r in slots.items() if MASTER[n]["size"] == 5]
        assert len({r.top for r in rows}) > 1, \
            "5-cell group did not wrap at zoom %s" % z
        for r in rows:
            assert r.right <= wide.W - wide.MARGIN + 1
            assert r.left >= wide.HAND_X0 - 1


def test_best_scale_respects_desktop():
    desktop = (1920, 1080)
    s = best_scale(desktop, 2.0)
    assert s in SCALES
    assert Layout(s, 2.0).fits(desktop)
    # a small screen can never pick a bigger scale than a large one
    assert best_scale((800, 600), 2.0) <= s


def test_bigger_hand_never_yields_a_bigger_board():
    """The documented trade-off, pinned so the two knobs stay independent."""
    desktop = (1920, 1080)
    small = Layout(best_scale(desktop, HAND_ZOOMS[0]), HAND_ZOOMS[0])
    big = Layout(best_scale(desktop, HAND_ZOOMS[-1]), HAND_ZOOMS[-1])
    assert big.board_px <= small.board_px
    assert big.hand_h > small.hand_h


def test_scale_change_resizes_and_stays_valid(monkeypatch):
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = make_ui(scale=1.0)
    start_cell = u.L.CELL
    u.bump_scale(1)
    assert u.L.CELL > start_cell, "board did not grow"
    big = u.L.CELL
    u.bump_scale(-1)
    assert u.L.CELL < big, "board did not shrink"
    # everything must stay inside the window at any scale
    for _ in range(len(SCALES)):
        u.bump_scale(1)
        assert u.L.BO[0] + u.L.board_px <= u.L.W
        assert u.L.BTN_Y + u.L.BTN_H <= u.L.H


def test_hand_zoom_is_independent_of_board_scale(monkeypatch):
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = make_ui(scale=1.4, hand_zoom=1.4)
    cell_before = u.L.CELL
    small_before = u.L.HAND_CELL
    u.bump_hand_zoom(1)
    assert u.L.HAND_CELL > small_before, "raising the zoom must enlarge pieces"
    assert u.L.CELL == cell_before, "hand zoom must not resize the board"
    u.bump_hand_zoom(-1)
    assert u.L.HAND_CELL == small_before
    # clamped at both ends, and never above the top of the allowed range
    for _ in range(6):
        u.bump_hand_zoom(-1)
    assert u.L.hand_zoom == HAND_ZOOMS[0]
    for _ in range(6):
        u.bump_hand_zoom(1)
    assert HAND_ZOOMS[0] <= u.L.hand_zoom <= HAND_ZOOMS[-1]
    assert u.L.layout_hand(sorted(MASTER))[2] <= u.L.H


def test_bracket_keys_change_hand_zoom(monkeypatch):
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = make_ui(scale=1.0, hand_zoom=1.4)
    key(u, pygame.K_RIGHTBRACKET)
    assert u.hand_zoom == HAND_ZOOMS[2], u.hand_zoom
    key(u, pygame.K_LEFTBRACKET)
    assert u.hand_zoom == 1.4


def test_fit_scale_shrinks_hand_only_when_board_cannot_fit(monkeypatch):
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    monkeypatch.setattr(ui_mod, "auto_fit_area", lambda: (1920, 1080))
    u = make_ui(scale=1.0, hand_zoom=2.0)
    u.fit_scale()
    assert u.hand_zoom == 2.0, "hand should be left alone on a roomy screen"
    assert Layout(u.scale, u.hand_zoom).fits((1920, 1080))


def test_scale_is_clamped_to_desktop(monkeypatch):
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1000, 800))
    u = make_ui(scale=1.0)
    u.bump_scale(len(SCALES))
    assert Layout(u.scale, u.hand_zoom).fits((1000, 800))
    assert u.L.W <= 1000 and u.L.H <= 800


def test_hand_slots_are_inside_the_band_at_every_combination():
    u = start_game(make_ui())
    for s in SCALES:
        for z in HAND_ZOOMS:
            u.set_scale(s)
            u.set_hand_zoom(z)
            names = u.hand_order()
            assert len(names) == 21
            seen = []
            for name in names:
                r = u.hand_slot(name)
                assert r is not None, (s, z, name)
                assert r.left >= 0 and r.top >= 0, (s, z, name, r)
                assert r.right <= u.L.W, (s, z, name, r)
                # the hand band is now the bottom strip, below the buttons
                assert r.bottom <= u.L.H, (s, z, name, r)
                seen.append(r)
            for _bid, b in u.button_rects().items():
                assert b.bottom <= u.L.H, (s, z, b)


def test_column_dividers_run_the_full_window_height(monkeypatch):
    """Both column rules reach the bottom edge, not just the board's height."""
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = start_game(make_ui())
    for s in SCALES:
        u.set_scale(s)
        L = u.L
        drawn = []
        real = pygame.draw.rect

        def spy(surf, color, rect, *a, _d=drawn, _r=real, **kw):
            _d.append((color, pygame.Rect(rect)))
            return _r(surf, color, rect, *a, **kw)

        pygame.draw.rect = spy
        try:
            u.draw_left()
        finally:
            pygame.draw.rect = real
        for color, r in drawn:
            if r.width <= L.GUTTER and r.height >= L.H - 1:
                assert r.bottom == L.H, (s, r, L.H)
        # the rack rule is drawn by draw_hand, and must match
        drawn.clear()
        pygame.draw.rect = spy
        try:
            u.draw_hand()
        finally:
            pygame.draw.rect = real
        for _color, r in drawn:
            if r.height >= L.H - 1:
                assert r.bottom == L.H, (s, r, L.H)
                assert r.x == L.HAND_X0 - L.MARGIN, (s, r)


def test_sidebar_text_stays_inside_the_panel(monkeypatch):
    """The sidebar is only as wide as the board panel; a long hint used to run
    out over the first board column at small scales."""
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = start_game(make_ui())
    for s in SCALES:
        u.set_scale(s)
        rects = []
        real = u.label

        def spy(text, x, y, size, color, bg=None, center=False, right=False,
                _r=rects):
            _r.append((text, real(text, x, y, size, color, bg, center, right)))
            return _r[-1][1]

        u.label = spy
        u.draw_left()
        u.label = real
        for text, r in rects:
            assert r.left >= 0, (s, text, r)
            assert r.right <= u.L.PANEL_W, (s, text, r)


def test_sidebar_drops_secondary_rows_when_short(monkeypatch):
    """Optional hint rows disappear rather than spilling past the band."""
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = start_game(make_ui())
    for s in SCALES:
        u.set_scale(s)
        rects = []
        real = u.label

        def spy(text, x, y, size, color, bg=None, center=False, right=False,
                _r=rects):
            _r.append((text, real(text, x, y, size, color, bg, center, right)))
            return _r[-1][1]

        u.label = spy
        u.draw_left()
        u.label = real
        # The sidebar stops at the foot of the board column, so nothing may be
        # drawn over the button strip or past the panel into the rack.
        side_bottom = u.L.BO[1] + u.L.board_px + u.L.MARGIN
        for text, r in rects:
            assert r.bottom <= side_bottom, (s, text, r)
            assert r.right <= u.L.HAND_X0, (s, text, r)


def test_hand_order_is_by_size_then_name():
    u = start_game(make_ui())
    order = u.hand_order()
    keys = [(MASTER[n]["size"], n) for n in order]
    assert keys == sorted(keys)
    assert u.hand_index("I1") == keys.index((1, "I1"))
    assert u.hand_index("not-a-piece") == -1


def test_hand_reflows_without_leaving_holes():
    u = start_game(make_ui())
    before = [u.hand_slot(n) for n in u.hand_order()]
    u.game.hands[0].names.remove("L5")
    after = [u.hand_slot(n) for n in u.hand_order()]
    assert len(after) == len(before) - 1
    # the remaining pieces are still packed left to right inside the window
    for r in after:
        assert r.right <= u.L.W
        assert r.bottom <= u.L.H


# ---------------------------------------------------------------- setup flow

def test_the_seat_screen_starts_a_game_without_picking_anything():
    """Pressing start with the defaults gives the game this screen replaced:
    one person against three chess players, every colour dealt at random."""
    u = make_ui()
    u.tick([])
    assert u.state == "SETUP_SEATS"
    assert u.seat_keys == ["human", "chess", "chess", "chess"]
    assert u.seat_colours == [seats_mod.RANDOM] * 4
    click(u, u.seat_rects()["start_game"].center)
    assert u.state == "PLAYING"
    assert len(set(u.game.colors.values())) == 4
    assert u.game.seat_kinds[0] == seats_mod.KIND_HUMAN
    assert [u.game.seat_kinds[o] for o in (1, 2, 3)] == [seats_mod.KIND_AI] * 3
    assert len(u.game.hands[0].names) == 21


def test_every_seat_option_renders_on_the_seat_screen():
    """The screen looks each option up by key, so a new option with a missing
    name or description only blows up at runtime. Checked for every one of
    them, with the size taken from the pool rather than typed (stage 8)."""
    options = seats_mod.seat_menu_options()
    assert len(options) == len(seats_mod.seat_options(True)) + 1
    for opt in options:
        u = make_ui()
        at_seat_screen(u)
        for seat in range(4):
            u.seat_keys[seat] = opt
            u.draw_setup_seats()
        assert ui.seat_label(opt)
        if seats_mod.kind_of(opt) in (seats_mod.KIND_AI, seats_mod.KIND_IMITATION,
                                      seats_mod.KIND_RL):
            assert ui.seat_desc(opt), opt
    for opt in options:
        if seats_mod.kind_of(opt) == seats_mod.KIND_AI:
            assert I[opt] and I[opt + "_desc"]


def test_an_option_may_be_repeated_and_a_colour_may_not():
    """The two rules the screen exists to express."""
    u = make_ui()
    at_seat_screen(u)
    u.seat_keys = ["wolf"] * 4
    u.seat_colours = ["blue", "green", "red", "yellow"]
    click(u, u.seat_rects()["start_game"].center)
    assert u.state == "PLAYING"
    assert [u.game.owner_key[o] for o in range(4)] == ["wolf"] * 4
    assert sorted(u.game.colors.values()) == ["blue", "green", "red", "yellow"]

    # four humans, and two recording humans, are both legal
    u = make_ui()
    at_seat_screen(u)
    u.seat_keys = ["human"] * 4
    u.seat_colours = [seats_mod.RANDOM] * 4
    click(u, u.seat_rects()["start_game"].center)
    assert u.state == "PLAYING"
    assert u.game.humans() == [0, 1, 2, 3]

    u = make_ui()
    at_seat_screen(u)
    u.seat_keys = ["human_log", "chess", "human_log", "chess"]
    u.seat_colours = [seats_mod.RANDOM] * 4
    click(u, u.seat_rects()["start_game"].center)
    assert u.state == "PLAYING"
    assert u.game.recording_seats() == [0, 2]


def test_a_colour_another_seat_took_is_not_on_offer():
    """Options stay in the menu however many seats take them; colours do not."""
    u = make_ui()
    at_seat_screen(u)
    u.seat_colours = ["blue", "green", "red", seats_mod.RANDOM]
    u.seat_pick = (3, "colour")
    keys = {k.split(":", 1)[1] for k in u.menu_rects()
            if k.startswith("colour:")}
    assert keys == {"random", "yellow"}, keys
    # and the option list is untouched by repetition
    u.seat_keys = ["fox"] * 4
    u.seat_pick = (3, "option")
    # Stage 7 turned the option list into a window, so "untouched" can no
    # longer mean "all of it in the first window" - that was the old assertion.
    # What it says now: walking every scroll position yields exactly the
    # options, none missing and none invented, whatever the seats are set to.
    offered = set()
    for rects in menu_windows(u):
        offered |= {k.split(":", 1)[1] for k in rects
                    if k.startswith("option:")}
    assert offered == set(seats_mod.seat_menu_options())


def test_picking_from_the_menu_assigns_to_that_seat_only():
    u = make_ui()
    at_seat_screen(u)
    click(u, u.seat_rects()["2:option"].center)
    assert u.seat_pick == (2, "option")
    click(u, u.menu_rects()["option:wolf"].center)
    assert u.seat_pick is None
    assert u.seat_keys == ["human", "chess", "wolf", "chess"]
    assert u.state == "SETUP_SEATS"


def test_setup_starts_only_from_the_start_button():
    """Clicking anywhere else does not start; only the start button (or Enter)."""
    u = at_seat_screen(make_ui())
    rects = u.seat_rects()
    # Points that are genuinely on nothing. The seat screen is mostly buttons,
    # so these are the four window corners, and each is checked to be empty so
    # a future layout change cannot quietly turn this into a different test.
    empty = ((5, 5), (u.L.W - 5, 5), (5, u.L.H - 5), (u.L.W - 5, u.L.H - 5))
    for pos in empty:
        assert not any(r.collidepoint(pos) for r in rects.values()), pos
        click(u, pos)
        assert u.state == "SETUP_SEATS", pos
        assert u.seat_pick is None, pos
    click(u, rects["start_game"].center)
    assert u.state == "PLAYING"
    u2 = at_seat_screen(make_ui())
    key(u2, pygame.K_RETURN)
    assert u2.state == "PLAYING"


def test_the_imitation_mode_button_toggles():
    u = at_seat_screen(make_ui())
    assert u.imitation_mode == "argmax"
    click(u, u.seat_rects()["imitation_mode"].center)
    assert u.imitation_mode == "softmax"
    click(u, u.seat_rects()["imitation_mode"].center)
    assert u.imitation_mode == "argmax"


def test_seat_buttons_fit_on_screen_and_do_not_overlap():
    u = at_seat_screen(make_ui())
    rects = u.seat_rects()
    panel = u.setup_panel_rect()
    for r in rects.values():
        assert r.bottom <= u.L.H, r
        assert panel.x <= r.x and r.right <= panel.right, r
    named = list(rects.items())
    for i, (_ka, a) in enumerate(named):
        for _kb, b in named[i + 1:]:
            assert not a.colliderect(b), (_ka, _kb)


# ---------------------------------------------------------------- placement flow

def test_hover_then_click_locks_and_confirm_works_from_button():
    """The regression the user hit: confirm must stay clickable after locking."""
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    mv = best_player_move(u)
    name, oi, x, y = mv
    click(u, u.hand_slot(name).center)
    assert u.drag is not None and u.drag["name"] == name
    u.drag["oi"] = oi
    assert not u.drag["locked"]

    # hover: live preview follows the cursor
    move_mouse(u, cell_pos(u, x, y))
    assert (u.drag["x"], u.drag["y"]) == (x, y)
    assert u.drag["valid"]

    # click the board to lock it in place
    click(u, cell_pos(u, x, y))
    assert u.drag["locked"]

    # now walk the cursor all the way over to the confirm button
    move_mouse(u, u.button_rects()["confirm"].center)
    assert u.drag["x"] == x and u.drag["y"] == y, "lock lost while leaving the board"
    assert u.drag["valid"], "confirm disabled after leaving the board"

    before = len(u.game.hands[0].names)
    click(u, u.button_rects()["confirm"].center)
    assert u.drag is None
    assert len(u.game.hands[0].names) == before - 1


def test_locked_piece_does_not_slide_with_cursor():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    mv = best_player_move(u)
    name, oi, x, y = mv
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    assert u.drag["locked"]
    move_mouse(u, cell_pos(u, 2, 17))
    assert (u.drag["x"], u.drag["y"]) == (x, y), "locked piece drifted"
    # clicking the board again re-targets
    click(u, cell_pos(u, 2, 17))
    assert (u.drag["x"], u.drag["y"]) == (2, 17)
    assert not u.drag["locked"] or u.drag["locked"]  # re-locked at the new spot


def test_hover_outside_board_keeps_last_preview():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    mv = best_player_move(u)
    name, oi, x, y = mv
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    move_mouse(u, (u.L.W - 5, u.L.BTN_Y + 5))
    assert u.drag["x"] == x and u.drag["y"] == y
    assert u.drag["valid"]


def test_enter_key_confirms():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    mv = best_player_move(u)
    name, oi, x, y = mv
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    before = len(u.game.hands[0].names)
    key(u, pygame.K_RETURN)
    assert len(u.game.hands[0].names) == before - 1


def test_rotate_key_cycles_orientation():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    name = "L4"
    click(u, u.hand_slot(name).center)
    o0 = u.drag["oi"]
    key(u, pygame.K_x)
    assert u.drag["oi"] != o0


def test_right_click_rotates_not_cancels():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    # L4 has several orientations; I1 would make the rotation a no-op
    click(u, u.hand_slot("L4").center)
    assert u.drag is not None
    o0 = u.drag["oi"]
    click(u, (5, 5), button=3)
    assert u.drag is not None, "right click must not cancel the piece"
    assert u.drag["oi"] != o0, "right click must rotate"
    key(u, pygame.K_c)
    assert u.drag is None


def find_opening(u, name):
    """An (orientation, x, y) where `name` legally covers owner's own corner."""
    owner = u.game.current_owner()
    cx, cy = u.game.must_cover(owner)
    for oi, cells in enumerate(MASTER[name]["orientations"]):
        for y in range(20):
            for x in range(20):
                if u.game.can_act(name, oi, x, y, owner):
                    return oi, x, y
    raise AssertionError("no legal opening with " + name)


def test_opening_piece_must_cover_own_corner():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    assert u.game.must_cover(0) == (19, 19)
    name = "L5"
    oi, x, y = find_opening(u, name)
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    # somewhere legal-looking but off the corner -> must be refused
    move_mouse(u, cell_pos(u, 0, 0))
    assert (u.drag["x"], u.drag["y"]) == (0, 0)
    assert not u.drag["valid"], "off-corner opening move must be rejected"
    click(u, cell_pos(u, 0, 0))
    assert not u.try_confirm(), "confirm must refuse an off-corner opening move"
    assert len(u.game.hands[0].names) == 21
    # the real opening spot is accepted (click the board again to re-target)
    click(u, cell_pos(u, x, y))
    assert (u.drag["x"], u.drag["y"]) == (x, y)
    assert u.drag["valid"]
    assert u.try_confirm()
    assert u.game.must_cover(0) is None
    assert u.game.board.grid[19 + 19 * 20] == 0
    assert len(u.game.hands[0].names) == 20


def test_opening_marks_disappear_after_first_piece():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    assert (19, 19) in u.corner_marks()
    name = "L5"
    oi, x, y = find_opening(u, name)
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    assert u.try_confirm()
    assert (19, 19) not in u.corner_marks()


def test_corner_only_placement_rule_in_the_ui():
    """The rule has to be visible in the UI, not just enforced by Game.act.

    A spot with no corner contact is invalid, and a spot sharing an edge with
    one of the player's own stones is invalid even when it also corner-touches
    another. A legal corner-only spot does not exist next to a single convex
    piece, so it is searched for rather than assumed.
    """
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    oi, x, y = find_opening(u, "L5")
    click(u, u.hand_slot("L5").center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    assert u.try_confirm(), "the legal opening must go through"
    assert wait_for_player_turn(u)

    own = {(i % 20, i // 20) for i in range(400) if u.game.board.grid[i] == 0}
    name, oi = "I1", 0
    legal = edge_spot = far_spot = None
    for cx in range(20):
        for cy in range(20):
            if (cx, cy) in own or u.game.can_act(name, oi, cx, cy, 0):
                if u.game.can_act(name, oi, cx, cy, 0) and legal is None:
                    legal = (cx, cy)
                continue
            touch = any((cx + dx, cy + dy) in own
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy))
            if touch and edge_spot is None:
                edge_spot = (cx, cy)
            elif not touch and far_spot is None:
                far_spot = (cx, cy)
    assert edge_spot and far_spot, (edge_spot, far_spot)

    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    # a board click re-targets a locked drag, so use it for every spot
    for spot, should in ((edge_spot, False), (far_spot, False), (legal, True)):
        click(u, cell_pos(u, *spot))
        assert (u.drag["x"], u.drag["y"]) == spot
        assert u.drag["valid"] is should, (spot, should)
        if not should:
            assert not u.try_confirm(), (spot, "confirm must refuse")
            assert u.game.placed[0] == 1
            # Game.act rejects it even if the UI were bypassed
            try:
                u.game.act(name, oi, spot[0], spot[1])
            except ValueError:
                pass
            else:
                raise AssertionError("act() accepted an illegal placement at %s"
                                     % (spot,))
    # ... and the legal corner-contact spot goes through
    assert u.try_confirm()
    assert u.game.placed[0] == 2


def test_adjacent_placement_is_accepted():
    """Counterpart to the test above: a legal follow-up still previews green."""
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    oi, x, y = find_opening(u, "L5")
    click(u, u.hand_slot("L5").center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    assert u.try_confirm()
    assert wait_for_player_turn(u)

    mv = best_player_move(u, seed=3)
    name, oi, x, y = mv
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    assert u.drag["valid"], "an AI-chosen legal follow-up must preview valid"
    assert u.try_confirm()
    assert u.game.placed[0] == 2


def test_dragged_thumbnail_stays_flat_with_rotation_badge():
    """A rotated piece can be taller than its group's row (I5 upright is 1x5 in
    a 3-cell row), so the hand thumbnail keeps the flat orientation and the
    board preview is the authoritative shape view."""
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    slot = u.hand_slot("I5")
    click(u, slot.center)
    assert u.drag is not None and u.drag["name"] == "I5"
    assert u.drag["rot"] == 0
    key(u, pygame.K_x)
    assert u.drag["rot"] == 1
    # the slot geometry never changes with rotation
    assert u.hand_slot("I5") == slot
    u.tick([])
    # the badge is drawn inside the hand band, not over the board
    top = u.L.HAND_Y0 - int(round(8 * u.L.scale))
    badge_top = max(top + 1, slot.top - 12 + 2)
    assert badge_top >= top


def test_cancel_button_returns_piece():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    before = list(u.game.hands[0].names)
    click(u, u.hand_slot(u.hand_order()[0]).center)
    assert u.drag is not None
    click(u, u.button_rects()["cancel"].center)
    assert u.drag is None
    assert u.game.hands[0].names == before


def test_player_drag_confirm_cycle():
    u = start_game(make_ui())
    done = 0
    for _ in range(12):
        if not wait_for_player_turn(u):
            break
        mv = best_player_move(u, seed=done)
        if mv is None:
            continue
        name, oi, x, y = mv
        names = u.game.hands[0].names
        if name not in names:
            continue
        click(u, u.hand_slot(name).center)
        if u.drag is None:
            continue
        u.drag["oi"] = oi
        move_mouse(u, cell_pos(u, x, y))
        click(u, cell_pos(u, x, y))
        click(u, u.button_rects()["confirm"].center)
        if u.drag is None:
            done += 1
            if done >= 4:
                break
    assert done == 4, "player completed only %d placements" % done


# ---------------------------------------------------------------- remaining cells

def test_remaining_cells_starts_at_full_hand():
    u = start_game(make_ui())
    for owner in range(4):
        assert u.remaining_cells(owner) == HAND_CELLS_TOTAL
        assert len(u.game.hands[owner].names) == 21
    assert HAND_CELLS_TOTAL == 89


def test_remaining_cells_drops_by_piece_size():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    name, oi, x, y = best_player_move(u)
    size = MASTER[name]["size"]
    before = u.remaining_cells(0)
    click(u, u.hand_slot(name).center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    click(u, u.button_rects()["confirm"].center)
    assert u.remaining_cells(0) == before - size
    occupied = sum(1 for c in u.game.board.grid if c == 0)
    assert occupied == HAND_CELLS_TOTAL - u.remaining_cells(0)


# ---------------------------------------------------------------- end of game

def test_game_over_screen_renders():
    u = start_game(make_ui())
    u.game.state = "GAME_OVER"
    u.state = "GAME_OVER"
    u.tick([])
    assert u.state == "GAME_OVER"
    r = u.new_game_rect()
    assert r.bottom <= u.L.H
    click(u, r.center)
    assert u.state == "SETUP_SEATS"
    assert u.game.state == "SETUP_SEATS"


def test_pass_draws_no_board_change():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    u.game.hands[0].names = []
    snap = list(u.game.board.grid)
    u.tick([])
    assert u.game.board.grid == snap
    assert u.remaining_cells(0) == 0


# ---------------------------------------------------------------- game over

class _MemoryRecords:
    """Stands in for the on-disk leaderboard so tests never touch records.json."""

    def __init__(self):
        self.calls = []
        self.last = []
        self._rows = []

    def record(self, standings):
        self.calls.append(list(standings))
        self.last = [(k, 0, 0) for k, _ in standings]
        return self.last

    def rows(self, order=None):
        return self._rows


def make_finished_ui(seed=3, records=None):
    u = start_game(make_ui(seed=seed))
    u.records = records if records is not None else _MemoryRecords()
    return u


def _force_game_over(u):
    for o in range(4):
        u.game.hands[o].names = []
    u.game.state = "GAME_OVER"
    u.enter_game_over()


def test_game_over_keeps_the_board_visible():
    """The finished board must stay on screen so it can be studied."""
    u = make_finished_ui()
    _force_game_over(u)
    before = list(u.game.board.grid)
    u.tick([])
    assert u.state == "GAME_OVER"
    assert u.game.board.grid == before, "board must not be cleared"
    # the results panel is confined to the sidebar column, never over the board
    board = u.L.board_rect()
    panel = pygame.Rect(0, 0, u.L.PANEL_W, u.L.HAND_Y0)
    assert panel.right <= board.left
    # the board itself is still fully drawn
    u.draw_results()
    assert u.game.board.grid == before


def test_game_over_only_restarts_on_an_explicit_choice():
    u = make_finished_ui()
    _force_game_over(u)
    u.tick([])
    assert u.state == "GAME_OVER"
    for _ in range(5):
        u.tick([])
    assert u.state == "GAME_OVER", "an idle screen must not restart by itself"
    # a click that misses the button does nothing
    click(u, (u.L.W - 20, 20))
    assert u.state == "GAME_OVER"
    # Enter is the explicit choice
    key(u, pygame.K_RETURN)
    assert u.state == "SETUP_SEATS"
    assert u.game.state == "SETUP_SEATS"


def test_game_over_restart_button_starts_a_new_game():
    u = make_finished_ui()
    _force_game_over(u)
    r = u.new_game_rect()
    assert r.bottom <= u.L.H
    assert r.right <= u.L.W
    click(u, r.center)
    assert u.state == "SETUP_SEATS"
    assert len(u.game.hands[0].names) == 21


def test_result_is_recorded_exactly_once():
    rec = _MemoryRecords()
    u = make_finished_ui(records=rec)
    _force_game_over(u)
    for _ in range(10):
        u.tick([])
    assert len(rec.calls) == 1, "the result was banked %d times" % len(rec.calls)
    keys = [k for k, _ in rec.calls[0]]
    # The four keys are the four seat *options*, not "one player and three
    # distinct personalities": a seat may repeat an option, which is the whole
    # point of the seat screen, and a person is named by its own key so two
    # humans are distinguishable from a personality.
    assert sorted(keys) == sorted(["human", "chess", "chess", "chess"]), keys
    assert set(keys) <= set(seats_mod.seat_menu_options())
    # starting a new game must arm it again
    u.new_game()
    u.game.state = "GAME_OVER"
    u.enter_game_over()
    assert len(rec.calls) == 2


def test_sidebar_shows_no_score_column():
    """The old 3x corner score was confusing and is gone. 'Remaining' is now
    the score, and fewer remaining is better."""
    assert "score" not in I, "the 分數 (score) label should be gone"
    u = start_game(make_ui())
    rects = []
    real = u.label

    def spy(text, x, y, size, color, bg=None, center=False, right=False):
        r = real(text, x, y, size, color, bg, center, right)
        rects.append((text, r))
        return r

    u.label = spy
    u.draw_left()
    u.label = real
    shown = [t for t, _ in rects]
    # The standalone "remaining" / "fewer is better" header is gone too: it
    # overlapped the title and restated what the per-player rows already say.
    assert I["left"] not in shown
    assert I["less_is_better"] not in shown
    # every player row is labelled with what they still hold, nothing else
    for o in range(4):
        rem = u.remaining_cells(o)
        assert I["left_fmt"].format(len(u.game.hands[o].names), rem) in shown


def test_sidebar_rows_follow_the_turn_order():
    """Row one is whoever moves first this game, not seat 0."""
    u = start_game(make_ui())
    order = list(u.game.turn_order)
    rects = []
    real = u.label

    def spy(text, x, y, size, color, bg=None, center=False, right=False):
        r = real(text, x, y, size, color, bg, center, right)
        rects.append((text, r))
        return r

    u.label = spy
    u.draw_left()
    u.label = real
    # The name column of each row carries either the player name or a
    # personality name.
    names = {ui.seat_label(k) for k in u.game.owner_key.values()}
    named = sorted(((r.top, t) for t, r in rects if t in names))
    assert len(named) == 4, named
    expected = [ui.seat_label(u.game.owner_key[o]) for o in order]
    assert [n for _top, n in named] == expected, (expected, named)


# ---------------------------------------------------------------- piece switching

def test_clicking_another_piece_switches_to_it_without_cancelling():
    """With a piece already held, a left click on a different hand piece
    selects that piece directly - no cancel press in between."""
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    click(u, u.hand_slot("L5").center)
    assert u.drag is not None and u.drag["name"] == "L5"
    click(u, u.hand_slot("O4").center)
    assert u.drag is not None, "the held piece must survive the switch"
    assert u.drag["name"] == "O4"
    assert u.drag["x"] is None and not u.drag["locked"], "fresh selection"
    assert len(u.game.hands[0].names) == 21, "switching must not place a piece"


def test_switching_resets_rotation_and_preview():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    click(u, u.hand_slot("L5").center)
    key(u, pygame.K_x)
    assert u.drag["rot"] == 1
    oi, x, y = find_opening(u, "L5")
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    assert u.drag["valid"]
    click(u, u.hand_slot("I3").center)
    assert u.drag["name"] == "I3"
    assert u.drag["rot"] == 0
    # a fresh selection starts at the flat hand orientation
    assert u.drag["oi"] == HAND_ITEMS["I3"]
    assert u.drag["x"] is None and u.drag["valid"] is False


def test_clicking_the_held_piece_keeps_it():
    """A click on the piece already held must not drop the preview."""
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    oi, x, y = find_opening(u, "L5")
    click(u, u.hand_slot("L5").center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, cell_pos(u, x, y))
    assert u.drag["locked"]
    click(u, u.hand_slot("L5").center)
    assert u.drag is not None and u.drag["name"] == "L5"
    assert u.drag["x"] == x and u.drag["y"] == y, "preview must survive"
    assert u.drag["locked"], "still locked after re-clicking the held piece"


def test_clicking_empty_hand_band_keeps_the_held_piece():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    click(u, u.hand_slot("L5").center)
    u.drag["x"], u.drag["y"] = 5, 5
    # a spot in the band that is not on any piece
    empty = (u.L.W - 30, u.L.HAND_Y0 + 4)
    assert not any(u.hand_slot(n).collidepoint(empty) for n in u.hand_order())
    click(u, empty)
    assert u.drag is not None and u.drag["name"] == "L5"
    assert (u.drag["x"], u.drag["y"]) == (5, 5)


def test_switching_does_not_break_the_buttons():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    click(u, u.hand_slot("L5").center)
    o0 = u.drag["oi"]
    # the cancel button sits above the hand band and must still cancel
    click(u, u.button_rects()["cancel"].center)
    assert u.drag is None
    # and rotate still applies to the held piece
    click(u, u.hand_slot("L5").center)
    o0 = u.drag["oi"]
    click(u, u.button_rects()["rotate"].center)
    assert u.drag is not None and u.drag["oi"] != o0


def test_switch_then_place_places_the_new_piece():
    u = start_game(make_ui())
    assert wait_for_player_turn(u)
    oi, x, y = find_opening(u, "L5")
    click(u, u.hand_slot("L5").center)
    u.drag["oi"] = oi
    move_mouse(u, cell_pos(u, x, y))
    click(u, u.hand_slot("O4").center)      # switch away without cancelling
    assert u.drag["name"] == "O4"
    # L5 must still be in hand; nothing was placed
    assert "L5" in u.game.hands[0].names
    assert "O4" in u.game.hands[0].names
    # now place the O4 at a legal corner-only spot
    spot = None
    own = {(i % 20, i // 20) for i in range(400) if u.game.board.grid[i] == 0}
    for cx in range(20):
        for cy in range(20):
            if u.game.can_act("O4", 0, cx, cy, 0):
                spot = (cx, cy)
                break
        if spot:
            break
    assert spot is not None
    click(u, cell_pos(u, *spot))
    assert u.drag["valid"]
    assert u.try_confirm()
    assert "O4" not in u.game.hands[0].names
    assert "L5" in u.game.hands[0].names
    assert u.game.placed[0] == 1



# ----------------------------------------------- the scrolling picker (stage 7)

def test_the_wheel_scrolls_the_list_down(monkeypatch):
    """The direction, pinned: a wheel-down flick moves the list down.

    pygame's `MOUSEWHEEL.y` is positive upwards, so this is the one place a
    sign error could hide and still look right in a screenshot.
    """
    u, keys = open_fake_menu(monkeypatch, rows=24)
    assert u.menu_scroll == 0
    wheel(u, -1)                      # wheel down
    assert u.menu_scroll == 1, "wheel down must increase menu_scroll"
    wheel(u, 1)                       # wheel up
    assert u.menu_scroll == 0
    for _ in range(64):
        wheel(u, -1)
    at_end = u.menu_scroll
    assert at_end > 0, "24 rows cannot all be on screen"
    wheel(u, -1)
    assert u.menu_scroll == at_end, "clamped at the end"
    wheel(u, 1)
    assert u.menu_scroll == at_end - 1
    for _ in range(64):
        wheel(u, 1)
    assert u.menu_scroll == 0, "clamped at the top"


def test_the_arrow_page_and_home_end_keys_drive_the_list(monkeypatch):
    u, keys = open_fake_menu(monkeypatch, rows=24)
    page = len(u.menu_rects()) - 1          # minus the cancel button
    key(u, pygame.K_DOWN)
    assert u.menu_scroll == 1
    key(u, pygame.K_UP)
    assert u.menu_scroll == 0
    # a full page from the top is still clamped to `max_scroll`, which is the
    # rest of the list rather than the page size
    key(u, pygame.K_PAGEDOWN)
    assert u.menu_scroll == min(page, len(keys) - page)
    key(u, pygame.K_PAGEUP)
    assert u.menu_scroll == 0
    key(u, pygame.K_END)
    assert u.menu_scroll == len(keys) - page
    key(u, pygame.K_HOME)
    assert u.menu_scroll == 0


def test_a_long_menu_reaches_its_last_row_and_picks_it(monkeypatch):
    """Stage 7's acceptance: with more rows than fit, the last one is still
    reachable and still clickable."""
    u, keys = open_fake_menu(monkeypatch, rows=24)
    windows = menu_windows(u)
    assert len(windows) > 1, "24 rows must need more than one screenful"
    last = windows[-1]
    assert ("option:" + keys[-1]) in last, "the last row is never reachable"
    click(u, last["option:" + keys[-1]].center)
    assert u.seat_pick is None
    assert u.seat_keys[0] == keys[-1]


def test_every_window_of_a_long_menu_sits_inside_the_list_area(monkeypatch):
    """What the union claim rests on, checked at every scroll position.

    Three properties per window - rows never overlap each other, every row
    sits above the cancel button (inside the list area) and on screen, and no
    key is listed twice - plus the union over positions being the whole list,
    and no two positions showing the same window.
    """
    u, keys = open_fake_menu(monkeypatch, rows=24)
    windows = menu_windows(u)
    offered, signatures = set(), []
    for rects in windows:
        assert "menu_cancel" in rects, "the cancel button is never scrolled away"
        cancel = rects["menu_cancel"]
        assert 0 <= cancel.top and cancel.bottom <= u.L.H, cancel
        rows = [(k, r) for k, r in rects.items() if k != "menu_cancel"]
        names = [k for k, _r in rows]
        assert len(set(names)) == len(names), names
        for i, (_ka, a) in enumerate(rows):
            for _kb, b in rows[i + 1:]:
                assert not a.colliderect(b), (_ka, _kb)
        for _k, r in rows:
            assert r.top >= 0 and r.bottom <= cancel.top, r
            assert r.bottom <= u.L.H and r.right <= u.L.W, r
        offered |= set(names)
        signatures.append(tuple(sorted(
            (k, (r.x, r.y, r.w, r.h)) for k, r in rects.items())))
    assert offered == {"option:" + k for k in keys}
    assert len(set(signatures)) == len(signatures), \
        "two scroll positions showed the same window"


def test_the_cancel_button_stays_reachable_at_every_scroll_position(
        monkeypatch):
    u, keys = open_fake_menu(monkeypatch, rows=24)
    # at the top: present, on screen, and a click closes the picker
    rects = u.menu_rects()
    cancel = rects["menu_cancel"]
    assert 0 <= cancel.top and cancel.bottom <= u.L.H
    # at the bottom: the same three properties, then the click that matters
    for _ in range(64):
        wheel(u, -1)
    rects = u.menu_rects()
    cancel = rects["menu_cancel"]
    assert 0 <= cancel.top and cancel.bottom <= u.L.H, cancel
    click(u, cancel.center)
    assert u.seat_pick is None, "cancel must work at max scroll"
    assert u.menu_scroll == 0, "closing resets the scroll"


@pytest.mark.parametrize("height", [600, 720, 900])
def test_the_page_and_the_scroll_range_follow_the_window_height(
        monkeypatch, height):
    """Both numbers come from `L.H`: the page is what fits, and `max_scroll`
    is the rest of the list."""
    u, keys = open_fake_menu(monkeypatch, rows=24, scale=height / 720.0)
    assert u.L.H == height
    windows = menu_windows(u)
    page = len(windows[0]) - 1
    assert 1 <= page < len(keys), page
    for rects in windows:
        assert "menu_cancel" in rects
        cancel = rects["menu_cancel"]
        assert 0 <= cancel.top and cancel.bottom <= u.L.H, (height, cancel)
        for k, r in rects.items():
            assert 0 <= r.top and r.bottom <= u.L.H, (height, k, r)
            if k != "menu_cancel":
                assert r.bottom <= cancel.top, (height, k, r)
    # scrolled to the end by `menu_windows`, so this is `max_scroll` itself
    assert u.menu_scroll == len(keys) - page, (height, u.menu_scroll, page)
    assert ("option:" + keys[-1]) in windows[-1], height


def test_a_shorter_window_never_shows_more_rows_than_a_taller_one(monkeypatch):
    pages = []
    for height in (600, 900):
        u, _keys = open_fake_menu(monkeypatch, rows=24, scale=height / 720.0)
        pages.append(len(u.menu_rects()) - 1)
    assert pages[0] <= pages[1], pages


def test_the_picker_offers_random_ai_and_it_resolves_at_start():
    """The seat screen offers it as a seventeenth choice, shows a name and a
    description for it, and turns it into a specific opponent when the game
    starts."""
    u = make_ui()
    at_seat_screen(u)
    click(u, u.seat_rects()["0:option"].center)
    assert u.seat_pick == (0, "option")
    # Stage 7, claim by claim:
    #   `"option:random_ai" in menu` and `len(...) == 14`
    #       -> the first window is only the first screenful. What the picker
    #          offers is the union over scroll positions, and `random_ai` is
    #          the last row, so it is read off the last window below.
    #   `all(r.bottom <= u.L.H)`
    #       -> kept, but per window: windowing is what makes it true at every
    #          scroll position rather than only while the list happens to fit.
    #          Whether the *last* row is reachable is asserted by
    #          `test_a_long_menu_reaches_its_last_row_and_picks_it`.
    #   (the cancel button was not checked at all)
    #       -> new: present and on screen in every window.
    windows = menu_windows(u)
    offered = set()
    for rects in windows:
        assert "menu_cancel" in rects
        cancel = rects["menu_cancel"]
        assert 0 <= cancel.top and cancel.bottom <= u.L.H, cancel
        rows = [(k, r) for k, r in rects.items() if k.startswith("option:")]
        offered |= {k.split(":", 1)[1] for k, _r in rows}
        assert all(0 <= r.top and r.bottom <= u.L.H for _k, r in rows), rows
        named = list(rects.items())
        for i, (_ka, a) in enumerate(named):
            for _kb, b in named[i + 1:]:
                assert not a.colliderect(b), (_ka, _kb)
    assert offered == set(seats_mod.seat_menu_options())
    menu = windows[-1]
    assert "option:random_ai" in menu, "the last row is reached by scrolling"

    click(u, menu["option:random_ai"].center)
    assert u.seat_keys[0] == seats_mod.RANDOM_AI_KEY
    assert ui.seat_label(seats_mod.RANDOM_AI_KEY) == I["random_ai"]
    assert ui.seat_desc(seats_mod.RANDOM_AI_KEY)

    u.seat_keys = [seats_mod.RANDOM_AI_KEY] * 4
    u.seat_colours = [seats_mod.RANDOM] * 4
    click(u, u.seat_rects()["start_game"].center)
    assert u.state == "PLAYING"
    for o in range(4):
        key = u.game.owner_key[o]
        assert key != seats_mod.RANDOM_AI_KEY
        assert key in seats_mod.automated_options(), key
        assert u.game.brains[o].key == key
    # the sidebar now names a concrete opponent, not "random AI"
    assert I["random_ai"] not in [ui.seat_label(u.game.owner_key[o])
                                  for o in range(4)]


def test_the_league_pool_is_unchanged_by_the_new_option():
    """`match.py` already draws every seat from the pool, so it does not need
    the deferred choice - adding it there would only draw twice."""
    from match import league_options
    assert list(league_options()) == list(ai.registry.automated_options())
    assert seats_mod.RANDOM_AI_KEY not in league_options()


def test_every_selectable_option_has_a_name_and_a_description_of_its_own():
    """What plan9a stage 6 left of the old registry-vs-screen comparison.

    Stage 2 wrote `label` and `desc_key` into `ai/registry.json` while the
    screen still rendered names itself, and the old test rendered both sides
    and compared them - a hand-edited `desc_key` would otherwise sit there
    doing nothing until stage 6 wired it up. Stage 6 did wire it up: the screen
    now reads the registry, so there are no longer two sides to compare, and a
    bad `desc_key` changes what the screen shows instead of being silently
    ignored (which is what `test_the_label_and_description_keys_exist_in_config`
    now guards from the JSON side).

    What a seat screen does still need is checked here instead: every
    selectable option renders a name and a description, neither is blank, and
    no two options share either one. The distinctness is not decoration - four
    seats sit at step 1000, so a label that dropped the teacher would put four
    identical rows in the menu.
    """
    from ai import registry as R

    labels, descs = [], []
    for e in R.ENTRIES:
        if not e["selectable"]:
            continue
        label, desc = ui.seat_label(e["key"]), ui.seat_desc(e["key"])
        assert label and label.strip(), e["key"]
        assert desc and desc.strip(), e["key"]
        labels.append(label)
        descs.append(desc)
    assert len(set(labels)) == len(labels), labels
    assert len(set(descs)) == len(descs), descs


# --------------------------------- two seats on the same option, results screen

def _finish_with_remaining(u, targets, options, colours):
    """Force a finished game's four scores, so a result can be asserted without
    playing a whole game to a particular shape."""
    import itertools
    from pieces import MASTER
    items = [(n, v["size"]) for n, v in MASTER.items()]
    u.game.setup_seats(options, colours, random.Random(0))
    for owner, target in enumerate(targets):
        for k in range(1, 8):
            done = False
            for combo in itertools.combinations(items, k):
                if sum(sz for _n, sz in combo) == target:
                    u.game.hands[owner].names = [n for n, _sz in combo]
                    done = True
                    break
            if done:
                break
        assert u.game.remaining_cells(owner) == target, (owner, target)
    u.enter_game_over()
    return u


def test_two_seats_on_the_same_option_each_get_their_own_place(tmp_path):
    """The regression: the results screen keyed the table by option, so two
    seats on one checkpoint came back as a single row and both were shown with
    the same rank, the same points and the same colour."""
    u = make_ui()
    u.game = Game(random.Random(0))
    u.records = Records(str(tmp_path / "records.json"))
    _finish_with_remaining(u, [7, 15, 16, 17],
                           ["hc_10000", "human", "hc_10000", "human_log"],
                           ["yellow", "blue", "green", "red"])
    rows = u.seat_rows()
    assert len(rows) == 4, rows
    assert [r[2] for r in rows] == [7, 15, 16, 17], rows
    assert [r[3] for r in rows] == [1, 2, 3, 4], rows
    assert [r[4] for r in rows] == [4, 3, 2, 1], rows
    assert [r[1] for r in rows] == ["hc_10000", "human", "hc_10000",
                                    "human_log"]
    # the two matching seats must not carry the same place
    twins = [r[3] for r in rows if r[1] == "hc_10000"]
    assert twins == [1, 3], twins
    # and each row takes its colour from its own seat, not from the option
    assert [u.game.colors[r[0]] for r in rows] == ["yellow", "blue", "green",
                                                   "red"]


def test_the_results_screen_renders_four_distinct_rows(tmp_path):
    u = make_ui()
    u.game = Game(random.Random(0))
    u.records = Records(str(tmp_path / "records.json"))
    _finish_with_remaining(u, [7, 15, 16, 17],
                           ["hc_10000", "human", "hc_10000", "human_log"],
                           ["yellow", "blue", "green", "red"])
    real = u.label
    texts = []
    u.label = lambda t, *a, **kw: (texts.append(str(t)), real(t, *a, **kw))[1]
    u.draw_results()
    u.label = real
    shown = [t for t in texts if I["left_won_fmt"].split("{")[0] in t]
    assert len(shown) == 4, shown
    for rank, target in zip([1, 2, 3, 4], [7, 15, 16, 17]):
        assert I["left_won_fmt"].format(rank, target) in shown, (rank, shown)
    # the ranks must all differ - the bug put the same place on two rows
    places = [t for t in shown if t.count("排第") == 1]
    assert len({t.split("，")[0] for t in places}) == 4, places


def test_a_leaderboard_row_for_a_repeated_contestant_is_neutral():
    """A contestant holding two seats has no single colour, so its leaderboard
    swatch says so rather than borrowing one seat's."""
    u = make_ui()
    u.game = Game(random.Random(0))
    u.game.setup_seats(["hc_2000", "fox", "hc_2000", "wolf"],
                       ["blue", "green", "red", "yellow"], random.Random(0))
    assert u.contestant_color("hc_2000") == (200, 200, 210)
    assert u.contestant_color("fox") == COLORS["green"]


def test_seat_rows_survive_a_game_that_was_never_recorded():
    u = make_ui()
    u.game = Game(random.Random(0))
    u.game.setup_seats(["wolf"] * 4, ["blue", "green", "red", "yellow"],
                       random.Random(0))
    u.records.last = []
    assert u.seat_rows() == []
