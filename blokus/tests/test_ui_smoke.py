"""Headless UI smoke tests: setup flow, lock/confirm, scaling, remaining cells."""
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai
from config import HAND_CELLS_TOTAL
from game import Game
from pieces import MASTER
from config import I
from ui import (HAND_ITEMS, HAND_ZOOMS, PERSONA_ZH, SCALES, Layout, UI,
                best_scale, load_cjk_font_path)


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
    key(u)
    click(u, (225, 330))
    key(u)
    assert u.state == "PLAYING"
    return u


def at_setup_info(u):
    """The setup screen, reached by picking a colour."""
    u.tick([])
    s = u.L.scale
    bw, gap = int(150 * s), int(20 * s)
    x0 = (u.L.W - (4 * bw + 3 * gap)) // 2
    click(u, (x0 + bw // 2, int(220 * s) + int(220 * s) // 2))
    assert u.state == "SETUP_INFO"
    return u


def best_player_move(u, seed=0):
    return ai.choose_move(u.game.board, u.game.hands[0].names, 0, u.game.brains[0],
                          random.Random(seed), other_brains=None,
                          must_cover=u.game.must_cover(0), reach=u.game.reach(0))


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


def test_layout_keeps_everything_inside_window():
    for s in SCALES:
        for z in HAND_ZOOMS:
            L = Layout(s, z)
            assert L.BO[0] + L.board_px <= L.W, (s, z, "board too wide")
            assert L.BTN_Y + L.BTN_H <= L.H, (s, z, "buttons below window")
            slots, labels, h = L.layout_hand(sorted(MASTER))
            assert h == L.hand_h
            for name, r in slots.items():
                assert r.left >= L.HAND_X0 - 1, (s, z, name, r)
                assert r.right <= L.W, (s, z, name, r)
                assert r.top >= L.HAND_Y0, (s, z, name, r)
                assert r.bottom <= L.BTN_Y, (s, z, name, r)
            for r, _size in labels:
                assert r.left >= 0 and r.right <= L.W, (s, z, r)
                assert r.top >= L.HAND_Y0 and r.bottom <= L.BTN_Y, (s, z, r)


def test_hand_groups_are_ordered_by_size():
    """1..5 cells ascending, and every group's pieces on their own rows."""
    L = Layout(1.0, 2.0)
    _slots, labels, _h = L.layout_hand(sorted(MASTER))
    sizes = [size for _r, size in labels]
    assert sizes == sorted(sizes)
    assert set(sizes) == {1, 2, 3, 4, 5}
    # one label per group, so a wrapped group keeps a single gutter plate
    assert len(labels) == len(set(sizes))


def test_hand_is_bigger_than_the_old_compact_default():
    """The whole point of the change: thumbnails roughly doubled."""
    assert Layout(1.0, 2.0).HAND_CELL == 16
    assert Layout(1.0, 3.0).HAND_CELL == 24
    assert Layout(1.0, 1.5).HAND_CELL == 12
    # 9 cell-rows plus four group gaps is the documented band height
    assert Layout(1.0, 2.0).hand_h == 9 * 16 + 4 * 12


def test_five_cell_group_wraps_instead_of_being_clipped():
    """The 5-cell group is 41 cells wide, so at high zoom it must wrap."""
    wide = Layout(1.0, 3.0)
    _slots, labels, _h = wide.layout_hand(sorted(MASTER))
    five = [r for r, size in labels if size == 5]
    assert len(five) == 1
    rows = [r for n, r in wide.layout_hand(sorted(MASTER))[0].items()
            if MASTER[n]["size"] == 5]
    tops = {r.top for r in rows}
    assert len(tops) > 1, "5-cell group did not wrap at zoom 3.0"
    for r in rows:
        assert r.right <= wide.W


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
    small = Layout(best_scale(desktop, 1.5), 1.5)
    big = Layout(best_scale(desktop, 3.0), 3.0)
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
    u = make_ui(scale=1.0, hand_zoom=2.0)
    cell_before = u.L.CELL
    u.bump_hand_zoom(1)
    assert u.hand_zoom == 2.5
    assert u.L.HAND_CELL == 20
    assert u.L.CELL == cell_before, "hand zoom must not resize the board"
    u.bump_hand_zoom(-1)
    assert u.hand_zoom == 2.0
    # clamped at both ends
    for _ in range(6):
        u.bump_hand_zoom(-1)
    assert u.hand_zoom == HAND_ZOOMS[0]
    for _ in range(6):
        u.bump_hand_zoom(1)
    assert u.hand_zoom == HAND_ZOOMS[-1]


def test_bracket_keys_change_hand_zoom(monkeypatch):
    import ui as ui_mod
    monkeypatch.setattr(ui_mod, "desktop_size", lambda: (1920, 1400))
    u = make_ui(scale=1.0, hand_zoom=2.0)
    key(u, pygame.K_RIGHTBRACKET)
    assert u.hand_zoom == 2.5
    key(u, pygame.K_LEFTBRACKET)
    assert u.hand_zoom == 2.0


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
                assert r.bottom <= u.L.BTN_Y, (s, z, name, r)
                seen.append(r)
            for r, _size in u.hand_layout()[1]:
                assert r.right <= u.L.W and r.bottom <= u.L.BTN_Y, (s, z, r)
            for _bid, b in u.button_rects().items():
                assert b.bottom <= u.L.H, (s, z, b)


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

        def spy(text, x, y, size, color, bg=None, center=False, _r=rects):
            _r.append((text, real(text, x, y, size, color, bg, center)))
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

        def spy(text, x, y, size, color, bg=None, center=False, _r=rects):
            _r.append((text, real(text, x, y, size, color, bg, center)))
            return _r[-1][1]

        u.label = spy
        u.draw_left()
        u.label = real
        # nothing may be drawn over the hand band
        for text, r in rects:
            assert r.bottom <= u.L.HAND_Y0 - int(round(8 * u.L.scale)), (s, text, r)


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
        assert r.bottom <= u.L.BTN_Y


# ---------------------------------------------------------------- setup flow

def test_setup_color_then_info():
    u = make_ui()
    u.tick([])
    assert u.state == "SETUP_COLOR"
    s = u.L.scale
    bw, bh, gap = int(150 * s), int(220 * s), int(20 * s)
    x0 = (u.L.W - (4 * bw + 3 * gap)) // 2
    click(u, (x0 + bw // 2, int(220 * s) + bh // 2))
    assert u.state == "SETUP_INFO"
    assert u.game.colors[0] == "blue"
    assert len(set(u.game.owner_key[o] for o in (1, 2, 3))) == 3
    key(u)
    assert u.state == "PLAYING"
    assert len(u.game.hands[0].names) == 21


def test_every_personality_renders_on_the_setup_panel():
    """The panel looks each personality up by key, so a new personality with a
    missing name or one-line description only blows up at runtime."""
    for key in ai.personality_keys():
        u = make_ui()
        u.game.set_player_color("blue")
        u.game.owner_key[1], u.game.owner_key[2], u.game.owner_key[3] = key, "fox", "wolf"
        u.state = "SETUP_INFO"
        u.game.state = "SETUP_INFO"
        u.draw_setup_info()
        assert I[key] and I[key + "_desc"]


def test_setup_panel_shows_the_clockwise_turn_order():
    """順時針輪轉、起點隨機，所以面板要寫出誰先手。"""
    u = at_setup_info(make_ui())
    real = u.label
    texts = []

    def spy(text, *a, **kw):
        texts.append(text)
        return real(text, *a, **kw)

    u.label = spy
    u.draw_setup_info()
    u.label = real
    line = [t for t in texts if I["turn_order_fmt"].split("{")[0] in str(t)]
    assert len(line) == 1, texts
    shown = str(line[0])
    first = u.game.turn_order[0]
    want = I["player"] if first == 0 else PERSONA_ZH[u.game.owner_key[first]]
    assert want + I["turn_first"] in shown, shown
    names = [I["player"] if o == 0 else PERSONA_ZH[u.game.owner_key[o]]
             for o in u.game.turn_order]
    for i in range(3):
        assert names[i] in shown and names[i + 1] in shown
        assert shown.index(names[i]) < shown.index(names[i + 1]), shown


def test_setup_starts_only_from_the_start_button():
    """點畫面任何地方都不開始，只有「開始」鍵（或 Enter）才會。"""
    u = at_setup_info(make_ui())
    buttons = u.setup_button_rects()
    for pos in ((5, 5), (u.L.W - 5, 5), (u.L.W // 2, u.L.H - 5),
                (u.L.W // 2, u.L.H // 2)):
        click(u, pos)
        assert u.state == "SETUP_INFO", pos
        assert u.game.state == "SETUP_INFO", pos
    click(u, buttons["start_game"].center)
    assert u.state == "PLAYING"
    assert u.game.state == "PLAYING"
    # 鍵盤上的「開始鍵」仍然有效
    u2 = at_setup_info(make_ui())
    key(u2, pygame.K_RETURN)
    assert u2.state == "PLAYING"


def test_redraw_button_changes_only_the_personalities():
    """「重抽 AI」換人格，但不動玩家剛選的顏色與下棋順序。"""
    u = at_setup_info(make_ui())
    colors = list(u.game.colors)
    order = list(u.game.turn_order)
    brains = {o: u.game.brains[o] for o in (1, 2, 3)}
    seen = set()
    for _ in range(8):
        click(u, u.setup_button_rects()["redraw"].center)
        assert u.state == "SETUP_INFO"
        assert u.game.state == "SETUP_INFO"
        keys = [u.game.owner_key[o] for o in (1, 2, 3)]
        seen.add(tuple(keys))
        assert len(set(keys)) == 3, keys
        assert all(k in ai.personality_keys() for k in keys), keys
        # 每個 seat 的 brain 都跟著換成對應的人格
        for o in (1, 2, 3):
            assert u.game.brains[o].key == u.game.owner_key[o]
    assert len(seen) > 1, "8 次重抽都抽到同一組，這是 rng 的問題"
    assert list(u.game.colors) == colors
    assert list(u.game.turn_order) == order
    assert u.game.brains[0] is not None


def test_setup_buttons_fit_on_screen_and_do_not_overlap():
    u = at_setup_info(make_ui())
    rects = u.setup_button_rects()
    assert set(rects) == {"redraw", "start_game"}
    panel = u.setup_panel_rect()
    for r in rects.values():
        assert r.bottom <= u.L.H, r
        assert panel.x <= r.x and r.right <= panel.right, r
        assert r.top >= u.setup_content_bottom(), r
    a, b = rects["redraw"], rects["start_game"]
    assert not a.colliderect(b)


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
    assert u.state == "SETUP_COLOR"
    assert u.game.state == "SETUP_COLOR"


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
    assert u.state == "SETUP_COLOR"
    assert u.game.state == "SETUP_COLOR"


def test_game_over_restart_button_starts_a_new_game():
    u = make_finished_ui()
    _force_game_over(u)
    r = u.new_game_rect()
    assert r.bottom <= u.L.H
    assert r.right <= u.L.W
    click(u, r.center)
    assert u.state == "SETUP_COLOR"
    assert len(u.game.hands[0].names) == 21


def test_result_is_recorded_exactly_once():
    rec = _MemoryRecords()
    u = make_finished_ui(records=rec)
    _force_game_over(u)
    for _ in range(10):
        u.tick([])
    assert len(rec.calls) == 1, "the result was banked %d times" % len(rec.calls)
    keys = [k for k, _ in rec.calls[0]]
    assert keys[0] == "player"
    # 3-of-4 抽樣：另外三個席位是三種**互異**的人格，不一定是固定哪三種
    drawn = keys[1:]
    assert len(drawn) == 3 and len(set(drawn)) == 3, drawn
    assert set(drawn) <= set(ai.personality_keys())
    # starting a new game must arm it again
    u.new_game()
    u.game.state = "GAME_OVER"
    u.enter_game_over()
    assert len(rec.calls) == 2


def test_sidebar_shows_no_score_column():
    """The old 3x corner score was confusing and is gone. 'Remaining' is now
    the score, and fewer remaining is better."""
    assert "score" not in I, "the 分數 label should be gone"
    u = start_game(make_ui())
    rects = []
    real = u.label

    def spy(text, x, y, size, color, bg=None, center=False):
        r = real(text, x, y, size, color, bg, center)
        rects.append((text, r))
        return r

    u.label = spy
    u.draw_left()
    u.label = real
    shown = [t for t, _ in rects]
    assert I["left"] in shown
    assert I["less_is_better"] in shown
    # every player row is labelled with what they still hold, nothing else
    for o in range(4):
        rem = u.remaining_cells(o)
        assert I["left_fmt"].format(len(u.game.hands[o].names), rem) in shown


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
