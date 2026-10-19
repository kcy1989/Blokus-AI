"""Deterministic UI screenshots for the S9 `Layout` refactor.

Renders a fixed matrix of (scale, hand_zoom, screen) to PNG so a `Layout`
extraction can be checked pixel for pixel. `--tree` chooses which project root
to import from, so one byte-identical copy of this script serves both the
baseline and the refactored tree.

Two runtime files are neutralised on purpose: `ratings.json` and
`records.json` are not in version control, so a screenshot that read them would
not be reproducible on another machine.

Usage:
    # baseline, and a second pass in the same process for a self-check
    python tools/ui_shots.py --tree <root> --out A --also B
    # a separate process, for the cross-process self-check
    python tools/ui_shots.py --tree <root> --out C
    # pixel comparison (0 differences required)
    python tools/ui_shots.py --compare A B
    # counter-proof: shift a pixel constant by one, to show the comparator bites
    python tools/ui_shots.py --tree <root> --out J --jitter 1
"""
import argparse
import hashlib
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

SCALES = (1.0, 1.5, 2.0)
SCREENS = ("setup", "playing", "game_over")
SEED = 7
PLAY_PLIES = 10
GAME_OVER_LIMIT = 400
EXPLICIT_COLOURS = ["blue", "green", "red", "yellow"]


class _NoRecords:
    """A `Records` stand-in: `records.json` is a runtime file."""

    def __init__(self, last=None):
        self.last = last

    def rows(self):
        return []

    def record(self, *a, **k):
        return None

    def rank_rows(self, *a, **k):
        return []


def _load(root):
    sys.path.insert(0, root)
    import pygame
    pygame.init()
    import ai
    import ui
    from game import Game
    # `ratings.json` is not in version control either, so the Elo suffixes it
    # would add to seat labels are dropped rather than left to the machine.
    ui._ELO_BY_KEY = {}
    return pygame, ai, ui, Game


def _advance(ai, game, limit):
    n = 0
    while game.state != "GAME_OVER" and n < limit:
        owner = game.current_owner()
        brain = ai.make_brain("chess", random.Random(2000 + n))
        mv = ai.choose_move(game.board, game.hands[owner].names, owner, brain,
                            random.Random(3000 + n), other_brains=None,
                            must_cover=game.must_cover(owner),
                            reach=game.reach(owner))
        if mv is None:
            game.act_pass()
        else:
            game.act(mv[0], mv[1], mv[2], mv[3])
        n += 1
    return game


def _fixed_ranks(game):
    """Deterministic `(key, rank, points)` rows, so the results screen does not
    have to read `records.json`."""
    standing = game.standings()
    rems = [rem for _owner, rem in standing]
    return [(game.owner_key[owner], 1 + sum(1 for r in rems if r < rem), rem)
            for owner, rem in standing]


def _shot(pygame, ai, ui, Game, scale, hand_zoom, screen, playing, over, path,
          jitter):
    u = ui.UI(None, ui.load_cjk_font_path(), scale=scale, hand_zoom=hand_zoom,
              records=_NoRecords())
    if screen == "setup":
        u.game = Game(random.Random(SEED))
        u.state = "SETUP_SEATS"
        u.seat_keys = list(ui.DEFAULT_SEAT_KEYS)
        u.seat_colours = list(EXPLICIT_COLOURS)
        u.draw_setup_seats()
    elif screen == "playing":
        u.game = playing
        u.state = "PLAYING"
        u.draw_all()
    else:
        u.game = over
        u.state = "GAME_OVER"
        u.records = _NoRecords(last=_fixed_ranks(over))
        u.draw_results()
    if jitter:
        # Counter-proof hook: move the board origin by `jitter` pixels so the
        # comparator has something it must catch.
        u.L.BO = (u.L.BO[0] + jitter, u.L.BO[1] + jitter)
        if screen == "setup":
            u.draw_setup_seats()
        elif screen == "playing":
            u.draw_all()
        else:
            u.draw_results()
    pygame.image.save(u.screen, path)


def _new_game(ai, Game):
    g = Game(random.Random(SEED))
    g.set_player_color("blue")
    g.start()
    return g


def render_matrix(root, out, jitter=0):
    pygame, ai, ui, Game = _load(root)
    os.makedirs(out, exist_ok=True)
    playing = _advance(ai, _new_game(ai, Game), PLAY_PLIES)
    over = _advance(ai, _new_game(ai, Game), GAME_OVER_LIMIT)
    hz = (ui.HAND_ZOOMS[0], ui.HAND_ZOOMS[-1])
    names = []
    for scale in SCALES:
        for hand_zoom in hz:
            for screen in SCREENS:
                name = "s%.1f_hz%.1f_%s.png" % (scale, hand_zoom, screen)
                path = os.path.join(out, name)
                _shot(pygame, ai, ui, Game, scale, hand_zoom, screen,
                      playing, over, path, jitter)
                names.append(name)
    with open(os.path.join(out, "md5sums.txt"), "w", encoding="utf-8") as fh:
        for name in names:
            with open(os.path.join(out, name), "rb") as png:
                fh.write("%s  %s\n" % (hashlib.md5(png.read()).hexdigest(), name))
    print("wrote %d PNGs + md5sums.txt to %s" % (len(names), out))
    return names


def compare(a_dir, b_dir):
    import pygame
    pygame.init()
    names = sorted(n for n in os.listdir(a_dir) if n.endswith(".png"))
    bad = 0
    for name in names:
        pa, pb = os.path.join(a_dir, name), os.path.join(b_dir, name)
        if not os.path.exists(pb):
            print("%-28s MISSING in %s" % (name, b_dir))
            bad += 1
            continue
        sa = pygame.image.load(pa)
        sb = pygame.image.load(pb)
        if sa.get_size() != sb.get_size():
            print("%-28s SIZE %s vs %s" % (name, sa.get_size(), sb.get_size()))
            bad += 1
            continue
        w, h = sa.get_size()
        ba = pygame.image.tobytes(sa, "RGB")
        bb = pygame.image.tobytes(sb, "RGB")
        if ba == bb:
            print("%-28s OK   (%dx%d)" % (name, w, h))
            continue
        stride = w * 3
        rows = [y for y in range(h)
                if ba[y * stride:(y + 1) * stride] != bb[y * stride:(y + 1) * stride]]
        xmin, xmax, count, maxdelta = w, -1, 0, 0
        for y in rows:
            ra = ba[y * stride:(y + 1) * stride]
            rb = bb[y * stride:(y + 1) * stride]
            for x in range(w):
                i = x * 3
                if ra[i:i + 3] != rb[i:i + 3]:
                    count += 1
                    if x < xmin:
                        xmin = x
                    if x > xmax:
                        xmax = x
                    for k in range(3):
                        d = abs(ra[i + k] - rb[i + k])
                        if d > maxdelta:
                            maxdelta = d
        print("%-28s DIFF %d px  bbox x[%d..%d] y[%d..%d]  maxdelta %d"
              % (name, count, xmin, xmax, rows[0], rows[-1], maxdelta))
        bad += 1
    print("---")
    print("%d/%d files differ" % (bad, len(names)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out")
    ap.add_argument("--also")
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if not args.out:
        ap.error("--out is required unless --compare is used")
    render_matrix(args.tree, args.out, args.jitter)
    if args.also:
        render_matrix(args.tree, args.also, args.jitter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
