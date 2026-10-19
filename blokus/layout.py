"""Window geometry: the scale/zoom constants and the hand-rack orientation map.

This module answers one question - what geometry does the window have - so the
drawing code in `ui.py` does not also own the arithmetic. It imports only
`pieces`, so nothing here can pull in pygame, the game, or the UI.
"""
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
