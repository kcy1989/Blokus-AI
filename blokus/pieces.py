"""Piece definitions, orientation generation, hand management."""
from dataclasses import dataclass

from config import PIECES


def _orient_set(cells):
    cells = list(cells)
    result = set()
    for flip in (0, 1):
        if flip:
            mx = max(x for x, _ in cells)
            base = [(mx - x, y) for x, y in cells]
        else:
            base = [(x, y) for x, y in cells]
        cur = base
        for _ in range(4):
            mx = max(x for x, _ in cur)
            my = max(y for _, y in cur)
            cur = [(mx - y, x) for x, y in cur]
            nx0 = min(x for x, _ in cur)
            ny0 = min(y for _, y in cur)
            cur = frozenset((x - nx0, y - ny0) for x, y in cur)
            result.add(cur)
    return tuple(sorted(result, key=lambda s: sorted(s)))


MASTER = {}
for _p in PIECES:
    item = dict(_p)
    item["orientations"] = _orient_set(item["cells"])
    MASTER[item["name"]] = item


@dataclass(frozen=True)
class Move:
    name: str
    orient: int
    x: int
    y: int


class Hand:
    def __init__(self, names):
        self.names = list(names)

    def __len__(self):
        return len(self.names)

    def has(self, name):
        return name in self.names

    def remove(self, name):
        self.names.remove(name)

    def all(self):
        return [MASTER[name] for name in self.names]
