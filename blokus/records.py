"""Persistent per-contestant record: average points and average squares left.

The score of a game is the number of squares a player failed to place, so
fewer is better. Each finished game ranks the four contestants on that number,
gives 4 points for first place down to 1 for last, and folds the result into
running averages.

Contestants are keyed by role, not by seat: the three AI seats are handed a
different personality every game, so a record keyed by owner would mix wolves
with foxes. Keying by personality keeps each contestant's history meaningful.
"""
import json
import os

PLAYER_KEY = "player"

RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "records.json")

# points for 1st, 2nd, 3rd, 4th
POINTS_FOR_RANK = (4, 3, 2, 1)


def rank_by_remaining(standings):
    """Competition ranking: equal remaining squares share a place, and the
    following places skip accordingly.

    `standings` is an iterable of (key, remaining). Returns (key, rank,
    points) for each contestant, ordered by place then key.
    """
    rows = sorted(standings, key=lambda t: (t[1], t[0]))
    out = []
    prev_rem = None
    rank = 0
    for i, (key, rem) in enumerate(rows):
        if rem != prev_rem:
            rank = i + 1
            prev_rem = rem
        out.append((key, rank, POINTS_FOR_RANK[rank - 1]))
    return out


class Records:
    def __init__(self, path=RECORDS_PATH):
        self.path = path
        self.entries = {}
        self.last = []
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for key, rec in data.items():
            if not isinstance(rec, dict):
                continue
            try:
                games = int(rec["games"])
                total_points = float(rec["total_points"])
                total_remaining = float(rec["total_remaining"])
            except (KeyError, TypeError, ValueError):
                continue
            if games > 0:
                self.entries[key] = {"games": games,
                                     "total_points": total_points,
                                     "total_remaining": total_remaining}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def record(self, standings):
        """Fold one finished game into the averages.

        `standings` is an iterable of (key, remaining) for the four
        contestants. Returns the (key, rank, points) rows for display.
        """
        rows = rank_by_remaining(standings)
        self.last = rows
        for key, _rank, points in rows:
            remaining = dict(standings)[key]
            rec = self.entries.setdefault(key, {"games": 0, "total_points": 0.0,
                                                "total_remaining": 0.0})
            rec["games"] += 1
            rec["total_points"] += points
            rec["total_remaining"] += remaining
        self._save()
        return rows

    def reset(self):
        self.entries = {}
        self.last = []
        self._save()

    def rows(self, order=None):
        """Leaderboard rows as (key, games, avg_points, avg_remaining).

        Sorted by average points descending, then average remaining ascending.
        `order` optionally pins the sequence (used to keep the player first).
        """
        keys = list(self.entries)
        if order is not None:
            keys = [k for k in order if k in self.entries]
            keys += [k for k in sorted(self.entries) if k not in order]
        out = []
        for k in keys:
            rec = self.entries[k]
            g = rec["games"]
            out.append((k, g, rec["total_points"] / g,
                        rec["total_remaining"] / g))
        out.sort(key=lambda t: (-t[2], t[3], t[0]))
        return out
