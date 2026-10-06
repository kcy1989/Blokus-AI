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

# Where a contestant came from, per plan9 step 1.5. Absent from every record
# written before that step, and from any that cannot be filled in honestly.
#
# `None` therefore means **not known**, never "not applicable" and never a
# guess. Filling these in from inference is the failure mode worth guarding
# against: a `teacher` deduced from the name of a checkpoint is a claim about
# how a model was trained, and a wrong one is indistinguishable from a
# recorded one once it is in the file.
#
# None of this is read by the scoring. `rows()` and `rank_rows()` look only at
# `games`, `total_points` and `total_remaining`, and their arithmetic is
# unchanged - provenance is carried alongside a result, never mixed into it.
#
# `commit` means **the code that produced these weights**, not the code that was
# checked out when the contestant was registered, and not the commit that renamed
# it. Those are different questions with different answers, and answering the
# second one here is worse than leaving the field null: a registration commit
# moves every time anybody renames a seat, and by then it says nothing about the
# artefact it is filed next to.
#
# A trained policy can fill it in only if its own training run recorded one. See
# `rl/collect.py`'s `code_hash`, which does - it records the commit *and* whether
# the tree was dirty, because "the tree had uncommitted changes, so the commit
# alone does not identify this data". `rl/rl_train.py` records neither, so for a
# policy trained before that gap was closed the honest value is `None`.
META_FIELDS = ("history", "train_seed", "teacher", "pool_contains_teacher",
               "commit")


def _blank_record():
    """A fresh entry: zeroed totals and every provenance field unknown."""
    rec = {"games": 0, "total_points": 0.0, "total_remaining": 0.0}
    rec.update({field: None for field in META_FIELDS})
    return rec


def rank_rows(standings):
    """Competition ranking with every seat's own numbers kept attached.

    `standings` is an iterable of `(key, remaining)` for the four seats - one
    per **seat**, so the same key may appear twice when two seats play the same
    contestant. Returns `[(key, rank, points, remaining)]`, ordered best first.

    Carrying `remaining` out of here is what lets a caller fold a seat's own
    score in: looking it up by key afterwards finds only one of the two seats
    when a key repeats, which silently drops the other's result.
    """
    rows = sorted(standings, key=lambda t: (t[1], t[0]))
    out = []
    prev_rem = None
    rank = 0
    for i, (key, rem) in enumerate(rows):
        if rem != prev_rem:
            rank = i + 1
            prev_rem = rem
        out.append((key, rank, POINTS_FOR_RANK[rank - 1], rem))
    return out


def rank_by_remaining(standings):
    """Competition ranking: equal remaining squares share a place, and the
    following places skip accordingly.

    `standings` is an iterable of (key, remaining). Returns (key, rank, points)
    for each seat, ordered by place then key. This is `rank_rows` without the
    remaining value.
    """
    return [(k, r, p) for k, r, p, _rem in rank_rows(standings)]


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
                entry = {"games": games, "total_points": total_points,
                         "total_remaining": total_remaining}
                # Carried through as found, and defaulted rather than demanded:
                # a file written before this step has none of these keys, and one
                # written after has some of them. Both have to open.
                for field in META_FIELDS:
                    entry[field] = rec.get(field)
                self.entries[key] = entry

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def record(self, standings):
        """Fold one finished game into the averages.

        `standings` is an iterable of (key, remaining) **per seat**, so a key
        may appear twice when two seats play the same contestant. Each seat
        contributes its own points and its own remaining squares, and
        `games` counts seat appearances - which is what makes the averages
        comparable between a contestant that played one seat and one that
        played two.

        Returns the `(key, rank, points)` rows for display, one per seat.
        """
        rows = rank_rows(standings)
        self.last = [(key, rank, points) for key, rank, points, _rem in rows]
        for key, _rank, points, remaining in rows:
            rec = self.entries.setdefault(key, _blank_record())
            rec["games"] += 1
            rec["total_points"] += points
            rec["total_remaining"] += remaining
        self._save()
        return self.last

    def set_meta(self, key, **fields):
        """Attach provenance to a contestant, and save.

        Separate from `record` because the two answer different questions: a game
        result says how someone scored, this says who they are. Overwriting one
        with the other is how a leaderboard quietly loses its history - so
        `record` only ever touches the three totals, and this only ever touches
        the provenance fields.

        An unknown field name is an error rather than a silently stored extra:
        a typo in `train_seed` would otherwise write a file that looks complete
        and records nothing.
        """
        unknown = sorted(set(fields) - set(META_FIELDS))
        if unknown:
            raise ValueError("unknown records field(s) %s; expected any of %s"
                             % (", ".join(unknown), ", ".join(META_FIELDS)))
        if not fields:
            return self.entries.get(key)
        rec = self.entries.setdefault(key, _blank_record())
        rec.update(fields)
        self._save()
        return rec

    def meta(self, key=None):
        """Provenance for one contestant, or for all of them.

        Returns `{field: value}` for `key`, or `{key: {field: value}}` when `key`
        is None. Missing fields read as `None`, which is the same "not known"
        that a file without them produces.
        """
        if key is not None:
            rec = self.entries.get(key, {})
            return {f: rec.get(f) for f in META_FIELDS}
        return {k: {f: rec.get(f) for f in META_FIELDS}
                for k, rec in self.entries.items()}

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
            if g <= 0:
                # A contestant can be given provenance before it has ever played.
                # Its averages do not exist yet, and dividing by the zero here
                # would take the whole leaderboard down with it.
                continue
            out.append((k, g, rec["total_points"] / g,
                        rec["total_remaining"] / g))
        out.sort(key=lambda t: (-t[2], t[3], t[0]))
        return out
