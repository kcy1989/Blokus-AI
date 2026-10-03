"""The per-move log a "human (record)" seat produces.

A game is written only when at least one seat is "human (record)". Without one
there is no file at all - not an empty one, not a header - because a log nobody
asked for is just disk.

What is stored is the **position**, never the features. The features are a
function of the position, so storing them would freeze one particular
implementation of "what a position looks like" into the data and make the log
unusable by anything that featurises differently. Storing the position means the
log can be turned into training rows years later by whatever
`rl.caches.features_27` says at the time - which is exactly what `rows()` does.

One JSON object per line, under `data/`, which is already gitignored. Four
owner bitmasks, four hands, four stuck latches, the turn order, the legal action
set and the action actually taken. The legal set is stored rather than
recomputed so that a log stays checkable against the position it was taken from
even if the rules are ever revised; the round-trip test asserts the two agree.
"""
import json
import os
import time

from config import B, CLOCKWISE_OWNERS, PROJECT_DIR

import engine
from rl.actions import (legal_indices, legal_mask_view, move_to_index,
                        real_to_view)
from rl.caches import features_27

# Both are recorded on every line. A log whose engine version does not match
# the code reading it is a log whose legality may be a different question, and
# a log whose feature version does not match is a log whose columns are a
# different length - both fail loudly rather than silently.
from engine import ENGINE_VERSION
from rl.features import FEATURE_VERSION

LOG_DIR = os.path.join(PROJECT_DIR, "data", "humanlog")


def default_log_path(run_id=None):
    d = LOG_DIR
    os.makedirs(d, exist_ok=True)
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(d, "game-%s.jsonl" % run_id)


def action_index_of(name, oi, x, y):
    """`(name, oi, x, y)` -> the real action index the caches use."""
    return move_to_index((engine.PIECE_ORDER.index(name), int(oi),
                          int(x) + int(y) * B))


def state_from_row(row):
    """A logged row back into the `engine.State` it was taken from."""
    return engine.State(
        own_bits=tuple(int(b) for b in row["own_bits"]),
        hand_bits=tuple(int(h) for h in row["hand_bits"]),
        turn_order=tuple(int(t) for t in row["turn_order"]),
        to_move=int(row["to_move"]),
        stuck=tuple(bool(s) for s in row["stuck"]))


def row_mask(row):
    """The legal-action mask of a logged row, recomputed from its position."""
    return legal_mask_view(state_from_row(row))


def row_features(row):
    """The 27 channels a logged row featurises to, recomputed not stored."""
    return features_27([state_from_row(row)])


def row_label(row):
    """The view-frame action index, which is what the trainer's label is.

    `real_to_view` under the mover's seat, the same one line
    `rl.caches.build_label_cache` writes.
    """
    return int(real_to_view(int(row["action"]),
                            CLOCKWISE_OWNERS.index(int(row["to_move"]))))


class GameLog:
    """Collects one game's recorded moves. Owned by the `Game` that has a
    recording seat and written nowhere until `write()` is called."""

    def __init__(self, game, recording_seats, run_id=None):
        self._game = game
        self.recording_seats = [int(o) for o in recording_seats]
        self.run_id = run_id
        # The destination is fixed up front so `write()` is not the first thing
        # that decides it, and so a caller can flush a half-played game.
        # Nothing is created on disk here: the file appears on the first write.
        self.path = default_log_path(run_id)
        # Fixed for the whole game and copied onto every line, so a single line
        # is self-describing: which seat kinds were in play, who was recording,
        # and which code produced it.
        self.meta = {
            "engine_version": ENGINE_VERSION,
            "feature_version": FEATURE_VERSION,
            "n_recording_seats": len(self.recording_seats),
            "recording_seats": list(self.recording_seats),
            "seat_kinds": [game.seat_kinds[o] for o in range(4)],
            "seat_options": [game.owner_key[o] for o in range(4)],
            "colors": [game.colors[o] for o in range(4)],
            "turn_order": [int(o) for o in game.turn_order],
        }
        self.records = []

    # -- collection ----------------------------------------------------------

    def _snapshot(self, owner):
        """The position as it stands, plus the bookkeeping the spec asks for.

        Taken before the move is applied, which is the whole point: a training
        row has to be the position the choice was made *in*.
        """
        g = self._game
        state = engine.State(
            own_bits=tuple(int(b) for b in g.board.owner_bits),
            hand_bits=tuple(self._hands()),
            turn_order=tuple(int(o) for o in g.turn_order),
            to_move=int(owner),
            stuck=tuple(bool(s) for s in g.stuck))
        rec = {
            "turn": int(g.turn_count),
            "seat": int(owner),
            "turn_pos": g.turn_order.index(int(owner)),
            "color": g.colors[owner],
            "own_bits": [int(b) for b in state.own_bits],
            "hand_bits": [int(h) for h in state.hand_bits],
            "stuck": [bool(s) for s in state.stuck],
            "to_move": int(state.to_move),
            "turn_order": list(state.turn_order),
            "legal": [int(i) for i in legal_indices(state, owner)],
        }
        rec.update(self.meta)
        return rec

    def record_move(self, owner, name, oi, x, y):
        """One placement. Only recorded if a recording seat is playing."""
        if int(owner) not in self.recording_seats:
            return None
        rec = self._snapshot(owner)
        rec["action"] = action_index_of(name, oi, x, y)
        self.records.append(rec)
        return rec

    def record_pass(self, owner):
        """A pass. Logged for completeness, with no action.

        A pass is not a training row: the policy only ever chooses placements,
        and there is no slot to put a pass in.
        """
        if int(owner) not in self.recording_seats:
            return None
        rec = self._snapshot(owner)
        rec["action"] = None
        self.records.append(rec)
        return rec

    # -- output --------------------------------------------------------------

    def rows(self):
        """The recorded placements as training rows.

        A pass is skipped, so `len(rows())` is the number of placements the
        recorded seats actually made. Each row keeps its position and its
        action, and `row_mask` / `row_label` derive the rest.
        """
        return [r for r in self.records if r["action"] is not None]

    def write(self, path=None):
        """Append this game's lines to a JSONL file. Returns the path."""
        target = path or self.path or default_log_path(self.run_id)
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, sort_keys=True, default=str))
                fh.write("\n")
        self.path = target
        return target

    # the Game is attached after construction so the log can be built before
    # the brains exist, which is the order `setup_seats` runs in
    def _hands(self):
        from rl.imitation import hand_bits_from_names
        return tuple(hand_bits_from_names(self._game.hands[o].names)
                     for o in range(4))


def read_log(path):
    """Every line of a log, as dicts."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out