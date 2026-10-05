"""The per-move log a "human (record)" seat writes, and reading it back.

Two properties matter and they are different kinds of thing:

  * a log exists only when a recording seat is playing, and it stores the
    **position**, never the features - so a log stays usable by whatever
    featurises it later;
  * a log can be turned back into training rows, and those rows reproduce the
    same legal mask and carry a target inside it. That is the same invariant the
    trainer asserts on every batch, checked here on data that came from a human
    rather than from a shard.

The last test plays a game with **two** recording seats, because "how many seats
are recording" is one of the fields the log has to carry and it is easy to get
right for one and wrong for two.
"""
import json
import os
import random

import numpy as np
import pytest

import ai
import engine
import gamelog
import seats as seats_mod
from config import CLOCKWISE_OWNERS
from engine import ENGINE_VERSION
from game import Game
from rl.features import FEATURE_VERSION


def make_game(seed=0, options=None, colours=None):
    rng = random.Random(seed)
    g = Game(rng)
    g.setup_seats(options or ["chess"] * 4, colours, rng)
    g.start()
    return g, rng


def brain_for(g):
    """Any brain, borrowed.

    A "human (record)" seat has no brain - that is what makes it a person - so
    a test that plays on its behalf has to borrow one from a seat that has one.
    The log cannot tell the difference: it sees a legal move either way.
    """
    brains = g.brain_map()
    if not brains:
        raise AssertionError("no AI seat to borrow a brain from")
    return brains[min(brains)]


def play_one(g, rng):
    """Play a single turn for whoever is to move."""
    owner = g.current_owner()
    move = ai.choose_move(g.board, g.hands[owner].names, owner,
                          brain_for(g), rng, other_brains=None,
                          must_cover=g.must_cover(owner),
                          reach=g.reach(owner))
    if move is None:
        g.act_pass()
        return owner, None
    g.act(*move)
    return owner, move


def human_moves(g, rng, seat, limit=200):
    """Play `limit` of `seat`'s turns using a borrowed brain, so the log sees a
    real move sequence rather than a fixture."""
    played = 0
    while g.state == "PLAYING" and played < limit:
        owner, move = play_one(g, rng)
        if move is not None and owner == seat:
            played += 1
    return played


# ------------------------------------------------------------ when it writes

def test_no_log_is_opened_without_a_recording_seat():
    g, _rng = make_game(1, ["chess"] * 4)
    assert g.log is None
    g, _rng = make_game(1, ["human", "human", "chess", "fox"])
    assert g.log is None, "a plain human seat records nothing"
    g, _rng = make_game(1, ["chess", "human_log", "fox", "wolf"])
    assert g.log is not None
    assert g.log.recording_seats == [1]
    assert g.log.meta["n_recording_seats"] == 1


def test_a_game_with_no_recording_seat_writes_nothing(tmp_path):
    """Not an empty file, not a header: no file."""
    g, rng = make_game(2, ["chess"] * 4)
    human_moves(g, rng, 0, 8)
    assert g.log is None
    before = set(os.listdir(gamelog.LOG_DIR)) if os.path.isdir(gamelog.LOG_DIR) \
        else set()
    # four humans, so there is no brain to borrow and no log to write either
    g2, rng2 = make_game(2, ["human"] * 4)
    assert g2.log is None
    assert g2.brain_map() == {}
    after = set(os.listdir(gamelog.LOG_DIR)) if os.path.isdir(gamelog.LOG_DIR) \
        else set()
    assert before == after


def test_the_log_records_the_position_not_the_features():
    g, rng = make_game(3, ["chess", "human_log", "fox", "wolf"])
    log = g.log
    assert log.records == []
    while g.state == "PLAYING" and not log.records:
        play_one(g, rng)
    assert log.records, "the recording seat never moved"
    rec = log.records[0]
    for key in ("own_bits", "hand_bits", "stuck", "to_move", "turn_order",
                "legal", "action", "turn", "seat", "turn_pos", "color",
                "seat_kinds", "n_recording_seats", "engine_version",
                "feature_version"):
        assert key in rec, key
    # No derived features on a line. The check is on the keys, not on the text:
    # `feature_version` is a field the spec asks for and its name contains the
    # word.
    assert set(rec) == {
        "turn", "seat", "turn_pos", "color", "own_bits", "hand_bits",
        "stuck", "to_move", "turn_order", "legal", "action",
        "engine_version", "feature_version", "n_recording_seats",
        "recording_seats", "seat_kinds", "seat_options", "colors",
    }, sorted(rec)
    # the position is four masks, four hands and four latches - not 27 planes
    assert len(rec["own_bits"]) == 4
    assert len(rec["hand_bits"]) == 4
    assert len(rec["stuck"]) == 4
    assert rec["engine_version"] == ENGINE_VERSION
    assert rec["feature_version"] == FEATURE_VERSION


def test_the_meta_fields_name_the_game_the_row_came_from():
    g, rng = make_game(4, ["human_log", "chess", "human_log", "fox"],
                       ["blue", "green", "red", "yellow"])
    human_moves(g, rng, 0, 6)
    human_moves(g, rng, 2, 6)
    rec = g.log.records[0]
    assert rec["seat_kinds"] == ["human_log", "ai", "human_log", "ai"]
    assert rec["seat_options"] == ["human_log", "chess", "human_log", "fox"]
    assert rec["colors"] == ["blue", "green", "red", "yellow"]
    assert rec["turn_order"] == list(g.turn_order)
    assert rec["n_recording_seats"] == 2
    assert rec["recording_seats"] == [0, 2]


# ----------------------------------------------------------- back to rows

def _drive_until_logged(g, rng, limit=400):
    while g.state == "PLAYING" and not g.log.records and limit:
        limit -= 1
        play_one(g, rng)
    return g.log.records


def test_a_logged_row_reproduces_its_mask_and_carries_a_legal_target():
    g, rng = make_game(5, ["chess", "human_log", "fox", "wolf"])
    recs = _drive_until_logged(g, rng)
    assert recs
    for rec in recs:
        mask = gamelog.row_mask(rec)
        label = gamelog.row_label(rec)
        assert mask.dtype == bool and mask.shape == (91 * 20 * 20,)
        assert mask.any(), "a recorded move means at least one legal slot"
        assert mask[label], "the target slot is outside the row's own mask"
        x = gamelog.row_features(rec)
        assert x.shape == (1, 27, 20, 20)
        assert np.isfinite(x).all()


def test_the_logged_action_is_among_the_logged_legal_slots():
    """The stored legal set and the stored action must agree, or the log cannot
    be checked against its own position."""
    from rl.actions import real_to_view

    g, rng = make_game(6, ["human_log", "chess", "fox", "wolf"])
    recs = _drive_until_logged(g, rng)
    for rec in recs:
        p = CLOCKWISE_OWNERS.index(rec["to_move"])
        legal = set(real_to_view(np.asarray(rec["legal"], dtype=np.int64), p)
                    .tolist())
        assert int(real_to_view(int(rec["action"]), p)) in legal


def test_two_recording_seats_in_one_game():
    """The round trip, with both a "human (record)" seat at once."""
    g, rng = make_game(7, ["human_log", "chess", "human_log", "fox"])
    assert g.recording_seats() == [0, 2]
    human_moves(g, rng, 0, 4)
    human_moves(g, rng, 2, 4)
    rows = g.log.rows()
    assert rows, "neither recording seat produced a row"
    seats_seen = {r["seat"] for r in rows}
    assert seats_seen == {0, 2}, seats_seen
    for rec in rows:
        assert rec["n_recording_seats"] == 2
        mask = gamelog.row_mask(rec)
        label = gamelog.row_label(rec)
        assert mask[label], (rec["seat"], rec["turn"])
        assert gamelog.row_features(rec).shape == (1, 27, 20, 20)


def test_a_pass_is_logged_but_is_not_a_training_row():
    g, rng = make_game(8, ["chess", "human_log", "fox", "wolf"])
    log = g.log
    log.record_move(log.recording_seats[0], "X5", 0, 0, 0)  # synthetic
    log.record_pass(log.recording_seats[0])
    assert len(log.records) == 2
    assert log.records[1]["action"] is None
    assert len(log.rows()) == 1
    # a pass has no slot to put a label in, so it must not become a row
    assert all(r["action"] is not None for r in log.rows())


def test_only_recording_seats_are_written():
    g, rng = make_game(9, ["chess", "human_log", "fox", "wolf"])
    while g.state == "PLAYING":
        before = len(g.log.records)
        owner, move = play_one(g, rng)
        grew = len(g.log.records) > before
        # a pass is recorded too, so "grew" tracks the seat, not the move
        assert grew == (owner == 1), (owner, grew, move)


# ------------------------------------------------------------ on disk

def test_the_log_writes_and_reads_back_as_jsonl(tmp_path):
    path = str(tmp_path / "game.jsonl")
    g, rng = make_game(10, ["human_log", "chess", "fox", "wolf"])
    human_moves(g, rng, 0, 5)
    assert g.log.write(path) == path
    assert os.path.exists(path)
    recs = gamelog.read_log(path)
    assert recs == g.log.records
    assert all(json.loads(json.dumps(r, default=str)) is not None for r in recs)
    for rec in recs:
        assert gamelog.row_mask(rec).any()
        assert gamelog.row_mask(rec)[gamelog.row_label(rec)]


def test_the_default_path_is_under_the_gitignored_data_directory():
    path = gamelog.default_log_path("probe")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # <root>/data/humanlog/... - inside the repository but inside `data/`,
    # which is what `.gitignore` excludes.
    assert os.path.abspath(path).startswith(os.path.join(root, "data") + os.sep)
    assert os.path.basename(os.path.dirname(path)) == "humanlog"
    assert os.path.isdir(gamelog.LOG_DIR)
    # Asked git rather than reading `.gitignore`, because "this path is
    # ignored" is a property of the *path*, not of a spelling. Matching the
    # rules as text asserted that some rule read `data/`, which is one way to
    # ignore `data/` and stops being true the moment the directory is excluded
    # a different way - including the way it has to be once a file inside it
    # needs tracking, since excluding the directory itself makes every
    # negation below it dead. The question the name asks is whether a played
    # game can commit its log, and only git answers that.
    import subprocess
    rel = os.path.relpath(os.path.abspath(path), root)
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=root,
                       capture_output=True)
    # returncode 0 = ignored, 1 = not ignored, anything else = git could not
    # answer (no repo, no git) and the path assertions above are all that is
    # left to stand on.
    if r.returncode in (0, 1):
        assert r.returncode == 0, "%s would be committed" % rel


def test_action_index_round_trips_through_the_engine():
    from rl.actions import index_to_move, move_to_index
    name, oi, base = "L5", 3, 17 * 20 + 4
    idx = gamelog.action_index_of(name, oi, base % 20, base // 20)
    piece, o2, b2 = index_to_move(idx)
    assert engine.PIECE_ORDER[piece] == name
    assert o2 == oi and b2 == base
    assert move_to_index((piece, o2, b2)) == idx

def test_starting_a_game_with_a_recording_seat_creates_no_empty_file(tmp_path):
    """The file is written when there is something to write, so a recording
    game that nobody moved in does not leave a zero-byte file behind."""
    import pygame

    from ui import UI, load_cjk_font_path
    pygame.init()
    before = set(os.listdir(gamelog.LOG_DIR)) if os.path.isdir(gamelog.LOG_DIR) \
        else set()
    u = UI(None, load_cjk_font_path(), scale=1.0)
    u.game = Game(random.Random(0))
    u.seat_keys = ["human_log", "chess", "fox", "wolf"]
    u.seat_colours = [seats_mod.RANDOM] * 4
    u.seat_pick = None
    u.start_configured_game()
    assert u.state == "PLAYING"
    after = set(os.listdir(gamelog.LOG_DIR)) if os.path.isdir(gamelog.LOG_DIR) \
        else set()
    assert after == before, after - before

    # and once the recording seat has moved, the file appears and is not empty
    u.game.log.records.append(u.game.log._snapshot(0))
    u.flush_log()
    written = set(os.listdir(gamelog.LOG_DIR)) - before
    assert written, "flush_log wrote nothing after a move was recorded"
    path = gamelog.LOG_DIR + "/" + sorted(written)[0]
    assert os.path.getsize(path) > 0
    recs = gamelog.read_log(path)
    assert recs and recs[-1]["seat"] == 0
    os.remove(path)
