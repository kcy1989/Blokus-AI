"""Tests for the persistent leaderboard: ranking, points, running averages."""
import os

from records import POINTS_FOR_RANK, Records, rank_by_remaining


def test_ranking_by_fewest_remaining():
    rows = rank_by_remaining([("a", 30), ("b", 10), ("c", 20), ("d", 0)])
    assert rows == [("d", 1, 4), ("b", 2, 3), ("c", 3, 2), ("a", 4, 1)]


def test_equal_remaining_shares_a_place_and_skips_the_next():
    """Ties share a rank; the places after them skip, as in 1, 1, 3, 4."""
    rows = rank_by_remaining([("a", 10), ("b", 30), ("c", 10), ("d", 20)])
    got = {k: (r, p) for k, r, p in rows}
    assert got["a"] == (1, 4)
    assert got["c"] == (1, 4)
    assert got["d"] == (3, 2)
    assert got["b"] == (4, 1)


def test_all_four_equal_shares_first():
    rows = rank_by_remaining([("a", 5), ("b", 5), ("c", 5), ("d", 5)])
    assert {p for _, _, p in rows} == {4}
    assert {r for _, r, _ in rows} == {1}


def test_points_cover_first_to_last():
    assert POINTS_FOR_RANK == (4, 3, 2, 1)
    rows = rank_by_remaining([("a", 40), ("b", 30), ("c", 20), ("d", 10)])
    assert [p for _, _, p in rows] == [4, 3, 2, 1]


def _rec(tmpdir):
    return Records(os.path.join(tmpdir, "records.json"))


def test_averages_accumulate_across_games(tmp_path):
    r = _rec(str(tmp_path))
    # game 1: player 0, wolf 10, chess 20, fox 30  -> 4, 3, 2, 1
    r.record([("player", 0), ("wolf", 10), ("chess", 20), ("fox", 30)])
    # game 2: chess 0, player 10, fox 20, wolf 30  -> chess 4, player 3, fox 2, wolf 1
    r.record([("player", 10), ("wolf", 30), ("chess", 0), ("fox", 20)])
    rows = {k: (g, ap, ar) for k, g, ap, ar in r.rows()}
    assert rows["player"] == (2, 3.5, 5.0)     # 4 then 3, left 0 then 10
    assert rows["wolf"] == (2, 2.0, 20.0)      # 3 then 1, left 10 then 30
    assert rows["chess"] == (2, 3.0, 10.0)     # 2 then 4, left 20 then 0
    assert rows["fox"] == (2, 1.5, 25.0)       # 1 then 2, left 30 then 20


def test_leaderboard_sorted_by_average_points(tmp_path):
    r = _rec(str(tmp_path))
    r.record([("player", 40), ("wolf", 20), ("chess", 30), ("fox", 10)])
    r.record([("player", 50), ("wolf", 30), ("chess", 20), ("fox", 0)])
    order = [(k, ap) for k, _g, ap, _ar in r.rows()]
    # fox took first both times (4+4), player was last both times (1+1)
    assert order[0] == ("fox", 4.0)
    assert order[-1] == ("player", 1.0)
    points = [ap for _k, ap in order]
    assert points == sorted(points, reverse=True)


def test_average_remaining_breaks_a_points_tie(tmp_path):
    r = _rec(str(tmp_path))
    # game 1 ordered A<B<C<D, game 2 exactly reversed, so everyone scores 2.5
    r.record([("player", 0), ("wolf", 5), ("chess", 10), ("fox", 15)])
    r.record([("player", 70), ("wolf", 60), ("chess", 50), ("fox", 40)])
    assert {ap for _k, _g, ap, _ar in r.rows()} == {2.5}
    # but the averages differ: fox 27.5 left, the player 35
    order = [(k, ar) for k, _g, _p, ar in r.rows()]
    assert order == [("fox", 27.5), ("chess", 30.0),
                     ("wolf", 32.5), ("player", 35.0)]



def test_unequal_averages_rank_by_points(tmp_path):
    r = _rec(str(tmp_path))
    r.record([("player", 0), ("wolf", 10), ("chess", 20), ("fox", 30)])
    r.record([("player", 20), ("wolf", 0), ("chess", 30), ("fox", 10)])
    points = {k: ap for k, _g, ap, _ar in r.rows()}
    assert points["player"] == 3.0     # 4 then 2
    assert points["wolf"] == 3.5      # 3 then 4
    assert points["fox"] == 2.0       # 1 then 3
    assert points["chess"] == 1.5     # 2 then 1
    assert [k for k, _g, _p, _ar in r.rows()][0] == "wolf"


def test_records_persist_across_sessions(tmp_path):
    path = str(tmp_path / "records.json")
    Records(path).record([("player", 0), ("wolf", 10),
                          ("chess", 20), ("fox", 30)])
    again = Records(path)
    assert again.entries["player"]["games"] == 1
    assert again.entries["player"]["total_points"] == 4
    rows = {k: g for k, g, _p, _r in again.rows()}
    assert rows == {"player": 1, "wolf": 1, "chess": 1, "fox": 1}


def test_corrupt_file_is_ignored(tmp_path):
    path = str(tmp_path / "records.json")
    with open(path, "w") as f:
        f.write("{not json")
    r = Records(path)
    assert r.rows() == []
    r.record([("player", 1), ("wolf", 2), ("chess", 3), ("fox", 4)])
    assert len(r.rows()) == 4


def test_last_game_is_available_for_display(tmp_path):
    r = _rec(str(tmp_path))
    assert r.last == []
    rows = r.record([("player", 0), ("wolf", 10), ("chess", 20), ("fox", 30)])
    assert r.last == rows
    assert {k: p for k, _rk, p in r.last}["player"] == 4


# ------------------------------------------- two seats on the same contestant

def test_rank_rows_keeps_each_seat_s_own_remaining():
    from records import rank_rows
    rows = rank_rows([("a", 10), ("a", 30), ("b", 20), ("c", 0)])
    # best first: fewer squares left is better
    assert rows == [("c", 1, 4, 0), ("a", 2, 3, 10), ("b", 3, 2, 20),
                    ("a", 4, 1, 30)]
    # the same key twice keeps two distinct rem values, which a lookup by key
    # could not do
    by_key = [r for r in rows if r[0] == "a"]
    assert sorted(r[3] for r in by_key) == [10, 30]


def test_a_repeated_key_is_recorded_as_two_appearances(tmp_path):
    r = Records(str(tmp_path / "records.json"))
    r.record([("wolf", 10), ("wolf", 30), ("fox", 20), ("human", 0)])
    # human(0) 1st, wolf(10) 2nd, fox(20) 3rd, wolf(30) 4th
    assert r.entries["wolf"] == {"games": 2, "total_points": 4.0,
                                 "total_remaining": 40.0}
    # both seats' squares, not one seat's - the bug was `dict(standings)[key]`,
    # which kept only the last
    assert r.entries["wolf"]["total_remaining"] == 40.0
    assert r.entries["fox"]["games"] == 1
    rows = {k: (g, ap, ar) for k, g, ap, ar in r.rows()}
    # averages are per appearance, so they stay on the same scale as everyone
    assert rows["wolf"] == (2, 2.0, 20.0)
    assert rows["human"] == (1, 4.0, 0.0)
    assert 0.0 <= rows["fox"][1] <= 4.0


def test_the_returned_rows_are_one_per_seat_not_one_per_key(tmp_path):
    r = Records(str(tmp_path / "records.json"))
    out = r.record([("a", 10), ("a", 30), ("b", 20), ("c", 0)])
    assert len(out) == 4
    assert [p for _k, _r, p in out] == [4, 3, 2, 1]
    assert r.last == out


def test_a_repeated_key_still_ends_the_game_once(tmp_path):
    r = Records(str(tmp_path / "records.json"))
    r.record([("a", 5), ("a", 5), ("b", 5), ("c", 5)])
    assert r.entries["a"]["games"] == 2
    assert r.entries["a"]["total_points"] == 8.0
    ranks = {k: rank for k, rank, _p in r.last}
    assert set(ranks.values()) == {1}, ranks
