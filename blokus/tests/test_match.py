"""全 AI 比賽：抽四個人格互比，結果餵進排行榜。"""
import random

from config import PERSONALITY_ORDER
from game import Game
from match import play_match, run_league
from records import Records


def test_setup_match_seats_four_distinct_personalities():
    """6 個人格抽 4 個，所以每局剛好輪空兩個人格。"""
    pool = set(PERSONALITY_ORDER)
    assert len(pool) == 6
    benched = set()
    for seed in range(60):
        g = Game(random.Random(seed))
        g.setup_match()
        keys = [g.owner_key[o] for o in range(4)]
        assert len(set(keys)) == 4, keys
        assert set(keys) <= pool, keys
        assert len(pool - set(keys)) == 2, "剛好輪空兩個人格"
        benched |= pool - set(keys)
        # 每個 seat 的 brain 都是對應的人格
        for o in range(4):
            assert g.brains[o].key == g.owner_key[o]
        assert sorted(g.colors.values()) == ["blue", "green", "red", "yellow"]
        assert len(set(g.turn_order)) == 4
    assert benched == pool, benched


def test_setup_match_rejects_duplicate_personalities():
    g = Game(random.Random(0))
    for bad in (["wolf", "wolf", "fox", "intruder"], ["wolf", "fox"],
                ["wolf", "fox", "intruder", "optimizer", "chess"]):
        try:
            g.setup_match(bad)
        except ValueError:
            continue
        raise AssertionError("should have refused %r" % (bad,))


def test_play_match_ends_and_scores_by_personality():
    g = Game(random.Random(4))
    g.setup_match()
    standings = play_match(g, random.Random(4))
    assert g.state == "GAME_OVER"
    assert len(standings) == 4
    assert sorted(k for k, _ in standings) == sorted(g.owner_key[o] for o in range(4))
    assert all(isinstance(r, int) and 0 <= r <= 89 for _, r in standings)
    # 每一手都合法：逐手檢查角對角規則
    rng = random.Random(9)
    g = Game(rng)
    g.setup_match()
    g.start()
    seen = 0
    while g.state == "PLAYING" and seen < 12:
        import ai
        owner = g.current_owner()
        before = {i for i in range(400) if g.board.grid[i] == owner}
        move = ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], rng, other_brains=g.brains,
                              must_cover=g.must_cover(owner), reach=g.reach(owner))
        if move is None:
            g.act_pass()
            continue
        if g.placed[owner] > 0:
            from board import neighbors_of
            from pieces import MASTER
            name, oi, x, y = move
            cells = {x + dx + (y + dy) * 20
                     for dx, dy in MASTER[name]["orientations"][oi]}
            for i in cells:
                for n in neighbors_of()[i]:
                    assert n not in before, (name, oi, x, y, owner)
            assert any((i % 20 + dx, i // 20 + dy) in {(b % 20, b // 20) for b in before}
                       for i in cells for dx in (-1, 1) for dy in (-1, 1)), move
            seen += 1
        g.act(*move)


def test_league_records_only_personality_keys(tmp_path):
    """排行榜只會多出人格的條目——全 AI 比賽裡沒有玩家。"""
    rec = Records(str(tmp_path / "records.json"))
    rows = run_league(6, seed=3, records=rec)
    assert len(rows) == 6
    keys = {k for row in rows for k, _ in row}
    assert keys <= set(PERSONALITY_ORDER), keys
    assert "player" not in rec.entries
    assert set(rec.entries) == keys
    total = sum(r["games"] for r in rec.entries.values())
    assert total == 4 * 6, total
    # 每局最多 4 分、最少 1 分（並列名次會讓總分低於平均 2.5，所以下界用 1）
    assert all(r["games"] <= r["total_points"] <= 4 * r["games"]
               for r in rec.entries.values()), rec.entries
    for _key, games, pts, rem in rec.rows():
        assert games > 0 and 0 < pts <= 4 and 0 <= rem <= 89 * games


def test_run_league_without_records_touches_no_file(tmp_path):
    rows = run_league(2, seed=1)
    assert len(rows) == 2
    assert not list(tmp_path.iterdir())
