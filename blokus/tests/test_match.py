"""All-AI matches: draw four personalities to compete, and feed the results
into the leaderboard."""
import random

import ai.registry as REG
import seats as seats_mod
from config import PERSONALITY_ORDER
from game import Game
from match import league_options, play_match, run_league
from records import Records


def test_setup_match_seats_four_distinct_personalities():
    """4 are drawn from the 7 personalities, so exactly three sit out each game."""
    pool = set(PERSONALITY_ORDER)
    assert len(pool) == 7
    benched = set()
    for seed in range(60):
        g = Game(random.Random(seed))
        g.setup_match()
        keys = [g.owner_key[o] for o in range(4)]
        assert len(set(keys)) == 4, keys
        assert set(keys) <= pool, keys
        assert len(pool - set(keys)) == 3, "exactly three personalities sat out"
        benched |= pool - set(keys)
        # each seat's brain is the matching personality
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
    assert sorted(k for k, _c, _r, _rk, _p in standings) == sorted(
        g.owner_key[o] for o in range(4))
    assert sorted(c for _k, c, _r, _rk, _p in standings) == [
        "blue", "green", "red", "yellow"]
    assert all(isinstance(r, int) and 0 <= r <= 89
               for _k, _c, r, _rk, _p in standings)
    # rank and points, per seat
    assert sorted(rk for _k, _c, _r, rk, _p in standings) == [1, 2, 3, 4]
    assert sorted(p for _k, _c, _r, _rk, p in standings) == [1, 2, 3, 4]
    # every move is legal: check the corner-contact rule move by move
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
                              g.brains[owner], rng, other_brains=g.brain_map(),
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
    """A league of the seven personalities, with no player in it.

    Run without the network options on purpose: the default fourteen would load
    six checkpoints and a trained policy - seven files, seven torch imports -
    and this test is about the leaderboard, not about torch.
    """
    rec = Records(str(tmp_path / "records.json"))
    rows = run_league(6, seed=3, records=rec, options=PERSONALITY_ORDER)
    assert len(rows) == 6
    keys = {k for game in rows for k, _c, _r, _rk, _p in game}
    assert keys <= set(PERSONALITY_ORDER), keys
    assert "player" not in rec.entries
    assert set(rec.entries) == keys
    total = sum(r["games"] for r in rec.entries.values())
    assert total == 4 * 6, total
    # each game is worth at most 4 points and at least 1 (ties push the total
    # below the 2.5 average, so the lower bound is 1)
    assert all(r["games"] <= r["total_points"] <= 4 * r["games"]
               for r in rec.entries.values()), rec.entries
    for _key, games, pts, rem in rec.rows():
        assert games > 0 and 0 < pts <= 4 and 0 <= rem <= 89 * games


def test_run_league_without_records_touches_no_file(tmp_path):
    rows = run_league(2, seed=1)
    assert len(rows) == 2
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------- the all-AI league

def test_the_league_offers_every_automated_option_and_no_humans():
    """What the league draws from, and what it never draws.

    Stage 8: the four numbers this used to type (14, 7, 6, 1) are now read
    from the registry, so a fifteenth seat changes them without touching the
    file. Two things still have to be *true* rather than equal to a roster:
    `match.league_options` and `ai.registry.automated_options` are the same
    list, and not one human kind can appear in it.
    """
    from match import league_options
    opts = league_options()
    assert list(opts) == list(REG.automated_options())
    assert len(set(opts)) == len(opts)
    families = {"personality": 0, "imitation": 0, "rl": 0}
    for e in REG.ENTRIES:
        families[e["family"]] += 1
    kinds = [seats_mod.kind_of(k, include_humans=False) for k in opts]
    assert kinds.count(seats_mod.KIND_AI) == families["personality"]
    assert kinds.count(seats_mod.KIND_IMITATION) == families["imitation"]
    assert kinds.count(seats_mod.KIND_RL) == families["rl"]
    assert not any(seats_mod.is_human_kind(k) for k in kinds)
    assert not any(seats_mod.is_human_kind(k) for k in kinds)


def test_every_league_seat_is_drawn_independently_so_options_repeat():
    """The old league drew four *distinct* personalities, which meant the three
    seats always showed a personality nobody else had. Drawing independently is
    what lets a league answer "is the imitation stronger"."""
    from match import league_options
    opts = [k for k in league_options() if not seats_mod.is_imitation_key(k)]
    seen_repeat = 0
    for seed in range(20):
        rows = run_league(1, seed=seed, options=opts)
        keys = [k for k, _c, _r, _rk, _p in rows[0]]
        assert set(keys) <= set(opts)
        assert len(keys) == 4
        if len(set(keys)) < 4:
            seen_repeat += 1
    assert seen_repeat > 0, "40 draws with 7 options and never a repeat"


def test_a_league_row_names_both_the_option_and_the_colour():
    """Needed to tell two seats apart when the same option is drawn twice, and
    to work out who played what later."""
    from match import league_options
    from records import POINTS_FOR_RANK
    opts = [k for k in league_options() if not seats_mod.is_imitation_key(k)]
    rows = run_league(3, seed=11, options=opts)
    for row in rows:
        assert len(row) == 4
        # Ranking is `records.rank_rows`, and this restates that rule from the
        # row's own `remaining` rather than copying its answer: competition
        # ranking over (remaining, key), a tie shares a place, the place after a
        # tie skips, and points are `POINTS_FOR_RANK[rank - 1]`.
        #
        # Measured on this pool, seed 11, game 1: `fox/blue` and `intruder/red`
        # both finish on 30 cells, both take rank 3 and **2 points**, there is
        # no 4th place, and the 1 point 4th would have carried goes to nobody -
        # 4 + 3 + 2 + 2 = 11, not 10. An earlier form of this assertion demanded
        # `[1, 2, 3, 4]` with `5 - rank`, which forbids a tie outright and was
        # true only for whichever game this seed happened to draw.
        standing = sorted(row, key=lambda r: (r[2], r[0]))
        prev_rem, prev_rank = None, 0
        for pos, (_k, _c, rem, rk, p) in enumerate(standing):
            if rem != prev_rem:
                prev_rank = pos + 1
                prev_rem = rem
            assert rk == prev_rank, (pos, rem, rk, prev_rank)
            assert p == POINTS_FOR_RANK[rk - 1]
        keys = [k for k, _c, _r, _rk, _p in row]
        colours = [c for _k, c, _r, _rk, _p in row]
        assert set(keys) <= set(opts)
        # four seats, four distinct colours, every pair identified
        assert sorted(colours) == ["blue", "green", "red", "yellow"]
        assert len(set(zip(keys, colours))) == 4


def test_every_league_game_deals_four_distinct_colours():
    """Colours come from the same `setup_seats` the UI uses, left on "random",
    so a league game is laid out exactly like a played one."""
    from match import league_options
    opts = [k for k in league_options() if not seats_mod.is_imitation_key(k)]
    for seed in range(25):
        rows = run_league(1, seed=seed, options=opts)
        assert sorted(c for _k, c, _r, _rk, _p in rows[0]) == \
            ["blue", "green", "red", "yellow"], seed


def test_the_league_draws_its_seats_through_setup_seats():
    """The opening player is drawn inside `setup_seats`, so a league game and a
    game a person set up go first the same way."""
    import random as _r

    from game import Game
    from match import league_options
    opts = [k for k in league_options() if not seats_mod.is_imitation_key(k)]
    firsts = set()
    for seed in range(40):
        rng = _r.Random(seed)
        g = Game(rng)
        keys = [rng.choice(opts) for _ in range(4)]
        g.setup_seats(keys, [None] * 4, rng)
        firsts.add(g.turn_order[0])
    assert firsts == {0, 1, 2, 3}, firsts


def test_only_imitation_options_can_be_asked_for():
    """The six imitation checkpoints and nothing else, in key order.

    Spelled out rather than derived, for the same reason `GOLDEN_POOL` is:
    since plan9a stage 6 the family is the registry's `family` field and
    `IMITATION_KEYS` only ever held H-C2's three steps - four seats share step
    1000 now - so deriving this from the code table would assert that the code
    table still knows the family, which is the thing beta stopped claiming.
    Key order is not step order: `hc_10000` sorts before `hc_2000`.
    """
    steps = [k for k in league_options() if seats_mod.is_imitation_key(k)]
    assert steps == ["hc_10000", "hc_2000", "rl_b1000_0k", "rl_h1000_0k",
                     "rl_i1000_0k", "rl_o1000_0k"]
