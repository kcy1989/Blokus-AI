"""Headless simulation: 100 all-AI games plus end/draw rules."""
import random

import ai
from game import Game
from board import Board, neighbors_of
from pieces import MASTER
from config import (CLOCKWISE_OWNERS, COLORS, OWNER_CORNER, PLAYER_OWNER)


def run_game(rng, on_move=None):
    g = Game(rng)
    g.set_player_color(rng.choice(list(COLORS)))
    g.start()
    while g.state == "PLAYING":
        owner = g.current_owner()
        # snapshot before the move: a piece must never be judged against its
        # own cells, only against stones that were already on the board
        before = {i for i in range(400) if g.board.grid[i] == owner}
        move = ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], rng, other_brains=g.brains,
                              must_cover=g.must_cover(owner),
                              reach=g.reach(owner),
                              other_must_cover={o: g.must_cover(o) for o in range(4)},
                              other_reach={o: g.reach(o) for o in range(4)})
        if move:
            if on_move is not None:
                on_move(g, owner, move, before)
            g.act(*move)
        else:
            g.act_pass()
    return g


def test_turn_order_is_clockwise_with_a_random_start():
    """輪轉必須順時針繞四角：左上 → 右上 → 右下 → 左下。

    隨機的只有起點（誰搶到下棋權），不是整個排列；用 shuffle 的話相鄰兩手
    有時會落在對角的兩個角上。
    """
    assert CLOCKWISE_OWNERS == (1, 2, 0, 3), CLOCKWISE_OWNERS
    corners = [OWNER_CORNER[o] for o in CLOCKWISE_OWNERS]
    assert corners == [(0, 0), (19, 0), (19, 19), (0, 19)], corners
    first_seen = set()
    for seed in range(200):
        g = Game(random.Random(seed))
        g.set_player_color("blue")
        order = g.turn_order
        assert sorted(order) == [0, 1, 2, 3], order
        assert order.count(PLAYER_OWNER) == 1, order
        idx = [CLOCKWISE_OWNERS.index(o) for o in order]
        for i in range(3):
            assert idx[i + 1] == (idx[i] + 1) % 4, order
        first_seen.add(order.index(PLAYER_OWNER))
        # 輪轉必須真的照著走，並在最後一位繞回第一位
        g.start()
        walked = [g.current_owner()]
        for _ in range(3):
            g.act_pass()
            walked.append(g.current_owner())
        assert walked == order, (walked, order)
    # 玩家四個位置都抽得到，也就是有時先手、有時後手
    assert first_seen == {0, 1, 2, 3}, first_seen


def test_100_games():
    rng = random.Random(42)
    total_turns = 0
    winners = []
    checked = [0]
    intruder_moves = [0, 0, 0, 0]
    keys_seen = set()
    neigh = neighbors_of()

    def check(g, owner, move, before):
        """Every stone after the opening must meet one of that owner's earlier
        stones at a corner, and must not share an edge with any of them."""
        if g.owner_key[owner] == "intruder":
            n = intruder_moves[owner]
            intruder_moves[owner] += 1
            # 規則 1：跨越棋先用完才換別的
            if n < 3:
                assert move[0] in ai.LEAPERS, \
                    "intruder move %d was %s, expected a leaper" % (n, move)
        if g.owner_key[owner] == "optimizer" and g.must_cover(owner) is None:
            # 沒有緊急棋塊時一定放能放下的最大棋塊
            counts = ai.placement_counts(g.board, g.hands[owner].names, owner, None)
            if not [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES]:
                biggest = max(MASTER[n]["size"] for n, k in counts.items() if k > 0)
                assert MASTER[move[0]]["size"] == biggest, (move, biggest, counts)
        if g.placed[owner] == 0:
            checked[0] += 1
            return
        name, oi, x, y = move
        cells = {(x + dx) + (y + dy) * 20
                 for dx, dy in MASTER[name]["orientations"][oi]}
        where = "%s oi=%d at (%d,%d) for owner %d" % (name, oi, x, y, owner)
        for i in cells:
            for n in neigh[i]:
                assert n not in before, where + " shares an edge with its own stone"
        before_xy = {(i % 20, i // 20) for i in before}
        assert any((i % 20 + dx, i // 20 + dy) in before_xy
                   for i in cells for dx in (-1, 1) for dy in (-1, 1)), \
            where + " has no corner contact"

    for i in range(100):
        g = run_game(rng, check)
        keys_seen |= {k for k in g.owner_key.values() if k}
        assert g.state == "GAME_OVER"
        assert g.turn_count < 300, g.turn_count
        total_turns += g.turn_count
        occupied = sum(1 for cell in g.board.grid if cell != -1)
        expected = sum(89 - sum(MASTER[n]["size"] for n in g.hands[o].names)
                       for o in range(4))
        assert occupied == expected, (occupied, expected)
        assert bin(g.board.empty_bits).count("1") == 400 - occupied
        # every player opened on their own corner square
        for owner in range(4):
            cx, cy = OWNER_CORNER[owner]
            assert g.board.grid[cx + cy * 20] == owner, (owner, "missed own corner")
        # The no-shared-edge rule cannot be re-checked on the final grid: a
        # single multi-cell piece is edge-adjacent to itself, and pieces are
        # not distinguishable once placed. It is checked per move above, where
        # the board as it stood before the move is still available.
        standings = g.standings()
        rem = [g.remaining_cells(o) for o in range(4)]
        # the score is the squares left unplaced, so fewer is better
        assert standings[0][1] == min(rem)
        assert [r for _, r in standings] == sorted(rem)
        # and it must agree with the hand accounting
        for o in range(4):
            assert rem[o] == sum(MASTER[n]["size"] for n in g.hands[o].names)
        # the game may only end once every one of the four is stuck
        assert not g.has_legal(0)
        for o in (1, 2, 3):
            assert not g.has_legal(o)
        winner = g.winner()
        assert winner in ("player", "draw", "1", "2", "3")
        best = min(rem)
        top = [j for j in range(4) if rem[j] == best]
        if winner == "player":
            assert rem[0] == best and len(top) == 1
        elif winner == "draw":
            assert 0 in top and len(top) > 1
        else:
            assert rem[int(winner)] == best and 0 not in top
        winners.append(winner)
    assert checked[0] == 400, "expected 400 opening moves, saw %d" % checked[0]
    # 3-of-4 抽樣跑了 100 局，四種人格都該出現過；狼／棋手／狐狸各局的
    # games 欄位自然不一致，排行榜照樣能加總。
    assert keys_seen == set(ai.personality_keys()), keys_seen
    assert any(intruder_moves), "no game drew the intruder"
    counts = {}
    for w in winners:
        counts[w] = counts.get(w, 0) + 1
    print("avg turns: %.1f winners=%s" % (total_turns / 100.0, counts))


def _pick_subset(names, target):
    """Pieces from `names` whose sizes sum to exactly `target`."""
    sizes = [MASTER[n]["size"] for n in names]
    # reach[i] = (prev_index, took) for hitting sum i, built back to front
    table = [None] * (target + 1)
    table[0] = (-1, False)
    for j in range(len(names) - 1, -1, -1):
        s = sizes[j]
        for t in range(target, s - 1, -1):
            if table[t] is None and table[t - s] is not None:
                table[t] = (j, True)
    if table[target] is None:
        raise AssertionError("cannot make %d from %r" % (target, sizes))
    out, t = [], target
    while t:
        j, took = table[t]
        assert took, "broken subset-sum chain"
        out.append(names[j])
        t -= sizes[j]
    return out


def _finished_game(remaining):
    """A GAME_OVER game with a prescribed number of unplaced squares each."""
    g = Game(random.Random(0))
    g.state = "GAME_OVER"
    for o in range(4):
        for n in _pick_subset(list(MASTER), 89 - remaining[o]):
            g.hands[o].names.remove(n)
        assert g.remaining_cells(o) == remaining[o], (o, g.remaining_cells(o))
    return g


def test_tie_declaration():
    g = _finished_game([10, 10, 10, 10])
    assert g.winner() == "draw", g.winner()


def test_player_wins_declaration():
    g = _finished_game([4, 20, 30, 40])
    assert g.winner() == "player", g.winner()


def test_ai_wins_declaration():
    g = _finished_game([40, 30, 20, 4])
    assert g.winner() == "3", g.winner()


def test_fewer_remaining_wins():
    """The score is the squares left unplaced, so more remaining is worse."""
    g = _finished_game([0, 5, 10, 15])
    assert g.standings() == [(0, 0), (1, 5), (2, 10), (3, 15)]
    assert g.winner() == "player"
    g = _finished_game([15, 10, 5, 0])
    assert g.standings() == [(3, 0), (2, 5), (1, 10), (0, 15)]
    assert g.winner() == "3"


def test_full_hand_scores_zero():
    g = _finished_game([0, 0, 0, 0])
    assert g.winner() == "draw"
    assert all(r == 0 for _, r in g.standings())


def test_game_ends_only_when_all_four_are_stuck():
    """The player being stuck must not end the game on its own."""
    b = Board()
    i1 = MASTER["I1"]["orientations"][0]
    g = Game(random.Random(0))
    g.start()
    # wall the player into its own corner: the player has no legal move, but
    # the three AI corners are still wide open
    g.board.place(19, 19, i1, 0)
    for x, y in ((19, 18), (18, 19), (18, 18)):
        g.board.place(x, y, i1, 1)
    g.placed[0] = 1
    assert not g.has_legal(0)
    assert g.has_legal(1) or g.has_legal(2) or g.has_legal(3)
    g.act_pass()
    assert g.state == "PLAYING", "must not end while the AIs can still move"


if __name__ == "__main__":
    test_100_games()
    test_tie_declaration()
    test_player_wins_declaration()
    test_ai_wins_declaration()
    test_fewer_remaining_wins()
    test_full_hand_scores_zero()
    test_game_ends_only_when_all_four_are_stuck()
    print("all simulation tests passed")
