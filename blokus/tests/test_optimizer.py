"""The Optimizer: place the biggest piece first, then preserve placeable space,
and in an emergency rescue the piece that is down to a single placement.

This personality's objective function has only one quantity - how many
placeable squares it will still have after the move - because scoring is
exactly "fewer remaining squares is better". Three things are pinned down here:
the priority of piece size, the maximisation of placeable space, and the fact
that the emergency rule overrides the first two (without overriding the opening
corner rule).
"""
import random
import time

import ai
from board import Board
from config import PERSONALITY_ORDER
from game import Game
from pieces import MASTER
from test_ai import assert_corner_contact

I1 = MASTER["I1"]["orientations"][0]


def playout(seed, moves, brains=None):
    """A position reached using only **weighted-style personalities**.

    The three AI seats are explicitly pinned, without going through
    `set_player_color`'s lottery: who gets drawn drifts as the personality pool
    changes, so touching the pool would change the position here too, and then
    every assertion below would no longer test the position it was meant to
    test. The position stopped here has nothing to do with the Optimizer, so it
    will not drift as the Optimizer's behaviour changes either.
    """
    rng = random.Random(seed)
    g = Game(rng)
    g.set_player_color("blue")
    for o, key in enumerate(("chess", "fox", "wolf"), start=1):
        g.owner_key[o] = key
        g.brains[o] = ai.make_brain(key, rng)
    if brains is not None:
        g.brains = brains
    g.start()
    n = 0
    while g.state == "PLAYING" and n < moves:
        owner = g.current_owner()
        mv = ai.choose_move(g.board, g.hands[owner].names, owner, g.brains[owner],
                            rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner), reach=g.reach(owner))
        if mv:
            g.act(*mv)
        else:
            g.act_pass()
        n += 1
    return g



def optimizer_brain():
    return ai.make_brain("optimizer", random.Random(0))


def pick(board, hand, owner, must_cover=None, reach=None, seed=0):
    return ai.choose_move(board, hand, owner, optimizer_brain(), random.Random(seed),
                         other_brains=None, must_cover=must_cover, reach=reach)


def legal_count(board, name, oi, base, ctx):
    return ai.place_state(board, name, oi, base, ctx)[4].bit_count()


# ------------------------------------------------------------ placeable space

def test_optimizer_maximises_placeable_squares():
    """Among the same piece, pick the cell that leaves us the most positions
    to play afterwards.

    Every legal candidate is independently enumerated and recomputed, without
    relying on `rescore`'s own result.
    """
    g = playout(0, 12)
    owner = 1
    hand = ["L5", "N5", "I1"]
    board, reach = g.board, g.reach(owner)
    mv = pick(board, hand, owner, reach=reach)
    assert mv is not None
    name, oi, x, y = mv
    assert MASTER[name]["size"] == 5
    ctx = ai.board_context(board, hand, owner, None)
    got = legal_count(board, name, oi, x + y * 20, ctx)
    best = got
    empt = board.empty_bits
    for cand in ("L5", "N5", "I1"):
        for o, od in ai.ODIRS[cand].items():
            for base in _bases(od, empt, reach):
                best = max(best, legal_count(board, cand, o, base, ctx))
    assert got == best, (got, best)


def _bases(od, empt, reach):
    mask = ai._legal_bases(od, empt, reach)
    out = []
    while mask:
        low = mask & -mask
        mask ^= low
        out.append(low.bit_length() - 1)
    return out


def test_optimizer_prefers_the_largest_piece():
    """With no urgent piece, place the biggest piece that fits first."""
    g = playout(0, 12)
    owner = 1
    board, reach = g.board, g.reach(owner)
    counts = ai.placement_counts(board, ["F5", "I5", "I1"], owner, None)
    assert counts["I1"] > 1, counts
    assert not [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES], counts
    mv = pick(board, ["F5", "I5", "I1"], owner, reach=reach)
    assert MASTER[mv[0]]["size"] == 5, mv
    # when only small pieces are left in hand it plays a small piece instead
    # (the phase is soft and will never jam up)
    mv = pick(board, ["I3", "I2", "I1"], owner, reach=reach)
    assert MASTER[mv[0]]["size"] == 3, mv


# ------------------------------------------------------------ emergency rule

def test_optimizer_places_the_urgent_piece_first():
    """A piece down to its only placement goes first, even when a 5-cell piece
    still has several placements.

    The position uses a fixed `playout(5, 46)`: this spot naturally grows into
    I4 / O4 / X5 each having exactly one placement left, while at the same time
    a 5-cell piece still has 5 placements. A rare situation must not use "just
    find some game" as a fixture - that would drift with the personality pool
    and the lottery.
    """
    g = playout(5, 46)
    owner = 1
    board, reach = g.board, g.reach(owner)
    hand = list(g.hands[owner].names)
    counts = ai.placement_counts(board, hand, owner, g.must_cover(owner))
    # situational premise: three pieces each have one placement left, while
    # another 5-cell piece still has several
    assert g.must_cover(owner) is None
    urgent = [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES]
    assert all(counts[n] == 1 for n in urgent) and len(urgent) == 3, counts
    five = {k: v for k, v in counts.items() if MASTER[k]["size"] == 5}
    assert max(five.values()) >= 5, five
    assert max(five.values()) > max(counts[n] for n in urgent), five
    brain = optimizer_brain()
    full = brain.context(board, hand, owner, g.must_cover(owner), reach)
    assert set(full["urgent"]) == set(urgent), full["urgent"]
    mv = pick(board, hand, owner, reach=reach)
    assert mv[0] in full["urgent"], (mv, full["urgent"])
    assert counts[mv[0]] == 1, (mv, counts[mv[0]])


def test_urgent_rescue_prefers_the_bigger_piece():
    """With several pieces down to one placement each, rescue the bigger one
    first: they are equally unsavable, and leaving 5 cells hurts less."""
    g = playout(5, 46)
    owner = 1
    board, reach = g.board, g.reach(owner)
    hand = list(g.hands[owner].names)
    brain = optimizer_brain()
    full = brain.context(board, hand, owner, g.must_cover(owner), reach)
    counts = ai.placement_counts(board, hand, owner, g.must_cover(owner))
    urgent = [n for n, k in counts.items() if 0 < k <= ai.URGENT_PLACES]
    assert len(urgent) > 1, urgent
    assert full["urgent"] == sorted(urgent, key=lambda n: -MASTER[n]["size"]), full
    assert MASTER[full["urgent"][0]]["size"] == max(MASTER[n]["size"] for n in urgent)


def test_opening_ignores_the_urgent_rule():
    """On the opening move the placement count is forced by the corner rule, it
    is not a signal that the board is getting crowded.

    At that point a piece like O4 has only one way to be placed, and treating
    that as "urgent" would spend the first move on a 4-cell piece, directly
    violating "prefer 5 cells".
    """
    b = Board()
    brain = optimizer_brain()
    ctx = brain.context(b, list(MASTER.keys()), 1, (0, 0), None)
    assert ctx["urgent"] == [], ctx["urgent"]
    assert ai.placement_counts(b, list(MASTER.keys()), 1, (0, 0))["O4"] == 1
    mv = pick(b, list(MASTER.keys()), 1, must_cover=(0, 0), reach=None)
    assert MASTER[mv[0]]["size"] == 5, mv
    assert b.can_place(mv[2], mv[3], MASTER[mv[0]]["orientations"][mv[1]], 1,
                       (0, 0), None)


def test_a_dead_piece_is_not_urgent():
    """A piece with 0 placements cannot be rescued and is not urgent (and it has
    no candidate placements in the first place)."""
    b = Board()
    b.place(0, 0, I1, 1)
    for y in range(4):
        for x in range(4):
            if (x, y) in ((0, 0), (1, 1)):
                continue
            b.place(x, y, I1, 2)
    reach = b.reach(1)
    hand = list(MASTER.keys())
    counts = ai.placement_counts(b, hand, 1, None)
    assert counts["I1"] == 1 and counts["F5"] == 0, counts
    ctx = optimizer_brain().context(b, hand, 1, None, reach)
    assert ctx["urgent"] == ["I1"], ctx["urgent"]
    mv = pick(b, hand, 1, reach=reach)
    assert mv == ("I1", 0, 1, 1), mv
    assert b.can_place(1, 1, I1, 1, None, reach)


# -------------------------------------------------------- legality / performance

def test_optimizer_moves_are_legal_in_a_real_game():
    """Every move must satisfy corner contact and share no edge. The final
    board cannot be re-verified, so it is checked move by move."""
    rng = random.Random(5)
    g = Game(rng)
    g.set_player_color("blue")
    g.brains[1] = ai.make_brain("optimizer", rng)
    g.owner_key[1] = "optimizer"
    g.start()
    checked = 0
    while g.state == "PLAYING" and checked < 10:
        owner = g.current_owner()
        before = {i for i in range(400) if g.board.grid[i] == owner}
        mv = ai.choose_move(g.board, g.hands[owner].names, owner, g.brains[owner],
                            rng, other_brains=g.brains,
                            must_cover=g.must_cover(owner), reach=g.reach(owner))
        if mv is None:
            assert not g.has_legal(owner)
            g.act_pass()
            continue
        if owner == 1 and g.placed[1] > 0:
            assert_corner_contact(g.board, 1, mv[0], mv[1], mv[2], mv[3], before)
            checked += 1
        g.act(*mv)


def test_optimizer_move_speed():
    board = playout(0, 40).board
    brain = optimizer_brain()
    reach = board.reach(1)
    t0 = time.perf_counter()
    for i in range(20):
        mv = ai.choose_move(board, list(MASTER.keys()), 1, brain,
                            random.Random(i), other_brains=None, reach=reach)
        assert mv is not None
    dt = (time.perf_counter() - t0) / 20
    assert dt < 1.5, dt


def test_optimizer_beats_the_weighted_personalities():
    """This personality is defined by the score, so it should really win on
    remaining cells."""
    keys = ["optimizer", "wolf", "fox"]
    left = dict.fromkeys(keys + ["player"], 0)
    games = 12
    for seed in range(games):
        rng = random.Random(1000 + seed)
        g = Game(rng)
        g.set_player_color("blue")
        g.owner_key[1], g.owner_key[2], g.owner_key[3] = keys
        g.brains = {0: ai.make_brain("chess", rng)}
        for i, k in enumerate(keys, start=1):
            g.brains[i] = ai.make_brain(k, rng)
        g.start()
        while g.state == "PLAYING":
            owner = g.current_owner()
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner), reach=g.reach(owner),
                                other_must_cover={o: g.must_cover(o) for o in range(4)},
                                other_reach={o: g.reach(o) for o in range(4)})
            if mv:
                g.act(*mv)
            else:
                g.act_pass()
        for owner, k in ((0, "player"), (1, keys[0]), (2, keys[1]), (3, keys[2])):
            left[k] += g.remaining_cells(owner)
    avg = {k: v / float(games) for k, v in left.items()}
    assert avg["optimizer"] == min(avg.values()), avg
    assert avg["optimizer"] < ai.SQ_CAP / 4, avg


def test_optimizer_is_part_of_the_personality_pool():
    assert "optimizer" in ai.personality_keys()
    assert "optimizer" in PERSONALITY_ORDER
    assert isinstance(optimizer_brain(), ai.OptimizerBrain)
    assert optimizer_brain().uses_lookahead is False
    assert optimizer_brain().mistake_rate == 0.0
