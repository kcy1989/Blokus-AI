"""The Builder: estimates how many squares will fail to fit into live regions.

This personality fills a hole in the Optimizer: the Optimizer only measures the
*quantity* of "how many placeable squares are left", so two live regions of 4
squares and one live region of 8 squares look equally good to it, yet no
5-cell piece fits into the former. Hence the main term is `pack_lost` - FFD-packs
the remaining hand into the live regions, and the squares that will not fit are
the estimated number of squares given away, while scoring is exactly "fewer
remaining squares is better".

The most likely place for this design to die is `pack_lost` not discriminating
(early on the closure saturates and `lost` is constantly 0, so it degenerates
into "a worse Optimizer"), so the last test watches that specifically.
"""
import random
import time

import ai
from board import dilate
from config import PERSONALITY_ORDER
from game import Game
from pieces import MASTER
from test_ai import assert_corner_contact


def playout(seed, moves):
    """A position reached using only **weighted-style personalities**.

    The three AI seats are explicitly pinned, without going through
    `set_player_color`'s lottery - once the personality pool changes the position
    drifts with it, and then every assertion below no longer tests the position
    it was meant to test. The position stopped here has nothing to do with the
    Builder, so it will not drift as the Builder's behaviour changes either.
    """
    rng = random.Random(seed)
    g = Game(rng)
    g.set_player_color("blue")
    for o, key in enumerate(("chess", "fox", "wolf"), start=1):
        g.owner_key[o] = key
        g.brains[o] = ai.make_brain(key, rng)
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


def builder_brain():
    return ai.make_brain("builder", random.Random(0))


def builder_ctx(board, hand, owner, must_cover=None):
    return builder_brain().context(board, hand, owner, must_cover, None)


def legal_cands(board, hand, owner, must_cover=None, reach=None):
    """Independently enumerate every legal candidate [(name, oi, base)], without
    relying on `choose_move`'s intermediate results."""
    cbit = 0
    if must_cover is not None:
        cbit = 1 << (must_cover[0] + must_cover[1] * 20)
    out = []
    for name in hand:
        for oi, od in ai.ODIRS[name].items():
            mask = ai._legal_bases(od, board.empty_bits, reach, cbit)
            while mask:
                low = mask & -mask
                mask ^= low
                out.append((name, oi, low.bit_length() - 1))
    return out


def pack_stats(board, hand, owner, must_cover=None, reach=None):
    """(lost, number of placeable squares, piece size, name, oi, base) for every
    legal candidate."""
    ctx = builder_ctx(board, hand, owner, must_cover)
    rows = []
    for name, oi, base in legal_cands(board, hand, owner, must_cover, reach):
        _own, empt_after, _need, avoid, legal = ai.place_state(board, name, oi,
                                                              base, ctx)
        bins = ai.fill_components(ai.fillable_closure(empt_after, avoid, legal))
        rows.append((ai.pack_lost(ctx["sizes_after"][name], bins),
                     legal.bit_count(), MASTER[name]["size"], name, oi, base))
    return rows


# ------------------------------------------------------------ pure functions

def test_fillable_closure_respects_the_rules():
    """The closure must contain every legal landing square, and land only on
    empty squares without touching `avoid`."""
    g = playout(0, 12)
    owner = 1
    hand = ["F5", "N5", "I3", "I2", "I1"]
    ctx = builder_ctx(g.board, hand, owner)
    name, oi, base = legal_cands(g.board, hand, owner, None, g.reach(owner))[0]
    _own, empt_after, _need, avoid, legal = ai.place_state(g.board, name, oi, base, ctx)
    fill = ai.fillable_closure(empt_after, avoid, legal)
    assert fill & legal == legal, "legal must lie entirely inside the closure"
    assert fill & ~empt_after == 0, "closure must not spill onto occupied squares"
    assert fill & avoid == 0, "closure must not cover the owner's own stones"
    # every square reaches some legal square through 4-adjacency (the definition
    # of a closure)
    reached = legal
    while True:
        grown = reached | (dilate(reached) & fill)
        if grown == reached:
            break
        reached = grown
    assert reached & fill == fill, "some reached square is not reachable back from legal"
    assert fill.bit_count() > legal.bit_count(), "closure should be larger than legal (the flood did work)"


def test_fill_components_partition_fill():
    """Live regions are pairwise disjoint, their union equals `fill`, each is
    4-connected, and the areas sum to the bit count."""
    g = playout(0, 12)
    owner = 1
    hand = ["L5", "N5", "I3", "I2", "I1"]
    ctx = builder_ctx(g.board, hand, owner)
    board = g.board
    for name, oi, base in legal_cands(board, hand, owner, None, g.reach(owner))[:40]:
        _own, empt_after, _need, avoid, legal = ai.place_state(board, name, oi,
                                                              base, ctx)
        fill = ai.fillable_closure(empt_after, avoid, legal)
        assert fill, "the candidate must leave at least one legal base"
        bins = ai.fill_components(fill)
        assert sum(bins) == fill.bit_count(), (name, oi, base)
        assert bins == sorted(bins, reverse=True), bins
        # verify disjointness and connectivity region by region (rebuilt here
        # from scratch, not relying on `fill_components`'s result)
        rest = seen = 0
        rest = fill
        while rest:
            comp = rest & -rest
            rest ^= comp
            while True:
                grown = comp | (dilate(comp) & rest)
                if grown == comp:
                    break
                comp = grown
            assert not comp & seen, "components overlap"
            assert comp & ~fill == 0, "component escaped `fill`"
            seen |= comp
            rest &= ~comp
        assert seen == fill


def test_pack_lost_prefers_one_big_region():
    """**The single pivot test of the whole design**: with the same total, one
    whole region holds far more than two fragments.

    Hand [5,3,2,1]:
      A  two 4-cell live regions -> F5 does not fit (given away: 5), I3 goes in
         the first region (1 left), I2 goes in the second region (2 left), I1
         goes in -> 5 cells given away
      B  one 8-cell live region -> F5 goes in (3 left), I3 goes in (0 left),
         I2 given away 2, I1 given away 1 -> 3 cells given away

    The Optimizer's "total placeable squares" is almost identical for A and B,
    so it would pick A and forever give away those 5 cells. These are the actual
    computed results of the FFD algorithm - the diagram in the plan originally
    said A gives away 7, which is a hand calculation that forgot to account for
    "I2 also fits into the second 4-cell live region".
    """
    sizes = [5, 3, 2, 1]
    assert ai.pack_lost(sizes, [4, 4]) == 5
    assert ai.pack_lost(sizes, [8]) == 3
    assert ai.pack_lost(sizes, [4, 4]) > ai.pack_lost(sizes, [8])


def test_pack_lost_is_zero_when_everything_fits():
    """When the total capacity is sufficient and the shapes work out, the
    number of cells given away is 0."""
    hand = sorted((MASTER[n]["size"] for n in MASTER), reverse=True)
    assert len(hand) == 21
    assert sum(hand) == 89
    assert ai.pack_lost(hand, [89]) == 0
    assert ai.pack_lost(hand, [40, 49]) == 0
    assert ai.pack_lost(hand, [30, 59]) == 0
    # insufficient capacity means there must be leftovers (two 1-cell live
    # regions can only take the smallest piece)
    assert ai.pack_lost([5, 5], [4]) == 10
    assert ai.pack_lost(hand, [1, 1]) == 89 - 1


def test_pack_lost_is_first_fit_decreasing():
    """FFD requires sizes from large to small: with the wrong order, the result
    changes completely."""
    assert ai.pack_lost([5, 3, 3], [5, 6]) == 0
    assert ai.pack_lost([3, 3, 5], [5, 6]) == 5


# ------------------------------------------------------------ objective function

def test_builder_maximises_packability():
    """A real mid-game position: independently enumerate all candidates and
    recompute, then assert that the chosen move is the top-scoring one.

    The tie-breaker terms are deliberately compared as well (not just `lost`),
    because what gets chosen is the move with the highest total score; comparing
    only `lost` would loosen the test enough to allow a regression.
    """
    g = playout(0, 12)
    owner = 1
    hand = ["F5", "N5", "I3", "I2", "I1"]
    board, reach = g.board, g.reach(owner)
    cands = legal_cands(board, hand, owner, None, reach)
    assert len(cands) <= ai.BUILDER_SCAN, "fewer candidates than the scan cap, so the exhaustive pass is honest"
    mv = ai.choose_move(board, hand, owner, builder_brain(), random.Random(0),
                        other_brains=None, reach=reach)
    assert mv is not None
    name, oi, x, y = mv
    ctx = builder_ctx(board, hand, owner)
    brain = builder_brain()
    got = brain._score(name, oi, x + y * 20, ctx)[0]
    scores = [brain._score(n, o, b, ctx)[0] for n, o, b in cands]
    assert got == max(scores), (got, max(scores))


def test_builder_prefers_bigger_pieces_when_packability_ties():
    """When both `lost` and the placeable-square count are the same, the 5-cell
    move wins - proving that `BUILDER_W_SIZE` actually does something."""
    g = playout(0, 12)
    owner = 1
    hand = ["F5", "I1"]
    board, reach = g.board, g.reach(owner)
    rows = pack_stats(board, hand, owner, None, reach)
    tie = {}
    for lost, total, size, name, _oi, _base in rows:
        tie.setdefault((lost, total), set()).add(size)
    assert (0, 8) in tie and {1, 5} <= tie[(0, 8)], sorted(tie[(0, 8)])
    mv = ai.choose_move(board, hand, owner, builder_brain(), random.Random(0),
                        other_brains=None, reach=reach)
    assert MASTER[mv[0]]["size"] == 5, mv


def test_builder_never_returns_none_and_stays_legal():
    """Play a whole game to the end, verifying the corner-contact rule move by
    move."""
    rng = random.Random(5)
    g = Game(rng)
    g.set_player_color("blue")
    g.brains[1] = ai.make_brain("builder", rng)
    g.owner_key[1] = "builder"
    g.start()
    checked = 0
    while g.state == "PLAYING":
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
    assert checked > 0


def test_builder_move_speed():
    board = playout(0, 40).board
    brain = builder_brain()
    reach = board.reach(1)
    t0 = time.perf_counter()
    for i in range(20):
        mv = ai.choose_move(board, list(MASTER.keys()), 1, brain,
                            random.Random(i), other_brains=None, reach=reach)
        assert mv is not None
    dt = (time.perf_counter() - t0) / 20
    print("\nbuilder single move %.4f s" % dt)
    assert dt < 1.5, dt


def test_builder_is_in_the_pool_and_renders():
    assert "builder" in ai.personality_keys()
    assert "builder" in PERSONALITY_ORDER
    assert PERSONALITY_ORDER[-1] == "builder"
    from config import I
    from ui import PERSONA_ZH
    assert I["builder"] and I["builder_desc"]
    assert PERSONA_ZH["builder"] == I["builder"]
    brain = builder_brain()
    assert isinstance(brain, ai.BuilderBrain)
    assert brain.uses_lookahead is False
    assert brain.mistake_rate == 0.0


# ------------------------------------------------------------ anti-regression

def test_packing_term_actually_discriminates():
    """**The most likely place for this design to die**: if `lost` never
    discriminates between candidates, the Builder degenerates into "a worse
    Optimizer". So play 20 real games and measure the fraction of moves in
    which `pack_lost` discriminates, with a 20% threshold.
    """
    moves = discriminated = 0
    for seed in range(20):
        rng = random.Random(300 + seed)
        g = Game(rng)
        g.set_player_color("blue")
        g.brains[1] = ai.make_brain("builder", rng)
        g.owner_key[1] = "builder"
        g.start()
        while g.state == "PLAYING":
            owner = g.current_owner()
            if owner == 1:
                hand = g.hands[owner].names
                losts = {r[0] for r in pack_stats(g.board, hand, owner,
                                                  g.must_cover(owner),
                                                  g.reach(owner))}
                moves += 1
                if len(losts) > 1:
                    discriminated += 1
            mv = ai.choose_move(g.board, g.hands[owner].names, owner,
                                g.brains[owner], rng, other_brains=g.brains,
                                must_cover=g.must_cover(owner),
                                reach=g.reach(owner))
            if mv:
                g.act(*mv)
            else:
                g.act_pass()
    ratio = discriminated / float(moves)
    print("\npack_lost spread %.1f%% (%d/%d moves)" % (ratio * 100, discriminated, moves))
    assert ratio >= 0.20, ratio
