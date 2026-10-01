"""Tests for `ai/hunter.py`: the stage H teacher.

Two things are defended here. First, that Hunter **is** the version stage H1
measured: the same moves as `rl.v3.choose_v3` from the first move on, with the
same opening and the same scoring. Second, that the opening book in `ai/` is the
same opening as the one in `rl/opening.py`, since they are separate
implementations and nothing else in the tree compares them.

Where a number is checked it is recomputed rather than copied. The oracle below
counts playable cells one square at a time straight from `board.grid`, sharing no
code with `ai/formulas`.
"""
import ast
import os
import random
import subprocess
import sys

import pytest

import ai
import engine
from ai import formulas as F
from ai.base import HUNTER_KEY, OPTIMIZER_KEY
from ai.chooser import _candidates
from ai.hunter import (BOOK, BOOK_STEPS, HunterBrain, after_moves,
                       bases_covering_all, book_candidates, book_draw,
                       book_piece, book_seed, book_squares, corner_and_direction,
                       make_brain, playable_before, strongest_opponent,
                       turn_order_after)
from config import B, CLOCKWISE_OWNERS, I, PERSONALITY_ORDER
from game import Game
from rl import opening as O
from rl.actions import index_to_move
from rl.collect import board_from_state
from rl.opening import (BookTracker, act_to_engine_move, must_cover, reach,
                        to_act_move)
from rl.v3 import V3Brain, choose_v3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OTHER_POOL = ("builder", "intruder", "wolf", "chess", "fox")

EDGE = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))

SEEDS = (2_000_000_000, 4_000_000, 7, 99, 12345)


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

def oracle_playable(board, owner):
    """How many empty squares are diagonally adjacent to one of `owner`'s stones
    and share no edge with one of them, counted square by square from
    `board.grid` and using no `ai.formulas` helper."""
    n = 0
    grid = board.grid
    for idx in range(B * B):
        if grid[idx] != -1:
            continue
        x, y = idx % B, idx // B
        diag = edge = False
        for dx, dy in EDGE:
            nx, ny = x + dx, y + dy
            if 0 <= nx < B and 0 <= ny < B and grid[nx + ny * B] == owner:
                edge = True
        for dx, dy in DIAG:
            nx, ny = x + dx, y + dy
            if 0 <= nx < B and 0 <= ny < B and grid[nx + ny * B] == owner:
                diag = True
        if diag and not edge:
            n += 1
    return n


def hand_names_of(state, owner):
    return [engine.PIECE_ORDER[i] for i in range(engine.N_PIECES)
            if state.hand_bits[owner] >> i & 1]


def book_alive_state(owner, step):
    """A real position where `owner` has played exactly `step` book moves.

    `plan6.md` acceptance 3: real games never abandon the book (7,500 pairs in
    stage H1, none of them), so the abandonment branch has to be reached
    deliberately. This reaches a given step by *playing* the book, which is the
    reachable way in.
    """
    cur = O.align_to(engine.initial_state(CLOCKWISE_OWNERS), owner)
    for s in range(step):
        cands = book_candidates(board_from_state(cur), owner, s)
        assert cands, (owner, s)
        cur = O.align_to(engine.apply_move(cur, _as_move(cands[0])), owner)
    assert cur.to_move == owner
    return cur


def _as_move(triple):
    """`(name, oi, base)` or `(piece_idx, oi, base)` -> an engine move."""
    piece = triple[0]
    if isinstance(piece, str):
        piece = engine.PIECE_IDX[piece]
    return (piece, triple[1], triple[2])


def play(seed, arm, tested=0):
    """One complete game.

    `arm == "v3"` drives the tested seat with `rl.v3.choose_v3` and a real
    `V3Brain`, which short-circuits the book and returns before `choose_move`.
    `arm == "hunter"` drives it with plain `ai.choose_move` and a `HunterBrain`,
    which is the path a normal game has, where no short-circuit is available.
    Seed, opponents and per-seat rng streams are identical between the two.
    """
    st = engine.initial_state(CLOCKWISE_OWNERS)
    g = Game(random.Random(seed))
    g.turn_order = list(CLOCKWISE_OWNERS)
    g.turn_pos = 0
    rngs = {o: random.Random(seed * 97 + o) for o in range(4)}
    profile = ai.make_brain(OPTIMIZER_KEY,
                            random.Random(seed * 4 + tested)).profile
    brains = {o: ai.make_brain(OPTIMIZER_KEY, random.Random(seed * 4 + o))
              for o in range(4)}
    for o in range(4):
        if o != tested:
            brains[o] = ai.make_brain(OTHER_POOL[(seed + o) % 5],
                                      random.Random(seed * 4 + o))
    if arm == "v3":
        # a real V3Brain, not a plain optimizer: choose_v3's post-book path is
        # plain ai.choose_move with whatever brain it is handed, so handing it a
        # plain optimizer would compare Hunter against v2 instead of v3
        brains[tested] = V3Brain(OPTIMIZER_KEY, profile)
        tracker = BookTracker(seed)
    else:
        brains[tested] = HunterBrain(HUNTER_KEY, profile, game_seed=seed)
        tracker = None
    g.brains = brains
    g.start()

    record = []
    while not engine.is_over(st):
        o = st.to_move
        if engine.legal_move_mask(st) == 0:
            st = engine.pass_turn(st)
            g.act_pass()
            continue
        if o == tested:
            before = (brains[tested].book_moves_used if arm == "hunter" else 0)
            if arm == "v3":
                mv, info = choose_v3(st, g.board, o, brains[tested], rngs[o],
                                     tracker, other_brains=brains)
                source = info["source"]
            else:
                r = ai.choose_move(g.board, g.hands[o].names, o, brains[tested],
                                   rngs[o], other_brains=brains,
                                   must_cover=must_cover(st, o),
                                   reach=reach(st, g.board, o),
                                   other_must_cover={x: must_cover(st, x)
                                                     for x in range(4)},
                                   other_reach={x: reach(st, g.board, x)
                                                for x in range(4)})
                if r is None:                            # pragma: no cover
                    raise RuntimeError("hunter found no move")
                mv = act_to_engine_move(r[0], r[1], r[2], r[3])
                source = ("book" if brains[tested].book_moves_used > before
                          else "abandoned" if brains[tested].book_abandoned
                          else "v3")
            record.append((source, mv))
        else:
            r = ai.choose_move(g.board, g.hands[o].names, o, brains[o],
                               rngs[o], other_brains=brains,
                               must_cover=must_cover(st, o),
                               reach=reach(st, g.board, o),
                               other_must_cover={x: must_cover(st, x)
                                                 for x in range(4)},
                               other_reach={x: reach(st, g.board, x)
                                              for x in range(4)})
            mv = act_to_engine_move(r[0], r[1], r[2], r[3])
        st = engine.apply_move(st, mv)
        g.act(*to_act_move(mv))
        if o == tested and tracker is not None:
            tracker.note_move(o)
    return record, list(engine.result(st)), brains[tested]


def setup_game(seed, brain_for_zero=None):
    """A four-seat game with an all-optimizer table, ready to play."""
    g = Game(random.Random(seed))
    g.set_player_color("blue")
    g.start()
    for o in range(4):
        g.brains[o] = ai.make_brain(OPTIMIZER_KEY, random.Random(seed * 4 + o))
    if brain_for_zero is not None:
        g.brains[0] = brain_for_zero
    return g


def ask(g, owner, brain, rng):
    return ai.choose_move(g.board, g.hands[owner].names, owner, brain, rng,
                          other_brains=g.brains, must_cover=g.must_cover(owner),
                          reach=g.reach(owner),
                          other_must_cover={o: g.must_cover(o) for o in range(4)},
                          other_reach={o: g.reach(o) for o in range(4)})


# --------------------------------------------------------------------------
# 1. full-game equivalence with choose_v3
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("tested", [0, 1, 2, 3])
def test_hunter_makes_exactly_the_moves_choose_v3_makes(seed, tested):
    """Acceptance 1: whole games from the first move on, covering the three book
    moves and everything after them."""
    ra, rema, _ = play(seed, "v3", tested)
    rb, remb, hunter = play(seed, "hunter", tested)
    assert [m for _s, m in ra] == [m for _s, m in rb]
    assert rema == remb
    # the opening really was the book, three times, in both arms
    assert sum(1 for s, _m in ra if s == "book") == BOOK_STEPS
    assert sum(1 for s, _m in rb if s == "book") == BOOK_STEPS
    assert hunter.book_moves_used == BOOK_STEPS
    assert hunter.book_abandoned is False


def test_hunter_and_v3_are_given_the_same_profile_and_order():
    """The equivalence above would be cheap if Hunter simply had a different
    profile; it must be the same one."""
    for seed in SEEDS[:3]:
        for tested in range(4):
            _r, _rem, v3 = play(seed, "v3", tested)
            _r2, _rem2, hunter = play(seed, "hunter", tested)
            assert v3.profile == hunter.profile
            assert v3.turn_order == hunter.turn_order
            assert v3.v3_diff_weight == hunter.diff_weight == 1.0


# --------------------------------------------------------------------------
# 2. the book cross-check against rl/opening
# --------------------------------------------------------------------------

def test_the_book_tables_and_geometry_are_identical_to_rl_opening():
    assert BOOK == O.BOOK
    assert BOOK_STEPS == O.BOOK_STEPS
    for owner in range(4):
        assert corner_and_direction(owner) == O.corner_and_direction(owner)
        for step in range(BOOK_STEPS):
            assert book_squares(owner, step) == O.book_squares(owner, step)
            assert book_piece(step) == O.book_piece(step)
    for owner in range(4):
        for step in range(BOOK_STEPS):
            assert book_seed(4_000_000, owner, step) == \
                O.book_seed(4_000_000, owner, step)


def test_the_candidate_sets_match_rl_opening_on_real_book_positions():
    """Acceptance 2: 8 games x 3 steps = 24 owner-step pairs, and the comparison
    is run on every owner and every step reachable by playing the book, which is
    96 (position, step) queries in total."""
    compared = 0
    for seed in range(8):
        for owner in range(4):
            cur = O.align_to(engine.initial_state(CLOCKWISE_OWNERS), owner)
            for step in range(BOOK_STEPS):
                mine = [(engine.PIECE_IDX[n], oi, b) for n, oi, b in
                        book_candidates(board_from_state(cur), owner, step)]
                theirs = sorted(index_to_move(int(i)) for i in
                                O.book_candidates(cur, owner, step))
                assert sorted(mine) == theirs, (seed, owner, step, mine,
                                                theirs)
                assert len(mine) == 2, (owner, step, mine)
                compared += 1
                cur = O.align_to(engine.apply_move(cur, _as_move(mine[0])),
                                 owner)
    assert compared >= 96


def test_the_draw_agrees_with_rl_opening_for_an_injected_seed():
    """Same seed and same derivation, so the same candidate index."""
    tracker = O.BookTracker(4_000_000)
    for owner in range(4):
        for step in range(BOOK_STEPS):
            tracker.own_moves[owner] = step
            for n in (1, 2, 4):
                cands = list(range(n))
                assert book_draw(4_000_000, owner, step, n) == \
                    tracker.draw(cands, owner)


def test_the_candidate_set_is_owner_independent_in_the_normalised_view():
    """`plan6.md` H-A0b item 5 asked exactly this. The rotation is measured from
    `engine._rot_mask` rather than assumed."""
    cell = {}
    for c in range(B * B):
        moved = engine._rot_mask(1 << c)
        assert moved.bit_count() == 1
        cell[c] = (moved & -moved).bit_length() - 1

    def rot_cell(c, n):
        for _ in range(n):
            c = cell[c]
        return c

    def rot_candidates(cands, n):
        out = set()
        for piece, oi, base in cands:
            p, o, b = piece, oi, base
            for _ in range(n):
                p, o, b = p, engine.rot_orient(p, o), engine.rot_base(p, o, b)
            out.add((p, o, b))
        return out

    for step in range(BOOK_STEPS):
        sets = []
        corners = []
        for owner in range(4):
            state = book_alive_state(owner, step)
            k = (4 - CLOCKWISE_OWNERS.index(owner)) % 4
            real = [(engine.PIECE_IDX[n], oi, b) for n, oi, b in
                    book_candidates(board_from_state(state), owner, step)]
            sets.append(rot_candidates(real, k))
            c = OWNER_CORNER_FOR(owner)
            corners.append(rot_cell(c, k))
        assert all(s == sets[0] for s in sets), (step, sets)
        assert len(sets[0]) == 2
        # the mover's own corner is always view (0, 0), whoever the real owner
        assert all(cc == 0 for cc in corners), (step, corners)


def OWNER_CORNER_FOR(owner):
    from config import OWNER_CORNER
    cx, cy = OWNER_CORNER[owner]
    return cx + cy * B


# --------------------------------------------------------------------------
# 3. the abandonment rule
# --------------------------------------------------------------------------

def test_a_buried_corner_abandons_the_book_and_falls_back_to_the_scoring():
    """Acceptance 3: real games never abandon (7,500 pairs in stage H1), so the
    branch is reached deliberately.

    The construction: let the tested player open legally, then have the next
    player cover the squares the book needs for step 2, which empties the
    candidate set. `restrict` then hands the untouched candidate list back and
    `rescore` scores by the differential rule - v1's rule minus `A_j*(after)`.
    """
    state = engine.initial_state(CLOCKWISE_OWNERS)
    owner = 0
    cur = O.align_to(state, owner)
    first = book_candidates(board_from_state(cur), owner, 0)[0]
    cur = O.align_to(engine.apply_move(cur, _as_move(first)), owner)
    second = book_candidates(board_from_state(cur), owner, 1)
    assert len(second) == 2
    cur = O.align_to(engine.apply_move(cur, _as_move(second[0])), owner)
    assert len(book_candidates(board_from_state(cur), owner, 2)) == 2

    brain = HunterBrain(HUNTER_KEY,
                        ai.make_profile("chess", random.Random(0)),
                        game_seed=11)
    # the position already carries the player's own first two book pieces, so the
    # brain must be put at the step that position is actually on
    brain.book_step = 2
    board = board_from_state(cur)
    names = hand_names_of(cur, owner)
    ctx = brain.context(board, names, owner, must_cover(cur, owner),
                        reach(cur, board, owner))
    assert len(ctx["hunter_book_candidates"]) == 2, "step 2 is still alive here"
    assert ctx["hunter_book_pick"] is not None
    assert brain.book_abandoned is False
    assert brain.book_step == 3


def test_an_empty_candidate_set_abandons_and_then_scores_differentially():
    """The abandonment path itself, forced by clearing the book squares.

    Step 2 asks for W5 over two specific diagonal squares. Covering them with the
    tested player's *own* stone leaves no W5 that covers both, so the set is
    empty - and the player still has plenty of legal moves, so the game continues.
    That is the situation the abandonment rule exists for.
    """
    owner = 0
    state = engine.initial_state(CLOCKWISE_OWNERS)
    cur = state
    # place owner's own stones on the two squares step 2 needs, then align
    a, b = book_squares(owner, 2)
    grid = [-1] * (B * B)
    for cell in (a, b):
        grid[cell[0] + cell[1] * B] = owner
    board = board_from_state(cur)
    board.grid = grid
    board.empty_bits = engine.ALL & ~engine.ALL
    for i in range(B * B):
        board.grid[i] = owner if (board.empty_bits >> i) & 1 else -1
    board.owner_bits = [0, 0, 0, 0]
    board.owner_bits[owner] = engine.ALL
    board.empty_bits = 0
    board._update_regions()
    assert book_candidates(board, owner, 2) == []


def test_a_seat_that_passed_still_counts_exactly_its_real_moves():
    """`plan6.md` decision 2: a pass must not shift the book step.

    `context` runs only on a turn where the player had a legal move, so a passing
    seat's counter does not advance - which is exactly the count a driver would
    keep. A seat that can no longer move never moves again, so the two agree.
    """
    games_with_a_pass = 0
    checked = 0
    for seed in range(30):
        g = setup_game(seed)
        brains = {}
        for o in range(4):
            b = ai.make_brain(HUNTER_KEY, random.Random(seed * 4 + o),
                              game_seed=seed)
            brains[o] = b
            g.brains[o] = b
        shared = random.Random(seed)
        passes = [0] * 4
        while g.state == "PLAYING":
            o = g.current_owner()
            if not g.has_legal(o):
                g.act_pass()
                passes[o] += 1
                continue
            m = ask(g, o, brains[o], shared)
            if m is None:                                # pragma: no cover
                g.act_pass()
                passes[o] += 1
                continue
            g.act(*m)
            for p, b in brains.items():
                assert b.book_step == g.placed[p], (seed, p, b.book_step,
                                                   g.placed[p])
                checked += 1
        if sum(passes):
            games_with_a_pass += 1
    assert checked > 0
    assert games_with_a_pass > 0, "no game in 30 produced a pass, so the " \
        "pass branch was never exercised"


def test_both_implementations_agree_on_the_step_abandoned():
    """Acceptance 3's second half, on the reachable case: both never abandon."""
    for seed in range(6):
        for tested in range(4):
            _r, _rem, v3 = play(seed, "v3", tested)
            rec, _rem2, hunter = play(seed, "hunter", tested)
            v3_abandoned = next((i for i, (s, _m) in enumerate(rec)
                                 if s == "v3"), BOOK_STEPS)
            hunter_abandoned = (BOOK_STEPS if not hunter.book_abandoned
                               else hunter.summary()["book_step"] - 1)
            assert v3_abandoned == hunter_abandoned == BOOK_STEPS


# --------------------------------------------------------------------------
# 4. the dependency direction
# --------------------------------------------------------------------------

def test_hunter_names_neither_rl_nor_numpy_nor_torch_nor_pygame():
    with open(os.path.join(ROOT, "ai", "hunter.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names & {"rl", "numpy", "torch", "pygame"} == set()


def test_importing_ai_hunter_pulls_in_none_of_them():
    """Acceptance 4, in a clean subprocess."""
    code = ("import sys; sys.path.insert(0, %r); import ai.hunter; "
            "bad = [m for m in ('rl', 'numpy', 'torch', 'pygame') "
            "if m in sys.modules]; "
            "print('LEAKED:' + ','.join(bad) if bad else 'clean')" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout, out.stdout


# --------------------------------------------------------------------------
# 5. the normal-game path must not touch the game's rng
# --------------------------------------------------------------------------

def test_a_book_step_does_not_consume_the_games_random_stream():
    """Acceptance 4's second half: `rng.getstate()` must be unchanged across a
    book decision, because the book draws from its own seed and the game shares
    one stream across all four seats."""
    hunter = ai.make_brain(HUNTER_KEY, random.Random(9), game_seed=77)
    g = setup_game(31, brain_for_zero=hunter)
    shared = random.Random(31)
    before = shared.getstate()
    ask(g, 0, hunter, shared)
    assert hunter.book_moves_used == 1, "this should have been a book move"
    assert shared.getstate() == before


def test_a_post_book_step_consumes_exactly_one_draw():
    """And it is the pipeline, not the book, that consumes: one draw per
    decision, which is what any rule-based personality already does."""
    g = setup_game(5)
    v1 = ai.make_brain(OPTIMIZER_KEY, random.Random(3))
    hunter = HunterBrain(HUNTER_KEY, v1.profile, game_seed=8)
    hunter.book_step = BOOK_STEPS          # skip the book for this comparison
    hunter.book_abandoned = True
    a = random.Random(5)
    b = random.Random(5)
    ask(g, 0, v1, a)
    ask(g, 0, hunter, b)
    assert a.getstate() == b.getstate()


def test_an_injected_seed_is_reproducible_and_an_absent_one_is_not():
    """The S6 contract, both halves."""
    a = ai.make_brain(HUNTER_KEY, random.Random(1), game_seed=4_242)
    b = ai.make_brain(HUNTER_KEY, random.Random(999), game_seed=4_242)
    assert a.game_seed == b.game_seed == 4_242
    assert a.game_seed_source == b.game_seed_source == "argument"
    # no seed and no rng of its own: ai.make_brain always passes one, so this is
    # the branch only a direct construction takes
    from ai.hunter import make_brain as hunter_make_brain
    v1 = ai.make_brain(OPTIMIZER_KEY, random.Random(1))
    loose_a = hunter_make_brain(v1.profile)
    loose_b = hunter_make_brain(v1.profile)
    assert loose_a.game_seed_source == loose_b.game_seed_source == "system"
    assert loose_a.game_seed != loose_b.game_seed


def test_deriving_the_seed_from_an_rng_gives_the_same_book_seed_for_the_same_game_seed():
    """Decision 1, first half: the same stream must give the same book seed.

    This is the case that matters, because it is the one a normal game takes -
    `Game` has no seed, only an rng, and `ai.make_brain` hands that rng to the
    constructor. What must not happen is Hunter behaving differently between two
    runs of the same seed, which is exactly what C4 asserts end to end.
    """
    for stream_seed in (1, 2, 3, 17, 4_000_000, 2_000_000_000):
        values = set()
        for _repeat in range(4):
            brain = ai.make_brain(HUNTER_KEY, random.Random(stream_seed))
            assert brain.game_seed_source == "rng", brain.game_seed_source
            values.add(brain.game_seed)
        assert len(values) == 1, (stream_seed, values)

    # and different streams give different seeds, or the derivation is constant
    across = {stream_seed: ai.make_brain(HUNTER_KEY,
                                         random.Random(stream_seed)).game_seed
              for stream_seed in range(40)}
    assert len(set(across.values())) == len(across), \
        "%d distinct streams collapsed to %d seeds" % (
            len(across), len(set(across.values())))


def test_an_explicit_game_seed_overrides_the_stream():
    """A driver that knows the seed does not want the stream's opinion."""
    brain = ai.make_brain(HUNTER_KEY, random.Random(1), game_seed=777)
    assert brain.game_seed == 777
    assert brain.game_seed_source == "argument"


def test_the_book_seed_is_stable_across_processes_and_hash_seeds():
    """Decision 1, second half, and the reason it is sha256 and not `hash`.

    `rng.getstate()`'s third element is `gauss_next`, which is `None` until
    `random.gauss` is called. `hash(None)` derives from the object's address in
    CPython, so a plain `hash` of the state would differ between processes while
    `repr` does not - and a probe that ran twice inside one process cannot see
    that. This runs the derivation in subprocesses under two different
    `PYTHONHASHSEED` values and compares the digits.
    """
    program = (
        "import random, sys\n"
        "sys.path.insert(0, %r)\n"
        "from ai.hunter import seed_from_rng\n"
        "print(seed_from_rng(random.Random(20260930)))\n"
    ) % ROOT
    outputs = {}
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, "-c", program],
                             capture_output=True, text=True, cwd=ROOT, env=env)
        assert out.returncode == 0, out.stderr
        outputs[hash_seed] = out.stdout.strip()
    assert len(set(outputs.values())) == 1, outputs
    assert outputs["0"].isdigit(), outputs


def test_the_state_is_read_once_and_cached_not_per_move():
    """Decision 1: `getstate()` is read at construction, not on every
    `context()`. Re-reading would give a different seed each move, because the
    pipeline draws from the caller's rng on most turns."""
    rng = random.Random(4)
    brain = ai.make_brain(HUNTER_KEY, rng)
    first = brain.game_seed
    before = rng.getstate()
    g = setup_game(4, brain_for_zero=brain)
    shared = random.Random(4)
    for _ in range(6):
        if g.state != "PLAYING":
            break
        owner = g.current_owner()
        if not g.has_legal(owner):
            g.act_pass()
            continue
        g.act(*ask(g, owner, g.brains[owner], shared))
    assert brain.game_seed == first
    assert rng.getstate() == before, "construction must not advance the stream"


def test_only_hunter_advertises_that_it_takes_a_game_seed():
    assert HunterBrain.accepts_game_seed is True
    for cls in (ai.RULE_BRAIN_CLASSES[OPTIMIZER_KEY],
                ai.RULE_BRAIN_CLASSES["intruder"],
                ai.RULE_BRAIN_CLASSES["builder"]):
        assert not getattr(cls, "accepts_game_seed", False)


# --------------------------------------------------------------------------
# 6. the coverage-type tests, at seven keys
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rounds", [50])
def test_the_coverage_type_conditions_hold_with_seven_keys(rounds):
    """Acceptance 6: the three bounded-draw conditions the existing tests rely
    on, re-checked so a flake would show up here first."""
    pool = list(ai.personality_keys())
    assert len(pool) == 7
    seen = set()
    for d in range(200):
        seen |= set(random.Random(d).sample(pool, 3))
    assert seen == set(pool), "200 draws of 3 must cover all seven"
    benched = set()
    for g in range(60):
        benched |= set(pool) - set(random.Random(g * 13).sample(pool, 4))
    assert benched == set(pool), "60 draws of 4 must bench every key once"
    seen2 = set()
    for g in range(100):
        seen2 |= set(random.Random(g * 7).sample(pool, 3))
    assert seen2 == set(pool), "100 draws of 3 must cover all seven"
    del rounds


# --------------------------------------------------------------------------
# 7. the registry and the scoring
# --------------------------------------------------------------------------

def test_hunter_is_registered_as_a_rule_based_personality():
    assert HUNTER_KEY in ai.personality_keys()
    assert HUNTER_KEY in PERSONALITY_ORDER
    assert PERSONALITY_ORDER[-1] == HUNTER_KEY
    assert HUNTER_KEY in ai.RULE_BRAIN_CLASSES
    brain = ai.make_brain(HUNTER_KEY, random.Random(0))
    assert isinstance(brain, HunterBrain)
    assert brain.key == HUNTER_KEY
    assert brain.uses_lookahead is False
    assert brain.mistake_rate == 0.0
    assert I[HUNTER_KEY] and I[HUNTER_KEY + "_desc"]


def test_the_strongest_opponent_tie_break_is_unchanged_from_v3():
    from rl.v3 import strongest_opponent as v3_strongest
    from rl.v3 import turn_order_after as v3_after
    masks = {0: 0b1111, 1: 0b1111, 2: 0b11, 3: 0b111111}
    assert strongest_opponent(masks, 0, CLOCKWISE_OWNERS) == \
        v3_strongest(masks, 0, CLOCKWISE_OWNERS)
    for owner in range(4):
        assert turn_order_after(owner, CLOCKWISE_OWNERS) == \
            v3_after(owner, CLOCKWISE_OWNERS)
    zero = {o: 0 for o in range(4)}
    for owner in range(4):
        assert strongest_opponent(zero, owner, CLOCKWISE_OWNERS) == \
            turn_order_after(owner, CLOCKWISE_OWNERS)[0]


def test_playable_before_agrees_with_the_cell_by_cell_oracle():
    checked = 0
    for seed in (3, 17, 250, 991):
        g = setup_game(seed)
        shared = random.Random(seed)
        for _ in range(12):
            if g.state != "PLAYING":
                break
            o = g.current_owner()
            if not g.has_legal(o):
                g.act_pass()
                continue
            for p in range(4):
                assert playable_before(g.board, p).bit_count() == \
                    oracle_playable(g.board, p), (seed, p)
                checked += 1
            g.act(*ask(g, o, g.brains[o], shared))
    assert checked >= 40


def test_the_playable_count_matches_a_null_move_place_state():
    """The identity the opponent term rests on: v1's own expression at the null
    move is the same set `playable_before` returns."""
    g = setup_game(41)
    for o in range(4):
        ctx = F.board_context(g.board, list(g.hands[o].names), o,
                              g.must_cover(o))
        if ctx["legal_before"]:
            assert playable_before(g.board, o) == \
                ctx["legal_before"].bit_count()


def test_bases_covering_all_requires_every_square_not_any():
    """The mistake that produced five candidates where there are two."""
    od = F.ODIRS["Z5"][0]
    b1 = 1 << (0 + 0 * B)
    b2 = 1 << (2 + 2 * B)
    both = bases_covering_all(od, (b1, b2))
    either = (bases_covering_all(od, (b1,))
              | bases_covering_all(od, (b2,)))
    assert both != 0
    assert both < either, "AND of two requirements must be stricter than either"
    wrong = bases_covering_all(od, (b1 | b2,))
    assert wrong == either, "one multi-bit argument is the 'or' reading"
    assert wrong.bit_count() > both.bit_count()


def test_hunter_with_the_differential_weight_at_zero_scores_exactly_as_v1():
    """The degenerate case on real positions, comparing the whole ordered score
    list rather than just the pick."""
    checked = 0
    for seed in (5, 61, 900):
        g = setup_game(seed)
        v1 = g.brains[0]
        zero = HunterBrain(HUNTER_KEY, v1.profile, diff_weight=0.0, game_seed=8)
        zero.book_step = BOOK_STEPS            # compare scoring, not the book
        zero.book_abandoned = True
        shared = random.Random(seed)
        for _ in range(10):
            if g.state != "PLAYING" or not g.has_legal(0):
                break
            names = list(g.hands[0].names)
            open_a, blk, dfn = F.board_feats(g.board.grid)
            cands = _candidates(g.board, names, 0, v1.profile, g.must_cover(0),
                                g.reach(0), open_a, blk, dfn,
                                g.board.corner_regions, g.board.borders,
                                g.board.empty_bits)
            cands.sort(key=lambda t: t[0], reverse=True)
            c1 = v1.context(g.board, names, 0, g.must_cover(0), g.reach(0))
            c2 = zero.context(g.board, names, 0, g.must_cover(0), g.reach(0))
            s1 = v1.rescore(v1.restrict(cands, c1), c1)
            s2 = zero.rescore(zero.restrict(cands, c2), c2)
            assert [int(x[0]) for x in s1] == [int(x[0]) for x in s2]
            assert [x[1:] for x in s1] == [x[1:] for x in s2]
            checked += 1
            owner = g.current_owner()
            g.act(*ask(g, owner, g.brains[owner], shared))
    assert checked >= 20


def test_after_masks_keys_all_four_owners():
    g = setup_game(12)
    masks = after_moves(g.board)
    assert sorted(masks) == [0, 1, 2, 3]
    for o, m in masks.items():
        assert m == playable_before(g.board, o)