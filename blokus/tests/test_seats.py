"""Seat options, colour dealing, and who opens the game.

Two rules carry most of this file:

  * an **option** may be repeated without limit - four wolves, four humans, two
    recording humans are all legal;
  * a **colour** may not, and a colour another seat holds is dropped from the
    other seats' menus.

And one fact about the engine that makes the whole feature safe: the rules do
not care who goes first. `OWNER_CORNER` fixes a corner to an owner id and the
opening rule is "cover your own corner", so a seat's corner follows from its
colour assignment and nothing else. The tests below check that under all four
openers rather than taking it on trust.
"""
import os
import random

import pytest

import ai
import engine
import seats
from pieces import MASTER
from config import CLOCKWISE_OWNERS, COLORS, OWNER_CORNER, PLAYER_OWNER
from game import Game


# ------------------------------------------------------------------ options

def test_there_are_thirteen_options_and_eleven_without_humans():
    thirteen = seats.seat_options(True)
    assert len(thirteen) == 13
    assert len(set(thirteen)) == 13
    eleven = seats.seat_options(False)
    assert len(eleven) == 11
    assert not any(seats.is_human_kind(seats.kind_of(k, False)) for k in eleven)
    # the two human seats are exactly what the league leaves out
    assert set(thirteen) - set(eleven) == {"human", "human_log"}


def test_the_option_pool_is_seven_ai_three_checkpoints_and_one_policy():
    options = seats.seat_options(False)
    assert len(options) == 11
    ai_only = [k for k in options if seats.kind_of(k) == seats.KIND_AI]
    imitating = [k for k in options if seats.kind_of(k) == seats.KIND_IMITATION]
    trained = [k for k in options if seats.kind_of(k) == seats.KIND_RL]
    assert len(ai_only) == 7
    assert set(ai_only) == set(ai.personality_keys())
    assert [seats.imitation_step(k) for k in imitating] == [1000, 2000, 10000]
    # the trained policy is its own kind, so `imitation_only` keeps meaning the
    # checkpoints that were trained by imitation
    assert trained == list(seats.rl_keys())


def test_a_trained_policy_seat_names_a_file_that_exists():
    """The one thing that makes a pool entry safe to trust.

    Every other registered key resolves through `imitation_key`, and a mistake
    there shows up as a missing file at game start. This one resolves through a
    table, so the table is what has to be checked - and it is checked here
    rather than at the first game, where the failure would look like a training
    problem.
    """
    for key in seats.rl_keys():
        path = seats.rl_checkpoint(key)
        assert os.path.exists(path), (key, path)
        assert os.path.isabs(path) or path.startswith("data/")


def test_the_rl_key_prefix_alone_is_not_a_registered_seat():
    """`rl_9999` has the right shape and no file behind it."""
    assert not seats.is_rl_key("rl_9999")
    assert not seats.is_rl_key("rl_")
    with pytest.raises(ValueError):
        seats.rl_checkpoint("rl_9999")


def test_the_checkpoint_seats_are_the_hc2_run():
    """The pool names H-C2, and only H-C2.

    H-B2's four checkpoints are still on disk and `load_brain` would still open
    one, so this pins the *directory* as well as the steps: a pool that pointed
    at `data/hb2` with the new key names would pass every key-shape test above
    and quietly serve networks trained on the old `stuck` column.
    """
    assert seats.IMITATION_CHECKPOINT_DIR == "data/hc2"
    assert seats.IMITATION_STEPS == (1000, 2000, 10000)
    assert seats.IMITATION_KEY_PREFIX == "hc_"
    for step in seats.IMITATION_STEPS:
        key = seats.imitation_key(step)
        assert key == "hc_%d" % step
        assert seats.imitation_step(key) == step
        assert os.path.exists(os.path.join(
            seats.IMITATION_CHECKPOINT_DIR, "step_%06d.pt" % step))


def test_an_unknown_option_is_refused():
    with pytest.raises(ValueError):
        seats.kind_of("wizard")
    with pytest.raises(ValueError):
        seats.imitation_step("wolf")


def test_kind_of_covers_every_option():
    for key in seats.seat_options(True):
        assert seats.kind_of(key) in (seats.KIND_AI, seats.KIND_IMITATION,
                                      seats.KIND_RL, seats.KIND_HUMAN,
                                      seats.KIND_HUMAN_LOG)
    assert seats.is_human_kind(seats.KIND_HUMAN)
    assert seats.is_human_kind(seats.KIND_HUMAN_LOG)
    assert not seats.is_human_kind(seats.KIND_AI)
    assert not seats.is_human_kind(seats.KIND_IMITATION)
    assert not seats.is_human_kind(seats.KIND_RL)


# ------------------------------------------------------------------ colours

def test_random_seats_are_dealt_the_remaining_colours():
    out = seats.assign_colours([None, None, None, None])
    assert sorted(out) == sorted(COLORS)
    out = seats.assign_colours(["blue", None, None, None])
    assert out[0] == "blue"
    assert sorted(out) == sorted(COLORS)


def test_a_colour_given_twice_is_refused():
    with pytest.raises(ValueError):
        seats.assign_colours(["blue", "blue", None, None])
    with pytest.raises(ValueError):
        seats.assign_colours(["red", "green", "red", "yellow"])


def test_an_unknown_colour_is_refused():
    with pytest.raises(ValueError):
        seats.assign_colours(["purple", None, None, None])
    with pytest.raises(ValueError):
        seats.assign_colours(["blue", "green"])


def test_the_random_sentinel_means_random():
    assert seats.assign_colours([seats.RANDOM] * 4) == \
        seats.assign_colours([None] * 4)


def test_a_taken_colour_leaves_the_other_menus():
    choices = seats.colour_choices({"blue", "red"})
    assert choices == [seats.RANDOM, "green", "yellow"]
    assert "blue" not in choices
    # everything still on offer when nothing is taken
    assert seats.colour_choices(set()) == \
        [seats.RANDOM, "blue", "green", "red", "yellow"]


def test_the_dealing_is_random_but_complete():
    seen = set()
    for seed in range(200):
        out = seats.assign_colours([None] * 4, random.Random(seed))
        assert sorted(out) == sorted(COLORS)
        seen.add(tuple(out))
    assert len(seen) > 12, "the deal is not actually random"


# ------------------------------------------------------------- turn order

def test_the_turn_order_is_always_a_rotation_of_the_clockwise_cycle():
    for seed in range(300):
        order = seats.draw_turn_order(random.Random(seed))
        assert sorted(order) == [0, 1, 2, 3]
        # a rotation of the one cycle, not an arbitrary permutation
        start = CLOCKWISE_OWNERS.index(order[0])
        assert order == [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]


def test_every_colour_can_open_the_game():
    """The first player is uniform over the four owners."""
    firsts = set()
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for seed in range(400):
        order = seats.draw_turn_order(random.Random(seed))
        firsts.add(order[0])
        counts[order[0]] += 1
    assert firsts == {0, 1, 2, 3}
    for owner, n in counts.items():
        assert 60 < n < 140, (owner, n)


def test_the_opening_order_is_the_anchored_draw_it_always_was():
    """Adding `PLAYER_OWNER`'s corner to the draw is still uniform, and keeping
    the anchor is what stops every seeded game from landing on a new board."""
    for seed in range(20):
        rng = random.Random(seed)
        r = rng.randrange(4)
        want = [CLOCKWISE_OWNERS[(CLOCKWISE_OWNERS.index(PLAYER_OWNER) + r + i)
                                 % 4] for i in range(4)]
        assert seats.draw_turn_order(random.Random(seed)) == want


def test_the_corner_a_seat_owns_does_not_depend_on_its_option_or_colour():
    """The corner follows from the owner id alone, which is why any colour may
    open the game."""
    assert [OWNER_CORNER[o] for o in CLOCKWISE_OWNERS] == \
        [(0, 0), (19, 0), (19, 19), (0, 19)]
    for corner in OWNER_CORNER.values():
        assert corner in ((0, 0), (19, 0), (19, 19), (0, 19))
    assert seats.corner_of(0) == OWNER_CORNER[0]


# ------------------------------------------------------- setup_seats, in a Game

def test_setup_seats_takes_four_options_that_may_repeat():
    rng = random.Random(3)
    g = Game(rng)
    g.setup_seats(["wolf"] * 4, ["blue", "green", "red", "yellow"], rng)
    assert g.state == "SETUP_INFO"
    assert g.seat_kinds == {o: seats.KIND_AI for o in range(4)}
    assert [g.owner_key[o] for o in range(4)] == ["wolf"] * 4
    for o in range(4):
        assert g.brains[o].key == "wolf"
    assert sorted(g.colors.values()) == ["blue", "green", "red", "yellow"]


def test_four_humans_and_two_recording_humans_are_both_legal():
    rng = random.Random(4)
    g = Game(rng)
    g.setup_seats(["human"] * 4, None, rng)
    assert g.humans() == [0, 1, 2, 3]
    assert g.recording_seats() == []
    assert g.brain_map() == {}
    assert all(b is None for b in g.brains.values())

    rng = random.Random(5)
    g = Game(rng)
    g.setup_seats(["human_log", "chess", "human_log", "human"], None, rng)
    assert g.humans() == [0, 2, 3]
    assert g.recording_seats() == [0, 2]
    assert sorted(g.brain_map()) == [1]


def test_no_human_at_all_is_legal():
    rng = random.Random(6)
    g = Game(rng)
    g.setup_seats(["wolf", "fox", "chess", "hunter"], None, rng)
    assert g.humans() == []
    assert len(g.brain_map()) == 4


def test_setup_seats_refuses_the_wrong_number_of_seats():
    rng = random.Random(7)
    g = Game(rng)
    for bad in (["wolf"] * 3, ["wolf"] * 5, []):
        with pytest.raises(ValueError):
            g.setup_seats(bad, None, rng)
    with pytest.raises(ValueError):
        g.setup_seats(["wizard"] * 4, None, rng)
    with pytest.raises(ValueError):
        g.setup_seats(["wolf"] * 4, ["blue", "blue", "red", "yellow"], rng)


def test_is_human_falls_back_to_the_old_convention_before_setup():
    """`setup_match` and `set_player_color` do not set seat kinds, and seat 0 is
    the person in both. A caller that never used `setup_seats` still gets that."""
    g = Game(random.Random(8))
    assert g.seats_configured() is False
    assert g.is_human(PLAYER_OWNER)
    assert not g.is_human(1)
    g.set_player_color("blue")
    assert g.seats_configured() is False
    assert g.is_human(PLAYER_OWNER)


# ------------------------------------------- the rules under all four openers

@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_a_full_game_under_each_opening_player(start, seed=17):
    """Legality, the opening rule, scoring and the game-over test, with each of
    the four owners opening in turn.

    `Game.act` raises on an illegal move, so a completed game is itself the
    legality check; the explicit assertions below are the parts a completed
    game would not catch.
    """
    rng = random.Random(seed)
    g = Game(rng)
    order = [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]
    g.turn_order = order
    g.owner_key = {o: "chess" for o in range(4)}
    g.brains = {o: ai.make_brain("chess", rng) for o in range(4)}
    g.colors = {o: c for o, c in zip(CLOCKWISE_OWNERS, seats.COLOR_NAMES)}
    g.start()

    assert g.current_owner() == order[0]
    assert g.must_cover(order[0]) == OWNER_CORNER[order[0]]

    opened = set()
    steps = 0
    while g.state == "PLAYING" and steps < 400:
        steps += 1
        owner = g.current_owner()
        # a player who has not opened must have their own corner available
        if g.placed[owner] == 0:
            opened.add(owner)
            assert g.has_legal(owner), (owner, "stuck on the opening move")
        # the turn follows the clockwise cycle, one step per turn
        assert owner == order[(steps - 1) % 4]
        move = ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], rng,
                              other_brains=g.brain_map(),
                              must_cover=g.must_cover(owner),
                              reach=g.reach(owner))
        if move is None:
            g.act_pass()
        else:
            g.act(*move)

    assert g.state == "GAME_OVER"
    assert opened == {0, 1, 2, 3}, opened
    assert all(g.stuck[o] for o in range(4))
    remaining = [g.remaining_cells(o) for o in range(4)]
    assert all(isinstance(r, int) and 0 <= r <= 89 for r in remaining)
    # scoring is a ranking by squares left, and it agrees with the engine
    assert [r for _o, r in g.standings()] == sorted(remaining)


def test_the_engine_agrees_about_legality_under_each_opening_player():
    """`Game` and `engine` are separate implementations; the seat feature must
    not make them disagree about which moves are legal."""
    from rl.actions import legal_indices
    from rl.imitation import state_from_game

    for start in range(4):
        rng = random.Random(30 + start)
        g = Game(rng)
        g.turn_order = [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]
        g.owner_key = {o: "chess" for o in range(4)}
        g.brains = {o: ai.make_brain("chess", rng) for o in range(4)}
        g.colors = {o: c for o, c in zip(CLOCKWISE_OWNERS, seats.COLOR_NAMES)}
        g.start()
        for _ in range(25):
            owner = g.current_owner()
            move = ai.choose_move(g.board, g.hands[owner].names, owner,
                                  g.brains[owner], rng, other_brains=None,
                                  must_cover=g.must_cover(owner),
                                  reach=g.reach(owner))
            if move is None:
                g.act_pass()
                continue
            state = state_from_game(g)
            legal = set(int(i) for i in legal_indices(state, owner))
            assert len(legal) > 0, (start, owner)
            # every move the chooser can return must be one the engine has
            from rl.actions import move_to_index
            name, oi, x, y = move
            idx = move_to_index((engine.PIECE_ORDER.index(name), oi, x + y * 20))
            assert idx in legal, (start, owner, move)
            g.act(*move)


def test_every_owner_may_open_on_its_own_corner():
    """The opening rule is `OWNER_CORNER[owner]`, with no reference to colour,
    to turn order, or to who went first."""
    for owner in range(4):
        rng = random.Random(40 + owner)
        g = Game(rng)
        g.turn_order = [CLOCKWISE_OWNERS[(owner + i) % 4] for i in range(4)]
        g.owner_key = {o: "chess" for o in range(4)}
        g.brains = {o: ai.make_brain("chess", rng) for o in range(4)}
        g.colors = {o: c for o, c in zip(CLOCKWISE_OWNERS, seats.COLOR_NAMES)}
        g.start()
        assert g.current_owner() == CLOCKWISE_OWNERS[owner]
        cx, cy = OWNER_CORNER[g.current_owner()]
        move = ai.choose_move(g.board, g.hands[0].names, g.current_owner(),
                              g.brains[g.current_owner()], rng,
                              other_brains=None,
                              must_cover=g.must_cover(g.current_owner()),
                              reach=g.reach(g.current_owner()))
        assert move is not None
        name, oi, x, y = move
        # the opening piece covers this seat's own corner square
        assert (cx - x, cy - y) in set(MASTER[name]["orientations"][oi]), \
            (owner, move, (cx, cy))

# ------------------------------------------------------- the deferred "random AI"

def test_the_menu_offers_one_more_than_there_are_options():
    """Fifteen things to pick from, fourteen things to be. The fifteenth is a
    request rather than a setting."""
    menu = seats.seat_menu_options()
    assert len(menu) == 14
    assert len(set(menu)) == 14
    assert menu[-1] == seats.RANDOM_AI_KEY
    assert set(menu) - set(seats.seat_options(True)) == {seats.RANDOM_AI_KEY}
    assert seats.seat_options(True) == menu[:-1]
    assert len(seats.seat_options(False)) == 11
    assert seats.automated_options() == seats.seat_options(False)


def test_random_ai_is_a_kind_of_request_not_of_contestant():
    assert seats.kind_of(seats.RANDOM_AI_KEY) == seats.KIND_RANDOM_AI
    assert not seats.is_human_kind(seats.KIND_RANDOM_AI)


def test_random_ai_draws_from_the_eleven_automated_options():
    assert len(seats.automated_options()) == 11
    pool = seats.automated_options()
    assert len(pool) == 11 and len(set(pool)) == 11
    assert not any(k.startswith("human") for k in pool)
    assert any(seats.is_imitation_key(k) for k in pool)
    assert any(seats.is_rl_key(k) for k in pool)
    # neither a checkpoint nor a trained policy is a personality, and nothing
    # else is in the pool either
    assert set(k for k in pool
               if not seats.is_imitation_key(k) and not seats.is_rl_key(k)) == \
        set(ai.personality_keys())


def test_random_ai_covers_all_ten_and_never_a_human():
    seen = set()
    repeats = 0
    for seed in range(400):
        out = seats.resolve_random_ai([seats.RANDOM_AI_KEY] * 4,
                                      random.Random(seed))
        assert len(out) == 4
        assert set(out) <= set(seats.automated_options())
        seen |= set(out)
        repeats += int(len(set(out)) < 4)
    assert seen == set(seats.automated_options())
    # 4 draws from 11 collide often; zero collisions in 400 games would mean the
    # draws are not independent
    assert repeats > 40, repeats


def test_each_random_seat_draws_independently():
    """The same option landing on two seats is legal, exactly as it is for a
    colour - that is what "each draws separately, repeats allowed" means."""
    both = 0
    for seed in range(200):
        out = seats.resolve_random_ai([seats.RANDOM_AI_KEY] * 4,
                                      random.Random(seed))
        if len(set(out)) == 1:
            both += 1
    # four independent draws from 11 all landing the same: 4 * (1/11)^3 ≈ 0.3%
    assert both <= 3, both


def test_resolution_leaves_the_other_seats_alone():
    out = seats.resolve_random_ai(["wolf", seats.RANDOM_AI_KEY, "human",
                                   seats.RANDOM_AI_KEY], random.Random(9))
    assert out[0] == "wolf"
    assert out[2] == "human"
    assert out[1] in seats.automated_options()
    assert out[3] in seats.automated_options()


def test_resolution_is_deterministic_for_a_seed():
    for seed in (0, 1, 42, 20260903):
        keys = ["chess", seats.RANDOM_AI_KEY, seats.RANDOM_AI_KEY, "human"]
        a = seats.resolve_random_ai(keys, random.Random(seed))
        b = seats.resolve_random_ai(keys, random.Random(seed))
        assert a == b, (seed, a, b)


def test_a_game_never_contains_a_deferred_choice():
    """Everything downstream of `setup_seats` - the seat kinds, the brains, the
    log, the leaderboard - depends on every seat naming something specific."""
    for seed in range(25):
        rng = random.Random(seed)
        g = Game(rng)
        g.setup_seats([seats.RANDOM_AI_KEY] * 4, None, rng)
        assert all(k in seats.seat_options(False) for k in g.owner_key.values())
        assert all(k != seats.RANDOM_AI_KEY for k in g.owner_key.values())
        assert set(g.seat_kinds.values()) <= {seats.KIND_AI,
                                              seats.KIND_IMITATION,
                                              seats.KIND_RL}
        for o in range(4):
            brain = g.brains[o]
            assert brain is not None
            assert brain.key == g.owner_key[o]


def test_four_random_seats_mix_ai_and_checkpoints():
    kinds = set()
    for seed in range(60):
        rng = random.Random(seed)
        g = Game(rng)
        g.setup_seats([seats.RANDOM_AI_KEY] * 4, None, rng)
        kinds |= set(g.seat_kinds.values())
    assert kinds == {seats.KIND_AI, seats.KIND_IMITATION,
                     seats.KIND_RL}, kinds


def test_a_random_seat_beside_a_human_still_resolves():
    rng = random.Random(11)
    g = Game(rng)
    g.setup_seats([seats.RANDOM_AI_KEY, "human", seats.RANDOM_AI_KEY,
                   "human_log"],
                  ["blue", "green", "red", "yellow"], rng)
    assert g.owner_key[1] == "human" and g.owner_key[3] == "human_log"
    assert g.owner_key[0] in seats.automated_options()
    assert g.owner_key[2] in seats.automated_options()
    assert g.humans() == [1, 3]
    assert g.recording_seats() == [3]
    assert g.brains[1] is None and g.brains[3] is None
    assert sorted(g.brain_map()) == [0, 2]


def test_setup_seats_refuses_an_option_that_was_never_resolved():
    """`build_brain` is the backstop, for a caller that skips `setup_seats`."""
    rng = random.Random(12)
    with pytest.raises(ValueError):
        seats.build_brain(seats.RANDOM_AI_KEY, rng)
    with pytest.raises(ValueError):
        seats.build_brain(seats.RANDOM_AI_KEY, rng, seats.KIND_RANDOM_AI)
