"""An H-B2 checkpoint as a seat that plays.

Four things have to hold for a checkpoint to be a legal opponent, and each gets
its own test because each fails differently:

  * it reads the same 27 channels and masks the same way the loss did, so the
    weights mean what they were trained to mean;
  * it only ever names a move that is legal, under every opening player;
  * it carries no state, so one checkpoint can drive four colours at once and
    two seats on the same step do not interfere;
  * argmax is deterministic and softmax is a different, still legal, stream.

Every test here needs `data/hb2/step_002000.pt`, which H-B2 produced and which
is gitignored. The tests that need it are skipped when it is absent rather than
failing, because the repository is checked out without its data.
"""
import os
import random

import numpy as np
import pytest

import ai
import engine
import seats
from config import CLOCKWISE_OWNERS, PLAYER_OWNER
from game import Game

CHECKPOINT_DIR = seats.IMITATION_CHECKPOINT_DIR
STEP = seats.IMITATION_STEPS[0]


def _have_checkpoint(step=STEP):
    from rl.imitation import checkpoint_path
    return os.path.exists(checkpoint_path(step, CHECKPOINT_DIR))


needs_checkpoint = pytest.mark.skipif(
    not _have_checkpoint(), reason="no H-B2 checkpoint in %s" % CHECKPOINT_DIR)


def make_game(seed=0, options=None, colours=None, mode="argmax"):
    rng = random.Random(seed)
    g = Game(rng)
    keys = options or ["step_%d" % STEP] * 4
    g.setup_seats(keys, colours, rng, checkpoint_dir=CHECKPOINT_DIR,
                  mode=mode)
    g.start()
    return g, rng


def drive(g, rng, limit=60):
    """Play until `limit` turns.

    Returns `[(owner, move, state_before)]`. The state is captured **before**
    the move is played: legality is a property of the position a move was
    chosen in, and checking it against the position the move produced tests
    nothing.
    """
    import rl.imitation as I
    played = []
    for _ in range(limit):
        if g.state != "PLAYING":
            break
        owner = g.current_owner()
        state = I.state_from_game(g)
        move = ai.choose_move(g.board, g.hands[owner].names, owner,
                              g.brains[owner], rng,
                              other_brains=g.brain_map(),
                              must_cover=g.must_cover(owner),
                              reach=g.reach(owner))
        if move is None:
            g.act_pass()
            continue
        played.append((owner, move, state))
        g.act(*move)
    return played


# --------------------------------------------------------------- the basics

def test_the_module_does_not_import_torch():
    """A game with no imitation seat must not load torch, which is what lets
    `seats.build_brain` import this module lazily."""
    import subprocess
    import sys
    code = ("import sys; import rl.imitation; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.getcwd())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


def test_the_seat_pools_list_the_four_checkpoints():
    assert seats.IMITATION_STEPS == (2000, 10000, 22000, 38000)
    assert [seats.imitation_key(s) for s in seats.IMITATION_STEPS] == \
        ["step_2000", "step_10000", "step_22000", "step_38000"]


def test_an_imitation_seat_is_built_only_when_asked_for():
    """No torch, no checkpoint file: a game of personalities must not touch
    either."""
    rng = random.Random(0)
    g = Game(rng)
    g.setup_seats(["wolf", "fox", "hunter", "chess"], None, rng)
    assert all(b is not None for b in g.brains.values())
    assert all(b.key == k for b, k in
               zip((g.brains[o] for o in range(4)),
                   ("wolf", "fox", "hunter", "chess")))


# ----------------------------------------------------------- loading

@needs_checkpoint
def test_a_checkpoint_loads_read_only():
    import rl.imitation as I
    brain = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu")
    assert brain.step == STEP
    assert brain.key == "step_%d" % STEP
    assert brain.mode == "argmax"
    # read-only: eval mode, no gradients, no optimiser state carried
    assert not brain.net.training
    assert all(not p.requires_grad for p in brain.net.parameters())
    assert all(p.grad is None for p in brain.net.parameters())
    assert brain.mistake_rate == 0.0
    assert brain.uses_lookahead is False
    # the optimiser state is not even loaded
    blob = I._load_blob(I.checkpoint_path(STEP, CHECKPOINT_DIR))
    assert "optimizer" not in blob
    assert "state_dict" in blob


@needs_checkpoint
def test_the_input_is_the_same_27_channels_the_loss_saw():
    """The network's first convolution says how many channels it wants, and the
    features we hand it are that many."""
    import rl.imitation as I
    from rl.caches import features_27
    brain = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu")
    first = brain.net.trunk[0].weight
    g, _rng = make_game()
    x = features_27([I.state_from_game(g)])
    assert x.shape[1] == first.shape[1] == 27
    assert brain.net.policy.out_channels == 91


@needs_checkpoint
def test_illegal_slots_are_filled_with_the_loss_value():
    import torch

    import rl.imitation as I
    brain = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu")
    g, _rng = make_game()
    state = I.state_from_game(g)
    logits = I.logits_from_state(brain.net, state, brain.device)
    mask = I.legal_mask_view(state)
    assert mask.any()
    assert torch.allclose(logits[~mask],
                          torch.full((int((~mask).sum()),), I.MASK_FILL))
    assert int((logits <= I.MASK_FILL / 2).sum()) == int((~mask).sum())
    # and every legal slot keeps a real logit
    assert torch.all(logits[mask] > I.MASK_FILL / 2)


@needs_checkpoint
def test_the_mask_agrees_with_the_enumerated_candidates():
    """The network reads one set of legal slots and `choose_move` is offered
    another; a disagreement would either crash or let an illegal move through."""
    from rl.actions import legal_indices, real_to_view

    import rl.imitation as I
    g, rng = make_game()
    for _owner, move, state in drive(g, rng, 40):
        p = I.seat_of_mover(state)
        view_legal = set(real_to_view(legal_indices(state), p).tolist())
        name, oi, x, y = move
        from rl.actions import move_to_index
        real = move_to_index((engine.PIECE_ORDER.index(name), oi, x + y * 20))
        # the move is real, so it goes into the view before being looked up
        assert int(real_to_view(real, p)) in view_legal


# --------------------------------------------------------- it plays legally

@pytest.mark.parametrize("seed", [5, 9, 14, 22])
@needs_checkpoint
def test_a_full_game_of_four_imitation_seats(seed):
    g, rng = make_game(seed)
    first = g.turn_order[0]
    drive(g, rng, 500)
    assert g.state == "GAME_OVER", first
    assert all(g.stuck[o] for o in range(4))
    remaining = [g.remaining_cells(o) for o in range(4)]
    assert all(0 <= r <= 89 for r in remaining)


@needs_checkpoint
def test_it_plays_from_every_opening_player():
    """One checkpoint object driving all four seats, from each opening."""
    import rl.imitation as I
    shared = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu")
    for start in range(4):
        rng = random.Random(60 + start)
        g = Game(rng)
        g.turn_order = [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]
        g.owner_key = {o: "step_%d" % STEP for o in range(4)}
        g.colors = {o: c for o, c in zip(CLOCKWISE_OWNERS,
                                         seats.COLOR_NAMES)}
        g.brains = {o: shared for o in range(4)}
        g.attach_brains()
        g.state = "PLAYING"
        g.start()
        assert g.current_owner() == CLOCKWISE_OWNERS[start]
        drive(g, rng, 500)
        assert g.state == "GAME_OVER", start


@needs_checkpoint
def test_every_move_it_makes_is_legal():
    """`Game.act` raises on an illegal move, so a completed game is the check."""
    from rl.actions import legal_indices, move_to_index

    import rl.imitation as I
    g, rng = make_game(31)
    played = drive(g, rng, 60)
    assert played
    for owner, move, state in played:
        legal = set(int(i) for i in legal_indices(state, owner))
        name, oi, x, y = move
        assert move_to_index((engine.PIECE_ORDER.index(name), oi,
                              x + y * 20)) in legal, (owner, move)


# ------------------------------------------------------------ no state

@needs_checkpoint
def test_the_same_checkpoint_drives_four_colours_at_once():
    """One instance, four seats: the network keeps nothing between calls, so
    there is nothing for the colours to collide over."""
    import rl.imitation as I
    shared = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu")
    g, rng = make_game(11)
    g.brains = {o: shared for o in range(4)}
    g.attach_brains()
    played = drive(g, rng, 500)
    owners = {o for o, _m, _s in played}
    assert len(owners) == 4, owners
    assert g.state == "GAME_OVER"


@needs_checkpoint
def test_two_seats_on_the_same_step_get_independent_instances():
    """The weights are shared; the sampling stream is not, or one colour's draws
    would move another's."""
    import rl.imitation as I
    a = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu",
                     seed=1)
    b = I.load_brain(STEP, checkpoint_dir=CHECKPOINT_DIR, device="cpu",
                     seed=1)
    assert a is not b
    assert a.net is not b.net
    assert a.rng is not b.rng


@needs_checkpoint
def test_argmax_is_deterministic():
    g1, r1 = make_game(7)
    g2, r2 = make_game(7)
    assert drive(g1, r1, 30) == drive(g2, r2, 30)


@needs_checkpoint
def test_softmax_is_a_different_stream_and_still_legal():
    import rl.imitation as I
    g_arg, r_arg = make_game(13, mode="argmax")
    g_sm, r_sm = make_game(13, mode="softmax")
    a = drive(g_arg, r_arg, 500)
    b = drive(g_sm, r_sm, 500)
    # the two games need not have the same length - that is most of what
    # sampling buys - but they must differ and both must finish
    assert a != b, "softmax played exactly the same game as argmax"
    assert g_arg.state == g_sm.state == "GAME_OVER"


# ------------------------------------------------------- selection plumbing

def test_pick_action_argmax_and_softmax():
    import torch

    import rl.imitation as I
    logits = torch.tensor([0.0, 5.0, 1.0, -3.0])
    assert I.pick_action(logits, "argmax") == 1
    assert I.pick_action(logits, "argmax", random.Random(0)) == 1
    # a peaked distribution mostly returns the top slot under any seed
    for seed in range(20):
        assert I.pick_action(logits, "softmax", random.Random(seed),
                             temperature=0.01) == 1
    # a flat one spreads
    flat = torch.zeros(4)
    picks = {I.pick_action(flat, "softmax", random.Random(s)) for s in range(60)}
    assert len(picks) > 1
    with pytest.raises(ValueError):
        I.pick_action(logits, "greedy")


def test_checkpoint_path_is_the_one_hb2_wrote():
    from rl.imitation import checkpoint_path
    for step in seats.IMITATION_STEPS:
        assert checkpoint_path(step, CHECKPOINT_DIR).endswith(
            "step_%06d.pt" % step)


@needs_checkpoint
def test_stuck_from_state_never_contradicts_the_game_latch():
    """The features need the four stuck latches, and they are derived here
    rather than read off the `Game`.

    Two things are checked, and the asymmetry used to be the point:

      * `Game.stuck` claiming stuck while the engine still finds a legal move
        would be a real contradiction, and there is none - 0 across 10 games.
      * the other direction used to happen, and often. `Game._all_stuck`
        returned as soon as it met a seat that can still move, so seats later in
        the loop were not tested that pass and were not latched until every
        earlier seat was stuck. That was sound for the game-over test it was
        written for, but it meant the latch under-reported: measured over 10
        games it missed 246 seat-turns that the engine says were stuck.

        The latch no longer short-circuits (plan7-A). `Game._all_stuck` and
        `engine._advance` both scan every unlatched seat, so the two agree at
        every ply and the second assertion is now an equality.

    The cost of the fix is that the latch is no longer cheap: it calls
        `has_legal` once per unlatched seat instead of stopping early. That is
        a deliberate trade of a few percent of game time for four feature
        channels that no longer lie.
    """
    import rl.imitation as I
    under_reported = 0
    for seed in range(10):
        g, rng = make_game(seed)
        while g.state == "PLAYING":
            state = I.state_from_game(g)
            derived = I.stuck_from_state(state)
            latched = tuple(bool(x) for x in g.stuck)
            for o in range(4):
                assert not (latched[o] and not derived[o]), (seed, o)
                under_reported += int(derived[o] and not latched[o])
            drive(g, rng, 1)
        # by the end the sweep has seen everyone, so the two must agree
        assert tuple(bool(x) for x in g.stuck) == \
            I.stuck_from_state(I.state_from_game(g)), seed
    assert under_reported == 0, (
        "the latch under-reported %d seat-turns; the non-short-circuit scan in "
        "_all_stuck should have latched every stuck seat on the same move"
        % under_reported)


def test_the_features_carry_no_absolute_colour_or_seat():
    """The reason one checkpoint can play any colour: rotating the board
    renumbers every owner and changes nothing about what the network sees."""
    import rl.imitation as I
    from rl.caches import features_27
    g, rng = make_game(4)
    drive(g, rng, 12)
    state = I.state_from_game(g)
    before = features_27([state])
    after = features_27([engine.rotate_state(state)])
    assert np.array_equal(before, after)
    from rl.actions import legal_mask_view
    assert np.array_equal(legal_mask_view(state),
                          legal_mask_view(engine.rotate_state(state)))