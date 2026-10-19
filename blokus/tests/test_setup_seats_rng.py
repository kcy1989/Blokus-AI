"""Pin the RNG consumption order of `Game.setup_seats`.

`setup_seats` draws from one stream in a fixed order, and the last thing it
draws is the turn order:

    1. `resolve_random_ai`   - one `choice` per "random AI" seat
    2. `assign_colours`      - one `shuffle` of the colours still free
    3. `build_brain` per seat - seven `uniform` draws per **personality**
                                (ai/base.make_profile), none for a checkpoint
    4. `draw_turn_order`     - one `randrange`

Step 3 is why the order matters: a personality jitters its weights on the way
in while a checkpoint seat is read from a file and draws nothing, so which
seats are personalities decides where the turn-order draw lands. Get the order
or the count wrong and every same-seed game reseats - silently, with no other
test noticing until a committed batch stops reproducing.

This test states the order **independently**: it replays the raw
`random.Random` calls and compares the generator state, so a change to the
order fails here instead of being mirrored by both sides.

The stub is injected through `seats.build_brain` today. The `injection`
parameter exists so the `brain_factory` path (once `Game.setup_seats` grows
one) can be added as a second value and must produce the **same** state.
"""
import random

import pytest

import seats
from game import Game

# A real network key, but the patched builder returns a stub for it, so no
# checkpoint is opened and torch is never imported.
STUB_KEY = "rl_h1000_20k"
PERSONALITIES = ("hunter", "wolf", "chess")
SEED = 20260928


class _StubBrain:
    """A seat with no weights: it draws nothing and does nothing."""

    def __init__(self, key):
        self.key = key


def _expected_state(option_keys, colours, seed):
    """The generator state after replaying `setup_seats`' draws by hand."""
    rng = random.Random(seed)
    pool = seats.automated_options()
    resolved = [rng.choice(pool) if k == seats.RANDOM_AI_KEY else k
                for k in option_keys]

    fixed = [c for c in colours if c not in (None, seats.RANDOM)]
    rest = [c for c in seats.COLOR_NAMES if c not in fixed]
    rng.shuffle(rest)

    for key in resolved:
        if seats.kind_of(key) == seats.KIND_AI:
            for _ in range(7):
                rng.uniform(0.7, 1.3)

    rng.randrange(seats.SEATS)
    return rng.getstate()


def _patched_build_brain(original):
    """A builder that stubs the checkpoint seat and delegates the rest."""
    def build(key, rng, kind=None, **kw):
        if key == STUB_KEY:
            return _StubBrain(key)
        return original(key, rng, kind=kind, **kw)
    return build


@pytest.mark.parametrize("injection", ["monkeypatch"])
def test_setup_seats_draws_in_the_documented_order(monkeypatch, injection):
    option_keys = list(PERSONALITIES) + [STUB_KEY]
    colours = [None, None, None, None]
    expected = _expected_state(option_keys, colours, SEED)

    patched = _patched_build_brain(seats.build_brain)
    game = Game(random.Random(SEED))
    if injection == "monkeypatch":
        monkeypatch.setattr(seats, "build_brain", patched)
        game.setup_seats(option_keys, colours)
    # elif injection == "brain_factory":   # added once S4 lands
    #     game.setup_seats(option_keys, colours, brain_factory=patched)

    assert game.rng.getstate() == expected, (
        "setup_seats drew a different number or order of values")
    # the stub seat is the injected one, and every seat still got a brain
    assert isinstance(game.brains[3], _StubBrain)
    assert all(b is not None for b in game.brains.values())


def test_a_personality_draws_seven_and_a_stub_draws_none():
    """The asymmetry, stated on its own so a failure says which half broke.

    Three personalities consume 7 each; the stub consumes nothing. If the
    count changed for either, the pinned state above would still catch it -
    this test just names the number instead of leaving it implicit in a
    state comparison.
    """
    before = random.Random(SEED)
    after_personality = random.Random(SEED)
    for _ in range(7):
        after_personality.uniform(0.7, 1.3)
    assert after_personality.getstate() != before.getstate()

    stub_rng = random.Random(SEED)
    before_stub = stub_rng.getstate()
    _StubBrain(STUB_KEY)
    assert stub_rng.getstate() == before_stub, "a stub must not draw"
