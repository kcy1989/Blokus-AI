"""Pins for the two values the refactor must not move: the jittered weights
and the personality order.

These are *value* pins, not call-count pins. What must not change is the
number a profile carries and the sequence `personality_keys()` returns, so
that is what is written down. A test that counted `uniform` calls instead
would report a change in implementation while missing a change in the
values, and would hard-code the RNG's internals into the suite.

`make_profile` draws seven `uniform(0.7, 1.3)` values from the stream it is
handed, so with `random.Random(0)` the result is fixed for as long as
CPython's Mersenne Twister is - which is the same assumption every seeded
game in this repository already rests on.
"""
import dataclasses
import random

import ai


def test_make_profile_pins_the_seven_chess_weights():
    """`chess` is the spec the rule-based personalities borrow, so its
    jittered values sit under every rule-based seat as well as under chess."""
    prof = ai.make_profile("chess", random.Random(0))
    assert dataclasses.astuple(prof) == (
        1.206653110915029,
        1.1547726417641815,
        0.952342948498507,
        0.855350050175778,
        1.0067648328211651,
        0.9429604824702486,
        0.058513957671043185,
    )


def test_make_profile_pins_the_seven_wolf_weights():
    """A second, differently-scaled spec, so the pin is not passing by
    accident on a profile whose weights all start at 1.0."""
    prof = ai.make_profile("wolf", random.Random(0))
    assert dataclasses.astuple(prof) == (
        4.826612443660116,
        1.1547726417641815,
        1.904685896997014,
        0.2566050150527334,
        0.5033824164105826,
        2.8288814474107458,
        0.17554187301312954,
    )


def test_personality_keys_is_the_literal_seven():
    """The draw order `rng.sample` indexes into.

    Written out rather than compared to `PERSONALITY_ORDER`: the two are
    maintained by hand in different modules and a test that only says they
    agree would still pass if both were reordered together - which is
    exactly the change that reseats every same-seed game.
    """
    assert ai.personality_keys() == (
        "wolf", "chess", "fox", "intruder", "optimizer", "builder", "hunter",
    )
