"""A suspicion about `engine.normalize`, recorded rather than fixed.

The suspicion, from `plan3.md`: `normalize` renumbers the players so the mover
is owner 0, but the opening rule reads `OWNER_CORNER[owner]`, and owner 0's
corner is the bottom-right square. So calling the engine's rules on a
normalised view, before the mover has opened, may offer them owner 0's corner
instead of their own.

Whether that is true is decided here by measurement, and the answer is reported
either way. The test asserts nothing: it collects the counts and prints them,
and the numbers go into `reports/g_report.md` section G5. Nothing in `engine.py`
is touched either way, and nothing in `rl/` uses the result - `rl` always
computes legal moves on real coordinates and rotates them afterwards, which is
why this is a curiosity about the engine and not a bug in the new code.

Run it directly to see the numbers:

    python3 -m pytest tests/test_rl_normalize_probe.py -q -s
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
from bench_engine import random_playout  # noqa: E402
import rl.actions as A  # noqa: E402

WANT = 200


def collect(seed_games):
    """Positions where the mover has not opened yet, plus the opened ones."""
    unopened = []
    opened = []
    for seed in range(seed_games):
        tape = []
        random_playout(seed, tape=tape, max_passes=60)
        for s, mask, _n, _mv in tape:
            if mask == 0:
                continue
            mover_opened = s.own_bits[s.to_move] != 0
            (opened if mover_opened else unopened).append((seed, s))
            if len(unopened) >= WANT * 2 and len(opened) >= WANT:
                break
        if len(unopened) >= WANT * 2 and len(opened) >= WANT:
            break
    return unopened, opened


def compare(state):
    """A against B, per `plan3.md`.

        A = real_to_view(legal_indices(real state), p)
        B = legal_indices(engine.normalize(state)[0])

    Both are sets of action indices in the mover's frame. If the two agree,
    computing legality on the view is safe; if not, only A is.
    """
    p = A.seat_of_mover(state)
    a = sorted(A.real_to_view(A.legal_indices(state), p).tolist())
    view, _seat_of = engine.normalize(state)
    b = sorted(A.legal_indices(view).tolist())
    return a, b


def probe(pairs, label):
    equal = 0
    differ = 0
    per_owner = {o: {"equal": 0, "differ": 0, "positions": 0} for o in range(4)}
    examples = []
    for seed, state in pairs[:WANT]:
        a, b = compare(state)
        owner = state.to_move
        per_owner[owner]["positions"] += 1
        if a == b:
            equal += 1
            per_owner[owner]["equal"] += 1
        else:
            differ += 1
            per_owner[owner]["differ"] += 1
            if len(examples) < 2:
                examples.append({
                    "seed": seed,
                    "to_move": owner,
                    "real_legal": len(a),
                    "view_legal": len(b),
                    "only_in_real": len(set(a) - set(b)),
                    "only_in_view": len(set(b) - set(a)),
                    "serialize": engine.serialize(state),
                })
    return {
        "label": label,
        "positions": len(pairs[:WANT]),
        "equal": equal,
        "differ": differ,
        "share_differing": differ / len(pairs[:WANT]) if pairs[:WANT] else None,
        "by_mover": per_owner,
        "examples": examples,
    }


def test_the_normalize_opening_corner_suspicion():
    unopened, opened = collect(60)
    result = {"mover_not_yet_opened": probe(unopened, "mover has not opened"),
              "mover_already_opened": probe(opened, "mover has opened")}
    print("\n--- G5 normalize probe (measurement, not an assertion) ---")
    for key, row in result.items():
        print("%-28s positions %3d  equal %3d  differ %3d  share_differing %s"
              % (key, row["positions"], row["equal"], row["differ"],
                 "n/a" if row["share_differing"] is None
                 else "%.4f" % row["share_differing"]))
        for owner in range(4):
            c = row["by_mover"][str(owner)] if str(owner) in row["by_mover"] \
                else row["by_mover"][owner]
            print("    mover owner %d: %3d positions, %3d equal, %3d differ"
                  % (owner, c["positions"], c["equal"], c["differ"]))
        for ex in row["examples"]:
            print("    example seed %d to_move %d: real %d, view %d, "
                  "only-in-real %d, only-in-view %d"
                  % (ex["seed"], ex["to_move"], ex["real_legal"],
                     ex["view_legal"], ex["only_in_real"], ex["only_in_view"]))
    print("--- end G5 probe ---\n")


def test_the_probe_really_exercises_the_unopened_case():
    """If every sampled position had the mover already opened, the finding
    above would be about the wrong question."""
    unopened, opened = collect(60)
    assert unopened, "no position where the mover has not opened"
    for _seed, state in unopened[:20]:
        assert state.own_bits[state.to_move] == 0
    assert len(opened) >= WANT, \
        "only %d opened positions" % len(opened)


def test_the_view_really_puts_the_mover_in_the_top_left_corner():
    """The premise the suspicion rests on. If this failed, the whole probe
    would be measuring something else."""
    for _seed, state in collect(10)[0][:20]:
        view, seat_of = engine.normalize(state)
        assert view.to_move == 0
        assert seat_of[0] == state.to_move
        assert state.own_bits[state.to_move] == 0