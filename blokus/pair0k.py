"""Measure one 0k imitation student against hc_1000 on the same games, paired.

This layer answers one question: how does a single 0k checkpoint score against
`hc_1000` when both play the same games - same opponents, same colours, same
boards - once per selection mode?

The output's schema is the one `eval/imitation/rl_b1000_0k-pair.json` and
`eval/imitation/rl_i1000_0k-pair.json` already use, field for field, so the
three files can be read side by side. What neither existing file records is its
**pool**, and neither records how its entropy numbers were defined, so nothing
here can claim to be strictly comparable with them. That is written into
`protocol_check` inside the file rather than left for a reader to discover.

`protocol_check` also says that hc_1000's md5 and its four aggregates were
**recomputed in this run**. The two existing files asserted them against a
baseline stored in /tmp, which is outside the repository and is gone after a
reboot; a recomputed value is honest about what it is, and it cannot fail to
agree with a file nobody has.

Pairing is the whole design: `match.run_league` with `paired_rng=True` draws
each game's setup and each seat's decisions from streams named by
`(seed, game index)`, and `subject_draw` places the designated seat from the
same draw regardless of which model fills it. The control run and the student
run therefore differ in exactly one respect - which checkpoint sits in that
seat. The script verifies this per game and refuses to write a file if any
opponent or colour moved.

Usage:

    .venv-rl/bin/python pair0k.py \
        --student rl_o1000_0k --pool no_imitation \
        --games 200 --seed 6100000 --device cuda \
        --out eval/imitation/rl_o1000_0k-pair-no_imitation.json
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import match  # noqa: E402
import rl.imitation as I  # noqa: E402
import seats  # noqa: E402

# `hc_1000` is spelled exactly like this in the two files that already exist,
# block key and all, so the control is fixed rather than configurable: a second
# spelling would make the third file's schema differ from the other two.
CONTROL = "hc_1000"
MODES = ("argmax", "softmax")


def md5_of(path):
    """Hex digest of a file, read in chunks because checkpoints are megabytes."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_of(key):
    """The file `key` loads from - the registry's answer, not a guess."""
    return seats.registry_checkpoint(seats.canonical_key(key))


def softmax_entropy(logits, temperature=I.DEFAULT_TEMPERATURE):
    """Entropy in nats of the temperature-scaled distribution over legal moves.

    The logits arrive already masked to `-1e9` off the legal set, so the
    distribution is over the legal moves whether or not the seat is playing
    argmax - which is what makes the number comparable between the two modes.
    """
    import torch
    t = max(1e-6, float(temperature))
    probs = torch.softmax(logits.float() / t, dim=-1)
    probs = probs[probs > 0]
    if probs.numel() == 0:
        return 0.0
    return float(-(probs * probs.log()).sum())


def stderr_of(values):
    """Sample standard error of the mean (ddof=1).

    This is the definition the existing pair files use: their stored
    `points_stderr` equals `statistics.stdev(points) / sqrt(n)` to the last
    digit, so anything else here would put two different standard errors in
    three files that claim to be the same measurement.
    """
    if len(values) < 2:
        raise ValueError("a standard error needs at least two values")
    return statistics.stdev(values) / math.sqrt(len(values))


def paired_block(control, student):
    """The `paired` block: `student - control`, per game, both standard errors.

    `independent_se` is the unpaired one, `sqrt(se_c^2 + se_s^2)`, kept because
    the existing files keep it: it is the yardstick that shows what pairing
    bought, not an alternative judgement.
    """
    diffs = [s - c for c, s in zip(control, student)]
    mean_diff = statistics.fmean(diffs)
    paired_se = stderr_of(diffs)
    control_se = stderr_of(control)
    student_se = stderr_of(student)
    return {
        "independent_se": math.sqrt(control_se ** 2 + student_se ** 2),
        "mean_diff": mean_diff,
        "n": len(diffs),
        "paired_se": paired_se,
        "r": statistics.correlation(control, student),
        "t": mean_diff / paired_se,
    }


@contextlib.contextmanager
def record_decisions():
    """Collect one run's per-decision entropies from the network seat.

    `ImitationBrain.choose` calls `pick_action` as a module global, so replacing
    that name for the length of one run intercepts every network decision and
    nothing else: the rule-based personalities never come near it. The move is
    the original's, unchanged - this observes, it does not choose.
    """
    entropies = []
    original = I.pick_action

    def spy(logits, mode=I.DEFAULT_MODE, rng=None,
            temperature=I.DEFAULT_TEMPERATURE):
        move = original(logits, mode, rng, temperature)
        entropies.append(softmax_entropy(logits, temperature))
        return move

    I.pick_action = spy
    try:
        yield entropies
    finally:
        I.pick_action = original


def series_for(rows, key):
    """Per-game `points`, `rank` and `remaining` of the seat held by `key`.

    Exactly one row per game, because the key is the designated seat and the
    pool holds no network to be drawn as an opponent by mistake. Two rows would
    mean the subject leaked into the opponents, and silently averaging them
    would be the bug this raises on.
    """
    points, ranks, remaining = [], [], []
    for index, game in enumerate(rows):
        hits = [row for row in game if row[0] == key]
        if len(hits) != 1:
            raise RuntimeError(
                "game %d: expected exactly one %r seat, found %d"
                % (index, key, len(hits)))
        _key, _colour, rem, rank, pts = hits[0]
        points.append(pts)
        ranks.append(rank)
        remaining.append(rem)
    return {"points": points, "rank": ranks, "remaining": remaining}


def verify_pairing(control_rows, student_rows, control_key, student_key):
    """Raise unless the two runs played the same games with one seat swapped.

    Checked per game on what the standings actually carry: the designated
    seat's colour, and the multiset of `(opponent key, colour)` for the other
    three. The opening player and the board follow from the same setup stream
    and are covered by the suite's own paired-rng tests.
    """
    if len(control_rows) != len(student_rows):
        raise RuntimeError("the two runs have different game counts")
    for index, (a, b) in enumerate(zip(control_rows, student_rows)):
        seat_a = [row for row in a if row[0] == control_key]
        seat_b = [row for row in b if row[0] == student_key]
        if len(seat_a) != 1 or len(seat_b) != 1:
            raise RuntimeError("game %d: designated seat missing" % index)
        if seat_a[0][1] != seat_b[0][1]:
            raise RuntimeError("game %d: designated seat changed colour"
                               % index)
        opponents_a = sorted((row[0], row[1]) for row in a
                             if row[0] != control_key)
        opponents_b = sorted((row[0], row[1]) for row in b
                             if row[0] != student_key)
        if opponents_a != opponents_b:
            raise RuntimeError("game %d: opponents differ: %r vs %r"
                               % (index, opponents_a, opponents_b))
    return True


def run_one_mode(mode, pool, seed, games, device, key):
    """`(rows, entropies)` for one run of one mode with `key` in the subject seat."""
    with record_decisions() as entropies:
        rows = match.run_league(games=games, seed=seed, records=None,
                                options=pool, mode=mode, device=device,
                                paired_rng=True, subject=key)
    if not entropies:
        raise RuntimeError("no network decision was recorded for %r" % key)
    return rows, entropies


def mode_block(key, path, mode, seed, games, series, entropies):
    """The per-model stats block, in the field order the existing files use."""
    return {
        "checkpoint": path,
        "decisions": len(entropies),
        "entropy_sd_over_decisions": statistics.stdev(entropies),
        "games": games,
        "key": key,
        "mean_points": statistics.fmean(series["points"]),
        "mean_remaining": statistics.fmean(series["remaining"]),
        "mean_softmax_entropy": statistics.fmean(entropies),
        "mode": mode,
        "points_stderr": stderr_of(series["points"]),
        "rank1_share": (sum(1 for r in series["rank"] if r == 1) / games),
        "remaining_stderr": stderr_of(series["remaining"]),
        "seed_base": seed,
        "seed_range": [seed, seed + games - 1],
    }


def build_payload(student, pool_raw, pool, seed, games, device, blocks):
    """The pair file itself: twelve top-level fields, the same twelve as b and i."""
    control_path = checkpoint_of(CONTROL)
    student_path = checkpoint_of(student)
    protocol_check = (
        "pool = %s -> [%s]; recorded here because rl_b1000_0k-pair.json and "
        "rl_i1000_0k-pair.json record no pool and no entropy definition, so "
        "strict comparability with them is unproven (b、i 原池未記錄,故與其"
        "嚴格可比性未證). hc_1000's checkpoint md5 and its four aggregates - "
        "mean points, mean entropy, decision count, rank-1 share - are "
        "recomputed in this run and recorded as recomputed values, not "
        "asserted bit-for-bit against a stored baseline; that baseline lived "
        "in /tmp and is gone. Pairing was verified in this run: opponents and "
        "the designated seat's colour are identical in all %d games of both "
        "runs. device = %s; standard errors are sample standard errors "
        "(ddof=1); entropy is the natural log of the temperature-1.0 "
        "distribution over the masked legal moves, averaged over decisions."
        % (pool_raw, ", ".join(pool), games, device)
    )
    seed_rule = (
        "game i draws its five streams by blake2b over (seed, i, 'setup') and "
        "(seed, i, 'seat', owner), so a game's layout is a function of the seed "
        "alone; the designated seat index and the three opponents come from the "
        "setup stream and the draw does not read the subject's name, so the "
        "control run and the student run differ only in which checkpoint sits "
        "in that seat"
    )
    return {
        "checkpoint": student_path,
        "checkpoint_md5": md5_of(student_path),
        "control": CONTROL,
        "control_checkpoint": control_path,
        "control_md5": md5_of(control_path),
        "modes": blocks,
        "n_games": games,
        "protocol_check": protocol_check,
        "seed_base": seed,
        "seed_range": [seed, seed + games - 1],
        "seed_rule": seed_rule,
        "subject": student,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="成對量測一個 0k 模仿檢查點對 hc_1000")
    ap.add_argument("--student", default="rl_o1000_0k",
                    help="被測 0k 檢查點的 key（預設 rl_o1000_0k）")
    ap.add_argument("--pool", default="no_imitation",
                    help="對手池：preset 或逗號分隔名單（預設 no_imitation，"
                         "即 7 個人格）")
    ap.add_argument("--pool-order", choices=("sorted", "literal"),
                    default="sorted", help="池順序，與 match.py 同義")
    ap.add_argument("--games", type=int, default=200, help="局數（預設 200）")
    ap.add_argument("--seed", type=int, default=6_100_000,
                    help="種子基底；第 i 局用第 i 個派生流（預設 6100000）")
    ap.add_argument("--device", default="cuda",
                    help="模仿選項的 torch device（預設 cuda）")
    ap.add_argument("--out", default="eval/imitation/"
                                     "rl_o1000_0k-pair-no_imitation.json",
                    help="輸出檔；已存在則拒絕寫入，不覆寫")
    args = ap.parse_args(argv)

    if os.path.exists(args.out):
        raise SystemExit("%s already exists; refusing to overwrite it"
                         % args.out)
    student = seats.canonical_key(args.student)
    if seats.kind_of(student) not in (seats.KIND_IMITATION, seats.KIND_RL):
        raise SystemExit("--student must name a checkpoint seat, got %r"
                         % args.student)
    control_key = seats.canonical_key(CONTROL)
    pool = match.expand_pool(args.pool, order=args.pool_order)

    print("池        : %s（%d 席）" % (args.pool, len(pool)))
    print("被測      : %s" % student)
    print("對照      : %s" % CONTROL)
    print("局數/種子 : %d / %d" % (args.games, args.seed))
    print("device    : %s" % args.device)
    print()

    blocks = {}
    for mode in MODES:
        print("%s ..." % mode, flush=True)
        control_rows, control_ent = run_one_mode(
            mode, pool, args.seed, args.games, args.device, control_key)
        student_rows, student_ent = run_one_mode(
            mode, pool, args.seed, args.games, args.device, student)
        verify_pairing(control_rows, student_rows, control_key, student)
        control_series = series_for(control_rows, control_key)
        student_series = series_for(student_rows, student)
        blocks[mode] = {
            "games": {
                "points": {"hc_1000": control_series["points"],
                           "student": student_series["points"]},
                "rank": {"hc_1000": control_series["rank"],
                         "student": student_series["rank"]},
                "remaining": {"hc_1000": control_series["remaining"],
                              "student": student_series["remaining"]},
            },
            CONTROL: mode_block(CONTROL, checkpoint_of(CONTROL), mode,
                                args.seed, args.games, control_series,
                                control_ent),
            "paired": {
                "orientation": "%s - %s, same games" % (student, CONTROL),
                "points": paired_block(control_series["points"],
                                       student_series["points"]),
                "remaining": paired_block(control_series["remaining"],
                                          student_series["remaining"]),
            },
            "student": mode_block(student, checkpoint_of(student), mode,
                                  args.seed, args.games, student_series,
                                  student_ent),
        }
        paired = blocks[mode]["paired"]["points"]
        print("   mean_diff %+0.4f  paired_se %.4f  t %+.2f  (n=%d)"
              % (paired["mean_diff"], paired["paired_se"], paired["t"],
                 paired["n"]))

    payload = build_payload(student, args.pool, pool, args.seed, args.games,
                            args.device, blocks)
    directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(directory, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print()
    print("已寫入 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
