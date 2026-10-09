"""The committed evidence in `eval/`: schema, internal consistency, written limits.

These tests read evidence, they do not run evaluations - one 3000-game batch
takes half an hour and the suite has to stay minutes. What they pin is that
every number inside a pair file follows from the per-game rows inside that same
file, that the 2026-10-09 pair file's schema is the older pair files' schema,
that the opening files' counts add up and stop where they say they stop, and
that the limits the rulings require are actually written down - a limit nobody
wrote into the evidence is a limit a reader cannot see.
"""
import gzip
import hashlib
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIR_FILES = {
    "b": "eval/imitation/rl_b1000_0k-pair.json",
    "i": "eval/imitation/rl_i1000_0k-pair.json",
    "o": "eval/imitation/rl_o1000_0k-pair-no_imitation.json",
}
OPENING_FILES = {
    "eval/rl-single-train/openings-12-h.json": ("rl_h1000_20k", 12),
    "eval/rl-single-train/openings-12-o.json": ("rl_o1000_20k", 12),
    "eval/rl-single-train/openings-12-b.json": ("rl_b1000_20k", 12),
    "eval/rl-single-train/openings-12-i.json": ("rl_i1000_20k", 12),
    "eval/rl-single-train/openings-four20k.json": (None, 15),
}
PERSONALITIES = {"hunter", "optimizer", "builder", "intruder", "fox", "chess",
                 "wolf"}
ZERO_K = {"rl_h1000_0k", "rl_o1000_0k", "rl_b1000_0k", "rl_i1000_0k"}


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def load_batch(rel):
    with gzip.open(os.path.join(ROOT, rel), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def md5_of(rel):
    h = hashlib.md5()
    with open(os.path.join(ROOT, rel), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shape(value):
    """Type-and-key skeleton: what the file is shaped like, not what it says.

    Lists collapse to their element's shape, because two runs of the same
    protocol have the same keys whether they played 4 games or 200.
    """
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return ["list", shape(value[0])] if value else ["list"]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return "str"


def stderr_of(values):
    return statistics.stdev(values) / math.sqrt(len(values))


def close(a, b, tol=1e-12):
    """`pytest.approx` for one number, with a tolerance tight enough to catch a
    different definition rather than a different float path."""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# --------------------------------------------------------------------------- #
# pair files
# --------------------------------------------------------------------------- #

def test_the_three_pair_files_have_the_same_schema():
    """Field for field: the 2026-10-09 file must not grow or drop a column."""
    shapes = {tag: shape(load(path)) for tag, path in PAIR_FILES.items()}
    assert shapes["o"] == shapes["b"], "the new pair file's schema drifted"
    assert shapes["i"] == shapes["b"], "b and i disagree with each other"


def test_the_o_pair_file_records_its_pool_and_its_limits():
    """The ruling's three statements are inside the evidence, not just in chat."""
    payload = load(PAIR_FILES["o"])
    note = payload["protocol_check"]
    assert "pool = no_imitation" in note
    assert "b、i 原池未記錄,故與其嚴格可比性未證" in note
    assert "recomputed in this run" in note
    assert "not asserted bit-for-bit" in note
    assert "no_imitation" in note and "builder" in note, "pool not expanded"
    assert payload["control"] == "hc_1000"
    assert payload["subject"] == "rl_o1000_0k"
    assert payload["n_games"] == 200
    assert payload["seed_range"] == [6_100_000, 6_100_199]


def test_the_o_pair_file_checksums_name_files_that_exist():
    """`checkpoint_md5` has to be the md5 of the path the file names."""
    payload = load(PAIR_FILES["o"])
    assert payload["checkpoint_md5"] == md5_of(payload["checkpoint"])
    assert payload["control_md5"] == md5_of(payload["control_checkpoint"])
    # The bytes are the ones b and i measured: same checkpoints, newer paths.
    assert payload["control_md5"] == "95dc7c463c69f8d337fde4e3df00097b"
    assert payload["checkpoint_md5"] == "6a62081dec9d796c024600def0246714"


def _assert_mode_consistent(payload, tag, mode):
    """Every stored statistic in one mode block follows from its own rows."""
    block = payload["modes"][mode]
    games = block["games"]
    n = payload["n_games"]
    for series in ("points", "rank", "remaining"):
        assert len(games[series]["hc_1000"]) == n, (tag, mode, series)
        assert len(games[series]["student"]) == n, (tag, mode, series)

    for side, series_key in (("hc_1000", "hc_1000"), ("student", "student")):
        stored = block[side]
        points = games["points"][series_key]
        remaining = games["remaining"][series_key]
        ranks = games["rank"][series_key]
        assert stored["mode"] == mode
        assert stored["games"] == n
        assert stored["seed_base"] == payload["seed_base"]
        assert stored["seed_range"] == payload["seed_range"]
        assert close(stored["mean_points"], statistics.fmean(points))
        assert close(stored["mean_remaining"], statistics.fmean(remaining))
        assert close(stored["points_stderr"], stderr_of(points))
        assert close(stored["remaining_stderr"], stderr_of(remaining))
        assert close(stored["rank1_share"],
                     sum(1 for r in ranks if r == 1) / n)

    control = games["points"]["hc_1000"]
    student = games["points"]["student"]
    paired = block["paired"]["points"]
    diffs = [s - c for c, s in zip(control, student)]
    assert paired["n"] == n
    assert close(paired["mean_diff"], statistics.fmean(diffs))
    assert close(paired["paired_se"], stderr_of(diffs))
    assert close(paired["independent_se"],
                 math.sqrt(stderr_of(control) ** 2 + stderr_of(student) ** 2))
    assert close(paired["t"], statistics.fmean(diffs) / stderr_of(diffs))
    assert close(paired["r"], statistics.correlation(control, student))

    rem_c = games["remaining"]["hc_1000"]
    rem_s = games["remaining"]["student"]
    rem_paired = block["paired"]["remaining"]
    rem_diffs = [s - c for c, s in zip(rem_c, rem_s)]
    assert close(rem_paired["mean_diff"], statistics.fmean(rem_diffs))
    assert close(rem_paired["paired_se"], stderr_of(rem_diffs))
    assert close(rem_paired["t"],
                 statistics.fmean(rem_diffs) / stderr_of(rem_diffs))


def test_every_pair_file_is_internally_consistent():
    """b and i too: the definition used for the new file is the old definition.

    Recomputing the two committed 2026-10-06 files from their own per-game rows
    is how the formulas in `pair0k.py` are checked against history rather than
    against themselves - if the new file used a different standard error or a
    different orientation, this would fail on b or i first.
    """
    for tag in ("b", "i", "o"):
        payload = load(PAIR_FILES[tag])
        for mode in ("argmax", "softmax"):
            _assert_mode_consistent(payload, tag, mode)


def test_the_pair_files_differ_only_where_they_are_supposed_to():
    """Same protocol shape, different subject - and b/i leave their pool blank."""
    b = load(PAIR_FILES["b"])
    i = load(PAIR_FILES["i"])
    o = load(PAIR_FILES["o"])
    assert b["subject"] == "rl_b1000_0k"
    assert i["subject"] == "rl_i1000_0k"
    assert o["subject"] == "rl_o1000_0k"
    assert {b["seed_base"], i["seed_base"], o["seed_base"]} == {6_100_000}
    assert {b["n_games"], i["n_games"], o["n_games"]} == {200}
    # The two older files never name a pool; the new one does, on purpose.
    assert "pool =" not in b["protocol_check"]
    assert "pool =" not in i["protocol_check"]
    assert "pool = no_imitation" in o["protocol_check"]


# --------------------------------------------------------------------------- #
# opening files
# --------------------------------------------------------------------------- #

def test_every_opening_file_stops_where_it_says_it_stops():
    """Counts add up: selections == full games x 10, plies never past ten."""
    for rel, (subject, pool_size) in OPENING_FILES.items():
        payload = load(rel)
        assert payload["stop_after_moves"] == 10, rel
        assert payload["seed"] == 20261005, rel
        assert payload["mode"] == "argmax", rel
        assert payload["pool_size"] == pool_size == len(payload["pool"]), rel
        assert payload["rng_mode"] == (
            "paired-v1-subject" if subject else "paired-v1"), rel
        assert payload["subject"] == subject, rel

        # The pool is the point of the 2026-10-09 ruling: seven personalities,
        # the four 0k checkpoints, and the students' 20k seats.
        pool = set(payload["pool"])
        assert PERSONALITIES <= pool, rel
        assert ZERO_K <= pool, rel
        assert sum(payload["appearances"].values()) == payload["games"] * 4, rel
        if subject is not None:
            # A designated seat is in every game; it may also be drawn as an
            # opponent, which is why the count can exceed the game count.
            assert payload["appearances"][subject] >= payload["games"], rel

        total = 0
        for key, entry in payload["openings"].items():
            by_ply_sum = sum(count for ply, counts in entry["by_ply"].items()
                             for count in counts.values())
            ply_numbers = {int(ply) for ply in entry["by_ply"]}
            assert ply_numbers <= set(range(1, 11)), (rel, key, ply_numbers)
            assert by_ply_sum == entry["moves"], (rel, key)
            assert sum(entry["pieces"].values()) == entry["moves"], (rel, key)
            assert set(entry["pieces"]) == {
                piece for counts in entry["by_ply"].values()
                for piece in counts}, (rel, key)
            total += entry["moves"]
        assert total == payload["games_with_moves"] * 10, rel


def test_the_opening_files_say_they_are_descriptive_only():
    """The plan forbids drawing conclusions; the evidence has to say so too."""
    for rel in OPENING_FILES:
        note = load(rel)["note"]
        assert "descriptive" in note, rel
        assert "no test is performed" in note, rel
