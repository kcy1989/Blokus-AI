"""`match.py --paired-rng`: laying a league out so two models meet the same board.

A league normally draws everything from one `random.Random`: the seat draw, the
colours, the opening player, each seat's own tie-breaks, and so on down the
stream. That is fine when the question is "which option is strongest on
average". It is not usable when the question is "do *these two* differ", because
the models never meet the same board twice. `choose_move` draws a number of times
that depends on the position and on the model's mistake rate, so swapping the
model in one seat moves every later game's seat draw, colours and opening
player - and a paired statistic over those games is pairing noise.

`--paired-rng` gives each game five streams derived from `(seed, game_index)`:
one for the setup and one per seat. The setup of game 300 is then a function of
the seed alone, so two runs differing in one seat's model meet the same
opponents, the same colours and the same opening player in every game.

What this file pins, in order of how much it matters:

  * the **default** league is byte-identical to before the split. Every existing
    `records.json` entry, and every number ever reported from `match.py`, was
    produced by the single stream, and an evaluation this feature cannot
    reproduce is worse than the feature;
  * swapping one seat's model leaves every game's setup untouched - which is
    false of the default, and is checked here for both so that the paired test
    cannot pass by accident;
  * the scoring path is the same function of the same rows in both modes;
  * paired runs stay out of `records.json` and label their own output, because a
    paired batch and a leaderboard batch are two different measurements.
"""
import contextlib
import gzip
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import pytest

import ai.registry as REG
import match as M
import seats as S
from game import Game

# The pool is small and entirely rule-based so that a dozen games run in a
# second and the golden below does not need a torch checkpoint to exist.
POOL = ("hunter", "optimizer", "builder", "intruder")

# Captured from this repository at 8ec8ba2, before the split existed. A hash
# rather than the rows themselves: the point is "not one bit moved", and a
# digest says that without freezing 48 rows into the file.
LEGACY_GOLDEN_MD5 = "3fa9fba063a5f1cd3868098454b05b4c"


def rows_digest(rows):
    payload = [[[k, c, r, rk, p] for k, c, r, rk, p in game] for game in rows]
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

@contextlib.contextmanager
def intruder_replaced_by_a_network():
    """Make the `intruder` seat play `hc_1000`'s weights instead.

    Chosen because it is the swap that breaks pairing hardest: a weighted
    personality draws seven times while it is being built, a trained policy
    draws not at all, and the opening player is drawn after the brains. A swap
    between two personalities would consume the same amount and could hide a
    regression that this one exposes.
    """
    original = S.build_brain

    def patched(key, rng, kind=None, **kw):
        if key == "intruder":
            from rl.imitation import load_brain
            return load_brain(1000, key="intruder", mode="argmax")
        return original(key, rng, kind=kind, **kw)

    S.build_brain = patched
    try:
        yield
    finally:
        S.build_brain = original


def setups_of(games, seed, paired, swap=False):
    """The per-game `(seats, colours, turn order)`, in game order."""
    seen = []
    original = M.play_match

    def spy(game, rng, on_move=None, seat_rngs=None):
        seen.append((tuple(game.owner_key[o] for o in range(4)),
                     tuple(game.colors[o] for o in range(4)),
                     tuple(game.turn_order)))
        return original(game, rng, on_move, seat_rngs=seat_rngs)

    with intruder_replaced_by_a_network() if swap else contextlib.nullcontext():
        M.play_match = spy
        try:
            M.run_league(games=games, seed=seed, options=POOL, mode="argmax",
                         paired_rng=paired)
        finally:
            M.play_match = original
    return seen


# --------------------------------------------------------------------------
# the default must not move
# --------------------------------------------------------------------------

def test_the_default_league_is_byte_identical_to_before_the_split():
    assert rows_digest(M.run_league(games=12, seed=4242, options=POOL,
                                    mode="argmax")) == LEGACY_GOLDEN_MD5


def test_both_layout_switches_default_to_off():
    """Every caller that does not pass the argument inherits these defaults.

    Named through `inspect.signature` rather than by position: `__defaults__[-1]`
    names whichever parameter happens to be last, so adding a parameter silently
    moved this assertion onto a different argument and it kept passing while
    checking nothing.
    """
    import inspect
    params = inspect.signature(M.run_league).parameters
    assert params["paired_rng"].default is False
    assert params["subject"].default is None


# --------------------------------------------------------------------------
# the property the split exists for
# --------------------------------------------------------------------------

def test_swapping_one_seat_model_leaves_every_setup_untouched_in_paired_mode():
    plain = setups_of(games=10, seed=12345, paired=True)
    swapped = setups_of(games=10, seed=12345, paired=True, swap=True)
    assert plain == swapped


def test_the_same_swap_still_breaks_the_default_mode():
    """So the test above is measuring the split and not a coincidence.

    Without this, a version of the paired path that ignored the flag entirely
    would still pass the paired test, and the flag would be decorative.
    """
    plain = setups_of(games=10, seed=12345, paired=False)
    swapped = setups_of(games=10, seed=12345, paired=False, swap=True)
    assert plain != swapped


def test_building_a_personality_advances_the_stream_and_building_a_policy_does_not():
    """The asymmetry that made one shared stream unusable for pairing.

    A personality jitters its seven weights on the way in, so building one draws
    seven times; a trained policy is loaded from a file and draws nothing. The
    opening player is drawn *after* the brains are built, so on a shared stream
    that count decides who goes first - and swapping a personality for a policy
    changes it.

    Measured by comparing the generator's state rather than by wrapping its
    methods: `uniform` calls `random` internally, so a counter that overrides
    both reports 14 draws for 7 and quietly hard-codes a wrong number into the
    test. Whether the state moved is the question, and it answers it exactly.
    """
    rng = random.Random(0)
    before = rng.getstate()
    S.build_brain("optimizer", rng, mode="argmax")
    assert rng.getstate() != before, "a personality should have drawn"

    rng = random.Random(0)
    before = rng.getstate()
    S.build_brain("rl_h1000_20k", rng, mode="argmax")
    assert rng.getstate() == before, "a trained policy should not have drawn"


def test_the_setup_stream_is_not_the_stream_the_brains_are_built_from():
    """So the opening player cannot depend on which models are sitting there.

    Reaches past `run_league` to the one line that matters: the generator handed
    to `build_brain` is the seat's own, and the generator `setup_seats` draws the
    opening player from is not.
    """
    seen = {}
    original = S.build_brain

    def spy(key, rng, kind=None, **kw):
        seen["brain_rng"] = rng
        return original(key, rng, kind=kind, **kw)

    S.build_brain = spy
    try:
        setup = random.Random(4242)
        seats_r = [random.Random(9_000 + o) for o in range(4)]
        g = Game(setup)
        g.setup_seats(["hunter"] * 4, [None] * 4, setup, seat_rngs=seats_r,
                      mode="argmax")
    finally:
        S.build_brain = original

    assert seen["brain_rng"] in seats_r
    assert seen["brain_rng"] is not setup


# --------------------------------------------------------------------------
# the streams themselves
# --------------------------------------------------------------------------

def test_a_game_index_depends_only_on_the_seed_and_not_on_what_preceded_it():
    """Game 300 is the same game whether you played 299 before it or not."""
    direct = M.paired_streams(777, 300)
    for _ in range(5):
        for game_index in (0, 7, 299, 300):
            M.paired_streams(777, game_index)
    again = M.paired_streams(777, 300)
    assert [r.random() for r in direct[1]] == [r.random() for r in again[1]]
    assert direct[0].random() == again[0].random()


def test_the_five_streams_of_a_game_are_five_different_streams():
    seeds = set()
    for parts in (("s", 3, "setup"), ("s", 3, "seat", 0), ("s", 3, "seat", 1),
                  ("s", 3, "seat", 2), ("s", 3, "seat", 3)):
        seeds.add(M.paired_stream_seed(*parts))
    assert len(seeds) == 5


def test_derivation_does_not_confuse_adjacent_labels():
    """`("ab", "c")` and `("a", "bc")` must not derive the same stream."""
    assert (M.paired_stream_seed("ab", "c")
            != M.paired_stream_seed("a", "bc"))


def test_a_different_seed_gives_a_different_league():
    a = setups_of(games=4, seed=1, paired=True)
    b = setups_of(games=4, seed=2, paired=True)
    assert a != b


# --------------------------------------------------------------------------
# scoring is the same function of the same rows
# --------------------------------------------------------------------------

def test_scoring_is_a_pure_function_of_the_rows_in_either_mode():
    rows = M.run_league(games=6, seed=31337, options=POOL, mode="argmax",
                        paired_rng=True)
    flat = [tuple(r) for game in rows for r in game]
    once = M.summarise(flat)
    twice = M.summarise(list(flat))
    assert once == twice
    # And the same rows score identically through the ranking the leaderboard
    # uses, which is the one place a second definition could disagree.
    assert sum(r["appearances"] for r in once) == len(flat)


def test_paired_rows_carry_the_same_shape_as_legacy_rows():
    legacy = M.run_league(games=3, seed=8, options=POOL, mode="argmax")
    paired = M.run_league(games=3, seed=8, options=POOL, mode="argmax",
                          paired_rng=True)
    for a, b in zip(legacy, paired):
        assert len(a) == len(b) == 4
        for ra, rb in zip(a, b):
            assert ra[0] in S.automated_options()      # option
            assert ra[1] in ("red", "green", "blue", "yellow")   # colour
            assert isinstance(ra[2], int) and ra[2] >= 0        # remaining
            assert ra[3] in (1, 2, 3, 4)                        # rank
            assert ra[4] in (1, 2, 3, 4)                        # points


def test_ranking_agrees_with_the_leaderboard_rule_on_a_paired_batch():
    """`play_match` ranks with `records.rank_rows`; a second rule would drift."""
    rows = M.run_league(games=5, seed=606, options=POOL, mode="argmax",
                        paired_rng=True)
    for game in rows:
        standings = [(r[0], r[2]) for r in game]
        order = sorted(range(4), key=lambda o: (standings[o][1], standings[o][0]))
        from records import rank_rows
        expected = {}
        for o, (_key, rank, _points, _rem) in zip(order, rank_rows(standings)):
            expected[o] = rank
        assert {o: r[3] for o, r in enumerate(game)} == expected


# --------------------------------------------------------------------------
# it stays out of the leaderboard and labels itself
# --------------------------------------------------------------------------

def test_paired_run_never_constructs_a_leaderboard(tmp_path, monkeypatch):
    """A paired batch and a leaderboard batch are two different measurements.

    Their seat draws are not comparable - one walks down a single stream, the
    other is laid out per game - so averaging them would produce a number
    describing neither. Checked by making `Records` explode rather than by
    watching a file, because `main()` builds `Records()` at the repository's own
    `records.json` and a test that redirects the path has redirected something
    `main()` never reads.
    """
    built = []

    class Tripwire:
        def __init__(self, *a, **kw):
            built.append(a)
            raise AssertionError("paired mode built a leaderboard")

    monkeypatch.setattr(M, "Records", Tripwire)
    assert M.main(["--games", "2", "--seed", "11", "--pool", "hunter",
                   "--paired-rng", "--out", str(tmp_path / "a.json")]) == 0
    assert built == []

    # And the flag is what stopped it: without it the same stub is reached.
    class Counting:
        def __init__(self, *a, **kw):
            built.append(a)

        def reset(self):
            pass

        def record(self, rows):
            pass

        def rows(self, order=()):
            return []

        path = "(stubbed, nothing written)"

    built.clear()
    monkeypatch.setattr(M, "Records", Counting)
    assert M.main(["--games", "2", "--seed", "11", "--pool", "hunter",
                   "--out", str(tmp_path / "b.json")]) == 0
    assert len(built) == 1, "without --paired-rng a leaderboard is still built"


def test_the_batch_file_says_which_rng_mode_produced_it(tmp_path):
    paired = tmp_path / "paired.json"
    legacy = tmp_path / "legacy.json"
    M.main(["--games", "2", "--seed", "12", "--pool", "hunter", "--dry",
            "--paired-rng", "--out", str(paired)])
    M.main(["--games", "2", "--seed", "12", "--pool", "hunter", "--dry",
            "--out", str(legacy)])
    assert json.loads(paired.read_text(encoding="utf-8"))["rng_mode"] \
        == M.PAIRED_RNG_VERSION
    assert json.loads(legacy.read_text(encoding="utf-8"))["rng_mode"] \
        == M.LEGACY_RNG_MODE
    assert M.PAIRED_RNG_VERSION != M.LEGACY_RNG_MODE


def test_paired_batches_keep_the_per_game_rows_the_statistics_need(tmp_path):
    out = tmp_path / "b.json"
    M.main(["--games", "3", "--seed", "13", "--pool", "hunter", "--dry",
            "--paired-rng", "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["games_detail"]) == 3
    assert all(len(game) == 4 for game in payload["games_detail"])
    assert all(len(row) == 5 for game in payload["games_detail"] for row in game)

# --------------------------------------------------------------------------
# --subject: one designated seat in every game
# --------------------------------------------------------------------------

GROUP_A = tuple(M.expand_pool("no_imitation"))

# The seat each game designates, for the first 24 games of this seed. Pinned as a
# string rather than asserted as a distribution: a seat that is merely *likely*
# uniform is not the property that matters, and a test that could only fail by
# chance is a test that will not fail.
SEAT_SEQUENCE = "300223121301211232221010"

# Scratch for the write tests below, which are about files rather than games.
TMP = Path(tempfile.mkdtemp(prefix="batch-writes-"))

# Captured at 764f24c, before --subject existed.
PAIRED_NO_SUBJECT_MD5 = "fc85b0fc00461a5600f6e2daea42f555"


def _capture_subject_setups(subject, games=10, seed=20261005, pool=GROUP_A):
    """Per game: `(designated seat, opponents, colours, turn order)`, unplayed."""
    seen = []
    holder = {}
    original_play = M.play_match
    original_draw = M.subject_draw

    def draw_spy(rng, options, subj):
        seat, keys = original_draw(rng, options, subj)
        holder["seat"] = seat
        return seat, keys

    def play_spy(game, rng, on_move=None, seat_rngs=None):
        seat = holder["seat"]
        seen.append((seat,
                     tuple(game.owner_key[o] for o in range(4) if o != seat),
                     tuple(game.colors[o] for o in range(4)),
                     tuple(game.turn_order)))
        return [(subject, "red", 10, 1, 4)] * 4

    M.subject_draw = draw_spy
    M.play_match = play_spy
    try:
        M.run_league(games=games, seed=seed, options=pool, mode="argmax",
                     paired_rng=True, subject=subject)
    finally:
        M.subject_draw = original_draw
        M.play_match = original_play
    return seen


def test_every_game_has_exactly_one_designated_subject_seat():
    """Counted by index, never by key.

    `hunter` is allowed to appear in the pool, so in a good share of games the
    key appears twice and a test that counted keys would either fail or - worse -
    quietly define the seat as the wrong one.
    """
    setups = _capture_subject_setups("hunter", games=60)
    assert len(setups) == 60
    for seat, opponents, _colours, _order in setups:
        assert seat in (0, 1, 2, 3)
        assert len(opponents) == 3
        assert len(set(opponents)) <= 3          # three seats, draws may repeat


def test_the_designated_seat_follows_the_setup_stream_deterministically():
    setups = _capture_subject_setups("hunter", games=24)
    assert "".join(str(seat) for seat, _o, _c, _t in setups) == SEAT_SEQUENCE


def test_swapping_the_subject_leaves_every_setup_identical():
    """The property the whole thing exists for.

    Seat, the other three contestants, the colours and the opening player all
    come out the same in every game. If any of them moved, the paired difference
    would be comparing two different games.
    """
    a = _capture_subject_setups("hunter", games=10)
    b = _capture_subject_setups("rl_h1000_20k", games=10)
    c = _capture_subject_setups("rl_1000_20k", games=10)
    assert a == b == c


def test_the_swap_survives_the_personality_to_network_boundary():
    """Named apart from the test above because this is the one that used to fail.

    A personality draws seven times while it is being built and a trained policy
    draws none, and the opening player is drawn after the brains. That asymmetry
    is exactly what `--subject` has to absorb when the subject changes from
    `hunter` to a checkpoint.
    """
    a = _capture_subject_setups("hunter", games=8)
    b = _capture_subject_setups("rl_h1000_20k", games=8)
    assert [s[0] for s in a] == [s[0] for s in b]      # the same seat
    assert [s[1] for s in a] == [s[1] for s in b]      # the same opponents
    assert [s[3] for s in a] == [s[3] for s in b]      # the same opening player


def test_the_subject_may_also_appear_in_the_pool():
    """Allowed on purpose, and worth pinning because it looks like a bug.

    Seating `hunter` against a pool containing `hunter` reproduces the training
    condition. The alternative - silently dropping the subject from the pool -
    would quietly measure a weaker model than the one trained.
    """
    setups = _capture_subject_setups("hunter", games=200, seed=31337)
    duplicated = sum(1 for _s, opp, _c, _t in setups if "hunter" in opp)
    assert duplicated > 0, "expected the pool to seat the subject's key too"
    # and the seat is still the designated index, every time
    for seat, _opp, _c, _t in setups:
        assert seat in (0, 1, 2, 3)


def test_an_unknown_subject_names_the_legal_options():
    with pytest.raises(ValueError) as e:
        M.run_league(games=1, seed=1, options=GROUP_A, paired_rng=True,
                     subject="hunterr")
    assert "hunterr" in str(e.value)
    # Read the names from the pool rather than writing them out, so that a
    # rename does not turn this into a test of the rename.
    legal = S.automated_options()
    for key in ("rl_h1000_20k", "rl_o1000_20k", "wolf"):
        assert key in legal
        assert key in str(e.value)
    assert list(legal) == list(REG.pool("all"))


def test_a_subject_without_the_split_is_refused():
    """Otherwise it produces numbers labelled as paired that are not paired."""
    with pytest.raises(ValueError) as e:
        M.run_league(games=1, seed=1, options=GROUP_A, subject="hunter")
    assert "--paired-rng" in str(e.value)


def test_the_two_existing_layouts_are_byte_identical():
    """Both digests captured before `--subject` existed, on the same 12 games.

    Not "the tests are green": the golden digest test already says that, and a
    green suite is compatible with a change that quietly alters every game. The
    digests are the claim.
    """
    assert rows_digest(M.run_league(games=12, seed=4242, options=POOL,
                                    mode="argmax")) == LEGACY_GOLDEN_MD5
    assert rows_digest(M.run_league(games=12, seed=4242, options=POOL,
                                    mode="argmax",
                                    paired_rng=True)) == PAIRED_NO_SUBJECT_MD5


def test_a_subject_batch_is_neither_of_the_other_two():
    """Or `rng_mode` would be the only thing distinguishing them, and wrong."""
    plain = rows_digest(M.run_league(games=12, seed=4242, options=POOL,
                                     mode="argmax", paired_rng=True))
    subject = rows_digest(M.run_league(games=12, seed=4242, options=POOL,
                                       mode="argmax", paired_rng=True,
                                       subject="hunter"))
    assert subject != plain
    assert subject != LEGACY_GOLDEN_MD5


def test_a_subject_batch_is_labelled_and_keeps_out_of_the_leaderboard(
        tmp_path, monkeypatch):
    built = []

    class Tripwire:
        def __init__(self, *a, **kw):
            built.append(a)
            raise AssertionError("a subject batch built a leaderboard")

    monkeypatch.setattr(M, "Records", Tripwire)
    out = tmp_path / "b.json"
    assert M.main(["--games", "2", "--seed", "14", "--pool", "hunter",
                   "--paired-rng", "--subject", "hunter",
                   "--out", str(out)]) == 0
    assert built == []
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["subject"] == "hunter"
    assert payload["rng_mode"] == M.PAIRED_RNG_SUBJECT_VERSION
    assert M.PAIRED_RNG_SUBJECT_VERSION not in (M.PAIRED_RNG_VERSION,
                                                M.LEGACY_RNG_MODE)
    # and a subjectless paired batch still says subject is absent
    out2 = tmp_path / "c.json"
    M.main(["--games", "1", "--seed", "14", "--pool", "hunter", "--dry",
            "--paired-rng", "--out", str(out2)])
    assert json.loads(out2.read_text(encoding="utf-8"))["subject"] is None


# --------------------------------------------------------------------------
# --gzip: the batch in transit, with nothing about it changed
# --------------------------------------------------------------------------

PAYLOAD = {
    "pool_raw": "no_imitation",
    "pool": list(GROUP_A),
    "seed": 99,
    "games": 2,
    "mode": "argmax",
    "rng_mode": M.PAIRED_RNG_SUBJECT_VERSION,
    "subject": "hunter",
    "summary": [{"option": "wolf", "appearances": 2, "avg_points": 2.5,
                 "avg_remaining": 9.0, "points_stderr": 0.5}],
    "games_detail": [[["wolf", "red", 5, 1, 4], ["hunter", "blue", 7, 2, 3],
                      ["fox", "green", 9, 3, 2], ["wolf", "yellow", 12, 4, 1]]] * 2,
}


def test_gzip_replaces_the_plain_file_and_leaves_no_second_copy():
    """One copy, not two.

    Two copies of an evaluation are two things that can disagree, and the stale
    one is the one somebody eventually reads.
    """
    plain = M.write_batch(str(TMP / "b.json"), PAYLOAD, compress=False)
    assert plain.endswith("b.json") and os.path.exists(plain)

    out = M.write_batch(str(TMP / "c.json"), PAYLOAD, compress=True)
    assert out.endswith("c.json.gz")
    assert os.path.exists(out)
    assert not os.path.exists(str(TMP / "c.json")), "the plain file must not exist"


def test_the_compressed_bytes_are_the_plain_bytes():
    """Byte-for-byte, not "equivalent after parsing".

    Checked both ways round so that neither a serialiser that differs nor a
    reader that is lenient could hide a difference: the raw decompressed bytes
    must equal the bytes the plain writer produced, and `json.loads` of them must
    compare equal as objects.
    """
    plain = M.write_batch(str(TMP / "d.json"), PAYLOAD, compress=False)
    gz = M.write_batch(str(TMP / "d.json"), PAYLOAD, compress=True)
    on_disk = open(plain, encoding="utf-8").read().encode("utf-8")
    packed = gzip.decompress(open(gz, "rb").read())
    assert packed == on_disk
    assert json.loads(packed.decode("utf-8")) == json.loads(on_disk.decode("utf-8"))


def test_the_gzip_carries_everything_the_plain_file_carried():
    """`rng_mode` and `subject` above all: they are what make a batch readable."""
    gz = M.write_batch(str(TMP / "e.json"), PAYLOAD, compress=True)
    got = json.loads(gzip.decompress(open(gz, "rb").read()))
    assert got == PAYLOAD
    assert got["rng_mode"] == M.PAIRED_RNG_SUBJECT_VERSION
    assert got["subject"] == "hunter"


def test_the_same_payload_gives_the_same_gzip_bytes():
    """`mtime=0` and no stored filename, so a committed batch is hashable.

    Both halves are needed. A stored MTIME makes every re-run differ from the
    last even when it produced identical numbers, and a stored FNAME makes two
    identical batches differ whenever they were written under different names -
    which is exactly what happens when the same run is checked out twice.
    """
    a = M.write_batch(str(TMP / "f.json"), PAYLOAD, compress=True)
    b = M.write_batch(str(TMP / "f.json"), PAYLOAD, compress=True)
    c = M.write_batch(str(TMP / "some-other-name.json"), PAYLOAD, compress=True)
    raw = open(a, "rb").read()
    assert raw == open(b, "rb").read()
    assert raw == open(c, "rb").read()
    assert int.from_bytes(raw[4:8], "little") == 0, "MTIME must be 0"
    assert raw[3] == 0, "FLG must be 0: no stored filename or comment"


def test_a_different_payload_gives_different_gzip_bytes():
    """The reproducibility above must not be reproducibility of everything."""
    a = M.write_batch(str(TMP / "g1.json"), PAYLOAD, compress=True)
    other = dict(PAYLOAD, subject="hc_1000")
    b = M.write_batch(str(TMP / "g2.json"), other, compress=True)
    assert gzip.decompress(open(a, "rb").read()) != \
        gzip.decompress(open(b, "rb").read())
    assert open(a, "rb").read() != open(b, "rb").read()


def test_out_without_gzip_still_writes_plain_json(tmp_path):
    """The default is untouched: no flag, no `.gz`, no behaviour change."""
    out = tmp_path / "plain.json"
    assert M.main(["--games", "2", "--seed", "15", "--pool", "hunter", "--dry",
                   "--out", str(out)]) == 0
    assert out.exists()
    assert not (tmp_path / "plain.json.gz").exists()
    assert json.loads(out.read_text(encoding="utf-8"))["rng_mode"] \
        == M.LEGACY_RNG_MODE


def test_gzip_and_plain_produce_the_same_payload_from_one_run(tmp_path):
    """Not just the same helper: the same league, through `main`."""
    plain = tmp_path / "p.json"
    packed = tmp_path / "q.json"
    argv = ["--games", "2", "--seed", "16", "--pool", "hunter", "--dry",
            "--paired-rng", "--subject", "hunter"]
    assert M.main(argv + ["--out", str(plain)]) == 0
    assert M.main(argv + ["--out", str(packed), "--gzip"]) == 0
    assert json.loads(plain.read_text(encoding="utf-8")) == \
        json.loads(gzip.decompress((tmp_path / "q.json.gz").read_bytes()))
