"""The balanced rollout schedule behind `--balanced-schedule`.

The 2026-10-13 decree, as a checklist of what every round must satisfy:
learner seats exactly a quarter each; pool seat appearances within one of each
other; the cell matrix (learner seat x pool seat) within one; three *distinct*
opponents per game; game order shuffled; the whole assignment a function of
seed and round number and replayable; and the *cumulative* spreads still
within one across rounds - for the pool sizes the frozen pools actually come
in, 10 and 11 seats.

Exact numbers are asserted where the arithmetic makes them exact (500 games
over 10 seats is 150 appearances per seat, period) and "within one" everywhere
the decree says spread. Pairwise co-occurrence is only *counted*: 3*G over
C(N,2) is not divisible for any of these shapes, so an exactly even pairwise
schedule does not exist - the decree asks for numbers, not for a constraint.

The flag-off tests carry the old draw's frozen literals so "off is bit-for-bit
unchanged" is pinned against the published history rather than against this
commit.
"""
import os

import pytest

import rl.ppo as ppo_mod
import rl.rl_train as T
import rl.rollout as R
import seats

# plan10 section 1's 11-seat frozen pool, spelled out. `cells[s][p]` is
# indexed by *position*, so a test pool has to be a literal order, not a set.
POOL_11 = ("fox", "chess", "wolf", "optimizer", "builder", "intruder",
           "hunter", "rl_h1000_20k", "rl_i1000_20k", "rl_o1000_20k",
           "rl_b1000_20k")
POOL_10 = POOL_11[:10]

# inside the registered train block, which is what the default `own` claims
LO = R.RL_SEED_BASE

SMALL = ppo_mod.Config(epochs=1, minibatch=32, kl_coef=0.0)
CHECKPOINT = os.path.join(seats.IMITATION_CHECKPOINT_DIR,
                          "step_%06d.pt" % seats.IMITATION_STEPS[0])
needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="no H-C2 checkpoint at %s"
    % CHECKPOINT)

COMBOS = [(500, POOL_10), (1000, POOL_10),
          (500, POOL_11), (1000, POOL_11)]

# The exact shape of round 1 for each combo: the sorted pool-coverage
# multiset and the cell value set. All four are arithmetic identities of
# 3*G/4 over N columns, not samples.
EXACT = {
    (500, 10): ([150] * 10, [37, 38]),
    (1000, 10): ([300] * 10, [75]),
    (500, 11): ([136] * 7 + [137] * 4, [34, 35]),
    (1000, 11): ([272] * 3 + [273] * 8, [68, 69]),
}


def make(round_no, games, pool, rng_seed=LO, start=LO):
    return R.make_specs(start, games, rng_seed=rng_seed,
                        opponent_pool=pool, balanced=True,
                        round_no=round_no)


# ---------------------------------------------------------- the flag being off

def test_the_flag_is_off_by_default_and_inert_when_off():
    """Three claims the decree's "flag off, bit-identical" rests on.

    The runbook defaults to the old draw; passing `balanced=False` plus a
    round number changes nothing (so `round_no` cannot leak into the default
    path); and the default draw itself still equals the literals read out of
    the code that recorded every published batch.
    """
    assert T.TrainConfig().balanced_schedule is False
    assert T.build_parser().parse_args([]).balanced_schedule is False
    assert T.build_parser().parse_args(
        ["--balanced-schedule"]).balanced_schedule is True
    cfg, _ = T._config_from_args(T.build_parser().parse_args([]))
    assert cfg.balanced_schedule is False

    a, ma = R.make_specs(LO, 9, rng_seed=4242)
    b, mb = R.make_specs(LO, 9, rng_seed=4242, balanced=False, round_no=7)
    assert a == b, "round_no changed the default draw"
    assert ma == mb and "balanced" not in ma

    c, _ = R.make_specs(7_000_000, 5, rng_seed=7_000_000)
    assert [(s.seed, s.learner_seat, s.opponent_names) for s in c] == [
        (7000000, 3, ('optimizer', 'intruder', 'fox')),
        (7000001, 3, ('chess', 'hunter', 'intruder')),
        (7000002, 0, ('optimizer', 'fox', 'builder')),
        (7000003, 3, ('optimizer', 'fox', 'builder')),
        (7000004, 0, ('wolf', 'fox', 'intruder')),
    ]


# ------------------------------------------------------------- one round, all

@pytest.mark.parametrize("games,pool", COMBOS,
                         ids=["10x500", "10x1000", "11x500", "11x1000"])
def test_a_balanced_round_holds_every_shape_the_decree_lists(games, pool):
    """Every per-round property, on both pool sizes and both round shapes."""
    n = len(pool)
    specs, man = make(1, games, pool)
    assert len(specs) == games
    assert [s.seed for s in specs] == list(range(LO, LO + games))

    # learner seats: exactly a quarter, in the manifest and in the specs
    assert man["learner_seat_counts"] == [games // 4] * 4
    seen = [0, 0, 0, 0]
    for s in specs:
        seen[s.learner_seat] += 1
        assert len(set(s.opponent_names)) == 3, "a game repeated an opponent"
        assert set(s.opponent_names) <= set(pool)
    assert seen == [games // 4] * 4

    # pool appearances and cells, exact where the arithmetic is exact
    want_cov, want_cells = EXACT[(games, n)]
    assert sorted(man["opponent_coverage"][k] for k in pool) == want_cov
    cells = man["cells"]
    assert man["balanced"] is True and man["round_no"] == 1
    assert cells == R.balanced_cells(1, n, games)
    assert sorted({row[p] for row in cells for p in range(n)}) == want_cells
    for p in range(n):
        col = [cells[s][p] for s in range(4)]
        assert max(col) - min(col) <= 1, (p, col)
    for s in range(4):
        assert sum(cells[s]) == 3 * games // 4, (s, sum(cells[s]))
    cols = [sum(cells[s][p] for s in range(4)) for p in range(n)]
    assert max(cols) - min(cols) <= 1, cols


# ------------------------------------------------------ seeds, rounds, replay

def test_the_schedule_is_a_function_of_seed_and_round_and_replays():
    """"Replayable from seed and round number" - all three dependencies.

    Same inputs twice: identical (replay). Another round: another schedule,
    including the constant-cell combo where only the shuffle can differ, or
    1,000 games would repeat every round of a 40-round run. Another rng_seed:
    another order over the same cells (two runs on one seed block would
    otherwise be one experiment recorded twice).
    """
    a, ma = make(3, 500, POOL_10)
    b, mb = make(3, 500, POOL_10)
    assert a == b and ma == mb, "the schedule did not replay"

    c, _mc = make(4, 500, POOL_10)
    assert a != c, "round 4 repeated round 3"

    d, _md = R.make_specs(LO, 500, rng_seed=99, opponent_pool=POOL_10,
                          balanced=True, round_no=3)
    assert a != d, "another rng_seed gave the same game order"

    e, _me = make(1, 1000, POOL_10)
    f, _mf = make(2, 1000, POOL_10)
    assert e != f, "constant cells collapsed two rounds into one shape"


# ------------------------------------------------------------ cross-round sum

@pytest.mark.parametrize("games,pool", COMBOS,
                         ids=["10x500", "10x1000", "11x500", "11x1000"])
def test_the_cumulative_schedule_never_spreads_by_more_than_one(games, pool):
    """Every prefix, not just the final totals.

    Balancing each round on its own is what the old remainder handling did,
    and the drift then grew with the round count (measured before this flag:
    60 rounds, cumulative pool spread 60). The check runs at *each* round
    because a construction that converges late would pass an end-point-only
    test. Rounds are capped at 20,000 / games so the whole loop stays inside
    the registered train block.
    """
    n = len(pool)
    rounds = 20_000 // games
    seat_cum = [0] * 4
    cover_cum = {k: 0 for k in pool}
    cell_cum = [[0] * n for _ in range(4)]
    for q in range(1, rounds + 1):
        specs, man = make(q, games, pool)
        for i, c in enumerate(man["learner_seat_counts"]):
            seat_cum[i] += c
        for s in specs:
            for name in s.opponent_names:
                cover_cum[name] += 1
        for s in range(4):
            for p in range(n):
                cell_cum[s][p] += man["cells"][s][p]
        assert len(set(seat_cum)) == 1, (q, seat_cum)
        spread = max(cover_cum.values()) - min(cover_cum.values())
        assert spread <= 1, (q, sorted(cover_cum.values()))
        for p in range(n):
            col = [cell_cum[s][p] for s in range(4)]
            assert max(col) - min(col) <= 1, (q, p, col)
    # and the final shape is the closed form's: exact quarters
    assert seat_cum == [rounds * games // 4] * 4


@pytest.mark.parametrize("games,pool", COMBOS,
                         ids=["10x500", "10x1000", "11x500", "11x1000"])
def test_consecutive_rounds_never_repeat_a_seat_sequence(games, pool):
    """The decree's anti-pattern has a name: the old `make_specs` seeded once
    from `rng_seed = seed_base` and drew the *same* round every round, so any
    imbalance multiplied by the round count instead of washing out."""
    for q in range(1, 6):
        a, _ma = make(q, games, pool)
        b, _mb = make(q + 1, games, pool)
        sa = [s.learner_seat for s in a]
        sb = [s.learner_seat for s in b]
        assert sa != sb, (games, len(pool), q)


# ----------------------------------------------------------- refused shapes

def test_shapes_the_schedule_cannot_serve_are_refused(tmp_path):
    """Errors a round-1 crash would otherwise deliver, with the directory
    already created - the same reasoning as the bad-pool test in
    `test_rl_train.py`."""
    with pytest.raises(ValueError) as exc:
        make(1, 501, POOL_10)
    assert "divisible by 4" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        R.make_specs(LO, 500, opponent_pool=POOL_10, balanced=True)
    assert "round number" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        R.make_specs(LO, 500, opponent_pool=("chess", "fox"), balanced=True,
                     round_no=1)
    assert "at least 3 pool seats" in str(exc.value)

    # the loop's own two pre-checks, before anything is opened
    out = str(tmp_path / "never-created")
    base = dict(games=8, per_round=10, rounds=1, n_procs=2,
                rollout_device="cpu", update_device="cpu",
                eval_every=0, eval_games=4, balanced_schedule=True)
    with pytest.raises(ValueError) as exc:
        T.train(T.TrainConfig(**base), out, ppo_cfg=SMALL,
                init_path="does/not/exist.pt", dry_run=True,
                log=lambda *_a, **_k: None, nice=0)
    assert "divisible by 4" in str(exc.value)
    assert not os.path.exists(out), "a refused run created its output dir"

    with pytest.raises(ValueError) as exc:
        T.train(T.TrainConfig(**{**base, "per_round": 8,
                                 "opponent_pool": ("chess", "fox")}),
                out, ppo_cfg=SMALL, init_path="does/not/exist.pt",
                dry_run=True, log=lambda *_a, **_k: None, nice=0)
    assert "at least 3 opponents" in str(exc.value)
    assert not os.path.exists(out)


# ------------------------------------------------------------- decree item 3

def test_the_in_training_evaluation_never_takes_the_flag():
    """`eval_specs` is untouched by decree, and this pins the *behaviour* the
    decree is protecting: the evaluation is scored on fixed seeds with the
    seven-personality draw whichever flag the run trained under, so two
    `eval.jsonl` rows stay comparable across a schedule change."""
    cfg = T.TrainConfig()
    a, ma = T.eval_specs(cfg)
    b, mb = T.eval_specs(cfg._replace(balanced_schedule=True,
                                      opponent_pool=POOL_11))
    assert a == b and ma == mb
    assert "balanced" not in ma


# ------------------------------------------------------------- decree item 4

def test_pairwise_co_occurrence_is_counted_and_not_enforced():
    """Count the pairs, assert only that the count is honest.

    3*500 = 1,500 over C(10,2) = 45 leaves remainder 15, and over C(11,2) = 55
    remainder 15: exact equality is arithmetically impossible, so the decree
    says report, not balance. The invariant worth asserting is that every game
    contributes exactly its three pairs and no pair is invented.
    """
    for games, pool in ((500, POOL_10), (500, POOL_11)):
        specs, _man = make(1, games, pool)
        counts = {}
        for s in specs:
            a, b, c = sorted(s.opponent_names)
            for pair in ((a, b), (a, c), (b, c)):
                counts[pair] = counts.get(pair, 0) + 1
        assert sum(counts.values()) == 3 * games
        all_pairs = [(x, y) for i, x in enumerate(pool) for y in pool[i + 1:]]
        vals = [counts.get(p, 0) for p in all_pairs]
        print("\n  N=%d G=%d: %d/%d pairs present, counts min %d max %d "
              "spread %d, zero-count pairs %d"
              % (len(pool), games, sum(1 for v in vals if v), len(vals),
                 min(vals), max(vals), max(vals) - min(vals),
                 sum(1 for v in vals if v == 0)))


# ------------------------------------------------------------------- the wire

@needs_checkpoint
def test_the_training_loop_hands_its_round_number_to_the_schedule(
        tmp_path, monkeypatch):
    """Wiring: the loop must pass its 1-based round number and the flag.

    The cells *are* that round's cells - a loop that passed nothing would
    start every round at schedule round 1 and repeat one shape forever, which
    is exactly the failure the flag exists to remove.
    """
    seen = []
    real = R.make_specs

    def spy(start_seed, n, **kw):
        seen.append((n, kw.get("balanced"), kw.get("round_no")))
        return real(start_seed, n, **kw)

    monkeypatch.setattr(R, "make_specs", spy)
    out = str(tmp_path / "run")
    cfg = T.TrainConfig(games=8, per_round=8, rounds=1, n_procs=2,
                        rollout_device="cpu", update_device="cpu",
                        eval_every=0, eval_games=4, balanced_schedule=True)
    T.train(cfg, out, ppo_cfg=SMALL, init_path=CHECKPOINT,
            log=lambda *_a, **_k: None, nice=0)
    assert seen == [(8, True, 1)], seen
    # and the round really was built as a balanced one
    assert os.path.exists(os.path.join(out, "step_000001.pt"))
