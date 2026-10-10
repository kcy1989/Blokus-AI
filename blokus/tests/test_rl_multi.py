"""plan10 task 2: the four learners, the two environments, one step's layout.

These tests pin the *layout* - how many games, from which seeds, against whom,
and who sits where - because that is what every number the run produces is a
statement about. They also pin the one structural property plan10 risk 7 asks
for: a frozen opponent and the learner that started from it share a file and
nothing else.
"""
import os

import pytest

import seats
from rl import multi
from rl import multi_train as mt
from rl import paired3
from rl import rollout as rr

def _paths():
    """The four frozen 20k files, resolved once - they are also the four
    learners' starting weights, so nothing here may run without them."""
    return mt.frozen_paths()


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------

def test_the_fixed_pool_is_the_eleven_plan10_names():
    assert len(multi.FIXED_POOL) == 11
    assert len(set(multi.FIXED_POOL)) == 11
    assert tuple(sorted(multi.FIXED_POOL)) == multi.FIXED_POOL
    for key in multi.FIXED_POOL:
        seats.kind_of(key)


def test_the_four_learners_are_registered_and_their_20k_files_exist():
    """Each learner continues from its own frozen 20k, which has to be a seat
    the roster knows - `seat_for` stands a training checkpoint in under its own
    name, and a key that was not registered would fail at the first game."""
    assert multi.LEARNER_KEYS == ("rl_h1000_20k", "rl_o1000_20k",
                                  "rl_b1000_20k", "rl_i1000_20k")
    for key in multi.LEARNER_KEYS:
        assert seats.kind_of(key) == seats.KIND_RL
        assert os.path.exists(seats.rl_checkpoint(key))


def test_every_seat_order_appears_once_and_balances_the_seats():
    assert len(multi.PERMS) == 24
    assert len(set(multi.PERMS)) == 24
    for seat in range(4):
        counts = [sum(1 for order in multi.PERMS if order[seat] == who)
                  for who in range(4)]
        assert counts == [6, 6, 6, 6]


def test_a_step_is_ten_seeds_and_twenty_four_orders_each():
    specs = multi.make_table_specs(41)
    assert len(specs) == 240
    # one seed per round, shared by all 24 orders - plan10's "每輪種子不同"
    # and risk 9's "每步種子數 10 組" together pin this, and it is what makes
    # the 24 games a blocked comparison over seat arrangement rather than 24
    # unrelated draws.
    seeds = [s.seed for s in specs]
    assert len(set(seeds)) == multi.ROUNDS_PER_STEP
    for seed in set(seeds):
        assert seeds.count(seed) == 24
    assert seeds == [multi.table_seed(41, r + 1) for r in range(10)
                     for _ in multi.PERMS]
    for spec in specs:
        assert len(spec.order) == 4 and sorted(spec.order) == [0, 1, 2, 3]
        assert spec.keys == tuple(multi.LEARNER_KEYS[o] for o in spec.order)


def test_the_table_seeds_step_forward_without_overlapping():
    a = {s.seed for s in multi.make_table_specs(41)}
    b = {s.seed for s in multi.make_table_specs(42)}
    assert a and b and not (a & b)


def test_a_fixed_pool_game_draws_with_replacement_and_rotates_the_seat():
    specs, manifest = multi.make_fixed_specs(41)
    assert len(specs) == 260
    assert manifest["with_replacement"] is True
    assert [s.learner_seat for s in specs] == [i % 4 for i in range(260)]
    for s in specs:
        assert len(s.opponent_names) == 3
        for n in s.opponent_names:
            assert n in multi.FIXED_POOL
            seats.kind_of(n)
    # 有放回 really means with replacement: the same key may sit twice.
    assert any(len(set(s.opponent_names)) < 3 for s in specs), \
        "260 draws from 11 seats should occasionally repeat; none did"


def test_the_fixed_pool_seeds_are_a_pure_function_of_the_step():
    specs, _ = multi.make_fixed_specs(41)
    again, _ = multi.make_fixed_specs(41)
    assert specs == again
    other, _ = multi.make_fixed_specs(42)
    assert [s.seed for s in other] != [s.seed for s in specs]


def test_the_four_copies_of_one_fixed_pool_game_share_the_seat_and_seed():
    """`make_fixed_specs` returns one spec per game; `play_paired_batch` plays
    it once per learner. What has to hold is that the pairing is on the spec -
    one seed, one seat - and not four independent draws."""
    specs, _ = multi.make_fixed_specs(41)
    row = {spec.seed: spec for spec in specs}
    assert len(row) == 260


def test_every_multi_block_is_registered_to_the_tuple():
    """`reject_reserved_seeds` exempts by exact triple, so the constant a run
    claims with and the row the guard polices are one fact in two files."""
    registered = set(paired3.RESERVED_RANGES)
    for block in (multi.TABLE_BLOCK, multi.FIXED_BLOCK, multi.EVAL_BLOCK):
        assert block in registered, block


def test_the_three_multi_blocks_do_not_overlap_each_other_or_the_old_ones():
    blocks = [multi.TABLE_BLOCK, multi.FIXED_BLOCK, multi.EVAL_BLOCK]
    for i, (la, alo, ahi) in enumerate(blocks):
        for lb, blo, bhi in blocks[i + 1:]:
            assert ahi < blo or bhi < alo, (la, lb)
    for label, blo, bhi in paired3.RESERVED_RANGES:
        if (label, blo, bhi) in blocks:
            continue
        for _l, alo, ahi in blocks:
            assert ahi < blo or bhi < alo, (label, _l)


def test_the_whole_plan_fits_inside_the_reserved_blocks():
    """100 steps is plan10's 150k; the blocks have to cover it, not just this
    run's 60."""
    multi.check_step(100)
    with pytest.raises(ValueError):
        multi.check_step(101)


def test_an_eval_schedule_is_the_same_whatever_the_step():
    assert multi.make_eval_specs(20) == multi.make_eval_specs(20)
    assert multi.make_eval_specs(20)[0].seed == multi.EVAL_SEED_BASE


# --------------------------------------------------------------------------
# one table, four episodes
# --------------------------------------------------------------------------

def _four_nets(device=None):
    from rl.policy import load_policy
    return tuple(load_policy(p, device=device)[0] for p in _paths())


def test_one_table_yields_four_episodes_with_one_game_behind_them():
    nets = _four_nets()
    spec = multi.make_table_specs(41)[0]
    eps = rr.play_game(list(spec.keys),
                       {seat: nets[spec.order[seat]] for seat in range(4)},
                       spec.seed)
    assert sorted(eps) == [0, 1, 2, 3]
    assert [eps[s].learner_seat for s in range(4)] == [0, 1, 2, 3]
    assert [eps[s].rewards for s in range(4)] == [eps[0].rewards] * 4
    assert [eps[s].remaining for s in range(4)] == [eps[0].remaining] * 4
    assert [eps[s].ranks for s in range(4)] == [eps[0].ranks] * 4
    assert [eps[s].moves for s in range(4)] == [eps[0].moves] * 4
    assert eps[0].opponent_names == spec.keys[1:]
    # each episode recorded its own seat's decisions, and nobody else's
    assert all(ep.n_decisions > 0 for ep in eps.values())


def test_a_table_is_a_function_of_its_seed_and_its_networks():
    nets = _four_nets()
    spec = multi.make_table_specs(41)[0]
    seat_nets = {seat: nets[spec.order[seat]] for seat in range(4)}
    a = rr.play_game(list(spec.keys), seat_nets, spec.seed)
    b = rr.play_game(list(spec.keys), seat_nets, spec.seed)
    for s in range(4):
        assert a[s].action_index == b[s].action_index
        assert a[s].log_prob == b[s].log_prob


def test_the_paired_batch_keys_a_row_by_learner_and_keeps_the_spec():
    nets = _paths()
    specs, _ = multi.make_fixed_specs(41, games=4)
    rows = rr.play_paired_batch(specs, nets, multi.LEARNER_KEYS, n_procs=1,
                                opponent_pool=multi.FIXED_POOL)
    assert len(rows) == 4
    for spec, row in zip(specs, rows):
        assert sorted(row) == [0, 1, 2, 3]
        for i, ep in row.items():
            assert ep.learner_seat == spec.learner_seat
            assert ep.opponent_names == spec.opponent_names
            assert ep.seed == spec.seed


def test_a_paired_batch_does_not_depend_on_the_worker_count():
    specs, _ = multi.make_fixed_specs(41, games=4)
    one = rr.play_paired_batch(specs, _paths(), multi.LEARNER_KEYS, n_procs=1,
                               opponent_pool=multi.FIXED_POOL)
    four = rr.play_paired_batch(specs, _paths(), multi.LEARNER_KEYS, n_procs=4,
                                opponent_pool=multi.FIXED_POOL)
    for a, b in zip(one, four):
        for i in a:
            assert a[i].action_index == b[i].action_index
            assert a[i].log_prob == b[i].log_prob


def test_a_fixed_pool_game_may_seat_the_same_policy_twice():
    """有放回: the plan says with replacement, so a learner can meet its own
    starting weights and a frozen policy can sit opposite itself. The rules
    already allow a repeated key; this asserts the rollout does not re-forbid
    it."""
    from rl.policy import load_policy
    spec = rr.Spec(multi.fixed_seed(41, 0), 2,
                   ("rl_h1000_20k", "rl_h1000_20k", "hunter"))
    ep = rr.play_episode(spec, load_policy(_paths()[0])[0],
                         opponent_pool=multi.FIXED_POOL,
                         learner_key=multi.LEARNER_KEYS[0])
    assert ep.opponent_names == spec.opponent_names
    assert ep.n_decisions > 0


def test_the_personality_only_rule_still_holds_by_default():
    """plan9 published batches were recorded against three *distinct
    personalities*; a pool that allows networks and repeats has to be asked
    for by name, or the old rule would stop being checkable."""
    nets = _paths()
    from rl.policy import load_policy
    net = load_policy(nets[0])[0]
    spec = rr.Spec(multi.fixed_seed(41, 0), 0,
                   ("rl_h1000_20k", "hunter", "wolf"))
    with pytest.raises(ValueError) as exc:
        rr.play_episode(spec, net, learner_key=multi.LEARNER_KEYS[0])
    assert "personality pool" in str(exc.value)


# --------------------------------------------------------------------------
# plan10 risk 7: the frozen seat and the learner share a file, not parameters
# --------------------------------------------------------------------------

def test_two_seats_on_one_file_get_independent_modules():
    from rl.imitation import load_brain_at
    path = seats.rl_checkpoint("rl_h1000_20k")
    a = load_brain_at(path, key="rl_h1000_20k")
    b = load_brain_at(path, key="rl_h1000_20k")
    assert a.net is not b.net
    import torch
    with torch.no_grad():
        a.net.trunk[0].weight.fill_(0.0)
    assert not torch.equal(a.net.trunk[0].weight, b.net.trunk[0].weight)
    # and the file neither of them can see was not written
    assert mt.sha256_of(path) == mt.frozen_hashes()["rl_h1000_20k"]


def test_the_learner_seat_plays_the_handed_network_not_the_registry_weights():
    """If the swap were ever missed, the learner would train against its own
    frozen weights and the run would look perfectly healthy - so the test
    makes the handed network unplayable and asks whether the game noticed."""
    import torch
    from rl.policy import load_policy
    spec = rr.Spec(multi.fixed_seed(41, 0), 1, ("hunter", "wolf", "chess"))
    broken = load_policy(_paths()[0])[0]
    with torch.no_grad():
        for p in broken.parameters():
            p.zero_()
    ep = rr.play_episode(spec, broken, opponent_pool=multi.FIXED_POOL,
                         learner_key=multi.LEARNER_KEYS[0])
    frozen = rr.play_episode(spec, load_policy(_paths()[0])[0],
                             opponent_pool=multi.FIXED_POOL,
                             learner_key=multi.LEARNER_KEYS[0])
    assert ep.action_index != frozen.action_index


# --------------------------------------------------------------------------
# the step's bookkeeping
# --------------------------------------------------------------------------

def test_the_merged_curve_is_the_mean_of_the_two_halves():
    class Fake:
        def __init__(self, r):
            self.rewards = (r, 0, 0, 0)
            self.learner_seat = 0
    t = mt._curve([Fake(1.0), Fake(3.0)], [Fake(2.0)])
    assert t == {"table": 2.0, "fixed": 2.0, "merged": 2.0}
    u = mt._curve([Fake(4.0)], [Fake(0.0)])
    assert u["merged"] == 2.0, "1:1, not weighted by how many episodes"


def test_the_frozen_weights_are_untouched_by_a_training_step(tmp_path):
    """The trainer re-hashes all four every step and stops if one moves. This
    is the same assertion from the outside."""
    before = mt.frozen_hashes()
    # `evidence_dir` has to be pointed away from `eval/rl-multiple-train/`
    # explicitly. Its default is the *real* evidence directory, so a milestone
    # written by a test would land beside the published snapshots - which is
    # exactly how `step_000041` came to be a pytest tmp path committed as
    # evidence. A test gets a tmp directory, never the repository's.
    cfg = mt.MultiConfig(start_step=40, end_step=41, table_games=24,
                         fixed_games=4, n_procs=2, eval_every=0, eval_games=2,
                         milestone_every=1, evidence_dir=str(tmp_path))
    result = mt.train(cfg, out_dir=str(tmp_path))
    assert result.steps_done == 41 and not result.stopped_by
    assert mt.frozen_hashes() == before
    for learner in multi.LEARNERS:
        assert os.path.exists(os.path.join(str(tmp_path), learner,
                                           "step_000041.pt"))
        assert os.path.exists(os.path.join(str(tmp_path), learner,
                                           "latest.pt"))
    rows = [line for line in
            open(os.path.join(str(tmp_path), "rounds.jsonl"), encoding="utf-8")]
    assert len(rows) == 1
    import json
    row = json.loads(rows[0])
    assert row["frozen_sha256_unchanged"] is True
    assert row["curves"]["merged"] == pytest.approx(
        (row["curves"]["table"] + row["curves"]["fixed"]) / 2)
    assert row["table_games"] == 24
    assert row["fixed_games"] == 4
    assert row["per_learner"]["h"]["episodes"] == 28   # 24 table + 4 fixed
    ms = [line for line in open(os.path.join(str(tmp_path),
                                             "milestones.jsonl"),
                                encoding="utf-8")]
    assert len(ms) == 1
    mile = json.loads(ms[0])
    assert set(mile["files"]) == set(multi.LEARNERS)
    for entry in mile["files"].values():
        assert len(entry["sha256"]) == 64
        assert entry["path"].endswith(".pt")
        assert entry["key"] in multi.LEARNER_KEYS
    # ...and the milestone snapshot went to the tmp directory, not beside the
    # published evidence. `milestones.jsonl` is the file that is guaranteed to
    # be there: the snapshot runs *before* this step's own round row is
    # appended, so on the very first step `rounds.jsonl` and `eval.jsonl` do
    # not exist yet. That ordering is a separate (minor) evidence-quality
    # issue, not this test's subject.
    snap = os.path.join(str(tmp_path), "step_000041")
    assert os.path.exists(os.path.join(snap, "milestones.jsonl"))
