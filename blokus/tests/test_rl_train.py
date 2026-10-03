"""Tests for `rl/train.py` and the schedule and checkpointing around the loop.

What is worth pinning here is the schedule's endpoints, that a checkpoint
round-trips exactly, and that the batch plan is a function of the seed alone.
The loop's arithmetic is not re-tested: `rl/caches.py` already checks that a
batch is internally consistent, and the A/B against the DataLoader variant is
what says the step itself is right.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import rl.caches as C  # noqa: E402
from rl.train import (SETTINGS, Trainer, lr_at, masked_loss,  # noqa: E402
                      plan_batches, validation_indices)

DATA = os.path.join(ROOT, "data", "hb1")
HAVE = os.path.isdir(DATA) and os.path.exists(
    os.path.join(DATA, C.LEGAL_CACHE_NAME))
needs_data = pytest.mark.skipif(
    not HAVE, reason="the H-B1 dataset and its caches are not present")
CPU = torch.device("cpu")


def test_the_settings_are_the_ruled_ones():
    """If these drift, a published number no longer describes this run."""
    assert SETTINGS["in_ch"] == 27
    assert SETTINGS["batch"] == 256
    assert SETTINGS["channels"] == 64 and SETTINGS["blocks"] == 6
    assert SETTINGS["optimizer"] == "AdamW"
    assert SETTINGS["weight_decay"] == 1e-4
    assert SETTINGS["lr_peak"] == 1e-3
    assert SETTINGS["lr_warmup_steps"] == 500
    assert SETTINGS["lr_final"] == 1e-5
    assert SETTINGS["lr_schedule_end_step"] == 50_000
    assert SETTINGS["total_steps"] == 50_000
    assert SETTINGS["checkpoint_every"] == 2_000
    assert SETTINGS["validation_rows"] == 20_000
    assert SETTINGS["value_aux"] is False
    assert SETTINGS["rotation_augmentation"] is False


def test_the_run_is_measured_in_steps_not_seconds():
    assert 50_000 % SETTINGS["checkpoint_every"] == 0
    assert SETTINGS["total_steps"] // SETTINGS["checkpoint_every"] == 25
    assert SETTINGS["total_steps"] * SETTINGS["batch"] == 12_800_000


# ------------------------------------------------------------------ schedule

def test_the_learning_rate_warms_up_then_decays_to_the_final_value():
    assert lr_at(0) == 0.0 or lr_at(1) > 0
    assert lr_at(1) == pytest.approx(1e-3 / 500)
    assert lr_at(500) == pytest.approx(1e-3)
    assert lr_at(500) == pytest.approx(lr_at(499), rel=0.01), \
        "the peak sits at the end of the warmup"
    assert lr_at(50_000) == pytest.approx(1e-5)
    assert lr_at(80_000) == pytest.approx(1e-5), "past the end it holds"
    mid = (500 + 50_000) / 2
    assert 1e-5 < lr_at(mid) < 1e-3
    vals = [lr_at(s) for s in range(501, 50_001, 1000)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "must be monotone down"


def test_the_schedule_is_monotone_up_during_warmup():
    vals = [lr_at(s) for s in range(0, 501, 25)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_the_schedule_does_not_depend_on_the_step_index_of_the_batch():
    a = lr_at(1234)
    b = lr_at(1234)
    assert a == b


# -------------------------------------------------------------- batch order

def test_the_batch_plan_is_determined_by_the_seed_alone():
    a = plan_batches(1_132_160, 50, 20_260_903, 256)
    b = plan_batches(1_132_160, 50, 20_260_903, 256)
    assert len(a) == 50
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    c = plan_batches(1_132_160, 50, 1, 256)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_every_planned_batch_is_inside_the_training_rows():
    plan = plan_batches(1_132_160, 200, 20_260_903, 256)
    n_train = 1_132_160
    for gidx in plan:
        assert gidx.size == 256
        assert gidx.min() >= 0 and gidx.max() < n_train


def test_validation_indices_are_a_fixed_seed_draw_of_the_right_size():
    v = validation_indices(59_445, 20_260_903, 20_000)
    assert v.size == 20_000
    assert np.array_equal(v, validation_indices(59_445, 20_260_903, 20_000))
    assert v.size == len(set(v.tolist())), "drawn without replacement"
    assert validation_indices(1_000, 20_260_903, 20_000).size == 1_000, \
        "asked for more than exist and gets what there is"


# ----------------------------------------------------------------- the guard

def test_masked_loss_ignores_illegal_slots():
    """A legal slot with a huge logit must not be punished for the illegal ones."""
    logits = torch.zeros(2, C.N_ACTIONS)
    mask = torch.zeros(2, C.N_ACTIONS, dtype=torch.bool)
    mask[:, 10] = True
    mask[:, 20] = True
    labels = torch.tensor([10, 20])
    plain = masked_loss(logits.clone(), mask, labels)
    spiked = logits.clone()
    spiked[0, 5_000] = 40.0
    spiked[1, 30_000] = -40.0
    assert masked_loss(spiked, mask, labels) == pytest.approx(float(plain))


def test_masked_loss_punishes_a_legal_slot_the_network_dislikes():
    """Two legal slots, or the softmax is 1 and the loss is 0 whatever the
    logits say - with one legal move there is genuinely nothing to choose."""
    logits = torch.zeros(1, C.N_ACTIONS)
    mask = torch.zeros(1, C.N_ACTIONS, dtype=torch.bool)
    mask[:, 7] = True
    mask[:, 9] = True
    good = logits.clone()
    bad = logits.clone()
    bad[:, 7] = -5.0
    assert masked_loss(bad, mask, torch.tensor([7])) > masked_loss(
        good, mask, torch.tensor([7]))


def test_one_legal_slot_gives_zero_loss_whatever_the_logit():
    """A consequence worth pinning, because it is easy to misread as a bug."""
    mask = torch.zeros(1, C.N_ACTIONS, dtype=torch.bool)
    mask[:, 7] = True
    for v in (-40.0, 0.0, 40.0):
        logits = torch.full((1, C.N_ACTIONS), v)
        assert masked_loss(logits, mask, torch.tensor([7])) == pytest.approx(0.0)


# ------------------------------------------------------------- checkpointing

@needs_data
def test_a_checkpoint_round_trips_weights_optimiser_and_step(tmp_path):
    s = dict(SETTINGS)
    s["batch"] = 8
    tr = Trainer(DATA, CPU, steps=1, settings=s)
    gidx = tr.batches[0]
    for i in range(3):
        tr._step_from_batch(gidx, C.make_batch(tr.source, gidx), i + 1,
                            "test")
    path = tr.save_checkpoint(7, str(tmp_path))
    assert os.path.basename(path) == "step_000007.pt"

    # state_dict, not parameters: the net has BatchNorm, so state_dict also
    # holds running_mean / running_var / num_batches_tracked and zipping it
    # against parameters() misaligns everything after the first entry
    before = {k: v.detach().clone() for k, v in tr.net.state_dict().items()}
    opt_before = tr.opt.state_dict()["state"]
    assert opt_before, "the optimiser must have state after stepping"

    # perturb every tensor, weights and buffers alike
    with torch.no_grad():
        for v in tr.net.state_dict().values():
            if v.dtype.is_floating_point:
                v.add_(1.0)
            else:
                v.add_(1)
    got_step = tr.load_checkpoint(path)
    assert got_step == 7

    after = tr.net.state_dict()
    assert set(after) == set(before)
    for k in before:
        assert torch.equal(before[k], after[k]), \
            "%s did not come back exactly" % k

    reloaded = tr.opt.state_dict()["state"]
    assert len(reloaded) == len(opt_before)
    for k in opt_before:
        assert torch.equal(opt_before[k]["exp_avg"],
                           reloaded[k]["exp_avg"]), k
    assert tr.opt.state_dict()["param_groups"][0]["weight_decay"] == 1e-4


@needs_data
def test_one_more_step_after_reload_reproduces_the_loss(tmp_path):
    """The optimiser state has to be back too, or training silently diverges."""
    s = dict(SETTINGS)
    s["batch"] = 8
    tr = Trainer(DATA, CPU, steps=1, settings=s)
    gidx = tr.batches[0]
    for i in range(3):
        tr._step_from_batch(gidx, C.make_batch(tr.source, gidx), i + 1, "test")
    path = tr.save_checkpoint(3, str(tmp_path))

    # the loss is read before the step, so the next step's loss is a function of
    # the saved weights and the saved optimiser state and nothing else
    loss_a, _ = tr._step_from_batch(gidx, C.make_batch(tr.source, gidx), 4, "t")
    tr.load_checkpoint(path)
    loss_b, _ = tr._step_from_batch(gidx, C.make_batch(tr.source, gidx), 4, "t")
    assert loss_b == pytest.approx(loss_a, rel=1e-6, abs=1e-6), \
        "weights or optimiser state came back different"


@needs_data
def test_the_trainer_knows_which_split_is_which():
    """Guards the bug where `train_rows` was computed as the validation count."""
    tr = Trainer(DATA, CPU, steps=1)
    assert tr.train_offset == 0, \
        "if this changes, every training index needs the offset applied"
    assert tr.train_rows == 1_132_160
    assert tr.valid_rows == 59_445
    assert tr.train_rows + tr.valid_rows == tr.source.n_rows


@needs_data
def test_a_validation_pass_is_chunked_not_one_giant_batch(tmp_path):
    """20,000 x 36,400 float logits is 2.9 GB and `masked_fill` doubles it."""
    s = dict(SETTINGS)
    s["batch"] = 8
    s["validation_chunk"] = 4
    tr = Trainer(DATA, CPU, steps=1, settings=s)
    val = tr.valid_offset + np.arange(20)
    seen = []

    real = C.make_batch

    def spy(source, gidx):
        seen.append(len(list(gidx)))
        return real(source, gidx)

    C.make_batch = spy
    try:
        value = tr.validation(val)
    finally:
        C.make_batch = real
    assert seen == [4] * 5, seen
    assert math.isfinite(value) and value > 0