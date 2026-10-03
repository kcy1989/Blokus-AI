"""Tests for `rl/caches.py`: the two caches, and the invariant that keeps a
training batch from being silently wrong.

The point of most of these is that a batch must be internally consistent. An
earlier smoke test read the caches by position within a split while taking the
features from a differently-indexed array, and the result was a validation loss
of 954,099,765 rather than an error. `assert_targets_legal` is the guard against
that, so it gets tested both ways: it must pass on aligned input and it must
fire on every way of misaligning.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine  # noqa: E402
from config import CLOCKWISE_OWNERS  # noqa: E402
from rl.actions import legal_mask_view, real_to_view, seat_of_mover  # noqa: E402
from rl.caches import (LABEL_CACHE_NAME, LEGAL_CACHE_NAME, N_ACTIONS,  # noqa: E402
                       Resident, ShardArrays, assert_targets_legal,
                       features_27, mask_from_cache, open_label_cache,
                       open_legal_cache, shard_order, states_from_rows)

DATA = os.path.join(ROOT, "data", "hb1")
HAVE = os.path.isdir(DATA) and os.path.exists(
    os.path.join(DATA, LEGAL_CACHE_NAME))
needs_data = pytest.mark.skipif(
    not HAVE, reason="the H-B1 dataset and its caches are not present")


# ------------------------------------------------------------------ no torch

def test_the_module_does_not_import_torch():
    """A DataLoader worker imports this; importing torch there would risk CUDA."""
    import rl.caches
    src = open(rl.caches.__file__, encoding="utf-8").read()
    body = src.split('"""', 2)[2]          # past the module docstring
    assert "import torch" not in body, "rl/caches.py must stay torch-free"


# ------------------------------------------------------------------ geometry

def test_shard_order_is_sorted_and_only_hb1_shards():
    names = shard_order(DATA)
    assert names == sorted(names)
    assert all(n.startswith("hb1_") and n.endswith(".npz") for n in names)


@needs_data
def test_split_offsets_partition_the_rows_without_gaps():
    from rl.caches import split_offsets
    offsets, total = split_offsets(DATA)
    assert offsets["train"] == 0
    assert offsets["valid"] > offsets["train"]
    assert total == 1_191_605, "the H-B1 dataset has a fixed size"
    # the splits must be adjacent with no gap, so a training index is already a
    # global index and needs no offset applied
    assert offsets["valid"] - offsets["train"] == 1_132_160
    assert offsets["valid"] + 59_445 == total


# -------------------------------------------------------------------- masks

@needs_data
def test_the_mask_cache_popcount_equals_n_legal():
    """If these disagreed, the storage estimate and the restore would both lie."""
    z = np.load(os.path.join(DATA, LEGAL_CACHE_NAME))
    counts = z["counts"]
    assert counts.size == 1_191_605
    assert counts.min() >= 1, "a row always has at least one legal move"
    assert counts.max() <= N_ACTIONS
    offsets = z["offsets"]
    assert offsets.size == counts.size + 1
    assert offsets[-1] == counts.sum()


@needs_data
def test_the_cached_mask_equals_a_live_legal_mask_view():
    flat, offsets = open_legal_cache(DATA, mmap=False)
    names = shard_order(DATA)
    d = np.load(os.path.join(DATA, names[0]))
    for local in (0, 1, 7, 50):
        states = states_from_rows({c: d[c][local:local + 1]
                                   for c in ("own_bits", "hand_bits",
                                             "stuck", "to_move",
                                             "own_move_count")})
        live = legal_mask_view(states[0])
        g = local
        cached = mask_from_cache(flat, offsets, [g])[0]
        assert np.array_equal(live, cached), local


@needs_data
def test_mmap_and_full_read_of_the_legal_cache_agree():
    a_flat, a_off = open_legal_cache(DATA, mmap=True)
    b_flat, b_off = open_legal_cache(DATA, mmap=False)
    assert np.array_equal(np.asarray(a_off[:5000]), np.asarray(b_off[:5000]))
    assert np.array_equal(np.asarray(a_flat[:5000]), np.asarray(b_flat[:5000]))


def test_a_compressed_cache_is_refused_rather_than_silently_copied(tmp_path):
    """Compressed zip members have no mappable bytes, so say so instead."""
    import numpy as np
    from rl.caches import _npz_member
    path = str(tmp_path / "compressed.npz")
    np.savez_compressed(path, flat=np.arange(10, dtype=np.int32))
    with pytest.raises(ValueError) as err:
        _npz_member(path, "flat", np.int32)
    assert "compressed" in str(err.value)


# ------------------------------------------------------------------- labels

@needs_data
def test_the_label_cache_equals_real_to_view_row_by_row():
    labels = open_label_cache(DATA, mmap=True)
    assert labels.size == 1_191_605
    assert labels.min() >= 0 and labels.max() < N_ACTIONS
    rng = np.random.default_rng(11)
    for g in rng.choice(1_191_605, size=200, replace=False):
        g = int(g)
        name = None
        row = g
        for n in shard_order(DATA):
            k = int(np.load(os.path.join(DATA, n))["action"].size)
            if row < k:
                name = n
                break
            row -= k
        d = np.load(os.path.join(DATA, name))
        st = engine.State(
            own_bits=tuple(int.from_bytes(bytes(d["own_bits"][row][o]), "little")
                           for o in range(4)),
            hand_bits=tuple(int(d["hand_bits"][row][o]) for o in range(4)),
            turn_order=CLOCKWISE_OWNERS, to_move=int(d["to_move"][row]),
            stuck=tuple(bool(d["stuck"][row][o]) for o in range(4)))
        live = int(real_to_view(np.array([int(d["action"][row])]),
                                seat_of_mover(st))[0])
        assert int(labels[g]) == live, (g, name, row)


# ----------------------------------------------------------------- features

@needs_data
def test_features_are_27_channels_with_the_scalars_broadcast():
    d = np.load(os.path.join(DATA, shard_order(DATA)[0]))
    states = states_from_rows({c: d[c][:4] for c in ("own_bits", "hand_bits",
                                                     "stuck", "to_move",
                                                     "own_move_count")})
    x = features_27(states)
    assert x.shape == (4, 27, 20, 20)
    assert x.dtype == np.float32
    # channels 14..26 are the 13 scalars, each constant over the board
    for c in range(14, 27):
        assert x[0, c].min() == x[0, c].max(), "channel %d is not broadcast" % c
    # and they agree with the 14-plane featurize's own scalars
    from rl.features import featurize_batch
    _planes, scalars, _hands = featurize_batch(states)
    assert np.allclose(x[0, 14:, 0, 0], scalars[0])


# --------------------------------------------------------------- the guard

@needs_data
def test_aligned_batches_pass_the_assertion():
    resident = Resident(DATA)
    for start in (0, 1_132_160):
        gidx = np.arange(start, start + 64)
        mask = resident.mask_for(gidx)
        labels = resident.labels_for(gidx)
        assert_targets_legal(mask, labels, "aligned", gidx)


@needs_data
def test_the_assertion_fires_on_every_way_of_misaligning():
    resident = Resident(DATA)
    base = np.arange(256)
    mask = resident.mask_for(base)
    for delta, name in ((1, "off-by-one"), (1000, "stale cache"),
                        (1_132_160, "train/valid confusion")):
        with pytest.raises(AssertionError) as err:
            assert_targets_legal(mask, resident.labels_for(base + delta),
                                 name, base)
        assert "outside their own legal mask" in str(err.value)
        assert name in str(err.value)


@needs_data
def test_a_past_end_cache_offset_does_not_silently_pass():
    """Reading past the end would return empty slices and an all-False mask."""
    flat, offsets = open_legal_cache(DATA, mmap=False)
    rows = 1_191_605
    # an out-of-range row must fail loudly: offsets has rows+1 entries, so row
    # `rows` indexes past the end. Returning an empty mask instead would let a
    # mis-sized batch produce an all-illegal row and a meaningless loss.
    with pytest.raises(IndexError):
        mask_from_cache(flat, offsets, [rows])


# ----------------------------------------------------- ShardArrays vs Resident

@needs_data
def test_shard_arrays_and_resident_agree_on_a_batch():
    """The two sources must be interchangeable, or the A/B is meaningless."""
    resident = Resident(DATA)
    lazy = ShardArrays(DATA, mmap=True)
    gidx = list(range(8)) + list(range(1_132_160, 1_132_168))
    xa, ma, ya = __import__("rl.caches", fromlist=["make_batch"]).make_batch(
        resident, gidx)
    xb, mb, yb = __import__("rl.caches", fromlist=["make_batch"]).make_batch(
        lazy, gidx)
    assert np.array_equal(xa, xb)
    assert np.array_equal(ma, mb)
    assert np.array_equal(ya, yb)