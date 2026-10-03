"""The DataLoader variant of the training loop. Same numbers, faster batches.

`rl/train.py` owns the schedule, the assertion, the checkpointing and the
validation. `rl/caches.py` owns turning rows into a batch. This file only
supplies batches, and it is the *only* thing it supplies: the step itself is
`Trainer._step_from_batch`, inherited unchanged, so the A/B comparison measures
the loader and not two different programs.

Three constraints shape it:

  * workers must not touch CUDA. `rl/caches.py` imports no torch at all, and
    this module's worker-side code is numpy, so `import rl.caches` in a spawn
    child cannot initialise a CUDA context.
  * the caches are 931 MiB and 4.5 MiB. Six workers each copying them would cost
    5.6 GiB, so both are memory-mapped: `labels_view.npy` through numpy's own
    `mmap_mode="r"`, and the legal cache through `caches.mmap_npz_array`, which
    maps the array straight out of the uncompressed zip member.
  * the batch order must be identical to the serial run's. The global row
    indices are chosen in the parent from the fixed seed before any worker
    starts, and the DataLoader is iterated in that order, so a difference in the
    loss sequence can only come from the loader.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import rl.caches as C
from rl.train import MASK_FILL, SETTINGS, Trainer

DL_PARAMS = {
    "multiprocessing_context": "spawn",
    "persistent_workers": True,
    "prefetch_factor": 2,
    "timeout": 60,
}


class GlobalBatchDataset(Dataset):
    """One item is one whole batch, already built.

    `batch_size=None` with a plain range sampler is the way to hand a DataLoader
    pre-batched items; collating a list of rows and building the batch in the
    parent would put the CPU cost back on the critical path, which is the whole
    point of this file.
    """

    def __init__(self, data_dir, batches, mmap=True):
        self.batches = batches
        self._source = None
        self._data_dir = data_dir
        self._mmap = mmap

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        if self._source is None:
            # built here, in the worker, on first use: a spawn child inherits
            # nothing, so this is what gives each worker its own mmap
            self._source = C.ShardArrays(self._data_dir, mmap=self._mmap)
        gidx = self.batches[i]
        return C.make_batch(self._source, gidx)


def _identity(item):
    """Hand the item through untouched.

    Without this the default convert turns every numpy array into a tensor,
    which puts a torch tensor in the parent's hand and quietly undoes the point
    of keeping `rl.caches` torch-free. The batch is built in the worker and
    stays numpy.
    """
    return item


class BatchOrderSampler(Sampler):
    """Yields 0..len(dataset)-1 in order, so batches arrive as planned."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


class DLLoader:
    """Feeds `Trainer` batches from a DataLoader instead of building them inline."""

    def __init__(self, data_dir, device, steps=None, settings=None,
                 num_workers=4, preload=True, mmap=True):
        self.s = dict(SETTINGS if settings is None else settings)
        self.device = device
        self.num_workers = num_workers
        self.trainer = Trainer(data_dir, device, steps=steps, settings=self.s)
        self.batches = self.trainer.batches
        self.dataset = GlobalBatchDataset(data_dir, self.batches, mmap=mmap)
        params = dict(DL_PARAMS)
        if num_workers > 0:
            self.loader = DataLoader(
                self.dataset,
                batch_size=None,
                num_workers=num_workers,
                sampler=BatchOrderSampler(len(self.batches)),
                collate_fn=_identity,
                **params)
        else:
            self.loader = None
        # the parent needs a source too, for validation; it shares the trainer's
        self._it = None
        if preload:
            self._it = iter(self.loader) if self.loader is not None else None

    def step(self, step):
        """The next planned batch, plus the assertion before it is used.

        `timeout=60` on the DataLoader is a hang guard: if a worker dies or
        stalls, iteration raises instead of blocking forever.
        """
        if self.loader is None:
            return self.trainer.serial_step(step)
        try:
            gidx = self.batches[step - 1]
            batch = next(self._it)
        except StopIteration:
            raise RuntimeError(
                "the DataLoader ran out after %d batches; the plan has %d"
                % (step - 1, len(self.batches)))
        C.assert_targets_legal(batch[1], batch[2],
                               "train batch at step %d" % step, gidx)
        return self.trainer._step_from_batch(gidx, batch, step,
                                             "train batch at step %d" % step)


def worker_rss_pids():
    """(pid, rss_bytes) for this process and its children."""
    out = []
    root = os.getpid()
    out.append((root, _rss(root)))
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open("/proc/%d/stat" % pid, encoding="utf-8") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            ppid = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == root:
            out.append((pid, _rss(pid)))
    return out


def _rss(pid):
    try:
        with open("/proc/%d/status" % pid, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def process_footprint():
    """`(rss_sum, pss_sum)` over this process and its children, in bytes.

    RSS is reported as well as PSS because they disagree here and the
    disagreement is the point: the caches are memory-mapped, so the same clean
    page is resident in every worker, and summing RSS counts it once per
    process. PSS divides each shared page by the number of processes mapping it,
    which is the number that says how much memory is actually being used.
    """
    rss = sum(r for _p, r in worker_rss_pids())
    pss = 0
    for pid, _r in worker_rss_pids():
        try:
            with open("/proc/%d/smaps_rollup" % pid, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        pss += int(line.split()[1]) * 1024
                        break
        except OSError:
            continue
    return rss, pss


__all__ = ["DLLoader", "GlobalBatchDataset", "BatchOrderSampler",
           "process_footprint", "worker_rss_pids", "DL_PARAMS"]