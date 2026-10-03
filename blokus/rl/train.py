"""This layer answers one question: how does the learner train, and what does it
write down?

One run is 50,000 steps of batch 256, which is 12.8 million samples, about 10.7
epochs over `data/hb1/`. Checkpoints every 2,000 steps - 25 of them - each
carrying weights, optimiser state and the step number, all kept. At step 50,000
it stops and waits; it does not extend itself.

The ruling that shapes the rest of this file: the run is measured in **steps,
not seconds**. There is no time budget and no early stop, so the only thing that
decides how long it takes is how fast a step is, and the only lever on that is
the loading path.

`rl/caches.py` owns everything about turning rows into a batch. This file owns
the schedule, the assertions, the checkpointing and the validation. The
DataLoader variant in `rl/train_dl.py` imports both, so the two cannot drift.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fnn

import rl.caches as C
from rl.smoke_gpu import build_model

# ---------------------------------------------------------------- the ruling

SETTINGS = {
    "channels": 64,
    "blocks": 6,
    "in_ch": 27,
    "batch": 256,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "lr_peak": 1e-3,
    "lr_warmup_steps": 500,
    "lr_final": 1e-5,
    "lr_schedule_end_step": 50_000,
    "dtype": "fp32",
    "value_aux": False,
    "rotation_augmentation": False,
    "validation_rows": 20_000,
    "validation_chunk": 512,
    "checkpoint_every": 2_000,
    "total_steps": 50_000,
    "seed": 20_260_903,
}

MASK_FILL = -1e9


def lr_at(step, s=None):
    """Linear warmup to the peak, then cosine down to `lr_final`.

    The schedule is defined over the whole run, so the learning rate at a given
    step does not depend on how long the run has taken or on anything measured
    at runtime.
    """
    s = s or SETTINGS
    peak, warm = s["lr_peak"], s["lr_warmup_steps"]
    end, final = s["lr_schedule_end_step"], s["lr_final"]
    if step <= warm:
        return peak * step / warm
    if step >= end:
        return final
    t = (step - warm) / (end - warm)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * t))


def plan_batches(n_rows, n_batches, seed, batch):
    """The global row index list for every batch, decided once up front.

    Both training loops consume this exact list, which is what makes the
    serial/DataLoader comparison meaningful: same seed, same order, same rows.

    Epochs are reshuffled permutations of `0..n_rows-1`, cut into batches, and
    the tail of an epoch is carried into the next one rather than dropped. An
    earlier version drew a random start offset per batch, which could place a
    batch past the last training row and read validation rows as training data.
    """
    if n_rows < batch:
        raise ValueError("n_rows %d < batch %d" % (n_rows, batch))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    out = []
    cursor = 0
    for _ in range(n_batches):
        if cursor + batch > n_rows:
            order = rng.permutation(n_rows)
            cursor = 0
        out.append(order[cursor:cursor + batch].astype(np.int64))
        cursor += batch
    return out


def validation_indices(n_rows, seed, want):
    """Validation rows: a fixed-seed draw from the validation split."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, min(want, n_rows), replace=False))


# ------------------------------------------------------------------ stepping

def make_model(device):
    return build_model(SETTINGS["channels"], SETTINGS["blocks"],
                       SETTINGS["in_ch"]).to(device)


def masked_loss(logits, mask, labels):
    return Fnn.cross_entropy(logits.masked_fill(~mask, MASK_FILL), labels)


def to_device(batch, device):
    x, mask, labels = batch
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(labels).to(device))


class Trainer:
    """The training loop, shared by both variants.

    The two differ only in how a batch arrives: `serial_step` is handed one
    directly, `dl_step` is handed one the DataLoader produced. Everything after
    that - the assertion, the schedule, the optimiser, the checkpoint - is the
    same code, so the comparison isolates the loader rather than comparing two
    programs.
    """

    def __init__(self, data_dir, device, steps=None, settings=None):
        self.s = dict(SETTINGS if settings is None else settings)
        self.device = device
        self.data_dir = data_dir
        self.steps = self.s["total_steps"] if steps is None else steps
        self.source = C.Resident(data_dir)
        self.net = make_model(device)
        self.net.train()
        self.opt = torch.optim.AdamW(self.net.parameters(),
                                     lr=self.s["lr_peak"],
                                     weight_decay=self.s["weight_decay"])
        offsets = self.source.offsets
        self.train_offset = offsets["train"]
        self.valid_offset = offsets["valid"]
        # the training split is [train, valid); the validation split is
        # [valid, n_rows). Subtracting the valid offset from n_rows gives the
        # *validation* count, which would plan batches out of only the first
        # 59,445 training rows and never see the other 95%.
        self.valid_rows = self.source.n_rows - offsets["valid"]
        self.train_rows = offsets["valid"] - offsets["train"]
        self.batches = plan_batches(self.train_rows, self.steps,
                                    self.s["seed"], self.s["batch"])
        # the training split starts at global row 0, so a training index is
        # already a global index; the assertion below is what keeps that true
        assert self.train_offset == 0, (
            "train split must start at global row 0, got %d; if that ever "
            "changes, every training index here needs the offset applied"
            % self.train_offset)

    # -- one step

    def _step_from_batch(self, gidx, batch, step, where):
        C.assert_targets_legal(batch[1], batch[2], where, gidx)
        x, mask, labels = to_device(batch, self.device)
        lr = lr_at(step, self.s)
        for g in self.opt.param_groups:
            g["lr"] = lr
        logits = self.net(x)[0].reshape(x.shape[0], -1)
        loss = masked_loss(logits, mask, labels)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.detach()), lr

    def serial_step(self, step):
        gidx = self.batches[step - 1]
        return self._step_from_batch(gidx, C.make_batch(self.source, gidx),
                                     step, "train batch at step %d" % step)

    # -- validation

    def validation(self, val_global, chunk=None):
        """Mean masked cross-entropy over the validation rows.

        Chunked on purpose: 20,000 x 36,400 float logits is 2.9 GB, and
        `masked_fill` doubles it, so the whole set at once does not fit beside
        the training state.
        """
        chunk = chunk or self.s["validation_chunk"]
        self.net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for lo in range(0, len(val_global), chunk):
                gidx = val_global[lo:lo + chunk]
                batch = C.make_batch(self.source, gidx)
                C.assert_targets_legal(batch[1], batch[2],
                                       "validation batch at row %d" % lo, gidx)
                x, mask, labels = to_device(batch, self.device)
                logits = self.net(x)[0].reshape(x.shape[0], -1)
                total += float(masked_loss(logits, mask, labels)) * len(gidx)
                n += len(gidx)
        self.net.train()
        return total / n

    # -- checkpoints

    def save_checkpoint(self, step, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "step_%06d.pt" % step)
        torch.save({"state_dict": self.net.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "step": step,
                    "settings": self.s}, path)
        return path

    def load_checkpoint(self, path):
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(blob["state_dict"])
        self.opt.load_state_dict(blob["optimizer"])
        self.net.train()
        return int(blob["step"])


# --------------------------------------------------------------------- entry

def run(data_dir, out_dir, steps=None, device=None, seed=None,
        settings=None, loader="serial"):
    s = dict(SETTINGS if settings is None else settings)
    if seed is not None:
        s["seed"] = seed
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if loader == "serial":
        tr = Trainer(data_dir, device, steps=steps, settings=s)
        stepper = tr.serial_step
    else:
        from rl.train_dl import DLLoader
        dl = DLLoader(data_dir, device, steps=steps, settings=s)
        tr = dl.trainer
        stepper = dl.step
    val_local = validation_indices(
        tr.source.n_rows - tr.valid_offset, s["seed"], s["validation_rows"])
    val_global = val_local + tr.valid_offset

    losses = []
    times = []
    curve = []
    saved = []
    for step in range(1, tr.steps + 1):
        t0 = time.perf_counter()
        loss, lr = stepper(step)
        times.append(time.perf_counter() - t0)
        losses.append(loss)
        if step % s["checkpoint_every"] == 0 or step == tr.steps:
            vl = tr.validation(val_global)
            curve.append({"step": step, "train_loss": loss,
                          "val_loss": vl, "lr": lr})
            saved.append(tr.save_checkpoint(step, out_dir))
        if step % 20 == 0 or step == 1:
            print("step %6d  loss %.4f  lr %.3g" % (step, loss, lr), flush=True)
    return {"trainer": tr, "losses": losses, "times": times, "curve": curve,
            "checkpoints": saved, "val_global": val_global}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hb1")
    ap.add_argument("--out-dir", default="data/hb2")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--loader", default="serial", choices=("serial", "dl"))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SETTINGS["seed"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.perf_counter()
    out = run(args.data_dir, args.out_dir, steps=args.steps, seed=args.seed,
              loader=args.loader)
    wall = time.perf_counter() - t0
    st = np.array(out["times"][20:])
    report = {"loader": args.loader, "steps": args.steps,
              "wall_seconds": wall,
              "ms_per_step_mean": float(st.mean() * 1000),
              "loss_first50_mean": float(np.mean(out["losses"][:50])),
              "loss_last50_mean": float(np.mean(out["losses"][-50:])),
              "checkpoints": out["checkpoints"], "curve": out["curve"],
              "projection_minutes_50k": 50_000 * float(st.mean()) / 60}
    print(json.dumps(report, indent=1, sort_keys=True, default=str))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())