"""Stage G0: does the GPU actually work, and how fast is it?

`plan3.md` asks for this as measurement rather than assumption, because the
card is Blackwell (compute capability 12.0) and the instruction says plainly
that "this card needs CUDA 12.8 or newer" was never verified. It is now: the
wheel is `2.11.0+cu128` and `get_arch_list()` contains `sm_120`.

Two halves, deliberately kept apart:

    results      which checks pass. A GPU either runs an fp16 convolution or
                 it does not, and that answer is a property of the install.
    measurement  throughput. Wall-clock, never reproducible, never claimed to
                 be. The model is built here and thrown away; nothing in this
                 repository trains anything yet.

Run it inside the project's own virtual environment:

    .venv-rl/bin/python rl/smoke_gpu.py
"""
import argparse
import json
import sys
import time


def gpu_facts():
    import torch
    out = {
        "torch_version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_is_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "arch_list": torch.cuda.get_arch_list(),
    }
    if out["cuda_is_available"]:
        props = torch.cuda.get_device_properties(0)
        out["device_0_name"] = torch.cuda.get_device_name(0)
        out["device_0_capability"] = list(torch.cuda.get_device_capability(0))
        out["device_0_total_memory_bytes"] = props.total_memory
        out["device_0_total_memory_gib"] = props.total_memory / 1024 ** 3
        out["device_0_multi_processor_count"] = props.multi_processor_count
    return out


def numpy_interop():
    """numpy arrays must survive a round trip through a tensor.

    The features are numpy arrays built by `engine.bits_to_plane`, so this is
    not a formality: if the two libraries could not exchange buffers, the
    feature pipeline and the model could not be connected at all.
    """
    import numpy as np
    import torch
    a = np.zeros((2, 2), np.float32)
    t = torch.from_numpy(a)
    back = t.numpy()
    return {
        "numpy_version": np.__version__,
        "from_numpy": isinstance(t, torch.Tensor),
        "shape_preserved": tuple(t.shape) == (2, 2),
        "dtype_preserved": str(t.dtype) == "torch.float32",
        "numpy_dtype_preserved": str(back.dtype) == "float32",
        "values_equal": bool(np.array_equal(a, back)),
        "shares_memory": bool(back.__array_interface__["data"][0]
                              == a.__array_interface__["data"][0]),
    }


def conv_smoke(dtype):
    """One forward and one backward of the shape the feature stack produces.

    `(256, 14, 20, 20)` is 256 positions of the 14-channel board, which is the
    largest batch this stage's measurements use, so it doubles as the upper
    bound on the smoke test.
    """
    return gradient_probe(dtype)["runs"][str(dtype).replace("torch.", "")]


def gradient_probe(seed=0, batch=256, in_ch=14, channels=64, size=20):
    """The same convolution in fp32, fp16 and bf16, from **one** set of inputs.

    Stage G ran `conv_smoke` once per dtype, and each call re-seeded and
    re-initialised, so in principle the three runs shared an input and a weight
    initialisation. They did not, in effect: the reported gradient magnitudes
    differed by about 6,000x between fp16 and bf16, which is far too large to be
    a property of either format and is the kind of gap that means the runs were
    not comparable rather than that fp16 is 6,000x less accurate.

    This builds `x`, the convolution and the batch norm **once** and runs the
    same loss on all three, so the only thing that varies is the dtype. It also
    reports the fp32 gradient as the reference the two reduced precisions are
    measured against, which the stage G table had no way to do: without a
    reference there is nothing to call a difference large or small.

    fp32 is run outside autocast, because autocast has no fp32 mode and passing
    one would silently fall back to the same fp16 path this is meant to
    distinguish from.
    """
    import torch
    from torch import nn
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    x = torch.randn(batch, in_ch, size, size, device=dev)
    conv = nn.Conv2d(in_ch, channels, 3, padding=1).to(dev)
    bn = nn.BatchNorm2d(channels).to(dev)

    out = {}
    runs = {}
    for name, dtype, autocast in (("float32", None, False),
                                  ("float16", torch.float16, True),
                                  ("bfloat16", torch.bfloat16, True)):
        conv.zero_grad(set_to_none=True)
        bn.zero_grad(set_to_none=True)
        if autocast:
            with torch.autocast("cuda", dtype=dtype):
                act = bn(torch.relu(conv(x)))
        else:
            act = bn(torch.relu(conv(x)))
        loss = act.float().sum()
        loss.backward()
        grads = {"conv.weight": conv.weight.grad, "conv.bias": conv.bias.grad,
                 "bn.weight": bn.weight.grad, "bn.bias": bn.bias.grad}
        entry = {
            "ok": all(g is not None and not bool(torch.isnan(g).any()
                                                 or torch.isinf(g).any())
                      for g in grads.values()),
            "loss": float(loss),
            "output_dtype": str(act.dtype),
            "grad_abs_mean": {k: float(g.abs().mean())
                              for k, g in grads.items()},
            "grad_abs_max": {k: float(g.abs().max())
                             for k, g in grads.items()},
        }
        if name == "float32":
            out["reference"] = entry
        runs[name] = entry

    # Relative error against fp32, per tensor. `mean` is the mean of |g - g32|
    # over the mean of |g32|, so it is scale-free and comparable across tensors.
    for name, entry in runs.items():
        rel = {}
        for k in entry["grad_abs_mean"]:
            ref = out["reference"]["grad_abs_mean"][k]
            got = entry["grad_abs_mean"][k]
            rel[k] = abs(got - ref) / abs(ref) if ref else None
        entry["relative_to_fp32"] = rel
    out["runs"] = runs
    out["note"] = ("one x, one conv, one batchnorm, one loss; only the dtype "
                   "differs between the three runs")
    return out


def _block(cin, cout):
    """One residual block: two 3x3 convolutions, each with BatchNorm and ReLU."""
    from torch import nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def build_model(channels=64, blocks=6, in_ch=14):
    """The shape this stage's plan asks for, and nothing bigger.

    Six residual blocks at 64 channels on a 14-channel 20x20 input, with a 1x1
    policy head producing the 91 orientation planes and a value head producing
    four numbers. The policy head keeps the spatial shape because the action
    index is `(orientation, base)`, so the mask has to stay a mask.
    """
    from torch import nn

    class TwinHead(nn.Module):
        """A shared trunk with two heads, because the two heads read the same
        features. Returning them as a tuple rather than one tensor is what
        stops the value head from accidentally consuming the policy head's 91
        planes."""

        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                *[_block(channels, channels) for _ in range(blocks)],
            )
            # Policy keeps the spatial shape: the action index is
            # `(orientation, base)`, so the mask has to stay a mask.
            self.policy = nn.Conv2d(channels, 91, 1)
            self.value = nn.Conv2d(channels, 4, 1)

        def forward(self, x):
            t = self.trunk(x)
            return self.policy(t), self.value(t)

    return TwinHead()


def timing_record(batch, times, warmup=None, extra=None):
    """One timing block, with throughput derived from the mean rather than the
    sum.

    The two have to be kept in this relationship, and it is written out here
    instead of at the two call sites because the stage G report got it wrong
    once: `positions_per_second` divided the batch by the *total* of every
    iteration, so it came out `iters` times too small - 363.1 instead of
    18,153.2 for a batch of 64 - while `mean_ms` on the row next to it was
    correct, which is what made the table internally inconsistent. One function
    that computes `mean_ms` from `times` and the rate from `mean_ms` cannot
    disagree with itself.

    `times` are per-iteration wall-clock seconds. The caller is responsible for
    synchronising the device before starting and stopping the clock; this
    function is pure so a test can hand it a fixed list and check the identity
    `positions_per_second == batch / (mean_ms / 1000)` without a GPU.
    """
    times = sorted(float(t) for t in times)
    n = len(times)
    total = sum(times)
    mean_ms = total / n * 1000.0
    record = {
        "batch": batch,
        "iterations": n,
        "mean_ms": mean_ms,
        "median_ms": times[n // 2] * 1000.0,
        "min_ms": times[0] * 1000.0,
        "max_ms": times[-1] * 1000.0,
        "positions_per_second": batch / (mean_ms / 1000.0),
    }
    if warmup is not None:
        record["warmup"] = warmup
    if extra:
        record.update(extra)
    return record


def throughput(batches=(64, 256, 1024), warmup=10, iters=50,
               channels=64, blocks=6, in_ch=14):
    """Inference positions per second, and one train step.

    `inference_mode` plus fp16 autocast is what an evaluation pass will look
    like. The optimiser step is measured separately because it is a different
    question: the inference number says how fast a position can be scored, and
    the step number says whether a batch of 256 fits the time budget of a
    training run at all.
    """
    import torch
    dev = torch.device("cuda")
    torch.manual_seed(0)
    net = build_model(channels, blocks, in_ch).to(dev).eval()
    n_params = sum(p.numel() for p in net.parameters())

    inference = []
    for bs in batches:
        x = torch.randn(bs, in_ch, 20, 20, device=dev)
        with torch.inference_mode():
            for _ in range(warmup):
                with torch.autocast("cuda", dtype=torch.float16):
                    net(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    out = net(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        policy_out, value_out = out
        inference.append(timing_record(bs, times, warmup, {
            "policy_shape": list(policy_out.shape),
            "value_shape": list(value_out.shape),
        }))

    # One real optimisation step: forward, backward, AdamW update. Measured
    # after the inference numbers so the caching allocator is warm and the
    # number is not dominated by the first allocation of a fresh shape.
    bs = 256
    net = build_model(channels, blocks, in_ch).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.randn(bs, in_ch, 20, 20, device=dev)
    target = torch.randn(bs, 91, 20, 20, device=dev)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            policy, _value = net(x)
            loss = torch.nn.functional.mse_loss(policy.float(), target)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = one_step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    train_step = timing_record(bs, times, warmup, {
        "last_loss": float(loss),
        "optimizer": "AdamW lr=1e-4",
        "loss": "MSE against a random (91, 20, 20) target, standing in for "
                "the real policy objective",
    })
    peak = None
    if dev.type == "cuda":
        peak = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        }
    del net, opt
    return {"model": {
                "architecture": ("%d residual blocks, %d channels, %d input "
                                 "channels, policy head 1x1 -> 91 planes, "
                                 "value head 1x1 -> 4 planes"
                                 % (blocks, channels, in_ch)),
                "parameters": n_params,
                "disposed": "built here and thrown away; nothing trains in "
                            "this repository yet"},
            "_measurement_inference": inference,
            "_measurement_train_step": train_step,
            "_measurement_memory": peak}


def main():
    ap = argparse.ArgumentParser(description="stage G0 GPU smoke test")
    ap.add_argument("--json", default=None, help="write the raw numbers here")
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    try:
        import torch
    except ImportError as exc:
        print("torch is not installed in this interpreter: %s" % exc,
              file=sys.stderr)
        print("use .venv-rl/bin/python", file=sys.stderr)
        return 2

    raw = {"facts": gpu_facts()}
    if not raw["facts"]["cuda_is_available"]:
        print(json.dumps(raw, indent=1, sort_keys=True))
        print("\nCUDA is not available; stopping here rather than pretending "
              "the rest ran.", file=sys.stderr)
        return 1

    raw["numpy_interop"] = numpy_interop()
    raw["gradient_probe"] = gradient_probe()
    raw["fp16"] = raw["gradient_probe"]["runs"]["float16"]
    raw["bf16"] = raw["gradient_probe"]["runs"]["bfloat16"]
    raw["fp32"] = raw["gradient_probe"]["runs"]["float32"]
    if not args.skip_throughput:
        raw.update(throughput(warmup=args.warmup, iters=args.iters))

    print(json.dumps(raw, indent=1, sort_keys=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
