"""Inference latency and throughput.

Measures the cost of the encoder on held-out windows. Timing is independent of the
learned parameter values, so a checkpoint is not required, but using real test-window
shapes avoids silently timing an unrelated synthetic interface.

Examples
--------
    python efficiency.py --dataset MotionSense --batch_size 8
    python efficiency.py --dataset MotionSense --batch_size 1 --device cpu
"""
import argparse
import os
import time

import numpy as np
import torch

from config import Config, DATASETS
from model import MomentHAR


def load_test_windows(dataset, data_root, n_iter, batch_size, seed):
    path = os.path.join(data_root, dataset, "x_test.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Prepare the canonical partitions described in data/README.md."
        )
    x = np.load(path, mmap_mode="r")
    expected = (500, DATASETS[dataset]["n_channels"])
    if x.ndim != 3 or tuple(x.shape[1:]) != expected:
        raise ValueError(f"{path} must have shape [N, {expected[0]}, {expected[1]}], got {x.shape}")
    if len(x) == 0:
        raise ValueError(f"{path} contains no test windows")

    rng = np.random.RandomState(seed)
    index_batches = rng.randint(0, len(x), size=(n_iter, batch_size))
    return [torch.from_numpy(np.asarray(x[idx], dtype=np.float32)) for idx in index_batches]


def measure(cfg, inputs, n_warmup=20):
    device = torch.device(cfg.device)
    model = MomentHAR(cfg).to(device).eval()
    batch_size = len(inputs[0])

    with torch.inference_mode():
        for i in range(n_warmup):
            model(inputs[i % len(inputs)].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        elapsed = []
        for cpu_x in inputs:
            x = cpu_x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

    elapsed = np.array(elapsed)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
    return {
        "batch_ms": elapsed.mean() * 1e3,
        "batch_ms_std": elapsed.std(ddof=1) * 1e3 if len(elapsed) > 1 else 0.0,
        "per_sample_ms": elapsed.mean() * 1e3 / batch_size,
        "samples_per_sec": batch_size / elapsed.mean(),
        "peak_mem_mb": peak,
        "trainable": model.count()[0],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MotionSense", choices=list(DATASETS))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--m_list", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15],
                   help="prompt lengths to evaluate; 0 means no prompt")
    p.add_argument("--reference", type=int, default=15,
                   help="prompt length used as the relative-speed baseline")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_root", default="data")
    p.add_argument("--n_iter", type=int, default=2000,
                   help="number of timed batches; the paper uses 2,000 with batch size 1")
    p.add_argument("--n_warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0,
                   help="seed used only to draw test-window indices")
    args = p.parse_args()

    print(f"{args.dataset} | batch size {args.batch_size} | device {args.device}")

    inputs = load_test_windows(
        args.dataset, args.data_root, args.n_iter, args.batch_size, args.seed
    )
    rows = []
    for m in args.m_list:
        cfg = Config(dataset=args.dataset, shot="5-shot", device=args.device,
                     finetune_type="ft_head" if m == 0 else "dspt")
        cfg.m = max(m, 1)
        rows.append((m, measure(cfg, inputs, n_warmup=args.n_warmup)))

    baseline = next((r["samples_per_sec"] for m, r in rows if m == args.reference),
                    rows[-1][1]["samples_per_sec"])

    header = (f"{'m':>4}{'batch mean±sd (ms)':>23}{'per sample (ms)':>18}"
              f"{'samples/s':>12}{'peak (MB)':>11}{'rel. speed':>12}")
    print(header)
    print("-" * len(header))
    for m, r in rows:
        batch = f"{r['batch_ms']:.2f}±{r['batch_ms_std']:.2f}"
        print(f"{m:>4}{batch:>23}{r['per_sample_ms']:>18.3f}"
              f"{r['samples_per_sec']:>12.1f}{r['peak_mem_mb']:>11.1f}"
              f"{r['samples_per_sec'] / baseline:>12.2f}")


if __name__ == "__main__":
    main()
