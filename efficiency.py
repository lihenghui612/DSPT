"""Inference latency and throughput.

Measures the cost of the encoder as a function of the prompt length, which controls
the ``O((m + s)^2)`` attention term. Timing is independent of the learned weights, so
no checkpoint is required.

Examples
--------
    python efficiency.py --dataset MotionSense --batch_size 8
    python efficiency.py --dataset MotionSense --batch_size 1 --device cpu
"""
import argparse
import time

import numpy as np
import torch

from config import Config, DATASETS
from model import MomentHAR


def measure(cfg, batch_size, n_warmup=10, n_iter=100):
    device = torch.device(cfg.device)
    model = MomentHAR(cfg).to(device).eval()
    x = torch.randn(batch_size, cfg.win_len, cfg.n_channels, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        elapsed = []
        for _ in range(n_iter):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

    elapsed = np.array(elapsed)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
    return {
        "batch_ms": elapsed.mean() * 1e3,
        "batch_ms_std": elapsed.std() * 1e3,
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
    p.add_argument("--n_iter", type=int, default=100)
    args = p.parse_args()

    print(f"{args.dataset} | batch size {args.batch_size} | device {args.device}")

    rows = []
    for m in args.m_list:
        cfg = Config(dataset=args.dataset, shot="5-shot", device=args.device,
                     finetune_type="ft_head" if m == 0 else "dspt")
        cfg.m = max(m, 1)
        rows.append((m, measure(cfg, args.batch_size, n_iter=args.n_iter)))

    baseline = next((r["samples_per_sec"] for m, r in rows if m == args.reference),
                    rows[-1][1]["samples_per_sec"])

    header = (f"{'m':>4}{'batch (ms)':>13}{'per sample (ms)':>18}"
              f"{'samples/s':>12}{'peak (MB)':>11}{'rel. speed':>12}")
    print(header)
    print("-" * len(header))
    for m, r in rows:
        print(f"{m:>4}{r['batch_ms']:>13.2f}{r['per_sample_ms']:>18.3f}"
              f"{r['samples_per_sec']:>12.1f}{r['peak_mem_mb']:>11.1f}"
              f"{r['samples_per_sec'] / baseline:>12.2f}")


if __name__ == "__main__":
    main()

