"""Evaluate robustness to independently sampled few-shot support sets.

Unlike the main fixed-support protocol, this analysis varies ``support_seed`` while
holding the initialization and optimization seed fixed. All compared methods should
be run with the same support seeds and optimization seed.

Example
-------
    python support_sampling.py --dataset HHAR --shot 5-shot \
      --finetune_type dspt --support_seeds 0 1 2 --optimization_seed 0
"""

import argparse
import json
import os

import numpy as np

from config import Config, DATASETS, FINETUNE_TYPES
from train import run_once, sample_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HHAR", choices=list(DATASETS))
    parser.add_argument(
        "--shot", default="5-shot", choices=("1-shot", "5-shot", "10-shot", "20-shot")
    )
    parser.add_argument("--finetune_type", default="dspt", choices=list(FINETUNE_TYPES))
    parser.add_argument("--support_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--optimization_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--result_dir", default="results/support_sampling")
    args = parser.parse_args()

    cfg = Config(dataset=args.dataset, shot=args.shot, finetune_type=args.finetune_type)
    if args.data_path is not None:
        cfg.data_path = os.path.join(os.path.abspath(args.data_path), args.shot)
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.num_epochs = args.epochs

    runs = []
    for support_seed in args.support_seeds:
        print(f"\n[support seed {support_seed} | optimization seed {args.optimization_seed}]")
        runs.append(
            run_once(
                cfg,
                args.optimization_seed,
                support_seed=support_seed,
                verbose=True,
            )
        )

    accuracy = np.asarray([run["test_acc"] for run in runs], dtype=float)
    macro_f1 = np.asarray([run["test"]["macro_f1"] for run in runs], dtype=float)
    summary = {
        "config": cfg.__dict__,
        "protocol": "vary support set while holding optimization randomness fixed",
        "support_seeds": args.support_seeds,
        "optimization_seed": args.optimization_seed,
        "runs": runs,
        "acc_mean": float(accuracy.mean()),
        "acc_std": sample_std(accuracy),
        "f1_mean": float(macro_f1.mean()),
        "f1_std": sample_std(macro_f1),
        "std_definition": "sample standard deviation (ddof=1)",
    }

    print("\n" + "=" * 74)
    print(
        f"support-set robustness | accuracy {summary['acc_mean']:.2f} +/- "
        f"{summary['acc_std']:.2f} | macro-F1 {summary['f1_mean']:.2f} +/- "
        f"{summary['f1_std']:.2f}"
    )
    print("=" * 74)

    os.makedirs(args.result_dir, exist_ok=True)
    output = os.path.join(
        args.result_dir,
        f"{args.dataset}_{args.shot}_{args.finetune_type}_opt{args.optimization_seed}.json",
    )
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"results written to {output}")


if __name__ == "__main__":
    main()
