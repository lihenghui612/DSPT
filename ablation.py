"""Ablation studies.

    prompt_len   sensitivity of standard prompt tuning to the prompt length l
    m            task-guidance length m, with r solved from the budget constraint
    lr           grid search over λ1 (prompt) and λ2 (sensor-embedding matrices)
    placement    where to inject the low-rank sensor-embedding update

Examples
--------
    python ablation.py --study m --dataset MotionSense --shot 5-shot
    python ablation.py --study placement --dataset MotionSense --shot 5-shot
"""
import argparse
import json
import os

import numpy as np

from config import Config, DATASETS
from train import run_once, sample_std

STUDIES = ("prompt_len", "m", "lr", "placement")


def rank_for(m, prompt_len=15, d_model=512, n_patches=64):
    """Solve ``l * d = m * d + (s + d) * r`` for the rank ``r``."""
    return max(int(round((prompt_len - m) * d_model / (n_patches + d_model))), 0)


def summarise(runs, label, override):
    acc = np.array([r["test_acc"] for r in runs])
    f1 = np.array([r["test"]["macro_f1"] for r in runs])
    return {
        "label": label,
        "override": override,
        "runs": [
            {
                "seed": int(run["seed"]),
                "accuracy": float(run["test_acc"]),
                "macro_f1": float(run["test"]["macro_f1"]),
            }
            for run in runs
        ],
        "acc_mean": float(acc.mean()),
        "acc_std": sample_std(acc),
        "f1_mean": float(f1.mean()),
        "f1_std": sample_std(f1),
        "minutes": float(np.mean([r["minutes"] for r in runs])),
        "peak_mem_mb": runs[0]["peak_mem_mb"],
        "trainable": runs[0]["trainable"],
    }


def settings_for(study, prompt_len):
    """Return ``[(label, config overrides), ...]`` for one study."""
    if study == "prompt_len":
        # l = 0 corresponds to tuning the classification head alone.
        settings = [("l=0", {"finetune_type": "ft_head"})]
        for l in (5, 10, 15, 20, 25, 30):
            settings.append((f"l={l}", {"finetune_type": "std_pt", "prompt_len": l}))
        return settings

    if study == "m":
        settings = []
        for m in (0, 3, 6, 9, 12, 15):
            r = rank_for(m, prompt_len)
            if m == 0:
                # Sensor-embedding adaptation only, without task guidance.
                settings.append((f"m=0, r={r}", {"finetune_type": "dspt", "m": 0, "r": r}))
            elif r == 0:
                # No budget left for sensor-embedding adaptation; reduces to standard PT.
                settings.append((f"m={m}, r=0", {"finetune_type": "std_pt", "prompt_len": m}))
            else:
                settings.append((f"m={m}, r={r}", {"finetune_type": "dspt", "m": m, "r": r}))
        return settings

    if study == "lr":
        grid = (1e-2, 1e-3, 1e-4, 1e-5)
        settings = [
            (f"alpha1={a1:.0e}, alpha2={a2:.0e}",
             {"finetune_type": "dspt", "alpha1": a1, "alpha2": a2, "dual_lr": True})
            for a1 in grid for a2 in grid
        ]
        for lr in (1e-3, 1e-4):
            settings.append((f"single lr={lr:.0e}",
                             {"finetune_type": "dspt", "learning_rate": lr, "dual_lr": False}))
        return settings

    if study == "placement":
        return [
            ("standard PT  [P ; H]", {"finetune_type": "std_pt"}),
            ("variant A    [P + AB ; H]", {"finetune_type": "variant_a"}),
            ("variant B    [P ; H] + AB", {"finetune_type": "variant_b"}),
            ("DSPT         [P ; H + AB]", {"finetune_type": "dspt"}),
        ]

    raise ValueError(f"unknown study: {study}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=list(STUDIES))
    p.add_argument("--dataset", default="MotionSense", choices=list(DATASETS))
    p.add_argument("--shot", default="5-shot")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--result_dir", default="results/ablation")
    args = p.parse_args()

    reference = Config(dataset=args.dataset, shot=args.shot)
    table = []

    for label, override in settings_for(args.study, reference.prompt_len):
        cfg = Config(dataset=args.dataset, shot=args.shot)
        for key, value in override.items():
            setattr(cfg, key, value)
        if args.epochs is not None:
            cfg.num_epochs = args.epochs

        print("\n" + "-" * 74)
        print(f"[{args.study}] {label}")
        runs = [run_once(cfg, seed, verbose=False) for seed in args.seeds]
        entry = summarise(runs, label, override)
        table.append(entry)
        print(f"  accuracy {entry['acc_mean']:.2f} +/- {entry['acc_std']:.2f} | "
              f"macro-F1 {entry['f1_mean']:.2f} | trainable {entry['trainable']:,} | "
              f"{entry['minutes']:.1f} min | peak {entry['peak_mem_mb'] or 0:.0f} MB")

    print("\n" + "=" * 74)
    print(f"{args.study} ablation on {args.dataset} / {args.shot}")
    print(f"{'setting':<28}{'accuracy':>16}{'macro-F1':>11}{'params':>11}{'min':>7}")
    for entry in table:
        print(f"{entry['label']:<28}"
              f"{entry['acc_mean']:>9.2f} +/-{entry['acc_std']:<4.2f}"
              f"{entry['f1_mean']:>11.2f}{entry['trainable']:>11,}{entry['minutes']:>7.1f}")
    print("=" * 74)

    os.makedirs(args.result_dir, exist_ok=True)
    path = os.path.join(args.result_dir, f"{args.dataset}_{args.shot}_{args.study}.json")
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"results written to {path}")


if __name__ == "__main__":
    main()
