"""Build paired, class-balanced few-shot support sets.

The canonical train/validation/test arrays must already exist under ``data/<dataset>``.
For each requested split seed, this script samples exactly ``k`` training windows per
class and stores them under ``<k>-shot/seed-<seed>``. Validation and test arrays remain
at the dataset root and are never sampled or copied.

Examples
--------
    python data/build_splits.py
    python data/build_splits.py --datasets HHAR --shots 1 5 --seeds 0 1 2
"""
import argparse
import json
import os

import numpy as np

DATASETS = ("MotionSense", "HHAR", "PAMAP2")
SHOTS = (1, 5, 10, 20)
SEEDS = (0, 1, 2)


def _validate_root(root):
    required = (
        "x_train.npy", "y_train.npy", "x_valid.npy", "y_valid.npy",
        "x_test.npy", "y_test.npy",
    )
    missing = [name for name in required if not os.path.isfile(os.path.join(root, name))]
    if missing:
        raise FileNotFoundError(
            f"{root} is missing {missing}. Prepare the canonical arrays described "
            "in data/README.md before building few-shot supports."
        )


def build(base_path, dataset, shots, seeds):
    root = os.path.join(base_path, dataset)
    _validate_root(root)

    x = np.load(os.path.join(root, "x_train.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, "y_train.npy"))
    if len(x) != len(y):
        raise ValueError(f"x_train and y_train have different lengths in {root}")

    classes = np.unique(y)
    indices = {str(c): np.flatnonzero(y == c) for c in classes}
    smallest = min(len(v) for v in indices.values())
    print(
        f"\n{dataset}: {len(y)} training windows, {len(classes)} classes, "
        f"{smallest} windows in the smallest class, shape {tuple(x.shape)}"
    )

    for seed in seeds:
        for k in shots:
            if k > smallest:
                print(f"  seed {seed}, {k}-shot: skipped; smallest class has {smallest}")
                continue

            rng = np.random.RandomState(seed)
            selected_by_class = {}
            for label, class_indices in indices.items():
                order = class_indices.copy()
                rng.shuffle(order)
                selected_by_class[label] = order[:k].tolist()

            support_idx = np.sort(
                np.concatenate(
                    [np.asarray(v, dtype=np.int64) for v in selected_by_class.values()]
                )
            )
            out = os.path.join(root, f"{k}-shot", f"seed-{seed}")
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, "x_train.npy"), np.ascontiguousarray(x[support_idx]))
            np.save(os.path.join(out, "y_train.npy"), y[support_idx])

            manifest = {
                "dataset": dataset,
                "shot": k,
                "split_seed": seed,
                "classes": [str(c) for c in classes],
                "source": "x_train.npy/y_train.npy at the dataset root",
                "support_indices_by_class": selected_by_class,
                "validation": "fixed x_valid.npy/y_valid.npy at the dataset root",
                "test": "fixed x_test.npy/y_test.npy at the dataset root",
            }
            with open(os.path.join(out, "split_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            print(
                f"  seed {seed}, {k}-shot: {len(support_idx)} support windows "
                f"({k} per class) -> {out}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", default=os.path.dirname(os.path.abspath(__file__)),
        help="directory containing one subdirectory per dataset",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    for dataset_name in args.datasets:
        build(args.base, dataset_name, args.shots, args.seeds)
