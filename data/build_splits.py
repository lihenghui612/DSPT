"""Build paired, class-balanced few-shot support/validation splits.

The canonical source and test arrays must already exist under ``data/<dataset>``.
For each requested seed, this script samples exactly ``k`` source windows per class
as the training support set. Every remaining source window becomes validation data
for that run. The held-out test arrays remain fixed at the dataset root.

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


def _validate_root(root, allow_unverified_root=False):
    required = (
        "x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy",
    )
    missing = [name for name in required if not os.path.isfile(os.path.join(root, name))]
    if missing:
        raise FileNotFoundError(
            f"{root} is missing {missing}. Prepare the canonical arrays described "
            "in data/README.md before building few-shot supports."
        )
    manifest_path = os.path.join(root, "temporal_split_manifest.json")
    if allow_unverified_root:
        return
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"{root} has no temporal_split_manifest.json. Build the canonical arrays "
            "with data/build_temporal_partitions.py, or pass --allow_unverified_root "
            "only for a deliberately external partition whose provenance you verified."
        )
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not (manifest.get("split_before_windowing")
            and manifest.get("raw_partitions_are_disjoint")):
        raise ValueError(
            f"{manifest_path} does not record a disjoint split before windowing"
        )


def build(base_path, dataset, shots, seeds, allow_unverified_root=False):
    root = os.path.join(base_path, dataset)
    _validate_root(root, allow_unverified_root)

    x = np.load(os.path.join(root, "x_train.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, "y_train.npy"))
    if len(x) != len(y):
        raise ValueError(f"x_train and y_train have different lengths in {root}")

    classes = np.unique(y)
    indices = {str(c): np.flatnonzero(y == c) for c in classes}
    all_indices = np.arange(len(y), dtype=np.int64)
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
            validation_mask = np.ones(len(y), dtype=bool)
            validation_mask[support_idx] = False
            validation_idx = all_indices[validation_mask]
            if validation_idx.size == 0:
                raise ValueError(
                    f"seed {seed}, {k}-shot leaves no source windows for validation"
                )

            out = os.path.join(root, f"{k}-shot", f"seed-{seed}")
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, "train_indices.npy"), support_idx)
            np.save(os.path.join(out, "valid_indices.npy"), validation_idx)

            manifest = {
                "dataset": dataset,
                "shot": k,
                "split_seed": seed,
                "classes": [str(c) for c in classes],
                "source": "x_train.npy/y_train.npy at the dataset root",
                "support_indices_by_class": selected_by_class,
                "support_indices_file": "train_indices.npy",
                "validation_indices_file": "valid_indices.npy",
                "validation": (
                    "all source indices not selected for this run's support set"
                ),
                "test": "fixed x_test.npy/y_test.npy at the dataset root",
            }
            with open(os.path.join(out, "split_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            print(
                f"  seed {seed}, {k}-shot: {len(support_idx)} support windows "
                f"({k} per class), {len(validation_idx)} validation windows -> {out}"
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
    parser.add_argument(
        "--allow_unverified_root", action="store_true",
        help="accept externally prepared root arrays without the temporal manifest",
    )
    args = parser.parse_args()
    for dataset_name in args.datasets:
        build(args.base, dataset_name, args.shots, args.seeds, args.allow_unverified_root)
