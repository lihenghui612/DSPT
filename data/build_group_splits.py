"""Build HHAR leave-one-device-model-out or leave-one-subject-out folds.

This script starts from recording-aware, already-windowed arrays under
``data/HHAR/domain_metadata``. It keeps complete held-out groups for testing, creates
a source-only validation partition, and generates paired few-shot support sets.

Examples
--------
    python data/build_group_splits.py --axis device --seeds 0 1 2
    python data/build_group_splits.py --axis subject --shots 5 10 --seeds 0 1 2
"""
import argparse
import json
import os
import re

import numpy as np


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _slug(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return text or "group"


def _validate_recordings(recordings, groups):
    for recording in np.unique(recordings):
        values = np.unique(groups[recordings == recording])
        if len(values) != 1:
            raise ValueError(
                f"recording {recording!r} maps to multiple group values: {values.tolist()}"
            )


def _source_partition(y, source_idx, validation_fraction, seed):
    rng = np.random.RandomState(seed)
    train_parts, valid_parts = [], []
    for label in np.unique(y[source_idx]):
        indices = source_idx[y[source_idx] == label].copy()
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"class {label!r} has fewer than two source windows")
        n_valid = max(1, int(round(len(indices) * validation_fraction)))
        n_valid = min(n_valid, len(indices) - 1)
        valid_parts.append(indices[:n_valid])
        train_parts.append(indices[n_valid:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(valid_parts))


def _save_pair(root, split, x, y, indices):
    np.save(os.path.join(root, f"x_{split}.npy"), np.ascontiguousarray(x[indices]))
    np.save(os.path.join(root, f"y_{split}.npy"), y[indices])


def _write_supports(fold_root, x, y, train_idx, shots, seeds):
    classes = np.unique(y[train_idx])
    class_indices = {str(c): train_idx[y[train_idx] == c] for c in classes}
    smallest = min(len(v) for v in class_indices.values())

    for seed in seeds:
        for k in shots:
            if k > smallest:
                raise ValueError(
                    f"cannot construct {k}-shot support: smallest source class has {smallest}"
                )
            rng = np.random.RandomState(seed)
            selected_by_class = {}
            for label, indices in class_indices.items():
                order = indices.copy()
                rng.shuffle(order)
                selected_by_class[label] = order[:k].tolist()
            support_idx = np.sort(
                np.concatenate(
                    [np.asarray(v, dtype=np.int64) for v in selected_by_class.values()]
                )
            )
            out = os.path.join(fold_root, f"{k}-shot", f"seed-{seed}")
            os.makedirs(out, exist_ok=True)
            _save_pair(out, "train", x, y, support_idx)
            with open(os.path.join(out, "split_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "shot": k,
                        "split_seed": seed,
                        "support_indices_by_class": selected_by_class,
                        "validation": "fixed source-only validation arrays at fold root",
                        "test": "complete held-out group arrays at fold root",
                    },
                    f,
                    indent=2,
                )


def build(args):
    metadata_root = os.path.abspath(args.metadata_root)
    x = _load(os.path.join(metadata_root, "x.npy"))
    y = _load(os.path.join(metadata_root, "y.npy"))
    groups = _load(os.path.join(metadata_root, f"{args.axis}.npy"))
    recordings = _load(os.path.join(metadata_root, "recording.npy"))

    lengths = {len(x), len(y), len(groups), len(recordings)}
    if len(lengths) != 1:
        raise ValueError(
            "x.npy, y.npy, the group array, and recording.npy must have equal lengths"
        )
    _validate_recordings(recordings, groups)

    output_root = os.path.abspath(
        args.output_root
        or os.path.join(os.path.dirname(metadata_root), f"cross-{args.axis}")
    )
    os.makedirs(output_root, exist_ok=True)

    fold_records = []
    for fold_index, held_group in enumerate(np.unique(groups)):
        test_idx = np.flatnonzero(groups == held_group)
        source_idx = np.flatnonzero(groups != held_group)
        train_idx, valid_idx = _source_partition(
            y, source_idx, args.validation_fraction, args.partition_seed + fold_index
        )

        fold_root = os.path.join(output_root, f"fold-{fold_index}-{_slug(held_group)}")
        os.makedirs(fold_root, exist_ok=True)
        _save_pair(fold_root, "train", x, y, train_idx)
        _save_pair(fold_root, "valid", x, y, valid_idx)
        _save_pair(fold_root, "test", x, y, test_idx)
        _write_supports(fold_root, x, y, train_idx, args.shots, args.seeds)

        record = {
            "fold": fold_index,
            "axis": args.axis,
            "held_out_group": str(held_group),
            "source_train_windows": int(len(train_idx)),
            "source_validation_windows": int(len(valid_idx)),
            "held_out_test_windows": int(len(test_idx)),
            "partition_seed": args.partition_seed + fold_index,
            "validation_fraction": args.validation_fraction,
        }
        fold_records.append(record)
        with open(os.path.join(fold_root, "fold_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(
            f"fold {fold_index}: hold out {held_group!r} | train {len(train_idx)} | "
            f"valid {len(valid_idx)} | test {len(test_idx)} -> {fold_root}"
        )

    with open(os.path.join(output_root, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(fold_records, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=("device", "subject"))
    parser.add_argument("--metadata_root", default="data/HHAR/domain_metadata")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--partition_seed", type=int, default=2026)
    build(parser.parse_args())
