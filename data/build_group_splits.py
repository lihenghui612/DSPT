"""Build HHAR leave-one-device-model-out or leave-one-user-out folds.

The input consists of per-reading continuous arrays. A complete device-model or
subject group is assigned to the test domain before 500-reading windows are
generated. This ordering guarantees that no overlapping window can cross the
source/test boundary.

Examples
--------
    python data/build_group_splits.py --axis device --support_seeds 0 1 2
    python data/build_group_splits.py --axis subject --shots 5 10
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


def _recording_blocks(recording, device, subject):
    if len(recording) == 0:
        raise ValueError("recording.npy is empty")
    boundaries = np.flatnonzero(recording[1:] != recording[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(recording)]))
    blocks = []
    seen = set()
    for start, end in zip(starts, ends):
        rec = recording[start]
        if str(rec) in seen:
            raise ValueError(f"recording {rec!r} occurs in multiple non-contiguous blocks")
        seen.add(str(rec))
        devices = np.unique(device[start:end])
        subjects = np.unique(subject[start:end])
        if len(devices) != 1 or len(subjects) != 1:
            raise ValueError(
                f"recording {rec!r} must map to exactly one device and one subject"
            )
        blocks.append((rec, devices[0], subjects[0], int(start), int(end)))
    return blocks


def _window_block(x, y, start, end, win_len, stride):
    windows, labels = [], []
    for window_start in range(start, end - win_len + 1, stride):
        window_end = window_start + win_len
        segment_y = y[window_start:window_end]
        if not np.all(segment_y == segment_y[0]):
            continue
        windows.append(x[window_start:window_end])
        labels.append(segment_y[0])
    return windows, labels


def _source_partition(y, validation_fraction, seed):
    rng = np.random.RandomState(seed)
    train_parts, valid_parts = [], []
    all_indices = np.arange(len(y), dtype=np.int64)
    for label in np.unique(y):
        indices = all_indices[y == label].copy()
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"class {label!r} has fewer than two source windows")
        n_valid = max(1, int(round(len(indices) * validation_fraction)))
        n_valid = min(n_valid, len(indices) - 1)
        valid_parts.append(indices[:n_valid])
        train_parts.append(indices[n_valid:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(valid_parts))


def _save_pair(root, split, x, y, indices=None):
    if indices is not None:
        x, y = x[indices], y[indices]
    np.save(os.path.join(root, f"x_{split}.npy"), np.ascontiguousarray(x))
    np.save(os.path.join(root, f"y_{split}.npy"), y)


def _write_supports(fold_root, y_train, shots, support_seeds):
    classes = np.unique(y_train)
    all_indices = np.arange(len(y_train), dtype=np.int64)
    class_indices = {str(label): all_indices[y_train == label] for label in classes}
    smallest = min(len(values) for values in class_indices.values())

    for support_seed in support_seeds:
        for k in shots:
            if k > smallest:
                raise ValueError(
                    f"cannot construct {k}-shot support: smallest source class has {smallest}"
                )
            rng = np.random.RandomState(support_seed)
            selected_by_class = {}
            for label, indices in class_indices.items():
                order = indices.copy()
                rng.shuffle(order)
                selected_by_class[label] = order[:k].tolist()
            support_idx = np.sort(
                np.concatenate(
                    [np.asarray(values, dtype=np.int64) for values in selected_by_class.values()]
                )
            )
            out = os.path.join(
                fold_root, f"{k}-shot", f"support-seed-{support_seed}"
            )
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, "train_indices.npy"), support_idx)
            with open(os.path.join(out, "split_manifest.json"), "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "shot": k,
                        "support_seed": support_seed,
                        "support_indices_by_class": selected_by_class,
                        "validation": "fixed source-only validation arrays at fold root",
                        "test": "complete held-out group arrays at fold root",
                    },
                    stream,
                    indent=2,
                )


def build(args):
    if args.win_len <= 0 or args.stride <= 0:
        raise ValueError("win_len and stride must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    input_root = os.path.abspath(args.input_root)
    x = _load(os.path.join(input_root, "x.npy"))
    y = _load(os.path.join(input_root, "y.npy"))
    device = _load(os.path.join(input_root, "device.npy"))
    subject = _load(os.path.join(input_root, "subject.npy"))
    recording = _load(os.path.join(input_root, "recording.npy"))

    if x.ndim != 2:
        raise ValueError(f"x.npy must have shape [T, C], got {x.shape}")
    if any(array.ndim != 1 for array in (y, device, subject, recording)):
        raise ValueError("metadata arrays must be one-dimensional")
    if len({len(x), len(y), len(device), len(subject), len(recording)}) != 1:
        raise ValueError("all input arrays must have equal lengths")

    groups = device if args.axis == "device" else subject
    output_root = os.path.abspath(
        args.output_root
        or os.path.join(os.path.dirname(input_root), f"cross-{args.axis}")
    )
    os.makedirs(output_root, exist_ok=True)
    blocks = _recording_blocks(recording, device, subject)
    fold_records = []

    for fold_index, held_group in enumerate(np.unique(groups)):
        source_x, source_y, test_x, test_y = [], [], [], []
        for _, block_device, block_subject, start, end in blocks:
            block_group = block_device if args.axis == "device" else block_subject
            windows, labels = _window_block(
                x, y, start, end, args.win_len, args.stride
            )
            if str(block_group) == str(held_group):
                test_x.extend(windows)
                test_y.extend(labels)
            else:
                source_x.extend(windows)
                source_y.extend(labels)

        source_x = np.asarray(source_x)
        source_y = np.asarray(source_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        if not len(source_y) or not len(test_y):
            raise ValueError(f"fold {fold_index} produced an empty source or test domain")

        train_idx, valid_idx = _source_partition(
            source_y, args.validation_fraction, args.partition_seed + fold_index
        )
        fold_root = os.path.join(output_root, f"fold-{fold_index}-{_slug(held_group)}")
        os.makedirs(fold_root, exist_ok=True)
        _save_pair(fold_root, "train", source_x, source_y, train_idx)
        _save_pair(fold_root, "valid", source_x, source_y, valid_idx)
        _save_pair(fold_root, "test", test_x, test_y)

        y_train = source_y[train_idx]
        _write_supports(
            fold_root, y_train, args.shots, args.support_seeds
        )

        record = {
            "fold": fold_index,
            "axis": args.axis,
            "held_out_group": str(held_group),
            "domain_assignment_before_windowing": True,
            "window_length": args.win_len,
            "stride": args.stride,
            "source_train_windows": int(len(train_idx)),
            "source_validation_windows": int(len(valid_idx)),
            "held_out_test_windows": int(len(test_y)),
            "partition_seed": args.partition_seed + fold_index,
            "validation_fraction": args.validation_fraction,
        }
        fold_records.append(record)
        with open(os.path.join(fold_root, "fold_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(
            f"fold {fold_index}: hold out {held_group!r} | train {len(train_idx)} | "
            f"valid {len(valid_idx)} | test {len(test_y)} -> {fold_root}"
        )

    with open(os.path.join(output_root, "folds.json"), "w", encoding="utf-8") as stream:
        json.dump(fold_records, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=("device", "subject"))
    parser.add_argument("--input_root", default="data/HHAR/continuous")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--support_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--partition_seed", type=int, default=2026)
    parser.add_argument("--win_len", type=int, default=500)
    parser.add_argument("--stride", type=int, default=250)
    build(parser.parse_args())
