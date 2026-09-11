"""Build subject-disjoint source/test windows from continuous recordings.

Subjects are assigned to source and test partitions *before* window generation.
Every recording must belong to exactly one subject and form one contiguous block in
the input arrays. Windows are then generated independently inside recordings, so a
window cannot cross an activity transition, recording boundary, or source/test
boundary.

Expected input arrays
---------------------
``x.npy``
    Raw readings with shape ``[T, C]``.
``y.npy``
    Per-reading integer activity labels with shape ``[T]``.
``subject.npy``
    Per-reading subject identifiers with shape ``[T]``.
``recording.npy``
    Per-reading continuous-recording identifiers with shape ``[T]``.

For exact reproduction, pass the paper's held-out subject identifiers explicitly:

    python data/build_subject_partitions.py \
      --input_root data/HHAR/continuous --output_root data/HHAR \
      --test_subjects SUBJECT_A SUBJECT_B

When ``--test_subjects`` is omitted, a deterministic approximately 80/20 subject
split is generated from ``--split_seed`` and recorded in the manifest.
"""

import argparse
import json
import os

import numpy as np


SPLITS = ("train", "test")


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _recording_blocks(recording, subject):
    """Return contiguous ``(recording, subject, start, end)`` blocks."""
    if len(recording) == 0:
        raise ValueError("recording.npy is empty")

    boundaries = np.flatnonzero(recording[1:] != recording[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(recording)]))
    blocks = []
    seen = set()
    for start, end in zip(starts, ends):
        rec = recording[start]
        key = str(rec)
        if key in seen:
            raise ValueError(
                f"recording {rec!r} occurs in multiple non-contiguous blocks; "
                "sort the raw arrays by recording and time before partitioning"
            )
        seen.add(key)
        owners = np.unique(subject[start:end])
        if len(owners) != 1:
            raise ValueError(
                f"recording {rec!r} maps to multiple subjects: {owners.tolist()}"
            )
        blocks.append((rec, owners[0], int(start), int(end)))
    return blocks


def _select_subjects(subject, test_subjects, test_fraction, split_seed):
    subjects = np.asarray(sorted(np.unique(subject), key=str), dtype=object)
    if len(subjects) < 2:
        raise ValueError("at least two subjects are required for a disjoint split")

    by_string = {str(value): value for value in subjects}
    if test_subjects:
        unknown = sorted(set(map(str, test_subjects)) - set(by_string))
        if unknown:
            raise ValueError(f"unknown --test_subjects values: {unknown}")
        test = np.asarray([by_string[str(value)] for value in test_subjects], dtype=object)
    else:
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        order = subjects.copy()
        np.random.RandomState(split_seed).shuffle(order)
        n_test = max(1, int(round(len(order) * test_fraction)))
        n_test = min(n_test, len(order) - 1)
        test = order[:n_test]

    test_keys = {str(value) for value in test}
    source = np.asarray(
        [value for value in subjects if str(value) not in test_keys], dtype=object
    )
    if len(source) == 0:
        raise ValueError("the test subject set leaves no source subjects")
    return source, test


def _window_recording(x, y, rec, owner, raw_start, raw_end, win_len, stride):
    rows = {key: [] for key in ("x", "y", "recording", "subject", "start", "end")}
    for start in range(raw_start, raw_end - win_len + 1, stride):
        end = start + win_len
        segment_y = y[start:end]
        if not np.all(segment_y == segment_y[0]):
            continue
        rows["x"].append(x[start:end])
        rows["y"].append(segment_y[0])
        rows["recording"].append(rec)
        rows["subject"].append(owner)
        rows["start"].append(start)
        rows["end"].append(end)
    return rows


def build(
    input_root,
    output_root,
    win_len=500,
    stride=250,
    test_fraction=0.2,
    split_seed=2026,
    test_subjects=None,
):
    if win_len <= 0 or stride <= 0:
        raise ValueError("win_len and stride must be positive")

    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    x = _load(os.path.join(input_root, "x.npy"))
    y = _load(os.path.join(input_root, "y.npy"))
    subject = _load(os.path.join(input_root, "subject.npy"))
    recording = _load(os.path.join(input_root, "recording.npy"))

    if x.ndim != 2:
        raise ValueError(f"x.npy must have shape [T, C], got {x.shape}")
    if any(array.ndim != 1 for array in (y, subject, recording)):
        raise ValueError("y.npy, subject.npy, and recording.npy must be one-dimensional")
    if len({len(x), len(y), len(subject), len(recording)}) != 1:
        raise ValueError("all input arrays must have equal lengths")

    source_subjects, held_out_subjects = _select_subjects(
        subject, test_subjects, test_fraction, split_seed
    )
    held_out_keys = {str(value) for value in held_out_subjects}
    collected = {
        split: {key: [] for key in ("x", "y", "recording", "subject", "start", "end")}
        for split in SPLITS
    }
    manifest_records = []

    for rec, owner, start, end in _recording_blocks(recording, subject):
        split = "test" if str(owner) in held_out_keys else "train"
        rows = _window_recording(x, y, rec, owner, start, end, win_len, stride)
        for key, values in rows.items():
            collected[split][key].extend(values)
        manifest_records.append(
            {
                "recording": str(rec),
                "subject": str(owner),
                "partition": split,
                "raw_start": start,
                "raw_end": end,
                "window_count": len(rows["x"]),
            }
        )

    os.makedirs(output_root, exist_ok=True)
    for split in SPLITS:
        rows = collected[split]
        if not rows["x"]:
            raise ValueError(f"no {split} windows were generated")
        np.save(
            os.path.join(output_root, f"x_{split}.npy"),
            np.ascontiguousarray(np.asarray(rows["x"])),
        )
        np.save(os.path.join(output_root, f"y_{split}.npy"), np.asarray(rows["y"]))
        np.save(
            os.path.join(output_root, f"recording_{split}.npy"),
            np.asarray(rows["recording"]),
        )
        np.save(
            os.path.join(output_root, f"subject_{split}.npy"),
            np.asarray(rows["subject"]),
        )
        np.save(
            os.path.join(output_root, f"start_{split}.npy"),
            np.asarray(rows["start"], dtype=np.int64),
        )
        np.save(
            os.path.join(output_root, f"end_{split}.npy"),
            np.asarray(rows["end"], dtype=np.int64),
        )

    manifest = {
        "protocol": "subject-disjoint source/test split before windowing",
        "split_before_windowing": True,
        "subject_partitions_are_disjoint": True,
        "window_length": win_len,
        "stride": stride,
        "overlap_fraction": 1.0 - stride / win_len,
        "split_seed": split_seed,
        "requested_test_fraction": test_fraction,
        "source_subjects": [str(value) for value in source_subjects],
        "test_subjects": [str(value) for value in held_out_subjects],
        "few_shot_validation": (
            "all source windows not selected for the fixed K-shot support set"
        ),
        "records": manifest_records,
        "window_counts": {split: len(collected[split]["x"]) for split in SPLITS},
    }
    manifest_path = os.path.join(output_root, "subject_split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print(f"wrote subject-disjoint partitions to {output_root}")
    print(f"  source subjects: {manifest['source_subjects']}")
    print(f"  test subjects:   {manifest['test_subjects']}")
    for split in SPLITS:
        print(f"  {split}: {len(collected[split]['x'])} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--win_len", type=int, default=500)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--test_subjects", nargs="+", default=None)
    build(**vars(parser.parse_args()))
