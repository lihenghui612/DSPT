"""Build boundary-aware train/validation/test windows from continuous records.

The split is performed on each continuous recording before window generation.
Windows are then created independently inside the three temporal partitions, so no
raw reading or overlapping window can cross a partition boundary.

Expected input arrays
---------------------
``x.npy``
    Raw readings with shape ``[T, C]``.
``y.npy``
    Per-reading activity labels with shape ``[T]``.
``recording.npy``
    Continuous-recording identifiers with shape ``[T]``. Each identifier must form
    one contiguous block in the input order.

Example
-------
    python data/build_temporal_partitions.py \
      --input_root data/HHAR/continuous --output_root data/HHAR
"""

import argparse
import json
import os

import numpy as np


SPLITS = ("train", "valid", "test")


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _recording_blocks(recording):
    """Return contiguous ``(recording_id, start, end)`` blocks."""
    if len(recording) == 0:
        raise ValueError("recording.npy is empty")

    boundaries = np.flatnonzero(recording[1:] != recording[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(recording)]))
    blocks = [(recording[start], int(start), int(end)) for start, end in zip(starts, ends)]

    seen = set()
    for rec, _, _ in blocks:
        key = str(rec)
        if key in seen:
            raise ValueError(
                f"recording {rec!r} occurs in multiple non-contiguous blocks; "
                "sort the raw arrays by recording and time before splitting"
            )
        seen.add(key)
    return blocks


def _temporal_ranges(start, end, test_fraction, validation_fraction):
    """Split one recording into train/validation/test raw-index ranges."""
    length = end - start
    source_end = start + int(np.floor(length * (1.0 - test_fraction)))
    source_length = source_end - start
    valid_length = int(np.floor(source_length * validation_fraction))
    valid_start = source_end - valid_length
    return {
        "train": (start, valid_start),
        "valid": (valid_start, source_end),
        "test": (source_end, end),
    }


def _window_partition(x, y, rec, raw_start, raw_end, win_len, stride):
    windows, labels, recordings, starts, ends = [], [], [], [], []
    for start in range(raw_start, raw_end - win_len + 1, stride):
        end = start + win_len
        segment_y = y[start:end]
        if not np.all(segment_y == segment_y[0]):
            continue
        windows.append(x[start:end])
        labels.append(segment_y[0])
        recordings.append(rec)
        starts.append(start)
        ends.append(end)
    return windows, labels, recordings, starts, ends


def build(input_root, output_root, win_len=500, stride=250,
          test_fraction=0.2, validation_fraction=0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if win_len <= 0 or stride <= 0:
        raise ValueError("win_len and stride must be positive")

    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    x = _load(os.path.join(input_root, "x.npy"))
    y = _load(os.path.join(input_root, "y.npy"))
    recording = _load(os.path.join(input_root, "recording.npy"))

    if x.ndim != 2:
        raise ValueError(f"x.npy must have shape [T, C], got {x.shape}")
    if y.ndim != 1 or recording.ndim != 1:
        raise ValueError("y.npy and recording.npy must be one-dimensional")
    if not (len(x) == len(y) == len(recording)):
        raise ValueError("x.npy, y.npy, and recording.npy must have equal lengths")

    collected = {
        split: {"x": [], "y": [], "recording": [], "start": [], "end": []}
        for split in SPLITS
    }
    manifest_records = []

    for rec, start, end in _recording_blocks(recording):
        ranges = _temporal_ranges(start, end, test_fraction, validation_fraction)
        record_entry = {
            "recording": str(rec),
            "raw_start": start,
            "raw_end": end,
            "partitions": {},
        }
        for split in SPLITS:
            raw_start, raw_end = ranges[split]
            windows = _window_partition(
                x, y, rec, raw_start, raw_end, win_len, stride
            )
            for key, values in zip(("x", "y", "recording", "start", "end"), windows):
                collected[split][key].extend(values)
            record_entry["partitions"][split] = {
                "raw_start": raw_start,
                "raw_end": raw_end,
                "window_count": len(windows[0]),
            }
        manifest_records.append(record_entry)

    os.makedirs(output_root, exist_ok=True)
    for split in SPLITS:
        part = collected[split]
        if not part["x"]:
            raise ValueError(
                f"no {split} windows were generated; check recording lengths and split fractions"
            )
        np.save(os.path.join(output_root, f"x_{split}.npy"),
                np.ascontiguousarray(np.asarray(part["x"])))
        np.save(os.path.join(output_root, f"y_{split}.npy"), np.asarray(part["y"]))
        np.save(os.path.join(output_root, f"recording_{split}.npy"),
                np.asarray(part["recording"]))
        np.save(os.path.join(output_root, f"start_{split}.npy"),
                np.asarray(part["start"], dtype=np.int64))
        np.save(os.path.join(output_root, f"end_{split}.npy"),
                np.asarray(part["end"], dtype=np.int64))

    manifest = {
        "protocol": "chronological split before windowing",
        "split_before_windowing": True,
        "raw_partitions_are_disjoint": True,
        "window_length": win_len,
        "stride": stride,
        "overlap_fraction": 1.0 - stride / win_len,
        "test_fraction_per_recording": test_fraction,
        "validation_fraction_of_source": validation_fraction,
        "records": manifest_records,
        "window_counts": {
            split: len(collected[split]["x"]) for split in SPLITS
        },
    }
    with open(os.path.join(output_root, "temporal_split_manifest.json"),
              "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print(f"wrote boundary-aware partitions to {output_root}")
    for split in SPLITS:
        print(f"  {split}: {len(collected[split]['x'])} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--win_len", type=int, default=500)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation_fraction", type=float, default=0.2,
        help="fraction of the pre-test source interval reserved for validation",
    )
    args = parser.parse_args()
    build(**vars(args))
