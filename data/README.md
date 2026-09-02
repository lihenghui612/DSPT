# Data preparation and evaluation protocol

## Why datasets are not included

MotionSense, HHAR, and PAMAP2 are public third-party datasets. Their raw archives
and processed derivatives are intentionally not redistributed here; users should
obtain them from the original providers and comply with the applicable terms.

| Dataset | Source | Channels used | Classes | Subjects |
| --- | --- | ---: | ---: | ---: |
| MotionSense | https://github.com/mmalekzadeh/motion-sense | 12 | 6 | 24 |
| HHAR | https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition | 12 | 6 | 9 |
| PAMAP2 | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | 36 | 12 | 9 |

The repository defines a stable NumPy interface between dataset-specific raw-file
parsing and model training. Exact reproduction requires using the same channel order,
class mapping, filtering rules, and fixed partition indices as the paper. These items
should be recorded when the canonical arrays are created; a merely shape-compatible
array is sufficient to run the code but does not guarantee the published numbers.

## Canonical array interface

```text
data/
├── MotionSense/
│   ├── x_train.npy     [N_train, 500, 12]
│   ├── y_train.npy     [N_train]
│   ├── x_valid.npy     [N_valid, 500, 12]
│   ├── y_valid.npy     [N_valid]
│   ├── x_test.npy      [N_test, 500, 12]
│   └── y_test.npy      [N_test]
├── HHAR/               same layout, 12 channels
└── PAMAP2/             same layout, 36 channels
```

`x_*` must use `[window, time, channel]` order. `y_*` contains integer labels in
`[0, num_classes)`. `float32` and `int64` are recommended. The loader casts arrays to
the appropriate PyTorch types at runtime.

## Windowing and in-domain partitions

1. Assemble sensor axes in one fixed, documented channel order.
2. Segment each continuous recording into windows of 500 readings with a stride of
   250 readings (50% overlap).
3. Discard windows that cross an activity transition.
4. Perform the sample-level in-domain split described in the paper: reserve 20% as
   the held-out test partition and reserve a disjoint validation subset from the
   remaining training portion. Use one fixed set of root partitions for every method.
5. Do not use validation or test samples for gradient updates or few-shot support
   construction.
6. Store raw sensor values. MOMENT applies reversible instance normalization inside
   the model.

Because splitting occurs at the window level in this in-domain benchmark, windows
from the same subject or recording session can occur in different root partitions,
and adjacent overlapping windows can retain temporal correlation. These results must
not be described as unseen-subject, unseen-session, or unseen-device generalization.

Reference windowing logic:

```python
import numpy as np

def sliding_windows(x, y, win_len=500, stride=250):
    """x: [T, C] sensor stream; y: [T] per-reading activity labels."""
    windows, labels = [], []
    for start in range(0, len(x) - win_len + 1, stride):
        segment_y = y[start:start + win_len]
        if not np.all(segment_y == segment_y[0]):
            continue
        windows.append(x[start:start + win_len])
        labels.append(segment_y[0])
    return np.asarray(windows), np.asarray(labels)
```

## Paired few-shot support sets

Generate the three class-balanced support-set draws after the fixed root partitions
have been prepared:

```bash
python data/build_splits.py \
  --datasets MotionSense HHAR PAMAP2 \
  --shots 1 5 10 20 \
  --seeds 0 1 2
```

The output layout is:

```text
data/MotionSense/
├── x_train.npy, y_train.npy, x_valid.npy, y_valid.npy, x_test.npy, y_test.npy
├── 1-shot/
│   ├── seed-0/x_train.npy, y_train.npy, split_manifest.json
│   ├── seed-1/...
│   └── seed-2/...
├── 5-shot/...
├── 10-shot/...
└── 20-shot/...
```

For split seed `s`, exactly `K` windows per class are sampled from the fixed training
partition. All compared methods use the same `seed-s` directory. Changing `s`
resamples the support set. The same integer seed also initializes model training,
which produces the paired protocol described in the paper. Validation and test arrays
remain fixed at the dataset root and are never copied into the support directories.

Each support directory contains `split_manifest.json`, including the source indices
selected for every class. This makes the draw auditable once the canonical root arrays
have been created.

## HHAR cross-device and cross-subject protocols

The domain-exclusive experiments require recording-aware metadata before the fold
split is constructed. Prepare the following arrays under `data/HHAR/domain_metadata/`:

```text
x.npy              [N, 500, 12]
y.npy              [N]
device.npy         [N] device-model identifier for each window
subject.npy        [N] participant identifier for each window
recording.npy      [N] continuous-recording identifier for each window
```

All windows originating from one continuous recording must have the same device and
subject identifiers. Create the leave-one-group-out folds with:

```bash
python data/build_group_splits.py --axis device --seeds 0 1 2
python data/build_group_splits.py --axis subject --seeds 0 1 2
```

For each fold, every window from the held-out group is used only for testing. Source
groups are divided into training and validation subsets, and few-shot supports are
sampled only from the source training subset. The script validates that a recording
does not map to multiple values of the held-out grouping variable. Consequently, no
recording can cross the source/test boundary.

The two protocols test different shifts:

- Cross-device holds out an entire device model; subjects may still overlap.
- Cross-subject holds out an entire participant; device models may still overlap.

Neither result should be described as explicit conditioning on device or participant
metadata. It evaluates whether the learned sensor-embedding adaptation remains useful
under a domain-exclusive shift.
