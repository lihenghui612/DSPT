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
│   ├── x_train.npy     [N_source, 500, 12]
│   ├── y_train.npy     [N_source]
│   ├── x_test.npy      [N_test, 500, 12]
│   └── y_test.npy      [N_test]
├── HHAR/               same layout, 12 channels
└── PAMAP2/             same layout, 36 channels
```

`x_*` must use `[window, time, channel]` order. `y_*` contains integer labels in
`[0, num_classes)`. `float32` and `int64` are recommended. The loader casts arrays to
the appropriate PyTorch types at runtime.

## Temporal partitioning and windowing

The canonical benchmark partitions each continuous recording before generating any
windows. This ordering prevents adjacent overlapping windows from sharing raw readings
across the source and test sets.

1. Assemble sensor axes in one fixed, documented channel order and retain the original
   temporal order within every continuous recording.
2. Split each recording chronologically: use the first 80% as the source interval and
   hold out the final 20% for testing.
3. Generate windows independently inside the two temporal partitions. Each window contains
   500 readings and the stride is 250 readings (50% overlap).
4. Discard windows that cross an activity transition. Because windowing is performed
   separately inside each partition, a window can never cross a partition boundary.
5. Use one fixed source/test partition for every method. Test samples are never used
   for gradient updates, validation, or few-shot support construction.
6. Store raw sensor values. MOMENT applies reversible instance normalization inside
   the model.

The same participant or recording session may contribute temporally disjoint portions
to more than one partition, so this remains an in-domain evaluation. However, the
partitions share neither raw timestamps nor overlapping windows. Unseen-device and
unseen-subject generalization are evaluated separately by the HHAR group-exclusive
protocols below.

Prepare per-reading continuous arrays under, for example,
`data/HHAR/continuous/`:

```text
x.npy           [T, C] raw readings in recording-time order
y.npy           [T] per-reading activity labels
recording.npy   [T] continuous-recording identifiers
```

Then create the canonical root arrays and an auditable temporal-boundary manifest:

```bash
python data/build_temporal_partitions.py \
  --input_root data/HHAR/continuous \
  --output_root data/HHAR \
  --win_len 500 --stride 250 \
  --test_fraction 0.2
```

Repeat the command for MotionSense and PAMAP2. The script writes window-level recording
identifiers and raw start/end indices in addition to the four canonical arrays. It also
writes `temporal_split_manifest.json`, which records the temporal boundary of every
partition and certifies that splitting occurred before window generation.

## Paired few-shot support sets

Generate the three class-balanced support-set draws after the boundary-aware root
partitions have been prepared:

```bash
python data/build_splits.py \
  --datasets MotionSense HHAR PAMAP2 \
  --shots 1 5 10 20 \
  --seeds 0 1 2
```

The output layout is:

```text
data/MotionSense/
├── x_train.npy, y_train.npy, x_test.npy, y_test.npy
├── 1-shot/
│   ├── seed-0/train_indices.npy, valid_indices.npy, split_manifest.json
│   ├── seed-1/...
│   └── seed-2/...
├── 5-shot/...
├── 10-shot/...
└── 20-shot/...
```

For split seed `s`, exactly `K` windows per class are sampled from the fixed source
partition as the training support set. Every source window not selected for support
forms that run's validation set. All compared methods use the same `seed-s` directory,
so their support and validation indices are paired. Changing `s` resamples the support
set and therefore changes its complementary validation set. The same integer seed also
initializes model training. The held-out test arrays remain fixed at the dataset root.

Each support directory contains `train_indices.npy`, `valid_indices.npy`, and
`split_manifest.json`, including the source indices selected for every class. This
makes both the support draw and its complementary validation set auditable once the
canonical root arrays have been created, without duplicating the source arrays.

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
