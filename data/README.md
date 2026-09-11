# Data preparation and evaluation protocol

This directory defines the stable interface between dataset-specific raw-file
parsing and DSPT training. The datasets themselves are not redistributed.

## Official dataset sources

| Dataset | Source | Channels | Classes | Subjects |
| --- | --- | ---: | ---: | ---: |
| MotionSense | [Official repository](https://github.com/mmalekzadeh/motion-sense) | 12 | 6 | 24 |
| HHAR | [UCI repository](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | 12 | 6 | 9 |
| PAMAP2 | [UCI repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 36 | 12 | 9 |

Users must download each dataset from its original provider and comply with the
applicable terms.

## Required continuous-data interface

After dataset-specific parsing, place the following per-reading arrays under
`data/<dataset>/continuous/`:

```text
x.npy           [T, C]  raw sensor readings in a fixed channel order
y.npy           [T]     integer activity label for every reading
subject.npy     [T]     subject identifier for every reading
recording.npy   [T]     continuous-recording identifier for every reading
```

For HHAR cross-domain evaluation, also provide:

```text
device.npy      [T]     sensing-device-model identifier for every reading
```

Requirements:

1. Each recording identifier must form one contiguous block.
2. A recording must map to exactly one subject and, when provided, one device model.
3. Samples inside every recording must preserve their original temporal order.
4. `y.npy` must use labels in `[0, num_classes)`.
5. Raw sensor values should be stored without test-set normalization; MOMENT applies
   reversible instance normalization inside the model.

Exact numerical reproduction additionally requires the same channel order, class
mapping, filtering rules, and fixed subject partition used in the paper. These
dataset-specific choices should be documented when the continuous arrays are created.

## Subject-disjoint split before windowing

The main evaluation first partitions participants into disjoint source and test sets,
with approximately 80% of the subjects assigned to the source set and the remaining
subjects held out for testing. Windowing is performed only after this subject-level
assignment.

For exact reproduction, pass the fixed held-out subject identifiers explicitly:

```bash
python data/build_subject_partitions.py \
  --input_root data/HHAR/continuous \
  --output_root data/HHAR \
  --test_subjects SUBJECT_A SUBJECT_B \
  --win_len 500 \
  --stride 250
```

If `--test_subjects` is omitted, the script generates a deterministic approximately
80/20 subject split using `--split_seed 2026`. This is useful for validating the
pipeline, but exact paper reproduction requires the canonical held-out IDs.

The script:

- assigns each complete subject to source or test before window generation;
- generates 500-reading windows separately inside each continuous recording;
- uses a stride of 250 readings (50% overlap);
- discards windows that cross an activity transition;
- writes the following canonical arrays:

```text
data/HHAR/
├── x_train.npy, y_train.npy
├── x_test.npy, y_test.npy
├── subject_train.npy, subject_test.npy
├── recording_train.npy, recording_test.npy
├── start_train.npy, end_train.npy
├── start_test.npy, end_test.npy
└── subject_split_manifest.json
```

`subject_split_manifest.json` records the source/test subject IDs, raw recording
boundaries, window settings, and generated sample counts.

## Class-balanced few-shot supports

After the fixed subject-disjoint root arrays have been created, generate the support
and validation indices:

```bash
python data/build_splits.py \
  --datasets MotionSense HHAR PAMAP2 \
  --shots 1 5 10 20 \
  --support_seeds 0 1 2
```

The output layout is:

```text
data/MotionSense/
├── x_train.npy, y_train.npy, x_test.npy, y_test.npy
├── 1-shot/
│   ├── support-seed-0/
│   │   ├── train_indices.npy
│   │   ├── valid_indices.npy
│   │   └── split_manifest.json
│   ├── support-seed-1/...
│   └── support-seed-2/...
├── 5-shot/...
├── 10-shot/...
└── 20-shot/...
```

For every support seed and every `K`:

- exactly `K` source windows per class form the support set;
- all remaining source windows form the validation set;
- support and validation indices are shared by every compared method;
- the held-out test partition remains unchanged.

## Two repeated-run protocols

### Main tables and figures: fixed support

The main results keep one class-balanced support set fixed across three runs and vary
only initialization and optimization randomness:

```bash
python train.py --dataset HHAR --shot 5-shot --finetune_type dspt \
  --support_seed 0 --seeds 0 1 2
```

The resulting sample standard deviation therefore measures initialization and
optimization variability under a fixed support set.

### Separate robustness analysis: varied support

The support-set robustness experiment varies the support selection while holding the
initialization and optimization seed fixed:

```bash
python support_sampling.py --dataset HHAR --shot 5-shot --finetune_type dspt \
  --support_seeds 0 1 2 --optimization_seed 0
```

Its sample standard deviation therefore characterizes support-set selection
variability. Do not mix these two uncertainty definitions when reporting results.

## HHAR cross-domain protocols

The cross-domain experiments assign the held-out device model or participant before
generating windows. Use the same per-reading arrays under `data/HHAR/continuous/` and
run:

```bash
python data/build_group_splits.py --axis device  --shots 5 10 --support_seeds 0 1 2
python data/build_group_splits.py --axis subject --shots 5 10 --support_seeds 0 1 2
```

The two protocols isolate complementary shifts:

- **Cross-device:** all recordings from one sensing-device model are held out for
  testing; participants may appear in both source and test domains.
- **Cross-subject:** all recordings from one participant are held out for testing;
  device models may appear in both source and test domains.

Within every fold, validation data are drawn only from source-domain windows. The
builders write fold and support manifests so the group assignment and every support
set can be audited.

Generated fold roots can be passed directly to `train.py`:

```bash
python train.py \
  --dataset HHAR \
  --data_path data/HHAR/cross-device/fold-0-LG-Nexus-4 \
  --shot 5-shot \
  --finetune_type dspt \
  --support_seed 0 \
  --seeds 0 1 2
```
