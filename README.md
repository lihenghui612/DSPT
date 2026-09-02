# DSPT: Decomposed Sensor Prompt Tuning

Official implementation of *Unlocking the Potential of General-Purpose Time Series
Foundation Models via Decomposed Sensor Prompt Tuning for Few-Shot Activity
Recognition*.

DSPT adapts a frozen MOMENT backbone in the input-embedding space:

| Component | Operation |
| --- | --- |
| Sensor-embedding update | `H' = H + AB`, where `A ∈ R^{s×r}` and `B ∈ R^{r×d}` |
| Task-guidance prompt | `Z = [P ; H']`, where `P ∈ R^{m×d}` |
| Budget constraint | `l·d ≈ m·d + (s+d)·r` |

The encoder and patch embedder remain frozen. The task prompt, low-rank matrices,
and classification head are optimized for each downstream dataset.

For MOMENT-SMALL (`d=512`, `patch_len=8`, `seq_len=512`, hence `s=64`), DSPT
with `m=9` and `r=5` uses 7,488 input-space adaptation parameters. Standard
prompt tuning with `l=15` uses 7,680 parameters.

## Installation

Python 3.10 is recommended. Install the pinned dependencies in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download MOMENT-SMALL separately and save it under `./models`:

```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small")
model.save_pretrained("./models")
```

Model weights are not redistributed in this repository and remain subject to the
license of the original model authors.

## Data availability and preparation

The raw and processed datasets are intentionally not redistributed. MotionSense,
HHAR, and PAMAP2 are public third-party datasets governed by their respective terms.
Download them from the original sources:

| Dataset | Official source |
| --- | --- |
| MotionSense | https://github.com/mmalekzadeh/motion-sense |
| HHAR | https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition |
| PAMAP2 | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring |

[`data/README.md`](data/README.md) specifies the array interface, windowing rules,
in-domain evaluation protocol, few-shot support-set construction, and the metadata
required for the HHAR domain-exclusive experiments. Dataset-specific raw-file
parsing is kept outside this repository because the source formats and redistribution
terms are controlled by the dataset providers; the training code begins from the
documented NumPy array interface.

After preparing the fixed train/validation/test arrays, construct the three paired
few-shot support sets used by all methods:

```bash
python data/build_splits.py \
  --datasets MotionSense HHAR PAMAP2 \
  --shots 1 5 10 20 \
  --seeds 0 1 2
```

This produces paths such as `data/HHAR/5-shot/seed-0/`. For a given seed, every
method reads exactly the same support indices. Across seeds, the support set is
resampled independently.

## Training

Run three paired few-shot experiments as follows:

```bash
python train.py \
  --dataset MotionSense \
  --shot 5-shot \
  --finetune_type dspt \
  --seeds 0 1 2
```

For few-shot experiments, each value passed to `--seeds` controls both the support
set and the optimization randomness. For fully supervised experiments, the training
partition is fixed and the values control optimization randomness only. Reported
deviations are sample standard deviations (`ddof=1`).

Available adaptation strategies are:

| Value | Method |
| --- | --- |
| `ft_head` | Classification head only |
| `ft_full` | Full backbone fine-tuning |
| `adapter` | Bottleneck adapters |
| `lora` | Low-rank updates to attention projections |
| `std_pt` | Standard prompt tuning |
| `dspt` | DSPT: `[P ; H + AB]` |
| `variant_a` | Placement variant A: `[P + AB ; H]` |
| `variant_b` | Placement variant B: `[P ; H] + AB` |

Defaults follow the paper: 200 epochs, Adam, batch size 8, a cosine learning-rate
schedule, `m=9`, `r=5`, `alpha1=1e-3` for the prompt, and `alpha2=1e-4` for the
low-rank matrices. The best epoch is selected using validation accuracy, after which
the held-out test set is evaluated once. A JSON record containing the configuration,
per-seed metrics, mean, and sample standard deviation is written to `results/`.

Frequently used options:

```text
--epochs 200 --batch_size 8
--m 9 --r 5 --prompt_len 15
--alpha1 1e-3 --alpha2 1e-4
--no_dual_lr
--save_ckpt
--data_path /path/to/an/alternative/dataset-root
```

## Reproducing the main experiments

```bash
# Every data regime for DSPT on MotionSense
for shot in 1-shot 5-shot 10-shot 20-shot full; do
  python train.py --dataset MotionSense --shot "$shot" \
    --finetune_type dspt --seeds 0 1 2
done

# Every fine-tuning method under one regime
for method in ft_head ft_full adapter lora std_pt dspt; do
  python train.py --dataset MotionSense --shot 5-shot \
    --finetune_type "$method" --seeds 0 1 2
done
```

## Ablations and analysis

```bash
python ablation.py --study prompt_len --dataset MotionSense --shot 5-shot
python ablation.py --study m          --dataset MotionSense --shot 5-shot
python ablation.py --study lr         --dataset MotionSense --shot 5-shot
python ablation.py --study placement  --dataset MotionSense --shot 5-shot

python efficiency.py --dataset MotionSense --batch_size 8
python efficiency.py --dataset MotionSense --batch_size 1 --device cpu
```

To generate t-SNE plots, first save the trainable tensors for each method:

```bash
for method in ft_head lora dspt; do
  python train.py --dataset MotionSense --shot 5-shot \
    --finetune_type "$method" --seeds 0 --save_ckpt
done

python tsne.py --dataset MotionSense --shot 5-shot \
  --finetune_type ft_head lora dspt --seed 0
```

## HHAR domain-exclusive evaluation

The revised evaluation includes leave-one-device-model-out and leave-one-user-out
HHAR protocols. Prepare the recording-aware HHAR arrays described in
[`data/README.md`](data/README.md), then build the folds with:

```bash
python data/build_group_splits.py --axis device --seeds 0 1 2
python data/build_group_splits.py --axis subject --seeds 0 1 2
```

Each generated fold is a standalone dataset root. For example:

```bash
python train.py --dataset HHAR --data_path data/HHAR/cross-device/fold-0-LG-Nexus-4 \
  --shot 5-shot --finetune_type dspt --seeds 0 1 2
```

Use the exact generated fold directory name printed by the fold builder; the suffix
is derived from the held-out group label.

The fold builder assigns complete device or participant groups to the test domain
and draws validation data only from source groups. It does not treat device-exclusive
evaluation as subject-exclusive, or vice versa.

## Repository layout

```text
config.py                    Hyperparameters and dataset dimensions
model.py                     DSPT, baseline modules, and MOMENT wrapper
train.py                     Training, evaluation, and paired-run driver
ablation.py                  Prompt, budget, learning-rate, and placement studies
efficiency.py                Latency and throughput measurement
tsne.py                      Feature visualization
data/README.md               Data interface and protocol
data/build_splits.py         Paired few-shot support-set generation
data/build_group_splits.py   HHAR domain-exclusive fold generation
```

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The datasets and
MOMENT backbone are distributed separately by their respective authors and remain
subject to their original licenses and terms.
