"""Configuration for DSPT training and evaluation.

Default values follow the experimental setup described in the paper.
"""
from dataclasses import dataclass

# Per-dataset input dimensionality and label space.
DATASETS = {
    "MotionSense": {"n_channels": 12, "num_class": 6},
    "HHAR": {"n_channels": 12, "num_class": 6},
    "PAMAP2": {"n_channels": 36, "num_class": 12},
}

FINETUNE_TYPES = (
    "ft_head",     # classification head only
    "ft_full",     # full fine-tuning
    "adapter",     # bottleneck adapters
    "lora",        # low-rank updates on attention projections
    "std_pt",      # standard prompt tuning
    "dspt",        # decomposed sensor prompt tuning
    "variant_a",   # [P + AB ; H]
    "variant_b",   # [P ; H] + AB
)


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    dataset: str = "MotionSense"
    shot: str = "5-shot"              # 1-shot | 5-shot | 10-shot | 20-shot | full
    data_root: str = "data"
    data_path: str = None
    n_channels: int = 12
    num_class: int = 6
    win_len: int = 500                # sliding-window length
    moment_seq_len: int = 512         # backbone input length

    # -------------------------------------------------------------- backbone
    model_path: str = "./models"      # MOMENT-SMALL: d_model=512, patch_len=8
    finetune_type: str = "dspt"

    # -------------------------------------------------------------- training
    num_epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-3       # classification head
    weight_decay: float = 0.0
    device: str = "cuda"
    seed: int = 42

    # ------------------------------------------------------------------ DSPT
    m: int = 9                        # task-guidance prompt length
    r: int = 5                        # rank of the sensor-embedding update matrices
    prompt_len: int = 15              # prompt length l of standard prompt tuning
    dual_lr: bool = True
    alpha1: float = 1e-3              # λ1: task-guidance prompt learning rate
    alpha2: float = 1e-4              # λ2: sensor-embedding matrix learning rate

    # ------------------------------------------------------------- baselines
    adapter_bottleneck: int = 24
    lora_r: int = 8
    lora_alpha: int = 16

    # ----------------------------------------------------------------- other
    save_dir: str = "checkpoints"
    result_dir: str = "results"
    save_ckpt: bool = False
    num_workers: int = 2

    def __post_init__(self):
        if self.dataset in DATASETS:
            self.n_channels = DATASETS[self.dataset]["n_channels"]
            self.num_class = DATASETS[self.dataset]["num_class"]
        if self.data_path is None:
            import os
            self.data_path = os.path.join(self.data_root, self.dataset)
            if self.shot != "full":
                self.data_path = os.path.join(self.data_path, self.shot)

    @property
    def n_patches(self):
        return self.moment_seq_len // 8


def parameter_budget(cfg):
    """Trainable input-space parameters of standard PT and of DSPT.

    Standard PT uses ``l * d`` parameters; DSPT uses ``m * d + (s + d) * r``.
    The two are matched so that both methods share the same budget.
    """
    d, s = 512, cfg.n_patches
    n_pt = cfg.prompt_len * d
    n_dspt = cfg.m * d + (s + d) * cfg.r
    return n_pt, n_dspt, abs(n_pt - n_dspt) / n_pt
