"""Training and evaluation entry point.

Examples
--------
    python train.py --dataset MotionSense --shot 5-shot --finetune_type dspt
    python train.py --dataset PAMAP2 --shot 1-shot --finetune_type std_pt --seeds 0 1 2
"""
import argparse
import json
import os
import random
import re
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

from config import Config, DATASETS, FINETUNE_TYPES, parameter_budget
from model import MomentHAR


class WindowDataset(Dataset):
    """Sliding windows stored as ``x_<split>.npy`` / ``y_<split>.npy``.

    Splits that are absent from a few-shot directory are read from the dataset root,
    so the test arrays are stored only once.
    """

    def __init__(self, path, split, dataset_root=None):
        candidates = [path]
        if dataset_root is not None and dataset_root not in candidates:
            candidates.append(dataset_root)

        selected = None
        for candidate in candidates:
            x_path = os.path.join(candidate, f"x_{split}.npy")
            y_path = os.path.join(candidate, f"y_{split}.npy")
            if os.path.isfile(x_path) and os.path.isfile(y_path):
                selected = (x_path, y_path)
                break

        if selected is None:
            searched = ", ".join(candidates)
            raise FileNotFoundError(
                f"Could not find x_{split}.npy and y_{split}.npy in: {searched}"
            )

        self.x = torch.from_numpy(np.load(selected[0])).float()
        self.y = torch.from_numpy(np.load(selected[1])).long()
        if len(self.x) != len(self.y):
            raise ValueError(f"Mismatched x/y lengths for split '{split}'")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_std(values):
    """Sample standard deviation (denominator S-1); zero for a single run."""
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def safe_tag(value):
    """Convert a path-derived experiment identifier into a filename-safe tag."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def data_paths_for_run(cfg, seed):
    """Return the run-specific data directory and its fixed dataset root.

    Few-shot runs use ``<dataset>/<k>-shot/seed-<seed>`` so that a run seed controls
    both the support-set draw and optimization randomness. Fully supervised runs use
    the fixed dataset root.
    """
    if cfg.shot == "full":
        return cfg.data_path, cfg.data_path

    dataset_root = os.path.dirname(cfg.data_path)
    run_path = os.path.join(cfg.data_path, f"seed-{seed}")
    if not os.path.isdir(run_path):
        raise FileNotFoundError(
            f"Missing paired support set: {run_path}. Run data/build_splits.py "
            f"with --shots {cfg.shot.removesuffix('-shot')} --seeds {seed}."
        )
    return run_path, dataset_root


@torch.no_grad()
def evaluate(model, loader, device, detailed=False):
    """Accuracy on ``loader``; with ``detailed`` also macro scores and the confusion matrix."""
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    accuracy = 100.0 * (preds == trues).mean()
    if not detailed:
        return accuracy

    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    cm = confusion_matrix(trues, preds)
    per_class = 100.0 * cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    macro = dict(average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy), 2),
        "macro_f1": round(100.0 * f1_score(trues, preds, **macro), 2),
        "macro_precision": round(100.0 * precision_score(trues, preds, **macro), 2),
        "macro_recall": round(100.0 * recall_score(trues, preds, **macro), 2),
        "per_class_accuracy": per_class.round(2).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def run_once(cfg, seed, verbose=True):
    """Train one model and evaluate it on the test split.

    The best epoch is selected on the validation split; the test split is evaluated
    once, with the selected weights.
    """
    set_seed(seed)
    device = torch.device(cfg.device)

    run_data_path, dataset_root = data_paths_for_run(cfg, seed)
    train_set = WindowDataset(run_data_path, "train", dataset_root)
    valid_set = WindowDataset(run_data_path, "valid", dataset_root)
    test_set = WindowDataset(run_data_path, "test", dataset_root)

    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, **loader_kwargs)

    model = MomentHAR(cfg).to(device)
    model.init_prompt(train_loader, device)
    trainable, total = model.count()

    if verbose:
        print(f"  samples: train {len(train_set)} | valid {len(valid_set)} | test {len(test_set)}")
        print(f"  trainable {trainable:,} ({trainable / 1e6:.4f}M) of {total:,}")
        if model.dspt is not None:
            print(f"  input-space parameters {model.dspt.n_params():,} "
                  f"(m={model.dspt.m}, r={model.dspt.r}, s={model.dspt.s})")

    optimizer = torch.optim.Adam(model.param_groups(), weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_valid, best_epoch, best_state = -1.0, -1, None
    started = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        valid_acc = evaluate(model, valid_loader, device)
        if valid_acc > best_valid:
            best_valid, best_epoch = valid_acc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:3d}/{cfg.num_epochs} "
                  f"loss {epoch_loss / len(train_loader):.4f} valid {valid_acc:.2f} "
                  f"(best {best_valid:.2f} @ {best_epoch})")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test = evaluate(model, test_loader, device, detailed=True)
    minutes = (time.time() - started) / 60

    if verbose:
        print(f"  seed {seed}: valid {best_valid:.2f} (epoch {best_epoch}) | "
              f"test accuracy {test['accuracy']:.2f} macro-F1 {test['macro_f1']:.2f} | "
              f"{minutes:.1f} min")

    if cfg.save_ckpt:
        os.makedirs(cfg.save_dir, exist_ok=True)
        keys = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save(
            {"cfg": cfg.__dict__, "seed": seed,
             "state_dict": {k: v for k, v in best_state.items() if k in keys}},
            os.path.join(cfg.save_dir, f"{cfg.dataset}_{cfg.shot}_{cfg.finetune_type}_s{seed}.pth"),
        )

    return {
        "seed": seed,
        "data_path": run_data_path,
        "best_valid": best_valid,
        "best_epoch": best_epoch,
        "test": test,
        "test_acc": test["accuracy"],
        "trainable": trainable,
        "minutes": minutes,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None,
    }


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MotionSense", choices=list(DATASETS))
    p.add_argument(
        "--shot", default="5-shot",
        choices=("1-shot", "5-shot", "10-shot", "20-shot", "full"),
    )
    p.add_argument(
        "--data_path", default=None,
        help="optional dataset root containing fixed train/valid/test arrays; useful for folds",
    )
    p.add_argument("--finetune_type", default="dspt", choices=list(FINETUNE_TYPES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--m", type=int, default=None, help="task-guidance prompt length")
    p.add_argument("--r", type=int, default=None, help="rank of the calibration matrices")
    p.add_argument("--prompt_len", type=int, default=None, help="prompt length l of standard PT")
    p.add_argument("--alpha1", type=float, default=None, help="learning rate of the prompt")
    p.add_argument("--alpha2", type=float, default=None, help="learning rate of the low-rank pair")
    p.add_argument("--no_dual_lr", action="store_true", help="use a single learning rate")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--tag", default="")
    return p


def config_from_args(args):
    cfg = Config(dataset=args.dataset, shot=args.shot, finetune_type=args.finetune_type)
    if args.data_path is not None:
        root = os.path.abspath(args.data_path)
        cfg.data_path = root if args.shot == "full" else os.path.join(root, args.shot)
    overrides = {"epochs": "num_epochs", "batch_size": "batch_size", "m": "m", "r": "r",
                 "prompt_len": "prompt_len", "alpha1": "alpha1", "alpha2": "alpha2"}
    for arg_name, cfg_name in overrides.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if args.no_dual_lr:
        cfg.dual_lr = False
    cfg.save_ckpt = args.save_ckpt
    return cfg


def main():
    args = build_argparser().parse_args()
    cfg = config_from_args(args)
    n_pt, n_dspt, gap = parameter_budget(cfg)

    print("=" * 74)
    print(f"{cfg.dataset} | {cfg.shot} | {cfg.finetune_type}")
    print(f"data {cfg.data_path} | channels {cfg.n_channels} | classes {cfg.num_class}")
    print(f"epochs {cfg.num_epochs} | batch size {cfg.batch_size} | seeds {args.seeds}")
    print(f"budget: standard PT (l={cfg.prompt_len}) {n_pt:,} vs "
          f"DSPT (m={cfg.m}, r={cfg.r}) {n_dspt:,} | difference {gap:.1%}")
    print(f"dual learning rates {cfg.dual_lr} | alpha1 {cfg.alpha1} | alpha2 {cfg.alpha2} | "
          f"head {cfg.learning_rate}")
    print("=" * 74)

    results = []
    for seed in args.seeds:
        print(f"\n[seed {seed}]")
        results.append(run_once(cfg, seed))

    acc = np.array([r["test_acc"] for r in results])
    f1 = np.array([r["test"]["macro_f1"] for r in results])
    print("\n" + "=" * 74)
    acc_std = sample_std(acc)
    f1_std = sample_std(f1)
    print(f"test over {len(acc)} runs | accuracy {acc.mean():.2f} +/- {acc_std:.2f} | "
          f"macro-F1 {f1.mean():.2f} +/- {f1_std:.2f}")
    print(f"per-seed accuracy {[round(a, 2) for a in acc.tolist()]}")
    print("=" * 74)

    os.makedirs(cfg.result_dir, exist_ok=True)
    tag = args.tag
    if not tag and args.data_path:
        normalized = os.path.normpath(os.path.abspath(args.data_path))
        parent, leaf = os.path.basename(os.path.dirname(normalized)), os.path.basename(normalized)
        tag = f"{parent}_{leaf}"
    suffix = f"_{safe_tag(tag)}" if tag else ""
    path = os.path.join(cfg.result_dir,
                        f"{cfg.dataset}_{cfg.shot}_{cfg.finetune_type}{suffix}.json")
    with open(path, "w") as f:
        json.dump({"config": cfg.__dict__, "runs": results,
                   "acc_mean": float(acc.mean()), "acc_std": acc_std,
                   "f1_mean": float(f1.mean()), "f1_std": f1_std,
                   "std_definition": "sample standard deviation (ddof=1)"}, f, indent=2)
    print(f"results written to {path}")


if __name__ == "__main__":
    main()


