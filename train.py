"""Training and evaluation entry point.

Examples
--------
    python train.py --dataset MotionSense --shot 5-shot --finetune_type dspt
    python train.py --dataset PAMAP2 --shot 1-shot --finetune_type std_pt \
      --support_seed 0 --seeds 0 1 2
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

    A few-shot directory may contain either materialized arrays or index files. For
    the main protocol, ``train_indices.npy`` and ``valid_indices.npy`` select the
    run-specific support and complementary validation subsets from the fixed source
    arrays at the dataset root. Test arrays are stored only once at the root.
    """

    def __init__(self, path, split, dataset_root=None):
        dataset_root = dataset_root or path
        index_path = os.path.join(path, f"{split}_indices.npy")
        x_path = os.path.join(path, f"x_{split}.npy")
        y_path = os.path.join(path, f"y_{split}.npy")

        # Prefer the current index-based protocol over any materialized arrays left
        # by an earlier preparation run in the same directory.
        if os.path.isfile(index_path):
            if split not in ("train", "valid"):
                raise ValueError(f"indexed split '{split}' is not supported")
            base_x = np.load(os.path.join(dataset_root, "x_train.npy"), mmap_mode="r")
            base_y = np.load(os.path.join(dataset_root, "y_train.npy"), mmap_mode="r")
            indices = np.load(index_path)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(f"{index_path} must contain a one-dimensional integer array")
            if indices.size and (indices.min() < 0 or indices.max() >= len(base_y)):
                raise IndexError(f"{index_path} contains an out-of-range source index")
            x, y = np.asarray(base_x[indices]), np.asarray(base_y[indices])
        elif os.path.isfile(x_path) and os.path.isfile(y_path):
            x, y = np.load(x_path), np.load(y_path)
        else:
            root_x = os.path.join(dataset_root, f"x_{split}.npy")
            root_y = os.path.join(dataset_root, f"y_{split}.npy")
            allow_root_fallback = path == dataset_root or split in ("valid", "test")
            if not (allow_root_fallback and os.path.isfile(root_x) and os.path.isfile(root_y)):
                raise FileNotFoundError(
                    f"Could not resolve split '{split}' from {path} or {dataset_root}"
                )
            x, y = np.load(root_x), np.load(root_y)

        self.x = torch.from_numpy(np.ascontiguousarray(x)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
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


def data_paths_for_run(cfg, support_seed):
    """Return the fixed-support directory and its dataset root.

    Main few-shot runs use one ``support-seed`` directory across all optimization
    seeds. Fully supervised runs use the complete fixed source partition.
    """
    if cfg.shot == "full":
        return cfg.data_path, cfg.data_path

    dataset_root = os.path.dirname(cfg.data_path)
    run_path = os.path.join(cfg.data_path, f"support-seed-{support_seed}")
    if not os.path.isdir(run_path):
        raise FileNotFoundError(
            f"Missing paired support set: {run_path}. Run data/build_splits.py "
            f"with --shots {cfg.shot.removesuffix('-shot')} "
            f"--support_seeds {support_seed}."
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


def run_once(cfg, seed, support_seed=0, verbose=True):
    """Train one model and evaluate it on the test split.

    ``seed`` controls initialization and optimization. ``support_seed`` independently
    selects the class-balanced support set, which remains fixed across the main three
    runs. For few-shot runs, the best epoch is selected on the complementary source
    validation split. The held-out test split is evaluated once.
    """
    set_seed(seed)
    device = torch.device(cfg.device)

    run_data_path, dataset_root = data_paths_for_run(cfg, support_seed)
    train_set = WindowDataset(run_data_path, "train", dataset_root)
    valid_set = None if cfg.shot == "full" else WindowDataset(
        run_data_path, "valid", dataset_root
    )
    test_set = WindowDataset(run_data_path, "test", dataset_root)

    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    valid_loader = None if valid_set is None else DataLoader(
        valid_set, batch_size=64, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, **loader_kwargs)

    model = MomentHAR(cfg).to(device)
    model.init_prompt(train_loader, device)
    trainable, total = model.count()

    if verbose:
        valid_count = "not used" if valid_set is None else str(len(valid_set))
        print(f"  samples: train {len(train_set)} | valid {valid_count} | test {len(test_set)}")
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

        valid_acc = None
        if valid_loader is not None:
            valid_acc = evaluate(model, valid_loader, device)
            if valid_acc > best_valid:
                best_valid, best_epoch = valid_acc, epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if verbose and (epoch + 1) % 20 == 0:
            status = (
                f"valid {valid_acc:.2f} (best {best_valid:.2f} @ {best_epoch})"
                if valid_acc is not None else "full-source training"
            )
            print(f"  epoch {epoch + 1:3d}/{cfg.num_epochs} "
                  f"loss {epoch_loss / len(train_loader):.4f} {status}")

    if valid_loader is None:
        best_epoch = cfg.num_epochs
        best_valid = None
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test = evaluate(model, test_loader, device, detailed=True)
    minutes = (time.time() - started) / 60

    if verbose:
        selection = (
            f"valid {best_valid:.2f} (epoch {best_epoch})"
            if best_valid is not None else f"final epoch {best_epoch}"
        )
        print(f"  seed {seed}: {selection} | test accuracy {test['accuracy']:.2f} "
              f"macro-F1 {test['macro_f1']:.2f} | {minutes:.1f} min")

    if cfg.save_ckpt:
        os.makedirs(cfg.save_dir, exist_ok=True)
        keys = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save(
            {"cfg": cfg.__dict__, "optimization_seed": seed,
             "support_seed": None if cfg.shot == "full" else support_seed,
             "state_dict": {k: v for k, v in best_state.items() if k in keys}},
            os.path.join(
                cfg.save_dir,
                f"{cfg.dataset}_{cfg.shot}_{cfg.finetune_type}"
                f"_support{support_seed}_opt{seed}.pth",
            ),
        )

    return {
        "seed": seed,
        "optimization_seed": seed,
        "support_seed": None if cfg.shot == "full" else support_seed,
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
        help="optional dataset root containing source/test arrays; useful for folds",
    )
    p.add_argument("--data_root", default=None, help="base directory containing datasets")
    p.add_argument("--model_path", default=None, help="local MOMENT-SMALL directory")
    p.add_argument("--device", default=None, help="PyTorch device, for example cuda or cpu")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--result_dir", default=None)
    p.add_argument("--finetune_type", default="dspt", choices=list(FINETUNE_TYPES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--support_seed", type=int, default=0,
        help="fixed support-set seed shared by all optimization runs",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--m", type=int, default=None, help="task-guidance prompt length")
    p.add_argument("--r", type=int, default=None, help="rank of the sensor-embedding update")
    p.add_argument("--prompt_len", type=int, default=None, help="prompt length l of standard PT")
    p.add_argument("--alpha1", type=float, default=None, help="learning rate of the prompt")
    p.add_argument("--alpha2", type=float, default=None, help="learning rate of the low-rank pair")
    p.add_argument("--no_dual_lr", action="store_true", help="use a single learning rate")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--tag", default="")
    return p


def config_from_args(args):
    cfg = Config(dataset=args.dataset, shot=args.shot, finetune_type=args.finetune_type)
    if args.data_root is not None:
        cfg.data_root = os.path.abspath(args.data_root)
        cfg.data_path = os.path.join(cfg.data_root, args.dataset)
        if args.shot != "full":
            cfg.data_path = os.path.join(cfg.data_path, args.shot)
    if args.data_path is not None:
        root = os.path.abspath(args.data_path)
        cfg.data_path = root if args.shot == "full" else os.path.join(root, args.shot)
    overrides = {"epochs": "num_epochs", "batch_size": "batch_size", "m": "m", "r": "r",
                 "prompt_len": "prompt_len", "alpha1": "alpha1", "alpha2": "alpha2",
                 "model_path": "model_path", "device": "device",
                 "num_workers": "num_workers", "result_dir": "result_dir"}
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
    print(
        f"epochs {cfg.num_epochs} | batch size {cfg.batch_size} | "
        f"optimization seeds {args.seeds} | "
        f"support seed {args.support_seed if cfg.shot != 'full' else 'n/a'}"
    )
    print(f"budget: standard PT (l={cfg.prompt_len}) {n_pt:,} vs "
          f"DSPT (m={cfg.m}, r={cfg.r}) {n_dspt:,} | difference {gap:.1%}")
    print(f"dual learning rates {cfg.dual_lr} | alpha1 {cfg.alpha1} | alpha2 {cfg.alpha2} | "
          f"head {cfg.learning_rate}")
    print("=" * 74)

    results = []
    for seed in args.seeds:
        print(f"\n[optimization seed {seed}]")
        results.append(run_once(cfg, seed, support_seed=args.support_seed))

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
    support_tag = f"_support{args.support_seed}" if cfg.shot != "full" else ""
    suffix = f"_{safe_tag(tag)}" if tag else ""
    path = os.path.join(cfg.result_dir,
                        f"{cfg.dataset}_{cfg.shot}_{cfg.finetune_type}"
                        f"{support_tag}{suffix}.json")
    with open(path, "w") as f:
        json.dump({"config": cfg.__dict__, "runs": results,
                   "acc_mean": float(acc.mean()), "acc_std": acc_std,
                   "f1_mean": float(f1.mean()), "f1_std": f1_std,
                   "protocol": (
                       "fixed support set across optimization runs"
                       if cfg.shot != "full" else "complete source set"
                   ),
                   "support_seed": None if cfg.shot == "full" else args.support_seed,
                   "optimization_seeds": args.seeds,
                   "std_definition": "sample standard deviation (ddof=1)"}, f, indent=2)
    print(f"results written to {path}")


if __name__ == "__main__":
    main()
