"""t-SNE visualisation of the features produced by the last encoder layer.

Requires checkpoints saved with ``train.py --save_ckpt``.

Example
-------
    python tsne.py --dataset MotionSense --shot 5-shot \
        --finetune_type ft_head lora dspt --seed 0
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, DATASETS, FINETUNE_TYPES
from train import WindowDataset, data_paths_for_run


def extract_features(cfg, ckpt_path, device, support_seed):
    from model import MomentHAR

    model = MomentHAR(cfg).to(device).eval()
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"  loaded {len(state)} tensors from {os.path.basename(ckpt_path)}")
    else:
        raise FileNotFoundError(
            f"{ckpt_path} not found; run train.py with --save_ckpt first")

    run_path, dataset_root = data_paths_for_run(cfg, support_seed)
    loader = DataLoader(
        WindowDataset(run_path, "test", dataset_root),
        batch_size=64,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    features, labels = [], []
    for x, y in loader:
        features.append(model.get_features(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MotionSense", choices=list(DATASETS))
    p.add_argument("--shot", default="5-shot")
    p.add_argument("--finetune_type", nargs="+", default=["ft_head", "lora", "dspt"],
                   choices=list(FINETUNE_TYPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--support_seed", type=int, default=0)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--model_path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--out", default="results/tsne.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n = len(args.finetune_type)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6))
    axes = np.atleast_1d(axes)

    for ax, finetune_type in zip(axes, args.finetune_type):
        print(f"[{finetune_type}]")
        cfg = Config(dataset=args.dataset, shot=args.shot, finetune_type=finetune_type)
        if args.model_path is not None:
            cfg.model_path = args.model_path
        ckpt = os.path.join(
            args.ckpt_dir,
            f"{args.dataset}_{args.shot}_{finetune_type}"
            f"_support{args.support_seed}_opt{args.seed}.pth",
        )
        features, labels = extract_features(cfg, ckpt, device, args.support_seed)
        embedded = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                        random_state=args.seed).fit_transform(features)
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(embedded[mask, 0], embedded[mask, 1], s=6, alpha=0.7,
                       label=f"class {label}")
        ax.set_title(finetune_type)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(markerscale=2, fontsize=8, loc="best")
    fig.suptitle(f"{args.dataset} {args.shot}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()
