"""Plot ablation means with sample-standard-deviation bands.

The script consumes JSON files written by ``ablation.py``. It never synthesizes
uncertainty: the shaded region is read directly from each entry's ``acc_std``.
"""
import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np


def numeric_value(label, key):
    match = re.search(rf"(?:^|[, ]+){re.escape(key)}=(\d+)", label)
    if not match:
        raise ValueError(f"could not parse {key}=... from label {label!r}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("output")
    parser.add_argument("--x", choices=("l", "m"), required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    with open(args.result_json, encoding="utf-8") as stream:
        rows = json.load(stream)

    points = []
    for row in rows:
        try:
            x = numeric_value(row["label"], args.x)
        except ValueError:
            continue
        points.append((x, float(row["acc_mean"]), float(row["acc_std"])))
    if not points:
        raise ValueError(f"no {args.x}=... settings found in {args.result_json}")

    points.sort()
    x, mean, std = (np.asarray(v, dtype=float) for v in zip(*points))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(x, mean, color="#1769aa", marker="o", linewidth=2, label="Mean accuracy")
    ax.fill_between(x, mean - std, mean + std, color="#1769aa", alpha=0.18,
                    label="±1 sample SD")
    ax.set_xlabel("Standard prompt length $l$" if args.x == "l" else "Task-guidance length $m$")
    ax.set_ylabel("Accuracy (%)")
    if args.title:
        ax.set_title(args.title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
