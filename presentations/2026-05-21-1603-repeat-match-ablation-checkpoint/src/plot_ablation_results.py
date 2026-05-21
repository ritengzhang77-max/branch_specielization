#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    summary_rows = read_rows(DATA / "condition_summary.csv")
    result_rows = read_rows(DATA / "ablation_results.csv")

    order = ["own_random", "source_same_index", "source_aligned", "own_top"]
    labels = ["own random", "same index", "aligned", "own top"]
    summary = {row["condition"]: row for row in summary_rows}

    means = [float(summary[name]["loss_delta_mean"]) for name in order]
    stds = [float(summary[name]["loss_delta_std"]) for name in order]
    colors = ["#9aa0a6", "#64748b", "#2563eb", "#c2410c"]

    plt.figure(figsize=(8.4, 4.6))
    plt.bar(labels, means, yerr=stds, color=colors, capsize=4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Loss delta after ablation")
    plt.title("Top repeat-match heads are causally important")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "condition_loss_delta.pdf")
    plt.close()

    rows_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in result_rows:
        if row["condition"] in {"source_same_index", "source_aligned"}:
            key = (row["target_seed"], row["source_seed"], row["condition"])
            rows_by_key[key] = row

    diffs = []
    for target in sorted({row["target_seed"] for row in result_rows}, key=int):
        for source in sorted({row["source_seed"] for row in result_rows if row["source_seed"]}, key=int):
            if source == target:
                continue
            aligned = rows_by_key[(target, source, "source_aligned")]
            same = rows_by_key[(target, source, "source_same_index")]
            diffs.append(float(aligned["loss_delta"]) - float(same["loss_delta"]))

    plt.figure(figsize=(8.4, 4.6))
    plt.hist(diffs, bins=16, color="#2563eb", edgecolor="white")
    plt.axvline(0.0, color="black", linewidth=0.9)
    plt.axvline(sum(diffs) / len(diffs), color="#c2410c", linewidth=1.4, label="mean")
    plt.xlabel("Aligned minus same-index loss delta")
    plt.ylabel("Target/source pairs")
    plt.title("Raw-score alignment transfers the causal role")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "paired_transfer_difference.pdf")
    plt.close()

    random_by_target: dict[str, list[float]] = defaultdict(list)
    own_by_target: dict[str, float] = {}
    for row in result_rows:
        if row["condition"] == "own_random":
            random_by_target[row["target_seed"]].append(float(row["loss_delta"]))
        elif row["condition"] == "own_top":
            own_by_target[row["target_seed"]] = float(row["loss_delta"])

    seeds = sorted(own_by_target, key=int)
    own_values = [own_by_target[seed] for seed in seeds]
    random_means = [sum(random_by_target[seed]) / len(random_by_target[seed]) for seed in seeds]

    x = range(len(seeds))
    plt.figure(figsize=(8.4, 4.6))
    plt.plot(x, own_values, marker="o", color="#c2410c", label="own top")
    plt.plot(x, random_means, marker="o", color="#9aa0a6", label="random mean")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(list(x), seeds)
    plt.xlabel("Target seed")
    plt.ylabel("Loss delta after ablation")
    plt.title("Own top heads beat random controls in every seed")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "own_top_vs_random_by_seed.pdf")
    plt.close()


if __name__ == "__main__":
    main()
