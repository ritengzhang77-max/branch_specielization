#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

ORDER = ["uniform4", "hetero4", "uniform2", "hetero4_64first"]
LABELS = ["uniform4", "hetero4", "uniform2", "64-first"]
COLORS = ["#64748b", "#2563eb", "#94a3b8", "#0f766e"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    config_rows = {row["config"]: row for row in read_rows(DATA / "config_summary.csv")}
    model_rows = read_rows(DATA / "model_summary.csv")
    head_rows = read_rows(DATA / "head_role_scores.csv")

    x = np.arange(len(ORDER))
    width = 0.36

    plt.figure(figsize=(8.4, 4.6))
    local_spec = [float(config_rows[name]["local_top_specialization_mean"]) for name in ORDER]
    induction_spec = [float(config_rows[name]["induction_top_specialization_mean"]) for name in ORDER]
    plt.bar(x - width / 2, local_spec, width=width, color="#2563eb", label="local")
    plt.bar(x + width / 2, induction_spec, width=width, color="#c2410c", label="induction")
    plt.xticks(x, LABELS, rotation=15, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Top specialization")
    plt.title("Heterogeneity increases role concentration, but not clean partition")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "role_specialization.pdf")
    plt.close()

    plt.figure(figsize=(8.4, 4.6))
    local_delta = [float(config_rows[name]["local_top_loss_delta_mean"]) for name in ORDER]
    induction_delta = [float(config_rows[name]["induction_top_loss_delta_mean"]) for name in ORDER]
    plt.bar(x - width / 2, local_delta, width=width, color="#2563eb", label="local")
    plt.bar(x + width / 2, induction_delta, width=width, color="#c2410c", label="induction")
    plt.xticks(x, LABELS, rotation=15, ha="right")
    plt.ylabel("Top-head ablation loss delta")
    plt.title("Induction remains the larger causal load")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "role_ablation_delta.pdf")
    plt.close()

    plt.figure(figsize=(8.4, 4.6))
    same_slot = [float(config_rows[name]["same_top_slot_rate"]) for name in ORDER]
    same_dim = [float(config_rows[name]["same_top_head_dim_rate"]) for name in ORDER]
    plt.bar(x - width / 2, same_slot, width=width, color="#64748b", label="same exact slot")
    plt.bar(x + width / 2, same_dim, width=width, color="#0f766e", label="same head dim")
    plt.xticks(x, LABELS, rotation=15, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Rate over seeds")
    plt.title("Local and induction roles often do not separate cleanly")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "role_overlap.pdf")
    plt.close()

    dim_values = [16, 32, 64]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, role in zip(axes, ["local", "induction"]):
        bottoms = np.zeros(len(ORDER))
        for dim in dim_values:
            values = []
            for config in ORDER:
                counts = json.loads(config_rows[config][f"{role}_top_head_dim_counts_json"])
                values.append(float(counts.get(str(dim), 0)))
            ax.bar(x, values, bottom=bottoms, label=str(dim))
            bottoms += np.asarray(values)
        ax.set_title(f"{role} top dims")
        ax.set_xticks(x, LABELS, rotation=15, ha="right")
        ax.set_ylabel("Seeds")
    axes[1].legend(title="head dim", frameon=False)
    fig.suptitle("Top role dimensions across seeds")
    fig.tight_layout()
    fig.savefig(FIGURES / "top_dim_counts.pdf")
    plt.close(fig)

    heatmap_configs = ["hetero4", "hetero4_64first"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2), sharex=True, sharey=True)
    for col, config in enumerate(heatmap_configs):
        for row_idx, role in enumerate(["local", "induction"]):
            ax = axes[row_idx, col]
            cfg_rows = [row for row in head_rows if row["config"] == config and row["role"] == role]
            seeds = sorted({int(row["seed"]) for row in cfg_rows})
            layers = sorted({int(row["layer"]) for row in cfg_rows})
            values = []
            for layer in layers:
                row_values = []
                for seed in seeds:
                    candidates = [
                        row for row in cfg_rows if int(row["seed"]) == seed and int(row["layer"]) == layer
                    ]
                    top = max(candidates, key=lambda row: float(row["role_score"]))
                    row_values.append(int(top["head"]))
                values.append(row_values)
            image = ax.imshow(values, aspect="auto", vmin=0, vmax=3, cmap="viridis")
            ax.set_title(f"{config} {role}")
            ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
            ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="top head index", shrink=0.8)
    fig.suptitle("Competition makes role assignment layout-sensitive")
    fig.savefig(FIGURES / "hetero_top_slots.pdf", bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(8.4, 4.6))
    for role, color in [("local", "#2563eb"), ("induction", "#c2410c")]:
        raw = [float(config_rows[name][f"{role}_raw_role_similarity_mean"]) for name in ORDER]
        random = [float(config_rows[name][f"{role}_random_role_similarity_mean"]) for name in ORDER]
        plt.plot(LABELS, raw, marker="o", color=color, label=f"{role} same slot")
        plt.plot(LABELS, random, marker="x", color=color, linestyle="--", label=f"{role} random")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Role distribution similarity")
    plt.title("Heterogeneous same-slot stability is above random, but layout-sensitive")
    plt.xticks(rotation=15, ha="right")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "role_stability.pdf")
    plt.close()


if __name__ == "__main__":
    main()
