#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


ORDER = ["uniform4", "hetero4", "uniform2", "hetero4_64first"]
LABELS = ["uniform4", "hetero4", "uniform2", "64-first"]
COLORS = ["#64748b", "#2563eb", "#94a3b8", "#0f766e"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def combined_config_rows() -> dict[str, dict[str, str]]:
    rows = read_rows(DATA / "main_config_summary.csv")
    rows += read_rows(DATA / "position_config_summary.csv")
    return {row["config"]: row for row in rows}


def combined_head_rows() -> list[dict[str, str]]:
    return read_rows(DATA / "main_head_role_scores.csv") + read_rows(DATA / "position_head_role_scores.csv")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    configs = combined_config_rows()

    top_spec = [float(configs[name]["top_specialization_mean"]) for name in ORDER]
    loss_delta = [float(configs[name]["own_top_loss_delta_mean"]) for name in ORDER]
    random_delta = [float(configs[name]["random_same_layer_loss_delta_mean"]) for name in ORDER]

    plt.figure(figsize=(8.4, 4.6))
    x = range(len(ORDER))
    plt.bar(x, top_spec, color=COLORS)
    plt.xticks(list(x), LABELS, rotation=15, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Top specialization")
    plt.title("Heterogeneous head dimensions concentrate causal specialization")
    plt.tight_layout()
    plt.savefig(FIGURES / "top_specialization.pdf")
    plt.close()

    plt.figure(figsize=(8.4, 4.6))
    width = 0.38
    plt.bar([i - width / 2 for i in x], loss_delta, width=width, color=COLORS, label="top head")
    plt.bar([i + width / 2 for i in x], random_delta, width=width, color="#cbd5e1", label="random same layer")
    plt.xticks(list(x), LABELS, rotation=15, ha="right")
    plt.ylabel("Ablation loss delta")
    plt.title("The heterogeneous 64-dim slot is causally load-bearing")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "ablation_delta.pdf")
    plt.close()

    raw_role = [float(configs[name]["raw_role_similarity_mean"]) for name in ORDER]
    random_role = [float(configs[name]["random_role_similarity_mean"]) for name in ORDER]
    top_match = [float(configs[name]["raw_top_head_match_rate"]) for name in ORDER]

    plt.figure(figsize=(8.4, 4.6))
    plt.plot(LABELS, raw_role, marker="o", color="#2563eb", label="same slot")
    plt.plot(LABELS, random_role, marker="o", color="#94a3b8", label="random")
    plt.plot(LABELS, top_match, marker="o", color="#c2410c", label="top-head match")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Cross-seed stability")
    plt.title("Heterogeneous slots are stable across seeds")
    plt.xticks(rotation=15, ha="right")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "stability_metrics.pdf")
    plt.close()

    head_rows = combined_head_rows()
    heatmap_configs = ["hetero4", "hetero4_64first", "uniform4"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
    for ax, config_name in zip(axes, heatmap_configs):
        cfg_rows = [row for row in head_rows if row["config"] == config_name]
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
        ax.set_title(config_name)
        ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
        ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
        ax.set_xlabel("Seed")
    axes[0].set_ylabel("Layer")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="top head index", shrink=0.8)
    fig.suptitle("The top causal slot follows the 64-dim head")
    fig.savefig(FIGURES / "top_head_slots.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
