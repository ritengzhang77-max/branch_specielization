#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def read_layer_summary(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def series(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows]


def main() -> None:
    raw = read_layer_summary(DATA / "raw_score_layer_summary.csv")
    prob = read_layer_summary(DATA / "probability_layer_summary.csv")
    layers = [int(row["layer"]) for row in raw]

    plt.figure(figsize=(8.4, 4.6))
    plt.plot(layers, series(raw, "raw_diag_mean"), marker="o", label="same head index")
    plt.plot(layers, series(raw, "matched_mean"), marker="o", label="Hungarian matched")
    plt.plot(layers, series(raw, "random_perm_mean"), marker="o", label="random permutation")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.title("Pythia-160M raw attention-score similarity across 9 seeds")
    plt.ylim(-0.05, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "raw_score_layer_profile.pdf")
    plt.close()

    plt.figure(figsize=(8.4, 4.6))
    plt.plot(layers, series(raw, "matched_minus_random_mean"), marker="o", label="raw pre-softmax scores")
    plt.plot(layers, series(prob, "matched_minus_random_mean"), marker="o", label="attention probabilities")
    plt.xlabel("Layer")
    plt.ylabel("Matched minus random similarity")
    plt.title("Raw scores expose stronger relabeled cross-seed structure")
    plt.ylim(0.0, 0.58)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "raw_vs_probability_gap.pdf")
    plt.close()


if __name__ == "__main__":
    main()
