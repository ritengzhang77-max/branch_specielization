#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict[str, str]], role: str) -> list[dict[str, str]]:
    return [row for row in rows if row["role"] == role]


def floats(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def main() -> None:
    consistency = read_rows(DATA / "role_layer_consistency_summary.csv")
    specialization = read_rows(DATA / "role_layer_specialization_summary.csv")

    repeat_consistency = select(consistency, "repeat_match")
    repeat_specialization = select(specialization, "repeat_match")
    layers = [int(row["layer"]) for row in repeat_consistency]

    plt.figure(figsize=(8.4, 4.6))
    plt.plot(layers, floats(repeat_consistency, "raw_distribution_similarity_mean"), marker="o", label="same index")
    plt.plot(
        layers,
        floats(repeat_consistency, "aligned_distribution_similarity_mean"),
        marker="o",
        label="raw-score aligned",
    )
    plt.plot(
        layers,
        floats(repeat_consistency, "random_distribution_similarity_mean"),
        marker="o",
        label="random",
    )
    plt.xlabel("Layer")
    plt.ylabel("Distribution similarity")
    plt.title("Repeat-match role consistency across Pythia-160M seeds")
    plt.ylim(0.0, 0.95)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "repeat_match_consistency.pdf")
    plt.close()

    plt.figure(figsize=(8.4, 4.6))
    for role in ["repeat_match", "previous_token", "bos"]:
        role_rows = select(specialization, role)
        plt.plot(
            [int(row["layer"]) for row in role_rows],
            floats(role_rows, "max_specialization_mean"),
            marker="o",
            label=role.replace("_", " "),
        )
    plt.xlabel("Layer")
    plt.ylabel("Mean max specialization S(h,t)")
    plt.title("Role concentration by layer")
    plt.ylim(0.0, 0.9)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "role_concentration.pdf")
    plt.close()

    plt.figure(figsize=(8.4, 4.6))
    plt.plot(layers, floats(repeat_consistency, "raw_top_head_match_rate"), marker="o", label="same index")
    plt.plot(
        layers,
        floats(repeat_consistency, "aligned_top_head_match_rate"),
        marker="o",
        label="raw-score aligned",
    )
    plt.xlabel("Layer")
    plt.ylabel("Top-head match rate")
    plt.title("Repeat-match top head is stable mainly after alignment")
    plt.ylim(0.0, 0.85)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "repeat_match_top_head_rate.pdf")
    plt.close()


if __name__ == "__main__":
    main()
