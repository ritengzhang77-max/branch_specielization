#!/usr/bin/env python3
"""Summarize explicit branch-isolation toy results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-summary",
        type=Path,
        default=Path("results/phase3_toy_branch_isolation_lw025/config_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_branch_isolation_analysis"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plot(output_dir: Path, rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    labels = [row["config"] for row in rows]
    x = np.arange(len(labels))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].bar(x - width, [float(row["same_top_branch_rate"]) for row in rows], width, label="same top branch")
    axes[0].bar(x, [float(row["routed_role_match_rate"]) for row in rows], width, label="local B0, induction B1")
    axes[0].bar(
        x + width,
        [float(row["branch_distribution_distance_mean"]) for row in rows],
        width,
        label="role-distance",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("rate / distance")
    axes[0].set_title("Branch-level modularity metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    branch0_local = [float(row["local_branch0_loss_delta_mean"]) for row in rows]
    branch1_local = [float(row["local_branch1_loss_delta_mean"]) for row in rows]
    branch0_induction = [float(row["induction_branch0_loss_delta_mean"]) for row in rows]
    branch1_induction = [float(row["induction_branch1_loss_delta_mean"]) for row in rows]
    role_labels = []
    values = []
    colors = []
    for idx, label in enumerate(labels):
        role_labels.extend([f"{label}\nlocal B0", "local B1", "ind B0", "ind B1"])
        values.extend([branch0_local[idx], branch1_local[idx], branch0_induction[idx], branch1_induction[idx]])
        colors.extend(["#4c78a8", "#4c78a8", "#f58518", "#f58518"])
    axes[1].bar(range(len(values)), values, color=colors)
    axes[1].set_xticks(range(len(values)))
    axes[1].set_xticklabels(role_labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("mean loss delta")
    axes[1].set_title("Branch ablation effect by role")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "branch_isolation_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.config_summary)
    write_csv(args.output_dir / "branch_isolation_summary.csv", rows)
    maybe_write_plot(args.output_dir, rows)
    for row in rows:
        print(row)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
