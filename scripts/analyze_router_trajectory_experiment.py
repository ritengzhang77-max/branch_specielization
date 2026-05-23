#!/usr/bin/env python3
"""Aggregate training-time router trajectory experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = [
    {
        "condition": "unlabeled_entropy_balance",
        "label": "unlabeled",
        "supervision_end_step": 0,
        "dir": "results/phase3_toy_trajectory_unlabeled_entropy_balance",
    },
    {
        "condition": "anneal_label_end400",
        "label": "end 400",
        "supervision_end_step": 400,
        "dir": "results/phase3_toy_trajectory_end400",
    },
    {
        "condition": "anneal_label_end800",
        "label": "end 800",
        "supervision_end_step": 800,
        "dir": "results/phase3_toy_trajectory_end800",
    },
    {
        "condition": "anneal_label_end1200",
        "label": "end 1200",
        "supervision_end_step": 1200,
        "dir": "results/phase3_toy_trajectory_end1200",
    },
    {
        "condition": "always_label_0.05",
        "label": "always",
        "supervision_end_step": 1600,
        "dir": "results/phase3_toy_trajectory_always_label",
    },
]


SUMMARY_FIELDS = [
    "train_loss",
    "eval_loss",
    "local_loss",
    "induction_loss",
    "local_accuracy",
    "induction_accuracy",
    "same_top_branch",
    "routed_role_match",
    "branch_distribution_distance",
    "gate_routed_role_match",
    "gate_distribution_distance",
    "local_gate_entropy_mean",
    "induction_gate_entropy_mean",
    "global_gate_entropy_mean",
    "global_gate_branch0_mean",
    "global_gate_branch1_mean",
    "global_gate_balance_error_mean",
    "gate_target_nll_mean",
    "local_gate_branch0_mean",
    "local_gate_branch1_mean",
    "induction_gate_branch0_mean",
    "induction_gate_branch1_mean",
    "local_branch0_loss_delta",
    "local_branch1_loss_delta",
    "induction_branch0_loss_delta",
    "induction_branch1_loss_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_router_trajectory_analysis"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float:
    if value == "":
        return float("nan")
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def mean_field(rows: list[dict[str, object]], field: str) -> float:
    values = [parse_float(str(row[field])) for row in rows if field in row]
    values = [value for value in values if not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def std_field(rows: list[dict[str, object]], field: str) -> float:
    values = [parse_float(str(row[field])) for row in rows if field in row]
    values = [value for value in values if not np.isnan(value)]
    return float(np.std(values, ddof=0)) if values else float("nan")


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in CONDITIONS:
        path = Path(str(spec["dir"])) / "trajectory_summary.csv"
        if not path.exists():
            print(f"missing {path}; skipping", flush=True)
            continue
        for row in read_rows(path):
            enriched: dict[str, object] = dict(row)
            enriched["condition"] = spec["condition"]
            enriched["label"] = spec["label"]
            enriched["supervision_end_step"] = spec["supervision_end_step"]
            rows.append(enriched)
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["condition"]), int(str(row["step"])))].append(row)

    summaries: list[dict[str, object]] = []
    for (condition, step), group in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1])):
        first = group[0]
        summary: dict[str, object] = {
            "condition": condition,
            "label": first["label"],
            "supervision_end_step": first["supervision_end_step"],
            "step": step,
            "n_models": len(group),
            "local_top_branch_counts_json": json.dumps(
                count_values(row["local_top_branch"] for row in group)
            ),
            "induction_top_branch_counts_json": json.dumps(
                count_values(row["induction_top_branch"] for row in group)
            ),
        }
        for field in SUMMARY_FIELDS:
            output_base = field[:-5] if field.endswith("_mean") else field
            summary[f"{output_base}_mean"] = mean_field(group, field)
            summary[f"{output_base}_std"] = std_field(group, field)
        summaries.append(summary)
    return summaries


def count_values(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plot(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        by_condition[str(row["condition"])].append(row)

    metrics = [
        ("routed_role_match_mean", "causal routed-role match"),
        ("branch_distribution_distance_mean", "causal branch distance"),
        ("gate_distribution_distance_mean", "gate role distance"),
        ("gate_target_nll_mean", "weak-label gate NLL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True)
    axes_flat = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color_idx, spec in enumerate(CONDITIONS):
        condition = str(spec["condition"])
        rows = sorted(by_condition.get(condition, []), key=lambda row: int(row["step"]))
        if not rows:
            continue
        color = colors[color_idx % len(colors)]
        steps = [int(row["step"]) for row in rows]
        label = str(spec["label"])
        for ax, (field, title) in zip(axes_flat, metrics):
            ax.plot(
                steps,
                [float(row[field]) for row in rows],
                marker="o",
                linewidth=1.8,
                markersize=3,
                label=label,
                color=color,
            )
            end_step = int(spec["supervision_end_step"])
            if 0 < end_step < max(steps):
                ax.axvline(end_step, linestyle=":", linewidth=1.1, color=color, alpha=0.5)
            ax.set_title(title)
            ax.grid(alpha=0.25)

    for ax in axes_flat[:3]:
        ax.set_ylim(-0.03, 1.05)
    for ax in axes_flat:
        ax.set_xlabel("optimizer updates completed")
    axes_flat[0].set_ylabel("mean over seeds")
    axes_flat[2].set_ylabel("mean over seeds")
    axes_flat[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "router_trajectory_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    summaries = summarize_rows(rows)
    write_csv(args.output_dir / "trajectory_rows.csv", rows)
    write_csv(args.output_dir / "trajectory_by_step.csv", summaries)
    maybe_write_plot(args.output_dir, summaries)
    print(f"read {len(rows)} trajectory rows", flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
