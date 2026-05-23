#!/usr/bin/env python3
"""Aggregate annealed weak-router supervision experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ANNEALED_CONDITIONS = [
    {
        "condition": f"anneal_label_end{end_step}",
        "config": "weak_token_router",
        "task_variant": "bidirectional_lookup",
        "branch_head_dim": 64,
        "supervision_weight": 0.05,
        "supervision_end_step": end_step,
        "dirs": [
            f"results/phase3_toy_conflict_wide64_anneal_label_end{end_step}_seed{seed}"
            for seed in range(1, 6)
        ],
    }
    for end_step in [50, 100, 200, 400, 800, 1200]
]


SUMMARY_FIELDS = [
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
    "local_gate_branch0_mean",
    "local_gate_branch1_mean",
    "induction_gate_branch0_mean",
    "induction_gate_branch1_mean",
    "local_branch0_loss_delta",
    "local_branch1_loss_delta",
    "induction_branch0_loss_delta",
    "induction_branch1_loss_delta",
]


REFERENCE_CONDITIONS = {
    "conflict_wide64_entropy_0.05_balance_1.0": {
        "condition": "unlabeled_entropy_0.05_balance_1.0",
        "supervision_end_step": 0,
    },
    "conflict_wide64_weak_label_0.05": {
        "condition": "always_label_0.05",
        "supervision_end_step": 1600,
    },
    "conflict_wide64_oracle_route": {
        "condition": "oracle_route",
        "supervision_end_step": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conflict-summary",
        type=Path,
        default=Path("results/phase3_toy_conflict_router_analysis/conflict_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_annealed_router_analysis"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_condition_rows(condition: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dir_text in condition["dirs"]:
        path = Path(str(dir_text)) / "model_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_rows(path))
    return rows


def maybe_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    if value == "":
        return float("nan")
    return float(value)


def mean_field(rows: list[dict[str, str]], field: str) -> float:
    values = [maybe_float(row, field) for row in rows if field in row]
    values = [value for value in values if not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def summarize_condition(condition: dict[str, object]) -> dict[str, object]:
    rows = read_condition_rows(condition)
    local_top = Counter(row["local_top_branch"] for row in rows)
    induction_top = Counter(row["induction_top_branch"] for row in rows)
    output: dict[str, object] = {
        "condition": condition["condition"],
        "source": "annealed_conflict_wide64",
        "config": condition["config"],
        "task_variant": condition["task_variant"],
        "branch_head_dim": condition["branch_head_dim"],
        "supervision_weight": condition["supervision_weight"],
        "entropy_weight": 0.0,
        "balance_weight": 0.0,
        "supervision_end_step": condition["supervision_end_step"],
        "supervision_fraction": float(condition["supervision_end_step"]) / 1600.0,
        "n_models": len(rows),
        "local_top_branch_counts_json": json.dumps(dict(sorted(local_top.items()))),
        "induction_top_branch_counts_json": json.dumps(dict(sorted(induction_top.items()))),
    }
    for field in SUMMARY_FIELDS:
        output_key = field if field.endswith("_mean") else f"{field}_mean"
        output[output_key] = mean_field(rows, field)
    return output


def read_references(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    refs = []
    for row in read_rows(path):
        if row["condition"] not in REFERENCE_CONDITIONS:
            continue
        spec = REFERENCE_CONDITIONS[row["condition"]]
        ref = dict(row)
        ref["condition"] = spec["condition"]
        ref["source"] = "conflict_reference"
        ref["supervision_end_step"] = spec["supervision_end_step"]
        if isinstance(spec["supervision_end_step"], int):
            ref["supervision_fraction"] = float(spec["supervision_end_step"]) / 1600.0
        else:
            ref["supervision_fraction"] = ""
        refs.append(ref)
    return refs


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


def maybe_write_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    selected_names = [
        "unlabeled_entropy_0.05_balance_1.0",
        "anneal_label_end50",
        "anneal_label_end100",
        "anneal_label_end200",
        "anneal_label_end400",
        "anneal_label_end800",
        "anneal_label_end1200",
        "always_label_0.05",
        "oracle_route",
    ]
    by_name = {str(row["condition"]): row for row in rows}
    plot_rows = [by_name[name] for name in selected_names if name in by_name]
    labels = [
        "unlabeled",
        "end 50",
        "end 100",
        "end 200",
        "end 400",
        "end 800",
        "end 1200",
        "always",
        "oracle",
    ][: len(plot_rows)]
    x = np.arange(len(plot_rows))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    axes[0].bar(
        x - width,
        [float(row["same_top_branch_mean"]) for row in plot_rows],
        width,
        label="same top branch",
    )
    axes[0].bar(
        x,
        [float(row["routed_role_match_mean"]) for row in plot_rows],
        width,
        label="causal routed match",
    )
    axes[0].bar(
        x + width,
        [float(row["branch_distribution_distance_mean"]) for row in plot_rows],
        width,
        label="causal branch distance",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("rate / distance")
    axes[0].set_title("Causal branch modularity")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(
        x - width,
        [float(row["gate_routed_role_match_mean"]) for row in plot_rows],
        width,
        label="gate routed match",
    )
    axes[1].bar(
        x,
        [float(row["gate_distribution_distance_mean"]) for row in plot_rows],
        width,
        label="gate role distance",
    )
    axes[1].bar(
        x + width,
        [float(row["global_gate_entropy_mean"]) for row in plot_rows],
        width,
        label="global gate entropy",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Router behavior")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "annealed_router_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annealed_rows = [summarize_condition(condition) for condition in ANNEALED_CONDITIONS]
    reference_rows = read_references(args.conflict_summary)
    rows = reference_rows[:1] + annealed_rows + reference_rows[1:]
    write_csv(args.output_dir / "annealed_summary.csv", annealed_rows)
    write_csv(args.output_dir / "annealed_with_references.csv", rows)
    maybe_write_plot(args.output_dir, rows)
    for row in rows:
        print(row)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
