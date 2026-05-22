#!/usr/bin/env python3
"""Aggregate bottlenecked branch-router experiments.

The experiment asks whether shrinking each branch's attention capacity makes
unlabeled routing pressure align with functional roles.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = [
    {
        "condition": "bottleneck16_unconstrained",
        "config": "learned_token_router",
        "branch_head_dim": 16,
        "supervision_weight": 0.0,
        "entropy_weight": 0.0,
        "balance_weight": 0.0,
        "dirs": [
            f"results/phase3_toy_bottleneck16_unlabeled_unconstrained_seed{seed}"
            for seed in range(1, 6)
        ],
    },
    {
        "condition": "bottleneck16_balance_only_1.0",
        "config": "learned_token_router",
        "branch_head_dim": 16,
        "supervision_weight": 0.0,
        "entropy_weight": 0.0,
        "balance_weight": 1.0,
        "dirs": [
            f"results/phase3_toy_bottleneck16_unlabeled_balance1_seed{seed}"
            for seed in range(1, 6)
        ],
    },
    {
        "condition": "bottleneck16_entropy_0.05_balance_1.0",
        "config": "learned_token_router",
        "branch_head_dim": 16,
        "supervision_weight": 0.0,
        "entropy_weight": 0.05,
        "balance_weight": 1.0,
        "dirs": [
            f"results/phase3_toy_bottleneck16_unlabeled_entropy005_balance1_seed{seed}"
            for seed in range(1, 6)
        ],
    },
    {
        "condition": "bottleneck16_weak_label_0.05",
        "config": "weak_token_router",
        "branch_head_dim": 16,
        "supervision_weight": 0.05,
        "entropy_weight": 0.0,
        "balance_weight": 0.0,
        "dirs": [f"results/phase3_toy_bottleneck16_weak_label005_seed{seed}" for seed in range(1, 6)],
    },
    {
        "condition": "bottleneck16_oracle_route",
        "config": "oracle_route",
        "branch_head_dim": 16,
        "supervision_weight": float("nan"),
        "entropy_weight": float("nan"),
        "balance_weight": float("nan"),
        "dirs": [f"results/phase3_toy_bottleneck16_oracle_route_seed{seed}" for seed in range(1, 6)],
    },
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


WIDE_REF_MAP = {
    "unconstrained_learned_token": "wide64_unconstrained",
    "entropy_0.05_balance_1.0": "wide64_entropy_0.05_balance_1.0",
    "weak_label_0.05": "wide64_weak_label_0.05",
    "oracle_route": "wide64_oracle_route",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-summary",
        type=Path,
        default=Path("results/phase3_toy_unlabeled_router_regularization_analysis/unlabeled_router_regularization_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_bottleneck16_router_analysis"),
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
        "source": "bottleneck16",
        "config": condition["config"],
        "branch_head_dim": condition["branch_head_dim"],
        "supervision_weight": condition["supervision_weight"],
        "entropy_weight": condition["entropy_weight"],
        "balance_weight": condition["balance_weight"],
        "n_models": len(rows),
        "local_top_branch_counts_json": json.dumps(dict(sorted(local_top.items()))),
        "induction_top_branch_counts_json": json.dumps(dict(sorted(induction_top.items()))),
    }
    for field in SUMMARY_FIELDS:
        output_key = field if field.endswith("_mean") else f"{field}_mean"
        output[output_key] = mean_field(rows, field)
    return output


def read_wide_refs(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = read_rows(path)
    refs = []
    for row in rows:
        source_condition = row["condition"]
        if source_condition not in WIDE_REF_MAP:
            continue
        refs.append(
            {
                "condition": WIDE_REF_MAP[source_condition],
                "source": "wide64_reference",
                "config": source_condition,
                "branch_head_dim": 64,
                "supervision_weight": float("nan"),
                "entropy_weight": row.get("entropy_weight", float("nan")),
                "balance_weight": row.get("balance_weight", float("nan")),
                "n_models": int(row["n_models"]),
                "local_top_branch_counts_json": row.get("local_top_branch_counts_json", ""),
                "induction_top_branch_counts_json": row.get("induction_top_branch_counts_json", ""),
                **{
                    field: row[field]
                    for field in row
                    if field.endswith("_mean") and field not in {"condition"}
                },
            }
        )
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
        for row in rows:
            writer.writerow(row)


def maybe_write_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    selected_names = [
        "wide64_entropy_0.05_balance_1.0",
        "bottleneck16_unconstrained",
        "bottleneck16_balance_only_1.0",
        "bottleneck16_entropy_0.05_balance_1.0",
        "bottleneck16_weak_label_0.05",
        "bottleneck16_oracle_route",
    ]
    by_name = {str(row["condition"]): row for row in rows}
    plot_rows = [by_name[name] for name in selected_names if name in by_name]
    labels = [
        "wide ent+bal",
        "b16 unconstr.",
        "b16 balance",
        "b16 ent+bal",
        "b16 weak",
        "b16 oracle",
    ][: len(plot_rows)]
    x = np.arange(len(plot_rows))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
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
        [float(row["global_gate_balance_error_mean"]) for row in plot_rows],
        width,
        label="global balance error",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Router behavior")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "bottleneck16_router_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bottleneck_rows = [summarize_condition(condition) for condition in CONDITIONS]
    wide_refs = read_wide_refs(args.wide_summary)
    all_rows = wide_refs + bottleneck_rows
    write_csv(args.output_dir / "bottleneck16_summary.csv", bottleneck_rows)
    write_csv(args.output_dir / "bottleneck16_with_wide_refs.csv", all_rows)
    maybe_write_plot(args.output_dir, all_rows)
    for row in bottleneck_rows:
        print(row)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
