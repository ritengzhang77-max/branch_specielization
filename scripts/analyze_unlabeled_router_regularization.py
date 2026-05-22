#!/usr/bin/env python3
"""Combine unlabeled token-router regularization runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = [
    {
        "condition": "entropy_only_0.05",
        "entropy_weight": 0.05,
        "balance_weight": 0.0,
        "dirs": [
            "results/phase3_toy_unlabeled_token_router_entropy005_lw025_seed1",
            "results/phase3_toy_unlabeled_token_router_entropy005_lw025_seed2",
            "results/phase3_toy_unlabeled_token_router_entropy005_lw025_seed3",
            "results/phase3_toy_unlabeled_token_router_entropy005_lw025_seed4",
            "results/phase3_toy_unlabeled_token_router_entropy005_lw025_seed5",
        ],
    },
    {
        "condition": "balance_only_1.0",
        "entropy_weight": 0.0,
        "balance_weight": 1.0,
        "dirs": [
            "results/phase3_toy_unlabeled_token_router_balance1_lw025_seed1",
            "results/phase3_toy_unlabeled_token_router_balance1_lw025_seed2",
            "results/phase3_toy_unlabeled_token_router_balance1_lw025_seed3",
            "results/phase3_toy_unlabeled_token_router_balance1_lw025_seed4",
            "results/phase3_toy_unlabeled_token_router_balance1_lw025_seed5",
        ],
    },
    {
        "condition": "entropy_0.05_balance_1.0",
        "entropy_weight": 0.05,
        "balance_weight": 1.0,
        "dirs": [
            "results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed1",
            "results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed2",
            "results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed3",
            "results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed4",
            "results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed5",
        ],
    },
    {
        "condition": "entropy_0.10_balance_1.0",
        "entropy_weight": 0.10,
        "balance_weight": 1.0,
        "dirs": [
            "results/phase3_toy_unlabeled_token_router_entropy010_balance1_lw025_seed1",
            "results/phase3_toy_unlabeled_token_router_entropy010_balance1_lw025_seed2",
            "results/phase3_toy_unlabeled_token_router_entropy010_balance1_lw025_seed3",
            "results/phase3_toy_unlabeled_token_router_entropy010_balance1_lw025_seed4",
            "results/phase3_toy_unlabeled_token_router_entropy010_balance1_lw025_seed5",
        ],
    },
]


FLOAT_FIELDS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-model-summary",
        type=Path,
        default=Path("results/phase3_toy_learned_router_lw025/model_summary.csv"),
    )
    parser.add_argument(
        "--weak-model-summary",
        type=Path,
        default=Path("results/phase3_toy_weak_router_w005_lw025/model_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_unlabeled_router_regularization_analysis"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_condition_rows(condition: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dir_text in condition["dirs"]:
        path = Path(str(dir_text)) / "model_summary.csv"
        rows.extend(read_rows(path))
    return rows


def maybe_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in {"", "True", "False"}:
        if value == "True":
            return 1.0
        if value == "False":
            return 0.0
        return float("nan")
    return float(value)


def summarize_rows(
    condition: str,
    entropy_weight: float,
    balance_weight: float,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    local_top = Counter(row["local_top_branch"] for row in rows)
    induction_top = Counter(row["induction_top_branch"] for row in rows)
    output: dict[str, object] = {
        "condition": condition,
        "entropy_weight": entropy_weight,
        "balance_weight": balance_weight,
        "n_models": len(rows),
        "local_top_branch_counts_json": json.dumps(dict(sorted(local_top.items()))),
        "induction_top_branch_counts_json": json.dumps(dict(sorted(induction_top.items()))),
    }
    for field in FLOAT_FIELDS:
        values = [maybe_float(row, field) for row in rows if field in row]
        values = [value for value in values if not np.isnan(value)]
        output_key = field if field.endswith("_mean") else f"{field}_mean"
        output[output_key] = float(np.mean(values)) if values else float("nan")
    return output


def build_summary(args: argparse.Namespace) -> list[dict[str, object]]:
    summary = []
    baseline_rows = [
        row for row in read_rows(args.baseline_model_summary) if row["config"] == "learned_token_router"
    ]
    summary.append(summarize_rows("unconstrained_learned_token", 0.0, 0.0, baseline_rows))
    for condition in CONDITIONS:
        summary.append(
            summarize_rows(
                str(condition["condition"]),
                float(condition["entropy_weight"]),
                float(condition["balance_weight"]),
                read_condition_rows(condition),
            )
        )
    weak_rows = [row for row in read_rows(args.weak_model_summary) if row["config"] == "weak_token_router"]
    summary.append(summarize_rows("weak_label_0.05", float("nan"), float("nan"), weak_rows))
    oracle_rows = [row for row in read_rows(args.baseline_model_summary) if row["config"] == "oracle_route"]
    summary.append(summarize_rows("oracle_route", float("nan"), float("nan"), oracle_rows))
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    labels = [
        "unconstr.",
        "entropy",
        "balance",
        "ent+bal",
        "2x ent+bal",
        "weak label",
        "oracle",
    ]
    x = np.arange(len(rows))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(
        x - width,
        [float(row["same_top_branch_mean"]) for row in rows],
        width,
        label="same top branch",
    )
    axes[0].bar(
        x,
        [float(row["routed_role_match_mean"]) for row in rows],
        width,
        label="causal routed match",
    )
    axes[0].bar(
        x + width,
        [float(row["branch_distribution_distance_mean"]) for row in rows],
        width,
        label="causal branch distance",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("rate / distance")
    axes[0].set_title("Causal modularity")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    router_rows = rows[:5]
    router_labels = labels[:5]
    router_x = np.arange(len(router_rows))
    axes[1].bar(
        router_x - width,
        [float(row["global_gate_balance_error_mean"]) for row in router_rows],
        width,
        label="global balance error",
    )
    axes[1].bar(
        router_x,
        [float(row["global_gate_entropy_mean"]) for row in router_rows],
        width,
        label="global gate entropy",
    )
    axes[1].bar(
        router_x + width,
        [float(row["gate_distribution_distance_mean"]) for row in router_rows],
        width,
        label="local-vs-ind gate dist.",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(router_x)
    axes[1].set_xticklabels(router_labels, rotation=25, ha="right")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Router metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "unlabeled_router_regularization.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary(args)
    write_csv(args.output_dir / "unlabeled_router_regularization_summary.csv", rows)
    maybe_write_plot(args.output_dir, rows)
    for row in rows:
        print(row)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
