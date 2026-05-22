#!/usr/bin/env python3
"""Combine weak-router supervision sweep summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("results/phase3_toy_learned_router_lw025/config_summary.csv"),
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        default=[
            "0.005:results/phase3_toy_weak_token_router_w0005_lw025/config_summary.csv",
            "0.01:results/phase3_toy_weak_token_router_w001_lw025/config_summary.csv",
            "0.02:results/phase3_toy_weak_token_router_w002_lw025/config_summary.csv",
            "0.03:results/phase3_toy_weak_token_router_w003_lw025/config_summary.csv",
            "0.04:results/phase3_toy_weak_token_router_w004_lw025/config_summary.csv",
            "0.045:results/phase3_toy_weak_token_router_w0045_lw025/config_summary.csv",
            "0.05:results/phase3_toy_weak_router_w005_lw025/config_summary.csv",
        ],
        help="Entries formatted as WEIGHT:CONFIG_SUMMARY_CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_weak_token_router_sweep_analysis"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def pick_config(path: Path, config: str) -> dict[str, str]:
    for row in read_rows(path):
        if row["config"] == config:
            return row
    raise ValueError(f"Missing config={config} in {path}")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def build_rows(args: argparse.Namespace) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    learned = pick_config(args.baseline_summary, "learned_token_router")
    rows.append(make_row("learned_token_router", 0.0, learned))
    for spec in args.sweep:
        weight_text, path_text = spec.split(":", maxsplit=1)
        row = pick_config(Path(path_text), "weak_token_router")
        rows.append(make_row("weak_token_router", float(weight_text), row))
    oracle = pick_config(args.baseline_summary, "oracle_route")
    rows.append(make_row("oracle_route", float("nan"), oracle))
    return rows


def make_row(condition: str, weight: float, source: dict[str, str]) -> dict[str, str | float]:
    return {
        "condition": condition,
        "router_supervision_weight": weight,
        "n_models": source["n_models"],
        "local_accuracy_mean": as_float(source, "local_accuracy_mean"),
        "induction_accuracy_mean": as_float(source, "induction_accuracy_mean"),
        "same_top_branch_rate": as_float(source, "same_top_branch_rate"),
        "routed_role_match_rate": as_float(source, "routed_role_match_rate"),
        "branch_distribution_distance_mean": as_float(source, "branch_distribution_distance_mean"),
        "gate_routed_role_match_rate": as_float(source, "gate_routed_role_match_rate"),
        "gate_distribution_distance_mean": as_float(source, "gate_distribution_distance_mean"),
        "local_gate_entropy_mean": as_float(source, "local_gate_entropy_mean"),
        "induction_gate_entropy_mean": as_float(source, "induction_gate_entropy_mean"),
        "induction_gate_branch1_mean": as_float(source, "induction_gate_branch1_mean"),
        "gate_target_nll_mean": as_float(source, "gate_target_nll_mean"),
        "local_branch0_loss_delta_mean": as_float(source, "local_branch0_loss_delta_mean"),
        "local_branch1_loss_delta_mean": as_float(source, "local_branch1_loss_delta_mean"),
        "induction_branch0_loss_delta_mean": as_float(source, "induction_branch0_loss_delta_mean"),
        "induction_branch1_loss_delta_mean": as_float(source, "induction_branch1_loss_delta_mean"),
        "local_top_branch_counts_json": source["local_top_branch_counts_json"],
        "induction_top_branch_counts_json": source["induction_top_branch_counts_json"],
    }


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plot(output_dir: Path, rows: list[dict[str, str | float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    sweep_rows = [row for row in rows if row["condition"] != "oracle_route"]
    weights = [float(row["router_supervision_weight"]) for row in sweep_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(
        weights,
        [float(row["routed_role_match_rate"]) for row in sweep_rows],
        marker="o",
        label="causal routed match",
    )
    axes[0].plot(
        weights,
        [float(row["branch_distribution_distance_mean"]) for row in sweep_rows],
        marker="o",
        label="causal branch distance",
    )
    axes[0].plot(
        weights,
        [float(row["gate_routed_role_match_rate"]) for row in sweep_rows],
        marker="o",
        label="gate routed match",
    )
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_xlabel("router supervision weight")
    axes[0].set_ylabel("rate / distance")
    axes[0].set_title("Modularity threshold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        weights,
        [float(row["induction_gate_branch1_mean"]) for row in sweep_rows],
        marker="o",
        label="induction gate on B1",
    )
    axes[1].plot(
        weights,
        [float(row["induction_branch1_loss_delta_mean"]) for row in sweep_rows],
        marker="o",
        label="induction B1 ablation delta",
    )
    axes[1].plot(
        weights,
        [float(row["induction_branch0_loss_delta_mean"]) for row in sweep_rows],
        marker="o",
        label="induction B0 ablation delta",
    )
    axes[1].set_xlabel("router supervision weight")
    axes[1].set_ylabel("gate weight / loss delta")
    axes[1].set_title("Induction is the hard role")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "weak_token_router_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    write_csv(args.output_dir / "weak_token_router_sweep_summary.csv", rows)
    maybe_write_plot(args.output_dir, rows)
    for row in rows:
        print(row)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
