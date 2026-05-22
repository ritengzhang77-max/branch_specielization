#!/usr/bin/env python3
"""Summarize role slots across all heterogeneous layouts and local weights."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_RUN_DIRS = [
    Path("results/phase3_toy_competition_all_layout_weights_lw00"),
    Path("results/phase3_toy_competition_all_layout_weights_lw001"),
    Path("results/phase3_toy_competition_all_layout_weights_lw010"),
    Path("results/phase3_toy_competition_all_layout_weights_lw025"),
    Path("results/phase3_toy_competition_weight_lw000"),
    Path("results/phase3_toy_competition_weight_lw001"),
    Path("results/phase3_toy_competition_weight_lw010"),
    Path("results/phase3_toy_competition_weight_lw025"),
    Path("results/phase3_toy_competition_head_dim_intervention"),
    Path("results/phase3_toy_competition_layout_permutations"),
]

DEFAULT_CONFIGS = ["hetero4_64first", "hetero4_64second", "hetero4_64third", "hetero4"]


@dataclass
class WeightSeedRow:
    config: str
    head_dims_json: str
    local_weight: float
    induction_weight: float
    seed: int
    local_accuracy: float
    induction_accuracy: float
    local_top_slot: str
    local_top_dim: int
    local_top_score: float
    local_top_specialization: float
    induction_top_slot: str
    induction_top_dim: int
    induction_top_score: float
    induction_top_specialization: float
    local_top_is_64: bool
    induction_top_is_64: bool
    same_top_slot: bool
    same_top_dim: bool


@dataclass
class WeightSummaryRow:
    config: str
    head_dims_json: str
    local_weight: float
    induction_weight: float
    n_models: int
    local_accuracy_mean: float
    induction_accuracy_mean: float
    local_top_64_rate: float
    induction_top_64_rate: float
    same_top_slot_rate: float
    same_top_dim_rate: float
    local_top_specialization_mean: float
    induction_top_specialization_mean: float
    local_top_score_mean: float
    induction_top_score_mean: float
    local_top_dim_counts_json: str
    induction_top_dim_counts_json: str
    local_top_slot_counts_json: str
    induction_top_slot_counts_json: str


@dataclass
class AggregateWeightRow:
    local_weight: float
    induction_weight: float
    n_models: int
    local_accuracy_mean: float
    induction_accuracy_mean: float
    local_top_64_rate: float
    induction_top_64_rate: float
    same_top_slot_rate: float
    same_top_dim_rate: float
    local_top_dim_counts_json: str
    induction_top_dim_counts_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", type=Path, default=DEFAULT_RUN_DIRS)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_competition_all_layout_weight_sweep"),
    )
    return parser.parse_args()


def read_run(run_dir: Path, configs: set[str]) -> list[WeightSeedRow]:
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    local_weight = float(summary["args"]["local_weight"])
    induction_weight = float(summary["args"]["induction_weight"])
    rows = []
    with (run_dir / "model_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["config"] not in configs:
                continue
            local_layer = int(row["local_top_layer"])
            local_head = int(row["local_top_head"])
            local_dim = int(row["local_top_head_dim"])
            induction_layer = int(row["induction_top_layer"])
            induction_head = int(row["induction_top_head"])
            induction_dim = int(row["induction_top_head_dim"])
            rows.append(
                WeightSeedRow(
                    config=row["config"],
                    head_dims_json=row["head_dims_json"],
                    local_weight=local_weight,
                    induction_weight=induction_weight,
                    seed=int(row["seed"]),
                    local_accuracy=float(row["local_accuracy"]),
                    induction_accuracy=float(row["induction_accuracy"]),
                    local_top_slot=f"L{local_layer}H{local_head}:d{local_dim}",
                    local_top_dim=local_dim,
                    local_top_score=float(row["local_top_role_score"]),
                    local_top_specialization=float(row["local_top_specialization"]),
                    induction_top_slot=f"L{induction_layer}H{induction_head}:d{induction_dim}",
                    induction_top_dim=induction_dim,
                    induction_top_score=float(row["induction_top_role_score"]),
                    induction_top_specialization=float(row["induction_top_specialization"]),
                    local_top_is_64=local_dim == 64,
                    induction_top_is_64=induction_dim == 64,
                    same_top_slot=row["same_top_slot"].lower() == "true",
                    same_top_dim=row["same_top_head_dim"].lower() == "true",
                )
            )
    return rows


def summarize_group(group: list[WeightSeedRow], config: str | None = None) -> dict[str, object]:
    local_dims = Counter(row.local_top_dim for row in group)
    induction_dims = Counter(row.induction_top_dim for row in group)
    payload = {
        "n_models": len(group),
        "local_accuracy_mean": float(np.mean([row.local_accuracy for row in group])),
        "induction_accuracy_mean": float(np.mean([row.induction_accuracy for row in group])),
        "local_top_64_rate": float(np.mean([row.local_top_is_64 for row in group])),
        "induction_top_64_rate": float(np.mean([row.induction_top_is_64 for row in group])),
        "same_top_slot_rate": float(np.mean([row.same_top_slot for row in group])),
        "same_top_dim_rate": float(np.mean([row.same_top_dim for row in group])),
        "local_top_dim_counts_json": json.dumps({str(dim): count for dim, count in sorted(local_dims.items())}),
        "induction_top_dim_counts_json": json.dumps(
            {str(dim): count for dim, count in sorted(induction_dims.items())}
        ),
    }
    if config is not None:
        local_slots = Counter(row.local_top_slot for row in group)
        induction_slots = Counter(row.induction_top_slot for row in group)
        payload.update(
            {
                "config": config,
                "head_dims_json": group[0].head_dims_json,
                "local_top_specialization_mean": float(
                    np.mean([row.local_top_specialization for row in group])
                ),
                "induction_top_specialization_mean": float(
                    np.mean([row.induction_top_specialization for row in group])
                ),
                "local_top_score_mean": float(np.mean([row.local_top_score for row in group])),
                "induction_top_score_mean": float(np.mean([row.induction_top_score for row in group])),
                "local_top_slot_counts_json": json.dumps(
                    {slot: count for slot, count in sorted(local_slots.items())}
                ),
                "induction_top_slot_counts_json": json.dumps(
                    {slot: count for slot, count in sorted(induction_slots.items())}
                ),
            }
        )
    return payload


def summarize(rows: list[WeightSeedRow]) -> tuple[list[WeightSummaryRow], list[AggregateWeightRow]]:
    by_config_weight: dict[tuple[str, float, float], list[WeightSeedRow]] = defaultdict(list)
    by_weight: dict[tuple[float, float], list[WeightSeedRow]] = defaultdict(list)
    for row in rows:
        by_config_weight[(row.config, row.local_weight, row.induction_weight)].append(row)
        by_weight[(row.local_weight, row.induction_weight)].append(row)

    config_summaries = []
    for (config, local_weight, induction_weight), group in sorted(by_config_weight.items()):
        payload = summarize_group(group, config=config)
        config_summaries.append(
            WeightSummaryRow(
                config=config,
                head_dims_json=payload["head_dims_json"],
                local_weight=local_weight,
                induction_weight=induction_weight,
                n_models=payload["n_models"],
                local_accuracy_mean=payload["local_accuracy_mean"],
                induction_accuracy_mean=payload["induction_accuracy_mean"],
                local_top_64_rate=payload["local_top_64_rate"],
                induction_top_64_rate=payload["induction_top_64_rate"],
                same_top_slot_rate=payload["same_top_slot_rate"],
                same_top_dim_rate=payload["same_top_dim_rate"],
                local_top_specialization_mean=payload["local_top_specialization_mean"],
                induction_top_specialization_mean=payload["induction_top_specialization_mean"],
                local_top_score_mean=payload["local_top_score_mean"],
                induction_top_score_mean=payload["induction_top_score_mean"],
                local_top_dim_counts_json=payload["local_top_dim_counts_json"],
                induction_top_dim_counts_json=payload["induction_top_dim_counts_json"],
                local_top_slot_counts_json=payload["local_top_slot_counts_json"],
                induction_top_slot_counts_json=payload["induction_top_slot_counts_json"],
            )
        )

    aggregate_summaries = []
    for (local_weight, induction_weight), group in sorted(by_weight.items()):
        payload = summarize_group(group)
        aggregate_summaries.append(
            AggregateWeightRow(
                local_weight=local_weight,
                induction_weight=induction_weight,
                n_models=payload["n_models"],
                local_accuracy_mean=payload["local_accuracy_mean"],
                induction_accuracy_mean=payload["induction_accuracy_mean"],
                local_top_64_rate=payload["local_top_64_rate"],
                induction_top_64_rate=payload["induction_top_64_rate"],
                same_top_slot_rate=payload["same_top_slot_rate"],
                same_top_dim_rate=payload["same_top_dim_rate"],
                local_top_dim_counts_json=payload["local_top_dim_counts_json"],
                induction_top_dim_counts_json=payload["induction_top_dim_counts_json"],
            )
        )
    return config_summaries, aggregate_summaries


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def maybe_write_plots(output_dir: Path, config_summaries: list[WeightSummaryRow], aggregate: list[AggregateWeightRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = DEFAULT_CONFIGS
    labels = {
        "hetero4_64first": "64 first",
        "hetero4_64second": "64 second",
        "hetero4_64third": "64 third",
        "hetero4": "64 last",
    }
    weights = sorted({row.local_weight for row in config_summaries})

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for config in configs:
        rows = sorted([row for row in config_summaries if row.config == config], key=lambda row: row.local_weight)
        ax.plot(
            [row.local_weight for row in rows],
            [row.induction_top_64_rate for row in rows],
            marker="o",
            label=labels.get(config, config),
        )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("local loss weight")
    ax.set_ylabel("induction top head is 64-dim")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "induction_top64_by_layout_and_weight.png", dpi=180)
    plt.close(fig)

    aggregate = sorted(aggregate, key=lambda row: row.local_weight)
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.plot(
        [row.local_weight for row in aggregate],
        [row.local_top_64_rate for row in aggregate],
        marker="o",
        label="local",
    )
    ax.plot(
        [row.local_weight for row in aggregate],
        [row.induction_top_64_rate for row in aggregate],
        marker="o",
        label="induction",
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("local loss weight")
    ax.set_ylabel("top head is 64-dim")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_top64_by_weight.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run_dir:
        rows.extend(read_run(run_dir, set(args.configs)))
    config_summaries, aggregate_summaries = summarize(rows)
    write_csv(args.output_dir / "weight_seed_rows.csv", rows)
    write_csv(args.output_dir / "weight_config_summary.csv", config_summaries)
    write_csv(args.output_dir / "weight_aggregate_summary.csv", aggregate_summaries)
    maybe_write_plots(args.output_dir, config_summaries, aggregate_summaries)
    print(json.dumps([asdict(row) for row in aggregate_summaries], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
