#!/usr/bin/env python3
"""Summarize local-vs-induction role slots across loss-weight settings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_RUN_DIRS = [
    Path("results/phase3_toy_competition_weight_lw000"),
    Path("results/phase3_toy_competition_weight_lw001"),
    Path("results/phase3_toy_competition_weight_lw010"),
    Path("results/phase3_toy_competition_weight_lw025"),
    Path("results/phase3_toy_competition_layout_permutations"),
]


@dataclass
class WeightSeedRow:
    config: str
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
    local_top_slot_counts_json: str
    induction_top_slot_counts_json: str
    local_top_dim_counts_json: str
    induction_top_dim_counts_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", type=Path, default=DEFAULT_RUN_DIRS)
    parser.add_argument("--config", default="hetero4_64second")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_competition_weight_sweep"))
    return parser.parse_args()


def read_run(run_dir: Path, config: str) -> list[WeightSeedRow]:
    summary = json.loads((run_dir / "summary.json").read_text())
    local_weight = float(summary["args"]["local_weight"])
    induction_weight = float(summary["args"]["induction_weight"])
    rows = []
    with (run_dir / "model_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["config"] != config:
                continue
            local_layer = int(row["local_top_layer"])
            local_head = int(row["local_top_head"])
            local_dim = int(row["local_top_head_dim"])
            induction_layer = int(row["induction_top_layer"])
            induction_head = int(row["induction_top_head"])
            induction_dim = int(row["induction_top_head_dim"])
            rows.append(
                WeightSeedRow(
                    config=config,
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


def summarize(rows: list[WeightSeedRow]) -> list[WeightSummaryRow]:
    summaries = []
    keys = sorted({(row.local_weight, row.induction_weight, row.config) for row in rows})
    for local_weight, induction_weight, config in keys:
        group = [
            row
            for row in rows
            if row.config == config
            and row.local_weight == local_weight
            and row.induction_weight == induction_weight
        ]
        local_slots = Counter(row.local_top_slot for row in group)
        induction_slots = Counter(row.induction_top_slot for row in group)
        local_dims = Counter(row.local_top_dim for row in group)
        induction_dims = Counter(row.induction_top_dim for row in group)
        summaries.append(
            WeightSummaryRow(
                config=config,
                local_weight=local_weight,
                induction_weight=induction_weight,
                n_models=len(group),
                local_accuracy_mean=float(np.mean([row.local_accuracy for row in group])),
                induction_accuracy_mean=float(np.mean([row.induction_accuracy for row in group])),
                local_top_64_rate=float(np.mean([row.local_top_is_64 for row in group])),
                induction_top_64_rate=float(np.mean([row.induction_top_is_64 for row in group])),
                same_top_slot_rate=float(np.mean([row.same_top_slot for row in group])),
                same_top_dim_rate=float(np.mean([row.same_top_dim for row in group])),
                local_top_specialization_mean=float(np.mean([row.local_top_specialization for row in group])),
                induction_top_specialization_mean=float(
                    np.mean([row.induction_top_specialization for row in group])
                ),
                local_top_score_mean=float(np.mean([row.local_top_score for row in group])),
                induction_top_score_mean=float(np.mean([row.induction_top_score for row in group])),
                local_top_slot_counts_json=json.dumps({slot: count for slot, count in sorted(local_slots.items())}),
                induction_top_slot_counts_json=json.dumps(
                    {slot: count for slot, count in sorted(induction_slots.items())}
                ),
                local_top_dim_counts_json=json.dumps(
                    {str(dim): count for dim, count in sorted(local_dims.items())}
                ),
                induction_top_dim_counts_json=json.dumps(
                    {str(dim): count for dim, count in sorted(induction_dims.items())}
                ),
            )
        )
    return summaries


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def maybe_write_plot(output_dir: Path, summaries: list[WeightSummaryRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    summaries = sorted(summaries, key=lambda row: row.local_weight)
    x = [row.local_weight for row in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    axes[0].plot(x, [row.local_top_64_rate for row in summaries], marker="o", label="local")
    axes[0].plot(x, [row.induction_top_64_rate for row in summaries], marker="o", label="induction")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("local loss weight")
    axes[0].set_ylabel("top head is 64-dim")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(x, [row.same_top_slot_rate for row in summaries], marker="o", color="#f58518")
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("local loss weight")
    axes[1].set_ylabel("local and induction same slot")
    axes[1].grid(alpha=0.25)

    fig.suptitle("hetero4_64second: role slots under local-weight sweep")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_sweep_top64_and_slot_overlap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run_dir:
        rows.extend(read_run(run_dir, args.config))
    summaries = summarize(rows)
    write_csv(args.output_dir / "weight_seed_rows.csv", rows)
    write_csv(args.output_dir / "weight_summary.csv", summaries)
    maybe_write_plot(args.output_dir, summaries)
    print(json.dumps([asdict(row) for row in summaries], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
