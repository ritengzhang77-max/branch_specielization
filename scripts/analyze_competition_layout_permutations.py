#!/usr/bin/env python3
"""Summarize local-vs-induction role slots across heterogeneous layouts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    Path("results/phase3_toy_competition_head_dim_intervention/model_summary.csv"),
    Path("results/phase3_toy_competition_layout_permutations/model_summary.csv"),
]


@dataclass
class RoleTopRow:
    config: str
    head_dims_json: str
    seed: int
    role: str
    accuracy: float
    loss: float
    top_layer: int
    top_head: int
    top_head_dim: int
    top_slot: str
    top_role_score: float
    top_specialization: float
    top_is_64: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-summary", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["hetero4", "hetero4_64first", "hetero4_64second", "hetero4_64third"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_competition_layout_analysis"))
    return parser.parse_args()


def read_rows(paths: list[Path], configs: set[str]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["config"] in configs:
                    rows.append(row)
    return rows


def role_rows(model_rows: list[dict[str, str]]) -> list[RoleTopRow]:
    rows: list[RoleTopRow] = []
    for row in model_rows:
        for role in ["local", "induction"]:
            top_layer = int(row[f"{role}_top_layer"])
            top_head = int(row[f"{role}_top_head"])
            top_dim = int(row[f"{role}_top_head_dim"])
            rows.append(
                RoleTopRow(
                    config=row["config"],
                    head_dims_json=row["head_dims_json"],
                    seed=int(row["seed"]),
                    role=role,
                    accuracy=float(row[f"{role}_accuracy"]),
                    loss=float(row[f"{role}_loss"]),
                    top_layer=top_layer,
                    top_head=top_head,
                    top_head_dim=top_dim,
                    top_slot=f"L{top_layer}H{top_head}:d{top_dim}",
                    top_role_score=float(row[f"{role}_top_role_score"]),
                    top_specialization=float(row[f"{role}_top_specialization"]),
                    top_is_64=top_dim == 64,
                )
            )
    return rows


def summarize(rows: list[RoleTopRow]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_config_role: dict[tuple[str, str], list[RoleTopRow]] = defaultdict(list)
    for row in rows:
        by_config_role[(row.config, row.role)].append(row)

    config_summaries = []
    for (config, role), group in sorted(by_config_role.items()):
        dim_counts = Counter(row.top_head_dim for row in group)
        slot_counts = Counter(row.top_slot for row in group)
        config_summaries.append(
            {
                "config": config,
                "role": role,
                "head_dims_json": group[0].head_dims_json,
                "n_models": len(group),
                "accuracy_mean": float(np.mean([row.accuracy for row in group])),
                "top_specialization_mean": float(np.mean([row.top_specialization for row in group])),
                "top_role_score_mean": float(np.mean([row.top_role_score for row in group])),
                "top_64_rate": float(np.mean([row.top_is_64 for row in group])),
                "top_head_dim_counts_json": json.dumps(
                    {str(dim): count for dim, count in sorted(dim_counts.items())}
                ),
                "top_slot_counts_json": json.dumps({slot: count for slot, count in sorted(slot_counts.items())}),
            }
        )

    aggregate = {}
    for role in ["local", "induction"]:
        role_group = [row for row in rows if row.role == role]
        aggregate[role] = {
            "n_models": len(role_group),
            "top_64_rate": float(np.mean([row.top_is_64 for row in role_group])),
            "top_specialization_mean": float(np.mean([row.top_specialization for row in role_group])),
            "top_role_score_mean": float(np.mean([row.top_role_score for row in role_group])),
            "top_head_dim_counts": {
                str(dim): count for dim, count in sorted(Counter(row.top_head_dim for row in role_group).items())
            },
            "top_slot_counts": {
                slot: count for slot, count in sorted(Counter(row.top_slot for row in role_group).items())
            },
        }
    return config_summaries, aggregate


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    first = rows[0]
    fieldnames = list(asdict(first).keys()) if not isinstance(first, dict) else list(first.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if not isinstance(row, dict) else row)


def maybe_write_plot(output_dir: Path, config_summaries: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = ["hetero4_64first", "hetero4_64second", "hetero4_64third", "hetero4"]
    labels = ["64 first", "64 second", "64 third", "64 last"]
    roles = ["local", "induction"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, role in zip(axes, roles):
        values = []
        for config in configs:
            row = next(item for item in config_summaries if item["config"] == config and item["role"] == role)
            values.append(float(row["top_64_rate"]))
        ax.bar(labels, values, color="#4c78a8")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{role.title()} top head is 64-dim")
        ax.set_ylabel("rate across 5 seeds")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "top_64_rate_by_layout.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = read_rows(args.model_summary, set(args.configs))
    tops = role_rows(model_rows)
    config_summaries, aggregate = summarize(tops)
    write_csv(args.output_dir / "role_top_rows.csv", tops)
    write_csv(args.output_dir / "layout_role_summary.csv", config_summaries)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))
    maybe_write_plot(args.output_dir, config_summaries)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
