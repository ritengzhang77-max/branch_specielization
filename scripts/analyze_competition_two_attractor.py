#!/usr/bin/env python3
"""Compare single-attractor and two-attractor layouts under competition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_RUN_DIRS = [
    Path("results/phase3_toy_competition_all_layout_weights_lw025"),
    Path("results/phase3_toy_competition_weight_lw025"),
    Path("results/phase3_toy_competition_two48_lw025"),
    Path("results/phase3_toy_competition_uniform2_lw025"),
]


@dataclass
class AttractorSeedRow:
    family: str
    config: str
    head_dims_json: str
    max_head_dim: int
    n_max_dim_heads: int
    local_weight: float
    seed: int
    local_accuracy: float
    induction_accuracy: float
    local_top_slot: str
    local_top_dim: int
    induction_top_slot: str
    induction_top_dim: int
    local_top_is_max_dim: bool
    induction_top_is_max_dim: bool
    both_top_max_dim: bool
    same_top_slot: bool
    distinct_max_dim_slots: bool


@dataclass
class AttractorSummaryRow:
    family: str
    n_models: int
    local_accuracy_mean: float
    induction_accuracy_mean: float
    local_top_max_dim_rate: float
    induction_top_max_dim_rate: float
    both_top_max_dim_rate: float
    same_top_slot_rate: float
    distinct_max_dim_slots_rate: float
    local_top_dim_counts_json: str
    induction_top_dim_counts_json: str
    local_top_slot_counts_json: str
    induction_top_slot_counts_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", type=Path, default=DEFAULT_RUN_DIRS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_competition_two_attractor"))
    return parser.parse_args()


def family_for_config(config: str, dims: list[int]) -> str:
    if config == "uniform2":
        return "two64_uniform"
    if max(dims) == 64 and dims.count(64) == 1:
        return "single64_hetero"
    if max(dims) == 48 and dims.count(48) == 2:
        return "two48_hetero"
    return "other"


def read_run(run_dir: Path) -> list[AttractorSeedRow]:
    summary = json.loads((run_dir / "summary.json").read_text())
    local_weight = float(summary["args"]["local_weight"])
    rows = []
    with (run_dir / "model_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            dims = [int(item) for item in json.loads(row["head_dims_json"])]
            max_dim = max(dims)
            local_layer = int(row["local_top_layer"])
            local_head = int(row["local_top_head"])
            local_dim = int(row["local_top_head_dim"])
            induction_layer = int(row["induction_top_layer"])
            induction_head = int(row["induction_top_head"])
            induction_dim = int(row["induction_top_head_dim"])
            local_slot = f"L{local_layer}H{local_head}:d{local_dim}"
            induction_slot = f"L{induction_layer}H{induction_head}:d{induction_dim}"
            local_is_max = local_dim == max_dim
            induction_is_max = induction_dim == max_dim
            rows.append(
                AttractorSeedRow(
                    family=family_for_config(row["config"], dims),
                    config=row["config"],
                    head_dims_json=row["head_dims_json"],
                    max_head_dim=max_dim,
                    n_max_dim_heads=dims.count(max_dim),
                    local_weight=local_weight,
                    seed=int(row["seed"]),
                    local_accuracy=float(row["local_accuracy"]),
                    induction_accuracy=float(row["induction_accuracy"]),
                    local_top_slot=local_slot,
                    local_top_dim=local_dim,
                    induction_top_slot=induction_slot,
                    induction_top_dim=induction_dim,
                    local_top_is_max_dim=local_is_max,
                    induction_top_is_max_dim=induction_is_max,
                    both_top_max_dim=local_is_max and induction_is_max,
                    same_top_slot=row["same_top_slot"].lower() == "true",
                    distinct_max_dim_slots=local_is_max
                    and induction_is_max
                    and local_slot != induction_slot,
                )
            )
    return rows


def summarize(rows: list[AttractorSeedRow]) -> list[AttractorSummaryRow]:
    by_family: dict[str, list[AttractorSeedRow]] = defaultdict(list)
    for row in rows:
        if row.family != "other":
            by_family[row.family].append(row)

    summaries = []
    for family, group in sorted(by_family.items()):
        local_dims = Counter(row.local_top_dim for row in group)
        induction_dims = Counter(row.induction_top_dim for row in group)
        local_slots = Counter(row.local_top_slot for row in group)
        induction_slots = Counter(row.induction_top_slot for row in group)
        summaries.append(
            AttractorSummaryRow(
                family=family,
                n_models=len(group),
                local_accuracy_mean=float(np.mean([row.local_accuracy for row in group])),
                induction_accuracy_mean=float(np.mean([row.induction_accuracy for row in group])),
                local_top_max_dim_rate=float(np.mean([row.local_top_is_max_dim for row in group])),
                induction_top_max_dim_rate=float(np.mean([row.induction_top_is_max_dim for row in group])),
                both_top_max_dim_rate=float(np.mean([row.both_top_max_dim for row in group])),
                same_top_slot_rate=float(np.mean([row.same_top_slot for row in group])),
                distinct_max_dim_slots_rate=float(
                    np.mean([row.distinct_max_dim_slots for row in group])
                ),
                local_top_dim_counts_json=json.dumps(
                    {str(dim): count for dim, count in sorted(local_dims.items())}
                ),
                induction_top_dim_counts_json=json.dumps(
                    {str(dim): count for dim, count in sorted(induction_dims.items())}
                ),
                local_top_slot_counts_json=json.dumps({slot: count for slot, count in sorted(local_slots.items())}),
                induction_top_slot_counts_json=json.dumps(
                    {slot: count for slot, count in sorted(induction_slots.items())}
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


def maybe_write_plot(output_dir: Path, summaries: list[AttractorSummaryRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = ["single64_hetero", "two48_hetero", "two64_uniform"]
    labels = {
        "single64_hetero": "single 64",
        "two48_hetero": "two 48s",
        "two64_uniform": "two 64s",
    }
    rows = [next(row for row in summaries if row.family == family) for family in order]
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width, [row.local_top_max_dim_rate for row in rows], width, label="local top max-dim")
    ax.bar(x, [row.induction_top_max_dim_rate for row in rows], width, label="induction top max-dim")
    ax.bar(x + width, [row.distinct_max_dim_slots_rate for row in rows], width, label="distinct max-dim slots")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[row.family] for row in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate across seeds")
    ax.set_title("Local weight 0.25: does a second attractor split roles?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "two_attractor_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run_dir:
        rows.extend(read_run(run_dir))
    summaries = summarize(rows)
    write_csv(args.output_dir / "attractor_seed_rows.csv", rows)
    write_csv(args.output_dir / "attractor_summary.csv", summaries)
    maybe_write_plot(args.output_dir, summaries)
    print(json.dumps([asdict(row) for row in summaries], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
