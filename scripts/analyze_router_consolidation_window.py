#!/usr/bin/env python3
"""Summarize a dense router-trajectory consolidation-window run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MEAN_FIELDS = [
    "local_accuracy",
    "induction_accuracy",
    "routed_role_match",
    "same_top_branch",
    "branch_distribution_distance",
    "gate_routed_role_match",
    "gate_distribution_distance",
    "gate_target_nll_mean",
    "local_gate_branch0_mean",
    "induction_gate_branch1_mean",
    "local_branch0_loss_delta",
    "local_branch1_loss_delta",
    "induction_branch0_loss_delta",
    "induction_branch1_loss_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/phase3_toy_trajectory_consolidation_end800"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3_toy_trajectory_consolidation_end800_analysis"),
    )
    parser.add_argument("--solved-accuracy-threshold", type=float, default=0.99)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def mean(values: list[float]) -> float:
    values = [value for value in values if value == value]
    return sum(values) / len(values) if values else float("nan")


def count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)

    summaries: list[dict[str, object]] = []
    for step, group in sorted(by_step.items()):
        out: dict[str, object] = {
            "step": step,
            "n_models": len(group),
            "local_top_branch_counts_json": json.dumps(
                count_values([row["local_top_branch"] for row in group])
            ),
            "induction_top_branch_counts_json": json.dumps(
                count_values([row["induction_top_branch"] for row in group])
            ),
        }
        for field in MEAN_FIELDS:
            out[f"{field}_mean"] = mean([parse_float(row[field]) for row in group])
        summaries.append(out)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_first_step(
    summaries: list[dict[str, object]], field: str, threshold: float
) -> int | None:
    for row in summaries:
        value = float(row[field])
        if value >= threshold:
            return int(row["step"])
    return None


def is_solved(row: dict[str, object], threshold: float) -> bool:
    return (
        float(row["local_accuracy_mean"]) >= threshold
        and float(row["induction_accuracy_mean"]) >= threshold
    )


def find_first_solved_step(
    summaries: list[dict[str, object]],
    field: str,
    threshold: float,
    solved_accuracy_threshold: float,
) -> int | None:
    for row in summaries:
        if not is_solved(row, solved_accuracy_threshold):
            continue
        value = float(row[field])
        if value >= threshold:
            return int(row["step"])
    return None


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_dir / "trajectory_summary.csv")
    summaries = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "consolidation_window_summary.csv", summaries)

    milestone = {
        "input_dir": str(args.input_dir),
        "n_rows": len(rows),
        "n_steps": len(summaries),
        "first_gate_match_all_seeds_step": find_first_step(
            summaries, "gate_routed_role_match_mean", 1.0
        ),
        "first_causal_match_all_seeds_step": find_first_step(
            summaries, "routed_role_match_mean", 1.0
        ),
        "first_branch_distance_ge_0.30_step": find_first_step(
            summaries, "branch_distribution_distance_mean", 0.30
        ),
        "first_branch_distance_ge_0.40_step": find_first_step(
            summaries, "branch_distribution_distance_mean", 0.40
        ),
        "solved_accuracy_threshold": args.solved_accuracy_threshold,
        "first_solved_gate_match_all_seeds_step": find_first_solved_step(
            summaries,
            "gate_routed_role_match_mean",
            1.0,
            args.solved_accuracy_threshold,
        ),
        "first_solved_causal_match_all_seeds_step": find_first_solved_step(
            summaries,
            "routed_role_match_mean",
            1.0,
            args.solved_accuracy_threshold,
        ),
        "first_solved_branch_distance_ge_0.30_step": find_first_solved_step(
            summaries,
            "branch_distribution_distance_mean",
            0.30,
            args.solved_accuracy_threshold,
        ),
        "first_solved_branch_distance_ge_0.40_step": find_first_solved_step(
            summaries,
            "branch_distribution_distance_mean",
            0.40,
            args.solved_accuracy_threshold,
        ),
        "final_step": int(summaries[-1]["step"]) if summaries else None,
        "final_routed_role_match": summaries[-1]["routed_role_match_mean"]
        if summaries
        else None,
        "final_branch_distribution_distance": summaries[-1][
            "branch_distribution_distance_mean"
        ]
        if summaries
        else None,
    }
    with (args.output_dir / "consolidation_window_milestones.json").open("w") as handle:
        json.dump(milestone, handle, indent=2)
        handle.write("\n")

    print(json.dumps(milestone, indent=2), flush=True)


if __name__ == "__main__":
    main()
