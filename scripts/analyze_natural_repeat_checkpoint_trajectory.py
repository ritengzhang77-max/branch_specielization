#!/usr/bin/env python3
"""Aggregate natural-repeat candidate-pool runs across checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = [
    Path("results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0"),
    Path("results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step4000"),
    Path("results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step16000"),
    Path("results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step64000"),
    Path("results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_wikitext103_ordinary_repeat_trajectory"),
    )
    return parser.parse_args()


def read_first_csv(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows[0]


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def revision_sort_key(revision: str) -> int:
    if revision.startswith("step"):
        return int(revision.removeprefix("step"))
    return 10**12


def collect_row(input_dir: Path) -> dict[str, object]:
    revision = read_first_csv(input_dir / "revision_summary.csv")
    significance = read_json(input_dir / "transfer_significance_summary.json")
    return {
        "revision": revision["revision"],
        "step": revision_sort_key(revision["revision"]),
        "n_target_seeds": revision.get("n_target_seeds", revision.get("n_seeds")),
        "selected_specialization_mean": revision["selected_specialization_mean_mean"],
        "own_top_excess_over_random": revision["own_top_excess_over_random_mean"],
        "same_index_transfer": revision["source_same_index_loss_delta_mean_mean"],
        "aligned_transfer": revision["source_aligned_loss_delta_mean_mean"],
        "aligned_minus_same": revision["aligned_minus_same_index_loss_delta_mean_mean"],
        "pair_aligned_minus_same_ci_low": significance[
            "pair_aligned_minus_same_bootstrap_ci_low"
        ],
        "pair_aligned_minus_same_ci_high": significance[
            "pair_aligned_minus_same_bootstrap_ci_high"
        ],
        "pair_aligned_better_count": significance[
            "pair_aligned_minus_same_positive_count"
        ],
        "pair_n": significance["pair_aligned_minus_same_n"],
        "target_aligned_minus_same_ci_low": significance[
            "target_aligned_minus_same_bootstrap_ci_low"
        ],
        "target_aligned_minus_same_ci_high": significance[
            "target_aligned_minus_same_bootstrap_ci_high"
        ],
        "target_aligned_positive_count": significance[
            "target_aligned_minus_same_positive_count"
        ],
        "target_n": significance["target_aligned_minus_same_n"],
        "target_own_excess_ci_low": significance["target_own_excess_bootstrap_ci_low"],
        "target_own_excess_ci_high": significance["target_own_excess_bootstrap_ci_high"],
        "target_own_excess_positive_count": significance["target_own_excess_positive_count"],
        "source_dir": str(input_dir),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [collect_row(input_dir) for input_dir in args.input_dirs]
    rows.sort(key=lambda row: int(row["step"]))
    write_csv(args.output_dir / "trajectory_summary.csv", rows)
    payload = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "output_dir": str(args.output_dir),
        "trajectory_summary": rows,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
