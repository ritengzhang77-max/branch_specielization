#!/usr/bin/env python3
"""Summarize local-copy candidate-pool alignment across checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = [
    Path("results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2_step0"),
    Path("results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2_step4000"),
    Path("results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2_step16000"),
    Path("results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_local_copy_candidate_pool_trajectory"),
    )
    return parser.parse_args()


def read_first_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def revision_sort_key(revision: str) -> int:
    if revision.startswith("step"):
        return int(revision.removeprefix("step"))
    return 10**12


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for input_dir in args.input_dirs:
        revision = read_first_csv(input_dir / "revision_summary.csv")
        transfer = read_first_csv(input_dir / "transfer_pair_summary.csv")
        if revision is None or transfer is None:
            continue
        row = {
            "revision": revision["revision"],
            "step": revision_sort_key(revision["revision"]),
            "n_target_seeds": revision.get("n_target_seeds", revision.get("n_seeds")),
            "n_pairs": transfer["n_pairs"],
            "selected_specialization_mean": revision["selected_specialization_mean_mean"],
            "own_top_excess_over_random": revision["own_top_excess_over_random_mean"],
            "same_index_transfer": transfer["same_index_loss_delta_mean"],
            "aligned_transfer": transfer["aligned_loss_delta_mean"],
            "aligned_minus_same": transfer["aligned_minus_same_index_mean"],
            "aligned_better_count": transfer["aligned_better_count"],
            "source_dir": str(input_dir),
        }
        rows.append(row)
    rows.sort(key=lambda row: int(row["step"]))
    if rows:
        with (args.output_dir / "trajectory_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "output_dir": str(args.output_dir),
        "trajectory_summary": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
