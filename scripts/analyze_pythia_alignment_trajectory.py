#!/usr/bin/env python3
"""Analyze aligned-vs-same-index transfer in Pythia repeat-match trajectories."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_repeat_match_alignment_trajectory"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_transfer_pair_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_pair: dict[tuple[str, int, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["condition"] not in {"source_same_index", "source_aligned"}:
            continue
        key = (
            row["revision"],
            int(row["revision_index"]),
            row["target_seed"],
            row["source_seed"],
        )
        by_pair[key][row["condition"]] = float(row["loss_delta"])

    by_revision: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for (revision, revision_index, target_seed, source_seed), values in by_pair.items():
        if "source_same_index" not in values or "source_aligned" not in values:
            continue
        same = values["source_same_index"]
        aligned = values["source_aligned"]
        by_revision[(revision_index, revision)].append(
            {
                "target_seed": float(target_seed),
                "source_seed": float(source_seed),
                "same_index_loss_delta": same,
                "aligned_loss_delta": aligned,
                "aligned_minus_same_index": aligned - same,
            }
        )

    output = []
    for (revision_index, revision), group in sorted(by_revision.items()):
        same_values = np.asarray([item["same_index_loss_delta"] for item in group], dtype=np.float64)
        aligned_values = np.asarray([item["aligned_loss_delta"] for item in group], dtype=np.float64)
        diffs = np.asarray([item["aligned_minus_same_index"] for item in group], dtype=np.float64)
        output.append(
            {
                "revision": revision,
                "revision_index": revision_index,
                "n_pairs": len(group),
                "same_index_loss_delta_mean": float(same_values.mean()),
                "aligned_loss_delta_mean": float(aligned_values.mean()),
                "aligned_minus_same_index_mean": float(diffs.mean()),
                "aligned_minus_same_index_std": float(diffs.std()),
                "aligned_better_count": int(np.sum(diffs > 0)),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir / "ablation_results.csv")
    summary = build_transfer_pair_summary(rows)
    write_csv(args.input_dir / "transfer_pair_summary.csv", summary)
    for row in summary:
        print(row, flush=True)
    print(f"wrote {args.input_dir / 'transfer_pair_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
