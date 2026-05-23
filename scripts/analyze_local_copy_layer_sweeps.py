#!/usr/bin/env python3
"""Combine local-copy layer causal sweep chunks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    Path("results/phase1_pythia160m_local_copy_layer_sweep_weak_targets"),
    Path("results/phase1_pythia160m_local_copy_layer_sweep_other_targets"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_local_copy_layer_sweep_combined"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def combine_rows(input_dirs: list[Path], filename: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        for row in read_csv(input_dir / filename):
            item: dict[str, object] = dict(row)
            item["source_dir"] = str(input_dir)
            rows.append(item)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_inputs = [path for path in args.input_dirs if (path / "seed_best_layer_summary.csv").exists()]
    layer_rows = combine_rows(existing_inputs, "layer_causal_sweep.csv")
    best_rows = combine_rows(existing_inputs, "seed_best_layer_summary.csv")

    best_counts = Counter(int(row["best_layer"]) for row in best_rows)
    best_excess = np.asarray([float(row["best_own_top_excess_over_random"]) for row in best_rows])
    layer3_excess = np.asarray([float(row["layer3_own_top_excess_over_random"]) for row in best_rows])
    best_minus_layer3 = best_excess - layer3_excess
    summary = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "existing_inputs": [str(path) for path in existing_inputs],
        "output_dir": str(args.output_dir),
        "n_target_seeds": len(best_rows),
        "best_layer_counts": {str(layer): count for layer, count in sorted(best_counts.items())},
        "layer3_best_count": int(best_counts.get(3, 0)),
        "best_excess_mean": float(best_excess.mean()),
        "best_excess_std": float(best_excess.std()),
        "layer3_excess_mean": float(layer3_excess.mean()),
        "layer3_excess_std": float(layer3_excess.std()),
        "best_minus_layer3_mean": float(best_minus_layer3.mean()),
        "best_minus_layer3_std": float(best_minus_layer3.std()),
    }

    write_csv(args.output_dir / "layer_causal_sweep.csv", layer_rows)
    write_csv(args.output_dir / "seed_best_layer_summary.csv", best_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
