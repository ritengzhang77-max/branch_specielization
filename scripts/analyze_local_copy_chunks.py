#!/usr/bin/env python3
"""Combine chunked local-copy alignment runs across target seed groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    Path("results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets1_3"),
    Path("results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets4_6"),
    Path("results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets7_9"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_local_copy_alignment_seed9_layer3_combined"),
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


def condition_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    output = []
    for condition, group in sorted(grouped.items()):
        loss = np.asarray([float(row["loss_delta"]) for row in group], dtype=np.float64)
        logit = np.asarray([float(row["target_logit_delta"]) for row in group], dtype=np.float64)
        output.append(
            {
                "revision": group[0]["revision"],
                "revision_index": group[0]["revision_index"],
                "condition": condition,
                "n": len(group),
                "loss_delta_mean": float(loss.mean()),
                "loss_delta_std": float(loss.std()),
                "target_logit_delta_mean": float(logit.mean()),
                "target_logit_delta_std": float(logit.std()),
            }
        )
    return output


def revision_summary(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not seed_rows:
        return []
    fields = [
        "selected_specialization_mean",
        "selected_specialization_max",
        "baseline_loss",
        "own_top_loss_delta",
        "random_loss_delta_mean",
        "own_top_excess_over_random",
        "source_same_index_loss_delta_mean",
        "source_aligned_loss_delta_mean",
        "aligned_minus_same_index_loss_delta_mean",
    ]
    item: dict[str, object] = {
        "revision": seed_rows[0]["revision"],
        "revision_index": seed_rows[0]["revision_index"],
        "n_target_seeds": len(seed_rows),
    }
    for field in fields:
        values = np.asarray([float(row[field]) for row in seed_rows], dtype=np.float64)
        item[f"{field}_mean"] = float(values.mean())
        item[f"{field}_std"] = float(values.std())
    return [item]


def transfer_summary(ablation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_pair: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in ablation_rows:
        condition = str(row["condition"])
        if condition not in {"source_same_index", "source_aligned"}:
            continue
        by_pair[(str(row["target_seed"]), str(row["source_seed"]))][condition] = float(row["loss_delta"])
    diffs = []
    same_values = []
    aligned_values = []
    for values in by_pair.values():
        if "source_same_index" not in values or "source_aligned" not in values:
            continue
        same = values["source_same_index"]
        aligned = values["source_aligned"]
        same_values.append(same)
        aligned_values.append(aligned)
        diffs.append(aligned - same)
    if not diffs:
        return []
    diffs_array = np.asarray(diffs, dtype=np.float64)
    return [
        {
            "revision": ablation_rows[0]["revision"],
            "revision_index": ablation_rows[0]["revision_index"],
            "n_pairs": len(diffs),
            "same_index_loss_delta_mean": float(np.mean(same_values)),
            "aligned_loss_delta_mean": float(np.mean(aligned_values)),
            "aligned_minus_same_index_mean": float(diffs_array.mean()),
            "aligned_minus_same_index_std": float(diffs_array.std()),
            "aligned_better_count": int(np.sum(diffs_array > 0)),
        }
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_inputs = [path for path in args.input_dirs if (path / "ablation_results.csv").exists()]
    ablation_rows = combine_rows(existing_inputs, "ablation_results.csv")
    seed_rows = combine_rows(existing_inputs, "revision_seed_summary.csv")
    probe_rows = combine_rows(existing_inputs, "probe_head_scores.csv")
    alignment_rows = combine_rows(existing_inputs, "alignment_rows.csv")

    condition_rows = condition_summary(ablation_rows)
    revision_rows = revision_summary(seed_rows)
    transfer_rows = transfer_summary(ablation_rows)

    write_csv(args.output_dir / "ablation_results.csv", ablation_rows)
    write_csv(args.output_dir / "revision_seed_summary.csv", seed_rows)
    write_csv(args.output_dir / "probe_head_scores.csv", probe_rows)
    write_csv(args.output_dir / "alignment_rows.csv", alignment_rows)
    write_csv(args.output_dir / "condition_summary.csv", condition_rows)
    write_csv(args.output_dir / "revision_summary.csv", revision_rows)
    write_csv(args.output_dir / "transfer_pair_summary.csv", transfer_rows)
    payload = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "existing_inputs": [str(path) for path in existing_inputs],
        "output_dir": str(args.output_dir),
        "revision_summary": revision_rows,
        "transfer_pair_summary": transfer_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
