#!/usr/bin/env python3
"""Combine selected-checkpoint Pythia-160M seed-9 alignment-transfer runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = [
    Path("results/phase1_pythia160m_repeat_match_alignment_seed9_step4000"),
    Path("results/phase1_pythia160m_repeat_match_alignment_seed9_step16000"),
    Path("results/phase1_pythia160m_repeat_match_alignment_seed9_step143000"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_repeat_match_alignment_seed9_selected"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def revision_sort_key(row: dict[str, object]) -> int:
    order = {"step4000": 0, "step16000": 1, "step143000": 2}
    return order.get(str(row["revision"]), int(row.get("revision_index", 0)))


def load_combined(input_dirs: list[Path], filename: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        path = input_dir / filename
        for row in read_csv(path):
            item: dict[str, object] = dict(row)
            item["source_dir"] = str(input_dir)
            rows.append(item)
    rows.sort(key=revision_sort_key)
    for idx, row in enumerate(rows):
        row["selected_revision_index"] = idx
    return rows


def maybe_write_plot(output_dir: Path, revision_rows: list[dict[str, object]], transfer_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [str(row["revision"]) for row in revision_rows]
    x = list(range(len(labels)))
    probe = [float(row["selected_specialization_mean_mean"]) for row in revision_rows]
    own_excess = [float(row["own_top_excess_over_random_mean"]) for row in revision_rows]
    same = [float(row["source_same_index_loss_delta_mean_mean"]) for row in revision_rows]
    aligned = [float(row["source_aligned_loss_delta_mean_mean"]) for row in revision_rows]
    aligned_minus = [float(row["aligned_minus_same_index_loss_delta_mean_mean"]) for row in revision_rows]
    aligned_better = [int(row["aligned_better_count"]) for row in transfer_rows]
    n_pairs = [int(row["n_pairs"]) for row in transfer_rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    axes[0, 0].plot(x, probe, marker="o")
    axes[0, 0].set_title("Repeat-match probe specialization")
    axes[0, 0].set_ylabel("mean selected-head specialization")

    axes[0, 1].plot(x, own_excess, marker="o", color="#2f855a")
    axes[0, 1].set_title("Own selected heads vs random controls")
    axes[0, 1].set_ylabel("loss-delta excess")

    axes[1, 0].plot(x, same, marker="o", label="same index", color="#718096")
    axes[1, 0].plot(x, aligned, marker="o", label="aligned", color="#2b6cb0")
    axes[1, 0].set_title("Source-head transfer")
    axes[1, 0].set_ylabel("loss delta")
    axes[1, 0].legend()

    axes[1, 1].plot(x, aligned_minus, marker="o", color="#c05621")
    axes[1, 1].set_title("Aligned transfer advantage")
    axes[1, 1].set_ylabel("aligned minus same-index")
    for i, (wins, total) in enumerate(zip(aligned_better, n_pairs)):
        axes[1, 1].annotate(f"{wins}/{total}", (x[i], aligned_minus[i]), textcoords="offset points", xytext=(0, 8), ha="center")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "seed9_alignment_selected_trajectory.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    revision_rows = load_combined(args.input_dirs, "revision_summary.csv")
    condition_rows = load_combined(args.input_dirs, "condition_summary.csv")
    transfer_rows = load_combined(args.input_dirs, "transfer_pair_summary.csv")

    write_csv(args.output_dir / "revision_summary.csv", revision_rows)
    write_csv(args.output_dir / "condition_summary.csv", condition_rows)
    write_csv(args.output_dir / "transfer_pair_summary.csv", transfer_rows)
    maybe_write_plot(args.output_dir, revision_rows, transfer_rows)

    payload = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "output_dir": str(args.output_dir),
        "revision_summary": revision_rows,
        "transfer_pair_summary": transfer_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
