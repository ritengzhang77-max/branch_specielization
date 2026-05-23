#!/usr/bin/env python3
"""Compute paired transfer significance summaries for alignment runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def exact_two_sided_sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    n = int(nonzero.shape[0])
    if n == 0:
        return 1.0
    positive = int(np.sum(nonzero > 0))
    smaller_tail = min(positive, n - positive)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(smaller_tail + 1)) / (2**n))


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_values(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    prefix: str,
) -> dict[str, object]:
    ci_low, ci_high = bootstrap_ci(values, n_bootstrap, rng)
    return {
        f"{prefix}_n": int(values.shape[0]),
        f"{prefix}_mean": float(values.mean()) if values.size else float("nan"),
        f"{prefix}_std": float(values.std()) if values.size else float("nan"),
        f"{prefix}_bootstrap_ci_low": ci_low,
        f"{prefix}_bootstrap_ci_high": ci_high,
        f"{prefix}_positive_count": int(np.sum(values > 0)),
        f"{prefix}_sign_p_two_sided": exact_two_sided_sign_p(values),
    }


def pair_diffs(ablation_rows: list[dict[str, str]]) -> np.ndarray:
    by_pair: dict[tuple[str, str], dict[str, float]] = {}
    for row in ablation_rows:
        condition = row["condition"]
        if condition not in {"source_same_index", "source_aligned"}:
            continue
        key = (row["target_seed"], row["source_seed"])
        by_pair.setdefault(key, {})[condition] = float(row["loss_delta"])
    values = []
    for item in by_pair.values():
        if "source_same_index" in item and "source_aligned" in item:
            values.append(item["source_aligned"] - item["source_same_index"])
    return np.asarray(values, dtype=np.float64)


def target_diffs(seed_rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [float(row["aligned_minus_same_index_loss_delta_mean"]) for row in seed_rows],
        dtype=np.float64,
    )


def own_excess(seed_rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [float(row["own_top_excess_over_random"]) for row in seed_rows],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ablation_rows = read_csv(args.input_dir / "ablation_results.csv")
    seed_rows = read_csv(args.input_dir / "revision_seed_summary.csv")
    pair_values = pair_diffs(ablation_rows)
    target_values = target_diffs(seed_rows)
    own_values = own_excess(seed_rows)
    summary = {
        "input_dir": str(args.input_dir),
        "revision": seed_rows[0]["revision"] if seed_rows else "",
        "model_size": seed_rows[0]["model_size"] if seed_rows else "",
    }
    summary |= summarize_values(pair_values, args.n_bootstrap, rng, "pair_aligned_minus_same")
    summary |= summarize_values(target_values, args.n_bootstrap, rng, "target_aligned_minus_same")
    summary |= summarize_values(own_values, args.n_bootstrap, rng, "target_own_excess")

    csv_path = args.input_dir / "transfer_significance_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    (args.input_dir / "transfer_significance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
