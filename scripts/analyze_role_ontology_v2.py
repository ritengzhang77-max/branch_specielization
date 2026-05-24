#!/usr/bin/env python3
"""Analyze Toy Ontology v2 head-dimension sweeps."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--title", default="Toy Ontology v2")
    return parser.parse_args()


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def load_inputs(results_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((results_dir / "summary.json").read_text())
    model = pd.read_csv(results_dir / "model_summary.csv")
    role = pd.read_csv(results_dir / "role_summary.csv")
    family = pd.read_csv(results_dir / "family_summary.csv")
    config = pd.read_csv(results_dir / "config_summary.csv")
    pair_path = results_dir / "role_pair_summary.csv"
    pair = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    return summary, model, role, family, config, pair


def model_metric_table(model: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "role_accuracy_mean",
        "role_accuracy_min",
        "role_specialization_mean",
        "role_effective_heads_mean",
        "role_top_dim_mass_mean",
        "family_gap",
    ]
    rows = []
    for config, group in model.groupby("config", sort=False):
        row = {"config": config, "n": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def affinity_table(summary: dict, role: pd.DataFrame) -> pd.DataFrame:
    head_dims = summary["head_dims_by_config"]
    rows = []
    for config, group in role.groupby("config", sort=False):
        dims = head_dims[config]
        max_dim = max(dims)
        min_dim = min(dims)
        is_uniform = len(set(dims)) == 1
        rows.append(
            {
                "config": config,
                "head_dims": str(dims),
                "n_role_rows": len(group),
                "uniform": is_uniform,
                "largest_dim": max_dim,
                "smallest_dim": min_dim,
                "largest_dim_top_rate": (group["top_dim"] == max_dim).mean(),
                "smallest_dim_top_rate": (group["top_dim"] == min_dim).mean(),
                "mean_top_dim_mass": group["top_dim_mass"].mean(),
            }
        )
    return pd.DataFrame(rows)


def family_metric_table(family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, fam), group in family.groupby(["config", "family"], sort=False):
        rows.append(
            {
                "config": config,
                "family": fam,
                "accuracy_mean": group["accuracy_mean"].mean(),
                "specialization_mean": group["specialization_mean"].mean(),
                "effective_heads_mean": group["effective_heads_mean"].mean(),
                "top_dim_mass_mean": group["top_dim_mass_mean"].mean(),
                "within_family_similarity_mean": group["within_family_similarity_mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def role_affinity_counts(role: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, role_name, family), group in role.groupby(["config", "role", "family"], sort=False):
        counts = group["top_dim"].value_counts().sort_index()
        rows.append(
            {
                "config": config,
                "family": family,
                "role": role_name,
                "n": len(group),
                "top_dim_counts": ", ".join(f"{int(dim)}:{int(count)}" for dim, count in counts.items()),
                "top_dim_mode": int(counts.idxmax()),
                "top_dim_mode_count": int(counts.max()),
                "accuracy_mean": group["accuracy"].mean(),
                "specialization_mean": group["global_top_specialization"].mean(),
                "effective_heads_mean": group["effective_heads"].mean(),
            }
        )
    return pd.DataFrame(rows)


def distribution_tv_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(set(first.tolist())) < 2 or len(set(second.tolist())) < 2:
        return 0.0
    value = spearmanr(first, second).correlation
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def alignment_stats_from_pairs(
    group: pd.DataFrame,
    *,
    rng_seed: int,
    n_permutations: int = 1000,
) -> dict:
    """Compare head-similarity geometry against the predefined ontology labels."""

    if group.empty:
        return {
            "n_pairs": 0,
            "within_family_similarity": 0.0,
            "between_family_similarity": 0.0,
            "family_gap": 0.0,
            "ontology_alignment_spearman": 0.0,
            "ontology_alignment_null_mean": 0.0,
            "ontology_alignment_null_std": 0.0,
            "ontology_alignment_z": 0.0,
            "ontology_alignment_p_ge": 1.0,
        }

    role_families: dict[str, str] = {}
    for _, item in group.iterrows():
        role_families[str(item["role_a"])] = str(item["family_a"])
        role_families[str(item["role_b"])] = str(item["family_b"])
    role_names = sorted(role_families)
    role_index = {role: idx for idx, role in enumerate(role_names)}
    family_labels = np.array([role_families[role] for role in role_names], dtype=object)
    similarities = group["similarity"].astype(float).to_numpy()
    first_idx = group["role_a"].map(lambda role: role_index[str(role)]).to_numpy()
    second_idx = group["role_b"].map(lambda role: role_index[str(role)]).to_numpy()
    same = (family_labels[first_idx] == family_labels[second_idx]).astype(float)
    same_bool = same.astype(bool)

    within = similarities[same_bool]
    between = similarities[~same_bool]
    within_mean = float(np.mean(within)) if len(within) else 0.0
    between_mean = float(np.mean(between)) if len(between) else 0.0
    family_gap = within_mean - between_mean
    ontology_alignment = safe_spearman(same, similarities)

    rng = np.random.default_rng(rng_seed)
    null_scores = []
    for _ in range(n_permutations):
        shuffled = family_labels.copy()
        rng.shuffle(shuffled)
        shuffled_same = (shuffled[first_idx] == shuffled[second_idx]).astype(float)
        null_scores.append(safe_spearman(shuffled_same, similarities))
    null = np.array(null_scores, dtype=np.float64)
    null_mean = float(null.mean()) if len(null) else 0.0
    null_std = float(null.std(ddof=0)) if len(null) else 0.0
    z = (ontology_alignment - null_mean) / null_std if null_std > 1e-12 else 0.0
    p_ge = float((1 + np.sum(null >= ontology_alignment)) / (len(null) + 1)) if len(null) else 1.0

    return {
        "n_pairs": int(len(group)),
        "within_family_similarity": within_mean,
        "between_family_similarity": between_mean,
        "family_gap": family_gap,
        "ontology_alignment_spearman": ontology_alignment,
        "ontology_alignment_null_mean": null_mean,
        "ontology_alignment_null_std": null_std,
        "ontology_alignment_z": float(z),
        "ontology_alignment_p_ge": p_ge,
    }


def stable_seed(config: str, seed: int, offset: int = 0) -> int:
    return 1729 + offset + int(seed) * 1009 + sum((idx + 1) * ord(char) for idx, char in enumerate(config))


def aggregate_alignment(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    aggregate_rows = []
    metrics = [
        "within_family_similarity",
        "between_family_similarity",
        "family_gap",
        "ontology_alignment_spearman",
        "ontology_alignment_null_mean",
        "ontology_alignment_z",
        "ontology_alignment_p_ge",
    ]
    for config, group in frame.groupby("config", sort=False):
        row = {"config": config, "n": len(group), "n_pairs": int(group["n_pairs"].iloc[0])}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows)


def ontology_alignment_tables(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (config, seed), group in pair.groupby(["config", "seed"], sort=False):
        row = {"config": config, "seed": int(seed)}
        row.update(alignment_stats_from_pairs(group, rng_seed=stable_seed(str(config), int(seed))))
        rows.append(row)
    return pd.DataFrame(rows), aggregate_alignment(rows)


def pairs_from_dimension_distributions(summary: dict, role: pd.DataFrame) -> pd.DataFrame:
    rows = []
    role_families = summary["role_families"]
    for (config, seed), group in role.groupby(["config", "seed"], sort=False):
        dims = sorted(set(summary["head_dims_by_config"][config]))
        dim_index = {int(dim): idx for idx, dim in enumerate(dims)}
        distributions = {}
        role_names = []
        for _, item in group.iterrows():
            role_name = str(item["role"])
            role_names.append(role_name)
            affinity = json.loads(item["head_dim_affinity_json"])
            vector = np.zeros(len(dims), dtype=np.float64)
            for dim, mass in affinity.items():
                vector[dim_index[int(dim)]] = float(mass)
            total = vector.sum()
            if total > 1e-12:
                vector = vector / total
            distributions[role_name] = vector
        for role_a, role_b in combinations(role_names, 2):
            tv_distance = distribution_tv_distance(distributions[role_a], distributions[role_b])
            rows.append(
                {
                    "config": config,
                    "seed": int(seed),
                    "role_set": str(group["role_set"].iloc[0]),
                    "role_a": role_a,
                    "family_a": role_families[role_a],
                    "role_b": role_b,
                    "family_b": role_families[role_b],
                    "same_family": role_families[role_a] == role_families[role_b],
                    "tv_distance": tv_distance,
                    "similarity": 1.0 - tv_distance,
                }
            )
    return pd.DataFrame(rows)


def role_neighbor_alignment_table(role: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pd.DataFrame()
    long_rows = []
    for _, item in pair.iterrows():
        same_family = bool_series(pd.Series([item["same_family"]])).iloc[0]
        long_rows.append(
            {
                "config": item["config"],
                "seed": int(item["seed"]),
                "role": item["role_a"],
                "family": item["family_a"],
                "other_role": item["role_b"],
                "other_family": item["family_b"],
                "same_family": same_family,
                "similarity": float(item["similarity"]),
            }
        )
        long_rows.append(
            {
                "config": item["config"],
                "seed": int(item["seed"]),
                "role": item["role_b"],
                "family": item["family_b"],
                "other_role": item["role_a"],
                "other_family": item["family_a"],
                "same_family": same_family,
                "similarity": float(item["similarity"]),
            }
        )

    neighbor = pd.DataFrame(long_rows)
    per_seed_rows = []
    for (config, seed, role_name, family), group in neighbor.groupby(["config", "seed", "role", "family"], sort=False):
        same = group[group["same_family"]]
        different = group[~group["same_family"]]
        top = group.sort_values("similarity", ascending=False).head(3)
        nearest = top.iloc[0]
        per_seed_rows.append(
            {
                "config": config,
                "seed": int(seed),
                "family": family,
                "role": role_name,
                "same_family_neighbor_similarity": same["similarity"].mean(),
                "different_family_neighbor_similarity": different["similarity"].mean(),
                "ontology_neighbor_margin": same["similarity"].mean() - different["similarity"].mean(),
                "top1_same_family": bool(nearest["same_family"]),
                "top3_same_family_rate": top["same_family"].mean(),
            }
        )
    per_seed = pd.DataFrame(per_seed_rows)

    role_metrics = role[
        [
            "config",
            "seed",
            "role",
            "accuracy",
            "global_top_specialization",
            "effective_heads",
            "top_dim",
            "top_dim_mass",
        ]
    ].copy()
    per_seed = per_seed.merge(role_metrics, on=["config", "seed", "role"], how="left")

    aggregate_rows = []
    for (config, family, role_name), group in per_seed.groupby(["config", "family", "role"], sort=False):
        counts = group["top_dim"].value_counts().sort_index()
        aggregate_rows.append(
            {
                "config": config,
                "family": family,
                "role": role_name,
                "n": len(group),
                "accuracy_mean": group["accuracy"].mean(),
                "specialization_mean": group["global_top_specialization"].mean(),
                "effective_heads_mean": group["effective_heads"].mean(),
                "top_dim_counts": ", ".join(f"{int(dim)}:{int(count)}" for dim, count in counts.items()),
                "same_family_neighbor_similarity_mean": group["same_family_neighbor_similarity"].mean(),
                "different_family_neighbor_similarity_mean": group["different_family_neighbor_similarity"].mean(),
                "ontology_neighbor_margin_mean": group["ontology_neighbor_margin"].mean(),
                "top1_same_family_rate": group["top1_same_family"].mean(),
                "top3_same_family_rate": group["top3_same_family_rate"].mean(),
            }
        )
    return pd.DataFrame(aggregate_rows)


def dimension_ontology_alignment_tables(summary: dict, role: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dimension_pairs = pairs_from_dimension_distributions(summary, role)
    rows = []
    for (config, seed), group in dimension_pairs.groupby(["config", "seed"], sort=False):
        n_dims = len(set(summary["head_dims_by_config"][config]))
        row = {"config": config, "seed": int(seed), "n_dims": n_dims}
        row.update(alignment_stats_from_pairs(group, rng_seed=stable_seed(str(config), int(seed), offset=50000)))
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    aggregate = aggregate_alignment(rows)
    if not aggregate.empty:
        n_dims_by_config = per_seed.groupby("config", sort=False)["n_dims"].first().to_dict()
        aggregate.insert(2, "n_dims", aggregate["config"].map(n_dims_by_config))
    return per_seed, aggregate


def write_markdown(
    path: Path,
    title: str,
    summary: dict,
    model_table: pd.DataFrame,
    affinity: pd.DataFrame,
    ontology_alignment: pd.DataFrame,
    role_neighbor_alignment: pd.DataFrame,
    family_table: pd.DataFrame,
    role_counts: pd.DataFrame,
    dimension_ontology_alignment: pd.DataFrame,
) -> None:
    args = summary["args"]
    lines = [
        f"# {title} Analysis",
        "",
        "## Setting",
        "",
        f"- Result directory: `{args['output_dir']}`",
        f"- Role set: `{args['role_set']}`",
        f"- Roles: `{len(summary['role_names'])}`",
        f"- Families: `{len(set(summary['role_families'].values()))}`",
        f"- Sequence length: `{summary['layout']['seq_len']}`",
        f"- Seeds: `{args['seeds']}`",
        f"- Steps: `{args['steps']}`",
        f"- Eval examples: `{args['eval_examples']}`",
        "",
        "## Baseline-To-Experiment Table",
        "",
        model_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Ontology Alignment Over Exact Attention Heads",
        "",
        "This replaces ARI as the main modularity diagnostic. It tests whether same-family role pairs are closer in head-usage geometry than shuffled ontology labels.",
        "",
        ontology_alignment.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Structural Role Affinity",
        "",
        affinity.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Per-Role Ontology Neighbor Margins",
        "",
        role_neighbor_alignment.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Per-Family Table",
        "",
        family_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Per-Role Top-Dimension Counts",
        "",
        role_counts.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Dimension-Level Ontology Alignment",
        "",
        dimension_ontology_alignment.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    path.write_text("\n".join(lines))


def plot_metric_bars(model_table: pd.DataFrame, ontology_alignment: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = model_table.merge(
        ontology_alignment[["config", "ontology_alignment_spearman_mean"]],
        on="config",
        how="left",
    )
    plot_cols = [
        ("role_accuracy_min_mean", "Minimum role accuracy"),
        ("role_specialization_mean_mean", "Mean specialization"),
        ("role_effective_heads_mean_mean", "Effective heads"),
        ("family_gap_mean", "Family gap"),
        ("ontology_alignment_spearman_mean", "Ontology alignment"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, plot_cols):
        ax.bar(plot_frame["config"], plot_frame[col], color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(plot_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "config_metric_bars.png", dpi=180)
    plt.close(fig)


def plot_family_heatmap(family_table: pd.DataFrame, output_dir: Path) -> None:
    pivot = family_table.pivot(index="family", columns="config", values="specialization_mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Mean specialization by family and config")
    fig.colorbar(image, ax=ax, label="specialization")
    fig.tight_layout()
    fig.savefig(output_dir / "family_specialization_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, model, role, family, _, pair = load_inputs(results_dir)
    model_table = model_metric_table(model)
    affinity = affinity_table(summary, role)
    family_table = family_metric_table(family)
    role_counts = role_affinity_counts(role)
    ontology_alignment_per_seed, ontology_alignment = ontology_alignment_tables(pair)
    role_neighbor_alignment = role_neighbor_alignment_table(role, pair)
    dimension_ontology_alignment_per_seed, dimension_ontology_alignment = dimension_ontology_alignment_tables(summary, role)

    model_table.to_csv(output_dir / "model_metric_table.csv", index=False)
    affinity.to_csv(output_dir / "affinity_table.csv", index=False)
    ontology_alignment_per_seed.to_csv(output_dir / "ontology_alignment_per_seed.csv", index=False)
    ontology_alignment.to_csv(output_dir / "ontology_alignment_table.csv", index=False)
    role_neighbor_alignment.to_csv(output_dir / "role_ontology_neighbor_table.csv", index=False)
    family_table.to_csv(output_dir / "family_metric_table.csv", index=False)
    role_counts.to_csv(output_dir / "role_top_dim_counts.csv", index=False)
    dimension_ontology_alignment_per_seed.to_csv(output_dir / "dimension_ontology_alignment_per_seed.csv", index=False)
    dimension_ontology_alignment.to_csv(output_dir / "dimension_ontology_alignment_table.csv", index=False)
    write_markdown(
        output_dir / "analysis_report.md",
        args.title,
        summary,
        model_table,
        affinity,
        ontology_alignment,
        role_neighbor_alignment,
        family_table,
        role_counts,
        dimension_ontology_alignment,
    )
    plot_metric_bars(model_table, ontology_alignment, output_dir)
    plot_family_heatmap(family_table, output_dir)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
