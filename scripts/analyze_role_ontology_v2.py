#!/usr/bin/env python3
"""Analyze Toy Ontology v2 head-dimension sweeps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--title", default="Toy Ontology v2")
    return parser.parse_args()


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def load_inputs(results_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((results_dir / "summary.json").read_text())
    model = pd.read_csv(results_dir / "model_summary.csv")
    role = pd.read_csv(results_dir / "role_summary.csv")
    family = pd.read_csv(results_dir / "family_summary.csv")
    config = pd.read_csv(results_dir / "config_summary.csv")
    return summary, model, role, family, config


def model_metric_table(model: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "role_accuracy_mean",
        "role_accuracy_min",
        "role_specialization_mean",
        "role_effective_heads_mean",
        "role_top_dim_mass_mean",
        "family_gap",
        "family_cluster_ari",
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


def adjusted_rand_index(true_labels: list[str], pred_labels: list[int]) -> float:
    def comb2(n: int) -> float:
        return n * (n - 1) / 2.0

    n = len(true_labels)
    if n < 2:
        return 0.0

    contingency = Counter(zip(true_labels, pred_labels))
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    sum_comb = sum(comb2(count) for count in contingency.values())
    sum_true = sum(comb2(count) for count in true_counts.values())
    sum_pred = sum(comb2(count) for count in pred_counts.values())
    total = comb2(n)
    expected = sum_true * sum_pred / total if total else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((sum_comb - expected) / denom)


def distribution_tv_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def cluster_ari(role_names: list[str], role_families: dict[str, str], distributions: dict[str, np.ndarray]) -> float:
    n_roles = len(role_names)
    if n_roles < 2:
        return 0.0
    distance = np.zeros((n_roles, n_roles), dtype=np.float64)
    for i, role_a in enumerate(role_names):
        for j, role_b in enumerate(role_names):
            distance[i, j] = distribution_tv_distance(distributions[role_a], distributions[role_b])
    if np.allclose(distance, 0.0):
        return 0.0
    condensed = squareform(distance, checks=False)
    n_families = len(set(role_families[role] for role in role_names))
    clusters = fcluster(linkage(condensed, method="average"), t=n_families, criterion="maxclust")
    return adjusted_rand_index([role_families[role] for role in role_names], [int(item) for item in clusters])


def dimension_modularity_table(summary: dict, role: pd.DataFrame) -> pd.DataFrame:
    role_families = summary["role_families"]
    rows = []
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

        within = []
        between = []
        for role_a, role_b in combinations(role_names, 2):
            similarity = 1.0 - distribution_tv_distance(distributions[role_a], distributions[role_b])
            if role_families[role_a] == role_families[role_b]:
                within.append(similarity)
            else:
                between.append(similarity)
        rows.append(
            {
                "config": config,
                "seed": int(seed),
                "n_dims": len(dims),
                "dimension_within_family_similarity": float(np.mean(within)) if within else 0.0,
                "dimension_between_family_similarity": float(np.mean(between)) if between else 0.0,
                "dimension_family_gap": (float(np.mean(within)) - float(np.mean(between))) if within and between else 0.0,
                "dimension_family_cluster_ari": cluster_ari(role_names, role_families, distributions),
            }
        )

    aggregate_rows = []
    for config, group in pd.DataFrame(rows).groupby("config", sort=False):
        aggregate_rows.append(
            {
                "config": config,
                "n": len(group),
                "n_dims": int(group["n_dims"].iloc[0]),
                "dimension_family_gap_mean": group["dimension_family_gap"].mean(),
                "dimension_family_gap_std": group["dimension_family_gap"].std(ddof=0),
                "dimension_family_cluster_ari_mean": group["dimension_family_cluster_ari"].mean(),
                "dimension_family_cluster_ari_std": group["dimension_family_cluster_ari"].std(ddof=0),
            }
        )
    return pd.DataFrame(aggregate_rows)


def write_markdown(
    path: Path,
    title: str,
    summary: dict,
    model_table: pd.DataFrame,
    affinity: pd.DataFrame,
    family_table: pd.DataFrame,
    role_counts: pd.DataFrame,
    dimension_modularity: pd.DataFrame,
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
        "## Structural Role Affinity",
        "",
        affinity.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Per-Family Table",
        "",
        family_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Per-Role Top-Dimension Counts",
        "",
        role_counts.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Dimension-Level Family Modularity",
        "",
        dimension_modularity.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    path.write_text("\n".join(lines))


def plot_metric_bars(model_table: pd.DataFrame, output_dir: Path) -> None:
    plot_cols = [
        ("role_specialization_mean_mean", "Mean specialization"),
        ("role_effective_heads_mean_mean", "Effective heads"),
        ("family_gap_mean", "Family gap"),
        ("family_cluster_ari_mean", "ARI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, plot_cols):
        ax.bar(model_table["config"], model_table[col], color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
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

    summary, model, role, family, _ = load_inputs(results_dir)
    model_table = model_metric_table(model)
    affinity = affinity_table(summary, role)
    family_table = family_metric_table(family)
    role_counts = role_affinity_counts(role)
    dimension_modularity = dimension_modularity_table(summary, role)

    model_table.to_csv(output_dir / "model_metric_table.csv", index=False)
    affinity.to_csv(output_dir / "affinity_table.csv", index=False)
    family_table.to_csv(output_dir / "family_metric_table.csv", index=False)
    role_counts.to_csv(output_dir / "role_top_dim_counts.csv", index=False)
    dimension_modularity.to_csv(output_dir / "dimension_modularity_table.csv", index=False)
    write_markdown(
        output_dir / "analysis_report.md",
        args.title,
        summary,
        model_table,
        affinity,
        family_table,
        role_counts,
        dimension_modularity,
    )
    plot_metric_bars(model_table, output_dir)
    plot_family_heatmap(family_table, output_dir)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
