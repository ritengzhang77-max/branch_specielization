#!/usr/bin/env python3
"""Compare ontology choices for Toy Ontology v2 modularity analysis.

This script is deliberately post-training: it does not change the trained toy
models. It asks whether the mixed modularity result is caused by the original
family labels being too coarse, and whether heterogeneous heads show cleaner
role geometry under defensible alternative ontologies or label-free clustering.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


TASK_PRIMITIVE_3 = {
    "local_copy": "direct_pointer",
    "previous_token": "direct_pointer",
    "kv_lookup": "direct_pointer",
    "duplicate_token": "direct_pointer",
    "bos_sink": "direct_pointer",
    "sep_sink": "direct_pointer",
    "fixed_offset_prev": "direct_pointer",
    "punctuation_boundary": "direct_pointer",
    "repeated_name_detection": "direct_pointer",
    "pronoun_antecedent": "direct_pointer",
    "induction_short": "sequence_repeat",
    "induction_long": "sequence_repeat",
    "induction_ngram": "sequence_repeat",
    "false_induction_control": "sequence_repeat",
    "anti_copy": "sequence_repeat",
    "distractor_suppression": "conflict_choice",
    "wrong_key_suppression": "conflict_choice",
    "recency_conflict": "conflict_choice",
    "simple_ioi_name_mover": "conflict_choice",
    "negative_name_control": "conflict_choice",
}


MECHANISM_GROUP = {
    "local_copy": "local_offset",
    "previous_token": "local_offset",
    "fixed_offset_prev": "local_offset",
    "kv_lookup": "key_value_lookup",
    "wrong_key_suppression": "key_value_lookup",
    "recency_conflict": "key_value_lookup",
    "duplicate_token": "repeat_detection",
    "repeated_name_detection": "repeat_detection",
    "induction_short": "induction",
    "induction_long": "induction",
    "induction_ngram": "induction",
    "false_induction_control": "induction",
    "anti_copy": "induction",
    "bos_sink": "boundary_anchor",
    "sep_sink": "boundary_anchor",
    "punctuation_boundary": "boundary_anchor",
    "distractor_suppression": "suppression_conflict",
    "pronoun_antecedent": "entity_coreference",
    "simple_ioi_name_mover": "entity_coreference",
    "negative_name_control": "entity_coreference",
}


MECHANISM_ATTRIBUTES = {
    "local_copy": {"local_offset", "copy"},
    "previous_token": {"local_offset", "copy"},
    "fixed_offset_prev": {"local_offset", "copy"},
    "kv_lookup": {"key_value_lookup", "copy"},
    "wrong_key_suppression": {"key_value_lookup", "conflict"},
    "recency_conflict": {"key_value_lookup", "conflict", "temporal_choice"},
    "duplicate_token": {"repeat_detection", "copy"},
    "repeated_name_detection": {"repeat_detection", "entity"},
    "induction_short": {"induction", "repeat_pattern", "copy"},
    "induction_long": {"induction", "repeat_pattern", "copy"},
    "induction_ngram": {"induction", "repeat_pattern", "copy"},
    "false_induction_control": {"induction", "repeat_pattern", "conflict"},
    "anti_copy": {"induction", "repeat_pattern", "conflict"},
    "bos_sink": {"boundary_anchor", "first_token"},
    "sep_sink": {"boundary_anchor", "marker_lookup", "copy"},
    "punctuation_boundary": {"boundary_anchor", "marker_lookup", "copy"},
    "distractor_suppression": {"suppression", "conflict"},
    "pronoun_antecedent": {"entity", "antecedent_lookup", "copy"},
    "simple_ioi_name_mover": {"entity", "name_mover", "conflict", "copy"},
    "negative_name_control": {"entity", "name_mover", "conflict"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--title", default="Toy Ontology Refinement")
    parser.add_argument("--permutations", type=int, default=1000)
    return parser.parse_args()


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(set(first.tolist())) < 2 or len(set(second.tolist())) < 2:
        return 0.0
    value = spearmanr(first, second).correlation
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def stable_seed(config: str, seed: int, candidate: str) -> int:
    text = f"{config}:{seed}:{candidate}"
    return 8675309 + sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


def build_original_family_mapping(pair: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for _, item in pair.iterrows():
        mapping[str(item["role_a"])] = str(item["family_a"])
        mapping[str(item["role_b"])] = str(item["family_b"])
    return mapping


def candidate_payloads(pair: pd.DataFrame) -> dict[str, dict[str, object]]:
    original = build_original_family_mapping(pair)
    roles = set(original)
    candidates = {
        "original_family": {
            "kind": "single_label",
            "description": "The predeclared family labels from the result file.",
            "mapping": original,
        }
    }
    if roles <= set(TASK_PRIMITIVE_3):
        candidates.update(
            {
                "task_primitive_3way": {
                    "kind": "single_label",
                    "description": "Coarser task primitives: direct pointer, repeated sequence, conflict choice.",
                    "mapping": TASK_PRIMITIVE_3,
                },
                "mechanism_group": {
                    "kind": "single_label",
                    "description": "Finer task-semantics groups defined from the synthetic scene generator.",
                    "mapping": MECHANISM_GROUP,
                },
                "mechanism_multilabel": {
                    "kind": "multilabel",
                    "description": "Multi-label task attributes with Jaccard ontology similarity.",
                    "mapping": MECHANISM_ATTRIBUTES,
                },
            }
        )
    return candidates


def ontology_similarity_for_pair(candidate: dict[str, object], role_a: str, role_b: str) -> float:
    mapping = candidate["mapping"]
    if candidate["kind"] == "single_label":
        return 1.0 if mapping[role_a] == mapping[role_b] else 0.0
    if candidate["kind"] == "multilabel":
        return jaccard(set(mapping[role_a]), set(mapping[role_b]))
    raise ValueError(f"Unknown candidate kind: {candidate['kind']}")


def shuffled_similarity_values(
    candidate: dict[str, object],
    roles: list[str],
    role_a: pd.Series,
    role_b: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    mapping = candidate["mapping"]
    shuffled_roles = roles.copy()
    rng.shuffle(shuffled_roles)
    shuffled_mapping = {role: mapping[assigned] for role, assigned in zip(roles, shuffled_roles)}
    shuffled_candidate = {**candidate, "mapping": shuffled_mapping}
    return np.array(
        [
            ontology_similarity_for_pair(shuffled_candidate, str(first), str(second))
            for first, second in zip(role_a, role_b)
        ],
        dtype=np.float64,
    )


def analyze_candidate(
    group: pd.DataFrame,
    candidate_name: str,
    candidate: dict[str, object],
    permutations: int,
) -> dict[str, float | int | str]:
    roles = sorted(set(group["role_a"]) | set(group["role_b"]))
    head_similarity = group["similarity"].astype(float).to_numpy()
    ontology_similarity = np.array(
        [
            ontology_similarity_for_pair(candidate, str(first), str(second))
            for first, second in zip(group["role_a"], group["role_b"])
        ],
        dtype=np.float64,
    )
    positive = ontology_similarity > 0
    high_gap = (
        float(head_similarity[positive].mean() - head_similarity[~positive].mean())
        if positive.any() and (~positive).any()
        else 0.0
    )
    exact_gap = (
        float(head_similarity[ontology_similarity >= 1.0].mean() - head_similarity[ontology_similarity <= 0.0].mean())
        if (ontology_similarity >= 1.0).any() and (ontology_similarity <= 0.0).any()
        else 0.0
    )
    observed = safe_spearman(ontology_similarity, head_similarity)

    rng = np.random.default_rng(stable_seed(str(group["config"].iloc[0]), int(group["seed"].iloc[0]), candidate_name))
    null_scores = []
    for _ in range(permutations):
        shuffled = shuffled_similarity_values(candidate, roles, group["role_a"], group["role_b"], rng)
        null_scores.append(safe_spearman(shuffled, head_similarity))
    null = np.array(null_scores, dtype=np.float64)
    null_mean = float(null.mean()) if len(null) else 0.0
    null_std = float(null.std(ddof=0)) if len(null) else 0.0
    z_score = (observed - null_mean) / null_std if null_std > 1e-12 else 0.0
    p_ge = float((1 + np.sum(null >= observed)) / (len(null) + 1)) if len(null) else 1.0

    return {
        "candidate": candidate_name,
        "kind": str(candidate["kind"]),
        "n_pairs": int(len(group)),
        "positive_pair_rate": float(positive.mean()),
        "ontology_alignment": observed,
        "ontology_null_mean": null_mean,
        "ontology_z": float(z_score),
        "ontology_p_ge": p_ge,
        "positive_vs_zero_gap": high_gap,
        "exact_one_vs_zero_gap": exact_gap,
    }


def distance_matrix(group: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    roles = sorted(set(group["role_a"]) | set(group["role_b"]))
    index = {role: idx for idx, role in enumerate(roles)}
    distance = np.zeros((len(roles), len(roles)), dtype=np.float64)
    for _, item in group.iterrows():
        i = index[str(item["role_a"])]
        j = index[str(item["role_b"])]
        distance[i, j] = distance[j, i] = float(item["tv_distance"])
    return roles, distance


def silhouette_score(distance: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for idx, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != idx]
        a_value = float(distance[idx, same].mean()) if len(same) else 0.0
        b_values = []
        for other_label in sorted(set(labels)):
            if other_label == label:
                continue
            other = np.where(labels == other_label)[0]
            if len(other):
                b_values.append(float(distance[idx, other].mean()))
        b_value = min(b_values) if b_values else 0.0
        denom = max(a_value, b_value)
        values.append((b_value - a_value) / denom if denom > 1e-12 else 0.0)
    return float(np.mean(values))


def cluster_contrast(distance: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    within = []
    between = []
    for i, j in combinations(range(len(labels)), 2):
        if labels[i] == labels[j]:
            within.append(distance[i, j])
        else:
            between.append(distance[i, j])
    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    return within_mean, between_mean, between_mean - within_mean


def analyze_clusterability(group: pd.DataFrame) -> dict[str, float | int]:
    _, distance = distance_matrix(group)
    if len(distance) < 3 or np.allclose(distance, 0.0):
        return {
            "mean_pair_tv": float(group["tv_distance"].mean()) if len(group) else 0.0,
            "silhouette_k3": 0.0,
            "silhouette_k5": 0.0,
            "best_silhouette": 0.0,
            "best_k": 0,
            "contrast_k5": 0.0,
            "clusterability_x_separation_k5": 0.0,
        }
    tree = linkage(squareform(distance, checks=False), method="average")
    best_score = -1.0
    best_k = 0
    silhouette_by_k = {}
    contrast_k5 = 0.0
    for k_value in range(2, min(8, len(distance) - 1) + 1):
        labels = fcluster(tree, t=k_value, criterion="maxclust")
        silhouette = silhouette_score(distance, labels)
        silhouette_by_k[k_value] = silhouette
        if k_value == 5:
            _, _, contrast_k5 = cluster_contrast(distance, labels)
        if silhouette > best_score:
            best_score = silhouette
            best_k = k_value
    mean_pair_tv = float(group["tv_distance"].mean())
    silhouette_k5 = float(silhouette_by_k.get(5, 0.0))
    return {
        "mean_pair_tv": mean_pair_tv,
        "silhouette_k3": float(silhouette_by_k.get(3, 0.0)),
        "silhouette_k5": silhouette_k5,
        "best_silhouette": float(best_score),
        "best_k": int(best_k),
        "contrast_k5": float(contrast_k5),
        "clusterability_x_separation_k5": float(silhouette_k5 * mean_pair_tv),
    }


def aggregate(frame: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        col
        for col in frame.columns
        if col not in index_cols + ["seed", "kind"] and pd.api.types.is_numeric_dtype(frame[col])
    ]
    rows = []
    for keys, group in frame.groupby(index_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(index_cols, keys)}
        row["n"] = len(group)
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def role_neighbor_table(pair: pd.DataFrame, candidate_name: str, candidate: dict[str, object]) -> pd.DataFrame:
    rows = []
    for _, item in pair.iterrows():
        ontology_similarity = ontology_similarity_for_pair(candidate, str(item["role_a"]), str(item["role_b"]))
        for side in [("role_a", "family_a", "role_b", "family_b"), ("role_b", "family_b", "role_a", "family_a")]:
            role_key, family_key, other_key, other_family_key = side
            rows.append(
                {
                    "config": item["config"],
                    "seed": int(item["seed"]),
                    "candidate": candidate_name,
                    "role": item[role_key],
                    "original_family": item[family_key],
                    "other_role": item[other_key],
                    "other_original_family": item[other_family_key],
                    "ontology_similarity": ontology_similarity,
                    "head_similarity": float(item["similarity"]),
                }
            )
    long = pd.DataFrame(rows)
    out_rows = []
    for (config, role, original_family), group in long.groupby(["config", "role", "original_family"], sort=False):
        positive = group[group["ontology_similarity"] > 0]
        zero = group[group["ontology_similarity"] <= 0]
        top = group.sort_values("head_similarity", ascending=False).head(3)
        out_rows.append(
            {
                "config": config,
                "candidate": candidate_name,
                "role": role,
                "original_family": original_family,
                "positive_neighbor_similarity": positive["head_similarity"].mean() if len(positive) else 0.0,
                "zero_neighbor_similarity": zero["head_similarity"].mean() if len(zero) else 0.0,
                "neighbor_margin": (
                    positive["head_similarity"].mean() - zero["head_similarity"].mean()
                    if len(positive) and len(zero)
                    else 0.0
                ),
                "top1_positive_rate": float(top.iloc[0]["ontology_similarity"] > 0) if len(top) else 0.0,
                "top3_positive_rate": float((top["ontology_similarity"] > 0).mean()) if len(top) else 0.0,
            }
        )
    return pd.DataFrame(out_rows)


def write_report(
    output_dir: Path,
    title: str,
    results_dir: Path,
    candidate_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    role_neighbors: pd.DataFrame,
    candidates: dict[str, dict[str, object]],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Result directory: `{results_dir}`",
        "",
        "## Candidate Ontologies",
        "",
    ]
    for name, payload in candidates.items():
        lines.extend([f"- `{name}`: {payload['description']}", ""])
    lines.extend(
        [
            "## Ontology Alignment Summary",
            "",
            candidate_summary.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Label-Free Clusterability Summary",
            "",
            cluster_summary.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Per-Role Neighbor Margins",
            "",
            role_neighbors.to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    )
    (output_dir / "ontology_refinement_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "analysis" / "ontology_refinement"
    output_dir.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(results_dir / "role_pair_summary.csv")
    candidates = candidate_payloads(pair)

    per_seed_rows = []
    cluster_rows = []
    role_neighbor_frames = []
    for (config, seed), group in pair.groupby(["config", "seed"], sort=False):
        for candidate_name, candidate in candidates.items():
            row = {"config": config, "seed": int(seed)}
            row.update(analyze_candidate(group, candidate_name, candidate, args.permutations))
            per_seed_rows.append(row)
        cluster_row = {"config": config, "seed": int(seed)}
        cluster_row.update(analyze_clusterability(group))
        cluster_rows.append(cluster_row)

    for candidate_name, candidate in candidates.items():
        role_neighbor_frames.append(role_neighbor_table(pair, candidate_name, candidate))

    per_seed = pd.DataFrame(per_seed_rows)
    candidate_summary = aggregate(per_seed, ["config", "candidate"])
    cluster_per_seed = pd.DataFrame(cluster_rows)
    cluster_summary = aggregate(cluster_per_seed, ["config"])
    role_neighbors = pd.concat(role_neighbor_frames, ignore_index=True)

    per_seed.to_csv(output_dir / "ontology_candidate_per_seed.csv", index=False)
    candidate_summary.to_csv(output_dir / "ontology_candidate_summary.csv", index=False)
    cluster_per_seed.to_csv(output_dir / "label_free_clusterability_per_seed.csv", index=False)
    cluster_summary.to_csv(output_dir / "label_free_clusterability_summary.csv", index=False)
    role_neighbors.to_csv(output_dir / "role_candidate_neighbor_table.csv", index=False)
    write_report(output_dir, args.title, results_dir, candidate_summary, cluster_summary, role_neighbors, candidates)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
