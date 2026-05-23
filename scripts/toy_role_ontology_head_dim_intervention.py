#!/usr/bin/env python3
"""Toy role-ontology head-dimension intervention.

This script keeps the unit of analysis as ordinary attention heads. It extends
the local-vs-induction pairwise test into a small role ontology with families
and subroles, then asks whether heterogeneous head dimensions affect:

1. structural role affinity: roles preferring a head type such as 64-dim heads;
2. functional specialization: role mass concentrating into few heads;
3. ontology-level modularity: related subroles having similar head distributions
   and unrelated role families separating.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from toy_head_dim_intervention import (
    TinyTransformer,
    resolve_config,
    resolve_device,
    specialization_distribution,
)


LOCAL_A_SEP = 1
LOCAL_B_SEP = 2
KV_A_SEP = 3
KV_B_SEP = 4
INDUCTION_SHORT_SEP = 5
INDUCTION_LONG_SEP = 6
TOKEN_LOW = 8

ROLE_FAMILIES = {
    "local_a": "local_copy",
    "local_b": "local_copy",
    "kv_a": "kv_lookup",
    "kv_b": "kv_lookup",
    "induction_short": "induction",
    "induction_long": "induction",
}


@dataclass
class ModelSummary:
    config: str
    seed: int
    head_dims_json: str
    eval_loss_mean: float
    role_accuracy_mean: float
    role_accuracy_min: float
    role_specialization_mean: float
    role_effective_heads_mean: float
    role_top_dim_mass_mean: float
    within_family_similarity_mean: float
    between_family_similarity_mean: float
    family_gap: float
    family_cluster_ari: float


@dataclass
class RoleSummary:
    config: str
    seed: int
    role: str
    family: str
    loss: float
    accuracy: float
    top_layer: int
    top_head: int
    top_head_dim: int
    top_role_score: float
    global_top_specialization: float
    effective_heads: float
    top_dim: int
    top_dim_mass: float
    head_dim_affinity_json: str


@dataclass
class HeadRoleScore:
    config: str
    seed: int
    role: str
    family: str
    layer: int
    head: int
    head_dim: int
    role_score: float
    role_mass: float
    ablation_accuracy_delta: float


@dataclass
class RolePairSummary:
    config: str
    seed: int
    role_a: str
    family_a: str
    role_b: str
    family_b: str
    same_family: bool
    tv_distance: float
    similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["uniform4", "hetero4", "hetero4_64first", "hetero4_64second", "hetero4_64third", "uniform2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--kv-pairs", type=int, default=4)
    parser.add_argument("--induction-short-length", type=int, default=8)
    parser.add_argument("--induction-long-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_role_ontology_head_dim"))
    return parser.parse_args()


def sequence_layout(kv_pairs: int, induction_short_length: int, induction_long_length: int) -> dict[str, object]:
    role_positions: dict[str, torch.Tensor] = {}
    cursor = 0

    role_positions["local_a"] = torch.tensor([cursor + 1], dtype=torch.long)
    cursor += 3
    role_positions["local_b"] = torch.tensor([cursor + 1], dtype=torch.long)
    cursor += 3

    role_positions["kv_a"] = torch.tensor([cursor + 1 + 2 * kv_pairs], dtype=torch.long)
    cursor += 1 + 2 * kv_pairs + 2
    role_positions["kv_b"] = torch.tensor([cursor + 1 + 2 * kv_pairs], dtype=torch.long)
    cursor += 1 + 2 * kv_pairs + 2

    short_second_start = cursor + 1 + induction_short_length
    role_positions["induction_short"] = torch.arange(
        short_second_start,
        short_second_start + induction_short_length - 1,
        dtype=torch.long,
    )
    cursor += 1 + 2 * induction_short_length

    long_second_start = cursor + 1 + induction_long_length
    role_positions["induction_long"] = torch.arange(
        long_second_start,
        long_second_start + induction_long_length - 1,
        dtype=torch.long,
    )
    cursor += 1 + 2 * induction_long_length

    return {
        "seq_len": cursor,
        "role_positions": role_positions,
    }


def append_local(row: list[int], rng: np.random.Generator, token_ids: np.ndarray, sep: int) -> None:
    token = int(rng.choice(token_ids))
    row.extend([token, sep, token])


def append_kv_lookup(
    row: list[int],
    rng: np.random.Generator,
    token_ids: np.ndarray,
    sep: int,
    kv_pairs: int,
) -> None:
    keys = rng.choice(token_ids, size=kv_pairs, replace=False)
    values = rng.choice(token_ids, size=kv_pairs, replace=False)
    query_idx = int(rng.integers(0, kv_pairs))
    row.append(sep)
    for key, value in zip(keys, values):
        row.extend([int(key), int(value)])
    row.extend([int(keys[query_idx]), int(values[query_idx])])


def append_induction(
    row: list[int],
    rng: np.random.Generator,
    token_ids: np.ndarray,
    sep: int,
    repeat_length: int,
) -> None:
    base = rng.choice(token_ids, size=repeat_length, replace=False)
    row.append(sep)
    row.extend(int(token) for token in base)
    row.extend(int(token) for token in base)


def make_ontology_dataset(
    rng: np.random.Generator,
    n_examples: int,
    kv_pairs: int,
    induction_short_length: int,
    induction_long_length: int,
    vocab_size: int,
    seq_len: int,
) -> torch.Tensor:
    required_unique = max(kv_pairs, induction_short_length, induction_long_length)
    if vocab_size - TOKEN_LOW < required_unique:
        raise ValueError("vocab_size is too small for the requested ontology task.")

    token_ids = np.arange(TOKEN_LOW, vocab_size)
    rows = []
    for _ in range(n_examples):
        row: list[int] = []
        append_local(row, rng, token_ids, LOCAL_A_SEP)
        append_local(row, rng, token_ids, LOCAL_B_SEP)
        append_kv_lookup(row, rng, token_ids, KV_A_SEP, kv_pairs)
        append_kv_lookup(row, rng, token_ids, KV_B_SEP, kv_pairs)
        append_induction(row, rng, token_ids, INDUCTION_SHORT_SEP, induction_short_length)
        append_induction(row, rng, token_ids, INDUCTION_LONG_SEP, induction_long_length)
        if len(row) != seq_len:
            raise RuntimeError(f"Bad generated row length: {len(row)} != {seq_len}")
        rows.append(row)
    return torch.tensor(np.asarray(rows), dtype=torch.long)


def objective_loss_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    positions = positions.to(logits.device)
    selected_logits = logits[:, positions, :]
    targets = input_ids[:, positions + 1]
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    accuracy = float((selected_logits.argmax(dim=-1) == targets).float().mean().detach().cpu())
    return loss, accuracy


def combined_loss_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    layout: dict[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    role_positions: dict[str, torch.Tensor] = layout["role_positions"]  # type: ignore[assignment]
    losses = []
    metrics = {}
    for role, positions in role_positions.items():
        role_loss, role_accuracy = objective_loss_accuracy(logits, input_ids, positions)
        losses.append(role_loss)
        metrics[f"{role}_loss"] = float(role_loss.detach().cpu())
        metrics[f"{role}_accuracy"] = role_accuracy
    loss = torch.stack(losses).mean()
    metrics["loss_mean"] = float(loss.detach().cpu())
    metrics["accuracy_mean"] = float(np.mean([metrics[f"{role}_accuracy"] for role in role_positions]))
    return loss, metrics


def iter_batches(input_ids: torch.Tensor, batch_size: int):
    for start in range(0, input_ids.shape[0], batch_size):
        yield input_ids[start : start + batch_size]


def train_model(
    model: TinyTransformer,
    train_rng: np.random.Generator,
    args: argparse.Namespace,
    layout: dict[str, object],
    device: torch.device,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    seq_len = int(layout["seq_len"])
    for step in range(args.steps):
        input_ids = make_ontology_dataset(
            train_rng,
            args.batch_size,
            args.kv_pairs,
            args.induction_short_length,
            args.induction_long_length,
            args.vocab_size,
            seq_len,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(input_ids)
        loss, _ = combined_loss_accuracy(logits, input_ids, layout)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"  step={step + 1} train_loss={last_loss:.4f}", flush=True)
    return last_loss


def evaluate(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, object],
    batch_size: int,
    device: torch.device,
    ablate: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    total = 0
    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            logits, _, _ = model(ids, ablate=ablate)
            _, metrics = combined_loss_accuracy(logits, ids, layout)
            batch_size_actual = ids.shape[0]
            for key, value in metrics.items():
                totals[key] += value * batch_size_actual
            total += batch_size_actual
    return {key: value / total for key, value in totals.items()}


def evaluate_all_single_head_ablations(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    baseline: dict[str, float],
) -> dict[str, list[np.ndarray]]:
    role_scores = {role: [] for role in ROLE_FAMILIES}
    role_accuracy_deltas = {role: [] for role in ROLE_FAMILIES}
    for layer_idx in range(len(model.blocks)):
        layer_scores = {role: [] for role in ROLE_FAMILIES}
        layer_accuracy_deltas = {role: [] for role in ROLE_FAMILIES}
        for head_idx in range(len(model.head_dims)):
            metrics = evaluate(model, input_ids, layout, args.batch_size, device, ablate=[(layer_idx, head_idx)])
            for role in ROLE_FAMILIES:
                layer_scores[role].append(metrics[f"{role}_loss"] - baseline[f"{role}_loss"])
                layer_accuracy_deltas[role].append(metrics[f"{role}_accuracy"] - baseline[f"{role}_accuracy"])
        for role in ROLE_FAMILIES:
            role_scores[role].append(np.asarray(layer_scores[role], dtype=np.float64))
            role_accuracy_deltas[role].append(np.asarray(layer_accuracy_deltas[role], dtype=np.float64))
    return {
        "role_scores": role_scores,
        "role_accuracy_deltas": role_accuracy_deltas,
    }


def flatten_role_scores(role_scores: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.maximum(layer, 0.0) for layer in role_scores]).astype(np.float64)


def entropy(distribution: np.ndarray, eps: float = 1e-12) -> float:
    clipped = np.clip(distribution.astype(np.float64), eps, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def distribution_tv_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def distribution_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(1.0 - distribution_tv_distance(first, second))


def head_dim_affinity(distribution: np.ndarray, head_dims: list[int], n_layers: int) -> dict[int, float]:
    affinity: dict[int, float] = defaultdict(float)
    n_heads = len(head_dims)
    for flat_idx, mass in enumerate(distribution):
        head_idx = flat_idx % n_heads
        affinity[int(head_dims[head_idx])] += float(mass)
    return dict(sorted(affinity.items()))


def top_role_summary(
    distribution: np.ndarray,
    raw_scores: np.ndarray,
    head_dims: list[int],
    n_layers: int,
) -> dict[str, float | int | dict[int, float]]:
    n_heads = len(head_dims)
    top_idx = int(np.argmax(distribution))
    top_layer = top_idx // n_heads
    top_head = top_idx % n_heads
    affinity = head_dim_affinity(distribution, head_dims, n_layers)
    top_dim = max(affinity.items(), key=lambda item: item[1])[0]
    return {
        "top_layer": top_layer,
        "top_head": top_head,
        "top_head_dim": int(head_dims[top_head]),
        "top_role_score": float(raw_scores[top_idx]),
        "global_top_specialization": float(distribution[top_idx]),
        "effective_heads": float(math.exp(entropy(distribution))),
        "top_dim": int(top_dim),
        "top_dim_mass": float(affinity[top_dim]),
        "head_dim_affinity": affinity,
    }


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


def cluster_roles(role_names: list[str], role_distributions: dict[str, np.ndarray]) -> tuple[list[int], float]:
    n_roles = len(role_names)
    distance = np.zeros((n_roles, n_roles), dtype=np.float64)
    for i, role_a in enumerate(role_names):
        for j, role_b in enumerate(role_names):
            distance[i, j] = distribution_tv_distance(role_distributions[role_a], role_distributions[role_b])
    condensed = squareform(distance, checks=False)
    clusters = fcluster(linkage(condensed, method="average"), t=len(set(ROLE_FAMILIES.values())), criterion="maxclust")
    ari = adjusted_rand_index([ROLE_FAMILIES[role] for role in role_names], [int(item) for item in clusters])
    return [int(item) for item in clusters], ari


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_config(
    config: str,
    model_rows: list[ModelSummary],
    role_rows: list[RoleSummary],
) -> dict[str, object]:
    config_models = [row for row in model_rows if row.config == config]
    config_roles = [row for row in role_rows if row.config == config]
    role_top_dim_counts: dict[str, dict[str, int]] = {}
    for role in ROLE_FAMILIES:
        counts = Counter(row.top_dim for row in config_roles if row.role == role)
        role_top_dim_counts[role] = {str(dim): count for dim, count in sorted(counts.items())}

    return {
        "n_models": len(config_models),
        "eval_loss_mean": float(np.mean([row.eval_loss_mean for row in config_models])),
        "role_accuracy_mean": float(np.mean([row.role_accuracy_mean for row in config_models])),
        "role_accuracy_min": float(np.min([row.role_accuracy_min for row in config_models])),
        "role_specialization_mean": float(np.mean([row.role_specialization_mean for row in config_models])),
        "role_effective_heads_mean": float(np.mean([row.role_effective_heads_mean for row in config_models])),
        "role_top_dim_mass_mean": float(np.mean([row.role_top_dim_mass_mean for row in config_models])),
        "within_family_similarity_mean": float(
            np.mean([row.within_family_similarity_mean for row in config_models])
        ),
        "between_family_similarity_mean": float(
            np.mean([row.between_family_similarity_mean for row in config_models])
        ),
        "family_gap_mean": float(np.mean([row.family_gap for row in config_models])),
        "family_cluster_ari_mean": float(np.mean([row.family_cluster_ari for row in config_models])),
        "role_top_dim_counts": role_top_dim_counts,
    }


def write_config_summary(path: Path, summary_by_config: dict[str, dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "n_models",
        "eval_loss_mean",
        "role_accuracy_mean",
        "role_accuracy_min",
        "role_specialization_mean",
        "role_effective_heads_mean",
        "role_top_dim_mass_mean",
        "within_family_similarity_mean",
        "between_family_similarity_mean",
        "family_gap_mean",
        "family_cluster_ari_mean",
        "role_top_dim_counts_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for config, values in summary_by_config.items():
            row = dict(values)
            role_top_dim_counts = row.pop("role_top_dim_counts")
            writer.writerow(
                {
                    "config": config,
                    **row,
                    "role_top_dim_counts_json": json.dumps(role_top_dim_counts),
                }
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    layout = sequence_layout(args.kv_pairs, args.induction_short_length, args.induction_long_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = make_ontology_dataset(
        np.random.default_rng(12345),
        args.eval_examples,
        args.kv_pairs,
        args.induction_short_length,
        args.induction_long_length,
        args.vocab_size,
        int(layout["seq_len"]),
    )

    model_rows: list[ModelSummary] = []
    role_rows: list[RoleSummary] = []
    head_rows: list[HeadRoleScore] = []
    pair_rows: list[RolePairSummary] = []
    head_dims_by_config: dict[str, list[int]] = {}

    for config_name in args.configs:
        head_dims = resolve_config(config_name)
        head_dims_by_config[config_name] = head_dims
        print(f"config={config_name} head_dims={head_dims} total_head_dim={sum(head_dims)}", flush=True)
        for seed in args.seeds:
            print(f"training config={config_name} seed={seed}", flush=True)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = TinyTransformer(
                vocab_size=args.vocab_size,
                seq_len=int(layout["seq_len"]),
                d_model=args.d_model,
                head_dims=head_dims,
                n_layers=args.n_layers,
                mlp_dim=args.mlp_dim,
            ).to(device)

            train_model(model, np.random.default_rng(seed), args, layout, device)
            baseline = evaluate(model, eval_ids, layout, args.batch_size, device)
            ablation = evaluate_all_single_head_ablations(model, eval_ids, layout, args, device, baseline)

            role_distributions: dict[str, np.ndarray] = {}
            role_specializations = []
            role_effective_heads = []
            role_top_dim_masses = []
            role_accuracies = []

            for role, family in ROLE_FAMILIES.items():
                flat_scores = flatten_role_scores(ablation["role_scores"][role])
                distribution = specialization_distribution(flat_scores)
                role_distributions[role] = distribution
                summary = top_role_summary(distribution, flat_scores, head_dims, args.n_layers)
                role_specializations.append(float(summary["global_top_specialization"]))
                role_effective_heads.append(float(summary["effective_heads"]))
                role_top_dim_masses.append(float(summary["top_dim_mass"]))
                role_accuracies.append(baseline[f"{role}_accuracy"])

                role_rows.append(
                    RoleSummary(
                        config=config_name,
                        seed=seed,
                        role=role,
                        family=family,
                        loss=baseline[f"{role}_loss"],
                        accuracy=baseline[f"{role}_accuracy"],
                        top_layer=int(summary["top_layer"]),
                        top_head=int(summary["top_head"]),
                        top_head_dim=int(summary["top_head_dim"]),
                        top_role_score=float(summary["top_role_score"]),
                        global_top_specialization=float(summary["global_top_specialization"]),
                        effective_heads=float(summary["effective_heads"]),
                        top_dim=int(summary["top_dim"]),
                        top_dim_mass=float(summary["top_dim_mass"]),
                        head_dim_affinity_json=json.dumps(
                            {str(k): v for k, v in summary["head_dim_affinity"].items()}
                        ),
                    )
                )

                n_heads = len(head_dims)
                for flat_idx, role_mass in enumerate(distribution):
                    layer_idx = flat_idx // n_heads
                    head_idx = flat_idx % n_heads
                    head_rows.append(
                        HeadRoleScore(
                            config=config_name,
                            seed=seed,
                            role=role,
                            family=family,
                            layer=layer_idx,
                            head=head_idx,
                            head_dim=head_dims[head_idx],
                            role_score=float(flat_scores[flat_idx]),
                            role_mass=float(role_mass),
                            ablation_accuracy_delta=float(
                                ablation["role_accuracy_deltas"][role][layer_idx][head_idx]
                            ),
                        )
                    )

            within = []
            between = []
            role_names = list(ROLE_FAMILIES)
            for role_a, role_b in combinations(role_names, 2):
                tv = distribution_tv_distance(role_distributions[role_a], role_distributions[role_b])
                similarity = 1.0 - tv
                same_family = ROLE_FAMILIES[role_a] == ROLE_FAMILIES[role_b]
                if same_family:
                    within.append(similarity)
                else:
                    between.append(similarity)
                pair_rows.append(
                    RolePairSummary(
                        config=config_name,
                        seed=seed,
                        role_a=role_a,
                        family_a=ROLE_FAMILIES[role_a],
                        role_b=role_b,
                        family_b=ROLE_FAMILIES[role_b],
                        same_family=same_family,
                        tv_distance=tv,
                        similarity=similarity,
                    )
                )

            _, ari = cluster_roles(role_names, role_distributions)
            within_mean = float(np.mean(within))
            between_mean = float(np.mean(between))
            model_rows.append(
                ModelSummary(
                    config=config_name,
                    seed=seed,
                    head_dims_json=json.dumps(head_dims),
                    eval_loss_mean=baseline["loss_mean"],
                    role_accuracy_mean=float(np.mean(role_accuracies)),
                    role_accuracy_min=float(np.min(role_accuracies)),
                    role_specialization_mean=float(np.mean(role_specializations)),
                    role_effective_heads_mean=float(np.mean(role_effective_heads)),
                    role_top_dim_mass_mean=float(np.mean(role_top_dim_masses)),
                    within_family_similarity_mean=within_mean,
                    between_family_similarity_mean=between_mean,
                    family_gap=within_mean - between_mean,
                    family_cluster_ari=ari,
                )
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_by_config = {
        config: summarize_config(config, model_rows, role_rows)
        for config in args.configs
    }

    write_csv(args.output_dir / "model_summary.csv", model_rows)
    write_csv(args.output_dir / "role_summary.csv", role_rows)
    write_csv(args.output_dir / "head_role_scores.csv", head_rows)
    write_csv(args.output_dir / "role_pair_summary.csv", pair_rows)
    write_config_summary(args.output_dir / "config_summary.csv", summary_by_config)
    payload = {
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "layout": {
            "seq_len": int(layout["seq_len"]),
            "role_positions": {
                role: [int(item) for item in positions.tolist()]
                for role, positions in layout["role_positions"].items()  # type: ignore[union-attr]
            },
        },
        "role_families": ROLE_FAMILIES,
        "head_dims_by_config": head_dims_by_config,
        "summary_by_config": summary_by_config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary_by_config, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
