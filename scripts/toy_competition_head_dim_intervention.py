#!/usr/bin/env python3
"""Toy local-vs-global competition head-dimension intervention.

This script tests the next Stage 3 question after the induction toy result:
does heterogeneous head dimension create a role partition, or does the largest
head simply dominate every task?

Each sequence contains two supervised regions:

1. Local copy positions: [x, SEP, x]. At SEP, predict x, which is available at
   the immediately previous position.
2. Global induction positions: [y_1, ..., y_n, y_1, ..., y_n]. At the second
   occurrence of y_i, predict y_{i+1}.

The script measures separate causal single-head ablation scores for the local
and induction objectives.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from toy_head_dim_intervention import (
    TinyTransformer,
    cosine_similarity_matrix,
    distribution_similarity,
    random_permutation_scores,
    resolve_config,
    resolve_device,
    specialization_distribution,
)


LOCAL_SEP = 1
GLOBAL_SEP = 2
TOKEN_LOW = 4


@dataclass
class ModelSummary:
    config: str
    seed: int
    head_dims_json: str
    eval_loss: float
    local_loss: float
    induction_loss: float
    local_accuracy: float
    induction_accuracy: float
    local_top_layer: int
    local_top_head: int
    local_top_head_dim: int
    local_top_role_score: float
    local_top_specialization: float
    local_random_same_layer_role_score_mean: float
    local_top_accuracy_delta: float
    induction_top_layer: int
    induction_top_head: int
    induction_top_head_dim: int
    induction_top_role_score: float
    induction_top_specialization: float
    induction_random_same_layer_role_score_mean: float
    induction_top_accuracy_delta: float
    same_top_slot: bool
    same_top_head_dim: bool


@dataclass
class HeadRoleScore:
    config: str
    seed: int
    layer: int
    head: int
    head_dim: int
    role: str
    local_prev_attention_score: float
    induction_match_attention_score: float
    induction_source_next_attention_score: float
    ablation_accuracy_delta: float
    role_score: float
    specialization: float


@dataclass
class PairStability:
    config: str
    role: str
    seed_a: int
    seed_b: int
    layer: int
    raw_diag_mean: float | None
    matched_mean: float
    random_perm_mean: float
    matched_minus_random: float
    raw_role_similarity: float
    aligned_role_similarity: float
    random_role_similarity: float
    raw_top_head_match: bool
    aligned_top_head_match: bool
    best_assignment_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["uniform4", "hetero4", "uniform2", "hetero4_64first"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--local-pairs", type=int, default=8)
    parser.add_argument("--repeat-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=160)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--induction-weight", type=float, default=1.0)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--random-permutations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_competition_head_dim"))
    return parser.parse_args()


def sequence_layout(local_pairs: int, repeat_length: int) -> dict[str, torch.Tensor | int]:
    local_positions = torch.tensor([3 * idx + 1 for idx in range(local_pairs)], dtype=torch.long)
    induction_offset = 3 * local_pairs + 1
    induction_positions = torch.arange(
        induction_offset + repeat_length,
        induction_offset + 2 * repeat_length - 1,
        dtype=torch.long,
    )
    return {
        "seq_len": induction_offset + 2 * repeat_length,
        "local_positions": local_positions,
        "local_prev_positions": local_positions - 1,
        "induction_positions": induction_positions,
        "induction_match_positions": induction_positions - repeat_length,
        "induction_source_next_positions": induction_positions - repeat_length + 1,
    }


def make_competition_dataset(
    rng: np.random.Generator,
    n_examples: int,
    local_pairs: int,
    repeat_length: int,
    vocab_size: int,
) -> torch.Tensor:
    required_unique = max(local_pairs, repeat_length)
    if vocab_size - TOKEN_LOW < required_unique:
        raise ValueError("vocab_size is too small for the requested task.")

    token_ids = np.arange(TOKEN_LOW, vocab_size)
    rows = []
    for _ in range(n_examples):
        local_tokens = rng.choice(token_ids, size=local_pairs, replace=False)
        induction_base = rng.choice(token_ids, size=repeat_length, replace=False)
        row = []
        for token in local_tokens:
            row.extend([int(token), LOCAL_SEP, int(token)])
        row.append(GLOBAL_SEP)
        row.extend(int(token) for token in induction_base)
        row.extend(int(token) for token in induction_base)
        rows.append(row)
    return torch.tensor(np.stack(rows), dtype=torch.long)


def objective_loss_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, float]:
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
    layout: dict[str, torch.Tensor | int],
    local_weight: float,
    induction_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    local_positions = layout["local_positions"].to(logits.device)
    induction_positions = layout["induction_positions"].to(logits.device)
    local_loss, local_accuracy = objective_loss_accuracy(logits, input_ids, local_positions)
    induction_loss, induction_accuracy = objective_loss_accuracy(logits, input_ids, induction_positions)
    loss = (local_weight * local_loss + induction_weight * induction_loss) / (local_weight + induction_weight)
    return loss, {
        "local_loss": float(local_loss.detach().cpu()),
        "induction_loss": float(induction_loss.detach().cpu()),
        "local_accuracy": local_accuracy,
        "induction_accuracy": induction_accuracy,
    }


def train_model(
    model: TinyTransformer,
    train_rng: np.random.Generator,
    args: argparse.Namespace,
    layout: dict[str, torch.Tensor | int],
    device: torch.device,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for step in range(args.steps):
        input_ids = make_competition_dataset(
            train_rng,
            args.batch_size,
            args.local_pairs,
            args.repeat_length,
            args.vocab_size,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(input_ids)
        loss, _ = combined_loss_accuracy(logits, input_ids, layout, args.local_weight, args.induction_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"  step={step + 1} train_loss={last_loss:.4f}", flush=True)
    return last_loss


def iter_batches(input_ids: torch.Tensor, batch_size: int):
    for start in range(0, input_ids.shape[0], batch_size):
        yield input_ids[start : start + batch_size]


def evaluate(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    local_weight: float,
    induction_weight: float,
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
            loss, metrics = combined_loss_accuracy(logits, ids, layout, local_weight, induction_weight)
            batch_size_actual = ids.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size_actual
            for key, value in metrics.items():
                totals[key] += value * batch_size_actual
            total += batch_size_actual
    return {key: value / total for key, value in totals.items()}


def evaluate_all_single_head_ablations(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    args: argparse.Namespace,
    device: torch.device,
    baseline: dict[str, float],
) -> dict[str, list[np.ndarray]]:
    local_loss_deltas = []
    induction_loss_deltas = []
    local_accuracy_deltas = []
    induction_accuracy_deltas = []
    for layer_idx in range(len(model.blocks)):
        layer_local_loss = []
        layer_induction_loss = []
        layer_local_accuracy = []
        layer_induction_accuracy = []
        for head_idx in range(len(model.head_dims)):
            metrics = evaluate(
                model,
                input_ids,
                layout,
                args.local_weight,
                args.induction_weight,
                args.batch_size,
                device,
                ablate=[(layer_idx, head_idx)],
            )
            layer_local_loss.append(metrics["local_loss"] - baseline["local_loss"])
            layer_induction_loss.append(metrics["induction_loss"] - baseline["induction_loss"])
            layer_local_accuracy.append(metrics["local_accuracy"] - baseline["local_accuracy"])
            layer_induction_accuracy.append(metrics["induction_accuracy"] - baseline["induction_accuracy"])
        local_loss_deltas.append(np.asarray(layer_local_loss, dtype=np.float64))
        induction_loss_deltas.append(np.asarray(layer_induction_loss, dtype=np.float64))
        local_accuracy_deltas.append(np.asarray(layer_local_accuracy, dtype=np.float64))
        induction_accuracy_deltas.append(np.asarray(layer_induction_accuracy, dtype=np.float64))
    return {
        "local_loss_delta": local_loss_deltas,
        "induction_loss_delta": induction_loss_deltas,
        "local_accuracy_delta": local_accuracy_deltas,
        "induction_accuracy_delta": induction_accuracy_deltas,
    }


def collect_attention_metrics(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    batch_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model.eval()
    seq_len = int(layout["seq_len"])
    causal_entries = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).reshape(-1)
    local_positions_cpu = layout["local_positions"]
    local_prev_positions_cpu = layout["local_prev_positions"]
    induction_positions_cpu = layout["induction_positions"]
    induction_match_positions_cpu = layout["induction_match_positions"]
    induction_source_next_positions_cpu = layout["induction_source_next_positions"]

    local_prev_chunks: list[list[torch.Tensor]] | None = None
    induction_match_chunks: list[list[torch.Tensor]] | None = None
    induction_source_next_chunks: list[list[torch.Tensor]] | None = None
    raw_chunks: list[list[torch.Tensor]] | None = None

    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            _, attentions, raw_scores = model(ids, return_attentions=True)
            if local_prev_chunks is None:
                local_prev_chunks = [[] for _ in attentions]
                induction_match_chunks = [[] for _ in attentions]
                induction_source_next_chunks = [[] for _ in attentions]
                raw_chunks = [[] for _ in raw_scores]

            local_positions = local_positions_cpu.to(device)
            local_prev_positions = local_prev_positions_cpu.to(device)
            induction_positions = induction_positions_cpu.to(device)
            induction_match_positions = induction_match_positions_cpu.to(device)
            induction_source_next_positions = induction_source_next_positions_cpu.to(device)

            for layer_idx, attention in enumerate(attentions):
                local_prev = attention[:, :, local_positions, local_prev_positions].mean(dim=-1)
                induction_match = attention[:, :, induction_positions, induction_match_positions].mean(dim=-1)
                induction_source_next = attention[
                    :, :, induction_positions, induction_source_next_positions
                ].mean(dim=-1)
                local_prev_chunks[layer_idx].append(local_prev.detach().float().cpu())
                induction_match_chunks[layer_idx].append(induction_match.detach().float().cpu())
                induction_source_next_chunks[layer_idx].append(induction_source_next.detach().float().cpu())

            for layer_idx, raw in enumerate(raw_scores):
                raw_flat = raw.detach().float().cpu().reshape(raw.shape[0], raw.shape[1], -1)
                raw_flat = raw_flat[:, :, causal_entries]
                raw_flat = raw_flat.permute(1, 0, 2).reshape(raw.shape[1], -1)
                raw_chunks[layer_idx].append(raw_flat)

    if (
        local_prev_chunks is None
        or induction_match_chunks is None
        or induction_source_next_chunks is None
        or raw_chunks is None
    ):
        raise RuntimeError("No attention metrics collected.")

    local_prev_scores = [torch.cat(layer, dim=0).mean(dim=0).numpy() for layer in local_prev_chunks]
    induction_match_scores = [
        torch.cat(layer, dim=0).mean(dim=0).numpy() for layer in induction_match_chunks
    ]
    induction_source_next_scores = [
        torch.cat(layer, dim=0).mean(dim=0).numpy() for layer in induction_source_next_chunks
    ]
    raw_vectors = [torch.cat(layer, dim=1).numpy() for layer in raw_chunks]
    return local_prev_scores, induction_match_scores, induction_source_next_scores, raw_vectors


def apply_assignment(vector: np.ndarray, assignment: list[tuple[int, int]]) -> np.ndarray:
    aligned = np.zeros_like(vector)
    for head_a, head_b in assignment:
        aligned[head_a] = vector[head_b]
    return aligned


def top_role(role_scores: list[np.ndarray], head_dims: list[int]) -> dict[str, float | int]:
    best = {"layer": 0, "head": 0, "head_dim": head_dims[0], "score": -1.0, "specialization": 0.0}
    for layer_idx, layer_scores in enumerate(role_scores):
        dist = specialization_distribution(layer_scores)
        for head_idx, score in enumerate(layer_scores):
            if float(score) > float(best["score"]):
                best = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "head_dim": head_dims[head_idx],
                    "score": float(score),
                    "specialization": float(dist[head_idx]),
                }
    return best


def random_same_layer_score(
    role_scores: list[np.ndarray],
    top_layer: int,
    top_head: int,
    n_heads: int,
    n_controls: int,
    rng: np.random.Generator,
) -> float:
    candidates = [head for head in range(n_heads) if head != top_head]
    if not candidates:
        candidates = [top_head]
    sampled = [int(rng.choice(candidates)) for _ in range(n_controls)]
    return float(np.mean([role_scores[top_layer][head] for head in sampled]))


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_role(config: str, role: str, pair_rows: list[PairStability]) -> dict[str, float | None]:
    role_pairs = [row for row in pair_rows if row.config == config and row.role == role]
    if not role_pairs:
        return {
            f"{role}_raw_role_similarity_mean": None,
            f"{role}_aligned_role_similarity_mean": None,
            f"{role}_random_role_similarity_mean": None,
            f"{role}_raw_top_head_match_rate": None,
            f"{role}_aligned_top_head_match_rate": None,
        }
    return {
        f"{role}_raw_role_similarity_mean": float(np.mean([row.raw_role_similarity for row in role_pairs])),
        f"{role}_aligned_role_similarity_mean": float(np.mean([row.aligned_role_similarity for row in role_pairs])),
        f"{role}_random_role_similarity_mean": float(np.mean([row.random_role_similarity for row in role_pairs])),
        f"{role}_raw_top_head_match_rate": float(np.mean([row.raw_top_head_match for row in role_pairs])),
        f"{role}_aligned_top_head_match_rate": float(np.mean([row.aligned_top_head_match for row in role_pairs])),
    }


def summarize_config(
    config: str,
    model_rows: list[ModelSummary],
    pair_rows: list[PairStability],
) -> dict[str, object]:
    config_models = [row for row in model_rows if row.config == config]
    local_dims = Counter(row.local_top_head_dim for row in config_models)
    induction_dims = Counter(row.induction_top_head_dim for row in config_models)
    local_slots = Counter(
        f"L{row.local_top_layer}H{row.local_top_head}:d{row.local_top_head_dim}"
        for row in config_models
    )
    induction_slots = Counter(
        f"L{row.induction_top_layer}H{row.induction_top_head}:d{row.induction_top_head_dim}"
        for row in config_models
    )
    summary = {
        "n_models": len(config_models),
        "eval_loss_mean": float(np.mean([row.eval_loss for row in config_models])),
        "local_loss_mean": float(np.mean([row.local_loss for row in config_models])),
        "induction_loss_mean": float(np.mean([row.induction_loss for row in config_models])),
        "local_accuracy_mean": float(np.mean([row.local_accuracy for row in config_models])),
        "induction_accuracy_mean": float(np.mean([row.induction_accuracy for row in config_models])),
        "local_top_specialization_mean": float(np.mean([row.local_top_specialization for row in config_models])),
        "induction_top_specialization_mean": float(
            np.mean([row.induction_top_specialization for row in config_models])
        ),
        "local_top_loss_delta_mean": float(np.mean([row.local_top_role_score for row in config_models])),
        "induction_top_loss_delta_mean": float(
            np.mean([row.induction_top_role_score for row in config_models])
        ),
        "local_random_same_layer_loss_delta_mean": float(
            np.mean([row.local_random_same_layer_role_score_mean for row in config_models])
        ),
        "induction_random_same_layer_loss_delta_mean": float(
            np.mean([row.induction_random_same_layer_role_score_mean for row in config_models])
        ),
        "same_top_slot_rate": float(np.mean([row.same_top_slot for row in config_models])),
        "same_top_head_dim_rate": float(np.mean([row.same_top_head_dim for row in config_models])),
        "local_top_head_dim_counts": {str(dim): count for dim, count in sorted(local_dims.items())},
        "induction_top_head_dim_counts": {str(dim): count for dim, count in sorted(induction_dims.items())},
        "local_top_slot_counts": {slot: count for slot, count in sorted(local_slots.items())},
        "induction_top_slot_counts": {slot: count for slot, count in sorted(induction_slots.items())},
    }
    summary.update(summarize_role(config, "local", pair_rows))
    summary.update(summarize_role(config, "induction", pair_rows))
    return summary


def write_config_summary(path: Path, summary_by_config: dict[str, dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "n_models",
        "eval_loss_mean",
        "local_loss_mean",
        "induction_loss_mean",
        "local_accuracy_mean",
        "induction_accuracy_mean",
        "local_top_specialization_mean",
        "induction_top_specialization_mean",
        "local_top_loss_delta_mean",
        "induction_top_loss_delta_mean",
        "local_random_same_layer_loss_delta_mean",
        "induction_random_same_layer_loss_delta_mean",
        "same_top_slot_rate",
        "same_top_head_dim_rate",
        "local_raw_role_similarity_mean",
        "local_aligned_role_similarity_mean",
        "local_random_role_similarity_mean",
        "local_raw_top_head_match_rate",
        "local_aligned_top_head_match_rate",
        "induction_raw_role_similarity_mean",
        "induction_aligned_role_similarity_mean",
        "induction_random_role_similarity_mean",
        "induction_raw_top_head_match_rate",
        "induction_aligned_top_head_match_rate",
        "local_top_head_dim_counts_json",
        "induction_top_head_dim_counts_json",
        "local_top_slot_counts_json",
        "induction_top_slot_counts_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config, values in summary_by_config.items():
            row = dict(values)
            local_counts = row.pop("local_top_head_dim_counts")
            induction_counts = row.pop("induction_top_head_dim_counts")
            local_slots = row.pop("local_top_slot_counts")
            induction_slots = row.pop("induction_top_slot_counts")
            writer.writerow(
                {
                    "config": config,
                    **row,
                    "local_top_head_dim_counts_json": json.dumps(local_counts),
                    "induction_top_head_dim_counts_json": json.dumps(induction_counts),
                    "local_top_slot_counts_json": json.dumps(local_slots),
                    "induction_top_slot_counts_json": json.dumps(induction_slots),
                }
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    layout = sequence_layout(args.local_pairs, args.repeat_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rng = np.random.default_rng(0)
    eval_ids = make_competition_dataset(
        np.random.default_rng(12345),
        args.eval_examples,
        args.local_pairs,
        args.repeat_length,
        args.vocab_size,
    )

    model_rows: list[ModelSummary] = []
    head_rows: list[HeadRoleScore] = []
    pair_rows: list[PairStability] = []
    raw_vectors_by_model: dict[tuple[str, int], list[np.ndarray]] = {}
    role_dist_by_model: dict[tuple[str, int, int, str], np.ndarray] = {}
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
            baseline = evaluate(
                model,
                eval_ids,
                layout,
                args.local_weight,
                args.induction_weight,
                args.batch_size,
                device,
            )
            (
                local_prev_attention,
                induction_match_attention,
                induction_source_next_attention,
                raw_vectors,
            ) = collect_attention_metrics(model, eval_ids, layout, args.batch_size, device)
            ablation = evaluate_all_single_head_ablations(model, eval_ids, layout, args, device, baseline)

            local_scores = [np.maximum(layer, 0.0) for layer in ablation["local_loss_delta"]]
            induction_scores = [np.maximum(layer, 0.0) for layer in ablation["induction_loss_delta"]]
            raw_vectors_by_model[(config_name, seed)] = raw_vectors

            for role_name, role_scores in [("local", local_scores), ("induction", induction_scores)]:
                for layer_idx, layer_scores in enumerate(role_scores):
                    role_dist_by_model[(config_name, seed, layer_idx, role_name)] = specialization_distribution(
                        layer_scores
                    )

            for layer_idx in range(len(local_scores)):
                for head_idx in range(len(head_dims)):
                    for role_name, role_scores, accuracy_key in [
                        ("local", local_scores, "local_accuracy_delta"),
                        ("induction", induction_scores, "induction_accuracy_delta"),
                    ]:
                        dist = role_dist_by_model[(config_name, seed, layer_idx, role_name)]
                        head_rows.append(
                            HeadRoleScore(
                                config=config_name,
                                seed=seed,
                                layer=layer_idx,
                                head=head_idx,
                                head_dim=head_dims[head_idx],
                                role=role_name,
                                local_prev_attention_score=float(local_prev_attention[layer_idx][head_idx]),
                                induction_match_attention_score=float(induction_match_attention[layer_idx][head_idx]),
                                induction_source_next_attention_score=float(
                                    induction_source_next_attention[layer_idx][head_idx]
                                ),
                                ablation_accuracy_delta=float(ablation[accuracy_key][layer_idx][head_idx]),
                                role_score=float(role_scores[layer_idx][head_idx]),
                                specialization=float(dist[head_idx]),
                            )
                        )

            local_top = top_role(local_scores, head_dims)
            induction_top = top_role(induction_scores, head_dims)
            model_rows.append(
                ModelSummary(
                    config=config_name,
                    seed=seed,
                    head_dims_json=json.dumps(head_dims),
                    eval_loss=baseline["loss"],
                    local_loss=baseline["local_loss"],
                    induction_loss=baseline["induction_loss"],
                    local_accuracy=baseline["local_accuracy"],
                    induction_accuracy=baseline["induction_accuracy"],
                    local_top_layer=int(local_top["layer"]),
                    local_top_head=int(local_top["head"]),
                    local_top_head_dim=int(local_top["head_dim"]),
                    local_top_role_score=float(local_top["score"]),
                    local_top_specialization=float(local_top["specialization"]),
                    local_random_same_layer_role_score_mean=random_same_layer_score(
                        local_scores,
                        int(local_top["layer"]),
                        int(local_top["head"]),
                        len(head_dims),
                        args.random_controls,
                        comparison_rng,
                    ),
                    local_top_accuracy_delta=float(
                        ablation["local_accuracy_delta"][int(local_top["layer"])][int(local_top["head"])]
                    ),
                    induction_top_layer=int(induction_top["layer"]),
                    induction_top_head=int(induction_top["head"]),
                    induction_top_head_dim=int(induction_top["head_dim"]),
                    induction_top_role_score=float(induction_top["score"]),
                    induction_top_specialization=float(induction_top["specialization"]),
                    induction_random_same_layer_role_score_mean=random_same_layer_score(
                        induction_scores,
                        int(induction_top["layer"]),
                        int(induction_top["head"]),
                        len(head_dims),
                        args.random_controls,
                        comparison_rng,
                    ),
                    induction_top_accuracy_delta=float(
                        ablation["induction_accuracy_delta"][int(induction_top["layer"])][
                            int(induction_top["head"])
                        ]
                    ),
                    same_top_slot=bool(
                        int(local_top["layer"]) == int(induction_top["layer"])
                        and int(local_top["head"]) == int(induction_top["head"])
                    ),
                    same_top_head_dim=bool(int(local_top["head_dim"]) == int(induction_top["head_dim"])),
                )
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for config_name in args.configs:
        seeds = list(args.seeds)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1 :]:
                layers_a = raw_vectors_by_model[(config_name, seed_a)]
                layers_b = raw_vectors_by_model[(config_name, seed_b)]
                for layer_idx, (vecs_a, vecs_b) in enumerate(zip(layers_a, layers_b)):
                    similarity = cosine_similarity_matrix(vecs_a, vecs_b)
                    raw_diag = np.diag(similarity) if similarity.shape[0] == similarity.shape[1] else None
                    rows, cols = linear_sum_assignment(-similarity)
                    assignment = [(int(row), int(col)) for row, col in zip(rows, cols)]
                    matched = similarity[rows, cols]
                    random_perm = random_permutation_scores(similarity, args.random_permutations, comparison_rng)

                    for role_name in ["local", "induction"]:
                        dist_a = role_dist_by_model[(config_name, seed_a, layer_idx, role_name)]
                        dist_b = role_dist_by_model[(config_name, seed_b, layer_idx, role_name)]
                        aligned_dist_b = apply_assignment(dist_b, assignment)
                        random_role_scores = [
                            distribution_similarity(dist_a, comparison_rng.permutation(dist_b))
                            for _ in range(args.random_permutations)
                        ]
                        pair_rows.append(
                            PairStability(
                                config=config_name,
                                role=role_name,
                                seed_a=seed_a,
                                seed_b=seed_b,
                                layer=layer_idx,
                                raw_diag_mean=None if raw_diag is None else float(raw_diag.mean()),
                                matched_mean=float(matched.mean()),
                                random_perm_mean=float(random_perm.mean()),
                                matched_minus_random=float(matched.mean() - random_perm.mean()),
                                raw_role_similarity=distribution_similarity(dist_a, dist_b),
                                aligned_role_similarity=distribution_similarity(dist_a, aligned_dist_b),
                                random_role_similarity=float(np.mean(random_role_scores)),
                                raw_top_head_match=bool(np.argmax(dist_a) == np.argmax(dist_b)),
                                aligned_top_head_match=bool(np.argmax(dist_a) == np.argmax(aligned_dist_b)),
                                best_assignment_json=json.dumps(assignment),
                            )
                        )

    summary_by_config = {
        config: summarize_config(config, model_rows, pair_rows)
        for config in args.configs
    }

    write_csv(args.output_dir / "model_summary.csv", model_rows)
    write_csv(args.output_dir / "head_role_scores.csv", head_rows)
    write_csv(args.output_dir / "pair_stability.csv", pair_rows)
    write_config_summary(args.output_dir / "config_summary.csv", summary_by_config)
    payload = {
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "layout": {
            key: int(value) if isinstance(value, int) else [int(item) for item in value.tolist()]
            for key, value in layout.items()
        },
        "head_dims_by_config": head_dims_by_config,
        "summary_by_config": summary_by_config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary_by_config, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
