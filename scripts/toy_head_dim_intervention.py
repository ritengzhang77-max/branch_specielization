#!/usr/bin/env python3
"""Toy heterogeneous-head-dimension intervention.

This script is a small Phase 3 pilot. It trains tiny decoder-only transformers
on a synthetic key-value recall task, comparing uniform head dimensions against
heterogeneous head dimensions under a matched total attention dimension.

The goal is not to claim language-model evidence. The goal is to quickly test
whether the planned structural intervention can produce measurable differences
in functional specialization and cross-seed stability before spending compute on
larger models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


CONFIG_PRESETS = {
    "uniform4": [32, 32, 32, 32],
    "hetero4": [16, 16, 32, 64],
    "hetero4_64first": [64, 16, 16, 32],
    "uniform2": [64, 64],
}


@dataclass
class ModelSummary:
    config: str
    seed: int
    head_dims_json: str
    eval_loss: float
    eval_accuracy: float
    top_layer: int
    top_head: int
    top_head_dim: int
    role_metric: str
    top_role_score: float
    top_specialization: float
    random_same_layer_role_score_mean: float
    own_top_loss_delta: float
    own_top_accuracy_delta: float
    random_same_layer_loss_delta_mean: float
    random_same_layer_accuracy_delta_mean: float


@dataclass
class HeadRoleScore:
    config: str
    seed: int
    layer: int
    head: int
    head_dim: int
    role: str
    attention_role_score: float
    ablation_accuracy_delta: float
    role_score: float
    specialization: float


@dataclass
class PairStability:
    config: str
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


class VariableHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, head_dims: list[int]) -> None:
        super().__init__()
        self.head_dims = list(head_dims)
        self.q_proj = nn.ModuleList([nn.Linear(d_model, dim, bias=False) for dim in head_dims])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, dim, bias=False) for dim in head_dims])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, dim, bias=False) for dim in head_dims])
        self.out_proj = nn.Linear(sum(head_dims), d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        ablate_heads: set[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_outputs = []
        attentions = []
        raw_scores = []
        for head_idx, head_dim in enumerate(self.head_dims):
            q = self.q_proj[head_idx](x)
            k = self.k_proj[head_idx](x)
            v = self.v_proj[head_idx](x)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            raw_scores.append(scores)
            masked_scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            attention = F.softmax(masked_scores, dim=-1)
            output = torch.matmul(attention, v)
            if ablate_heads and head_idx in ablate_heads:
                output = torch.zeros_like(output)
            head_outputs.append(output)
            attentions.append(attention)

        return (
            self.out_proj(torch.cat(head_outputs, dim=-1)),
            torch.stack(attentions, dim=1),
            torch.stack(raw_scores, dim=1),
        )


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, head_dims: list[int], mlp_dim: int) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = VariableHeadSelfAttention(d_model, head_dims)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        ablate_heads: set[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_output, attention, raw_scores = self.attn(self.ln_attn(x), causal_mask, ablate_heads)
        x = x + attn_output
        x = x + self.mlp(self.ln_mlp(x))
        return x, attention, raw_scores


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        head_dims: list[int],
        n_layers: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.head_dims = list(head_dims)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, head_dims, mlp_dim) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        ablate: list[tuple[int, int]] | None = None,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        device = input_ids.device
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        x = self.token_embed(input_ids) + self.pos_embed[:seq_len]

        ablate_by_layer: dict[int, set[int]] = defaultdict(set)
        for layer, head in ablate or []:
            ablate_by_layer[int(layer)].add(int(head))

        attentions = []
        raw_scores = []
        for layer_idx, block in enumerate(self.blocks):
            x, attention, raw = block(x, causal_mask, ablate_by_layer.get(layer_idx))
            if return_attentions:
                attentions.append(attention)
                raw_scores.append(raw)

        logits = self.unembed(self.final_ln(x))
        return logits, attentions, raw_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["uniform4", "hetero4", "uniform2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=1024)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--key-vocab", type=int, default=96)
    parser.add_argument("--value-vocab", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--random-permutations", type=int, default=200)
    parser.add_argument(
        "--role-target",
        default="value",
        choices=["key", "value"],
        help="Score final-query attention to the matching key token or its associated value token.",
    )
    parser.add_argument(
        "--role-metric",
        default="ablation",
        choices=["ablation", "attention"],
        help="Use causal ablation loss delta or attention mass as S(h,t).",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_head_dim_intervention"))
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_config(name: str) -> list[int]:
    if name in CONFIG_PRESETS:
        return CONFIG_PRESETS[name]
    try:
        dims = [int(item) for item in name.split(",") if item]
    except ValueError as exc:
        raise ValueError(f"Unknown config preset or head-dim list: {name}") from exc
    if not dims or min(dims) <= 0:
        raise ValueError(f"Invalid head-dim list: {name}")
    return dims


def make_kv_dataset(
    rng: np.random.Generator,
    n_examples: int,
    num_pairs: int,
    key_vocab: int,
    value_vocab: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_low = 4
    value_low = key_low + key_vocab
    input_rows = []
    targets = []
    key_positions = []
    key_tokens = np.arange(key_low, key_low + key_vocab)
    value_tokens = np.arange(value_low, value_low + value_vocab)

    for _ in range(n_examples):
        keys = rng.choice(key_tokens, size=num_pairs, replace=False)
        values = rng.choice(value_tokens, size=num_pairs, replace=True)
        query_idx = int(rng.integers(0, num_pairs))
        row = []
        for key, value in zip(keys, values):
            row.extend([int(key), int(value)])
        row.append(int(keys[query_idx]))
        input_rows.append(row)
        targets.append(int(values[query_idx]))
        key_positions.append(2 * query_idx)

    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(key_positions, dtype=torch.long),
    )


def iter_dataset_batches(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    key_positions: torch.Tensor,
    batch_size: int,
):
    for start in range(0, input_ids.shape[0], batch_size):
        yield (
            input_ids[start : start + batch_size],
            targets[start : start + batch_size],
            key_positions[start : start + batch_size],
        )


def train_model(
    model: TinyTransformer,
    train_rng: np.random.Generator,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for step in range(args.steps):
        input_ids, targets, _ = make_kv_dataset(
            rng=train_rng,
            n_examples=args.batch_size,
            num_pairs=args.num_pairs,
            key_vocab=args.key_vocab,
            value_vocab=args.value_vocab,
        )
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(input_ids)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"  step={step + 1} train_loss={last_loss:.4f}", flush=True)
    return last_loss


def evaluate_loss_accuracy(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    key_positions: torch.Tensor,
    batch_size: int,
    device: torch.device,
    ablate: list[tuple[int, int]] | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.inference_mode():
        for ids, y, _ in iter_dataset_batches(input_ids, targets, key_positions, batch_size):
            ids = ids.to(device)
            y = y.to(device)
            logits, _, _ = model(ids, ablate=ablate)
            final_logits = logits[:, -1, :]
            loss = F.cross_entropy(final_logits, y, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_correct += int((final_logits.argmax(dim=-1) == y).sum().detach().cpu())
            total += ids.shape[0]
    return total_loss / total, total_correct / total


def evaluate_all_single_head_ablations(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    key_positions: torch.Tensor,
    batch_size: int,
    device: torch.device,
    baseline_loss: float,
    baseline_accuracy: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    loss_deltas = []
    accuracy_deltas = []
    for layer_idx in range(len(model.blocks)):
        layer_loss_deltas = []
        layer_accuracy_deltas = []
        for head_idx in range(len(model.head_dims)):
            loss, accuracy = evaluate_loss_accuracy(
                model,
                input_ids,
                targets,
                key_positions,
                batch_size,
                device,
                ablate=[(layer_idx, head_idx)],
            )
            layer_loss_deltas.append(loss - baseline_loss)
            layer_accuracy_deltas.append(accuracy - baseline_accuracy)
        loss_deltas.append(np.asarray(layer_loss_deltas, dtype=np.float64))
        accuracy_deltas.append(np.asarray(layer_accuracy_deltas, dtype=np.float64))
    return loss_deltas, accuracy_deltas


def collect_attention_metrics(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    key_positions: torch.Tensor,
    batch_size: int,
    device: torch.device,
    role_target: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    del targets
    model.eval()
    seq_len = input_ids.shape[1]
    query_pos = seq_len - 1
    causal_entries = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).reshape(-1)
    role_chunks: list[list[torch.Tensor]] | None = None
    raw_chunks: list[list[torch.Tensor]] | None = None

    with torch.inference_mode():
        for ids, _, key_pos in iter_dataset_batches(input_ids, input_ids[:, 0], key_positions, batch_size):
            ids = ids.to(device)
            key_pos = key_pos.to(device)
            role_pos = key_pos if role_target == "key" else key_pos + 1
            _, attentions, raw_scores = model(ids, return_attentions=True)
            if role_chunks is None:
                role_chunks = [[] for _ in attentions]
                raw_chunks = [[] for _ in raw_scores]
            batch_idx = torch.arange(ids.shape[0], device=device)
            for layer_idx, attention in enumerate(attentions):
                # attention: [batch, heads, seq, seq]
                role = attention[
                    batch_idx[:, None],
                    torch.arange(attention.shape[1], device=device)[None, :],
                    query_pos,
                    role_pos[:, None],
                ]
                role_chunks[layer_idx].append(role.detach().float().cpu())
            for layer_idx, raw in enumerate(raw_scores):
                raw_flat = raw.detach().float().cpu().reshape(raw.shape[0], raw.shape[1], -1)
                raw_flat = raw_flat[:, :, causal_entries]
                raw_flat = raw_flat.permute(1, 0, 2).reshape(raw.shape[1], -1)
                raw_chunks[layer_idx].append(raw_flat)

    if role_chunks is None or raw_chunks is None:
        raise RuntimeError("No attention metrics collected.")

    role_scores = [torch.cat(layer, dim=0).mean(dim=0).numpy() for layer in role_chunks]
    raw_vectors = [torch.cat(layer, dim=1).numpy() for layer in raw_chunks]
    return role_scores, raw_vectors


def specialization_distribution(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.maximum(scores.astype(np.float64), 0.0)
    total = clipped.sum()
    if total <= eps:
        return np.ones_like(clipped) / clipped.size
    return clipped / total


def distribution_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.abs(a - b).sum())


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return row_normalize(a) @ row_normalize(b).T


def random_permutation_scores(similarity: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n_rows, n_cols = similarity.shape
    n = min(n_rows, n_cols)
    scores = []
    for _ in range(n_samples):
        cols = rng.permutation(n_cols)[:n]
        rows = np.arange(n)
        scores.append(float(similarity[rows, cols].mean()))
    return np.asarray(scores, dtype=np.float64)


def apply_assignment(vector: np.ndarray, assignment: list[tuple[int, int]]) -> np.ndarray:
    aligned = np.zeros_like(vector)
    for head_a, head_b in assignment:
        aligned[head_a] = vector[head_b]
    return aligned


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_config(
    config: str,
    model_rows: list[ModelSummary],
    pair_rows: list[PairStability],
) -> dict[str, object]:
    config_models = [row for row in model_rows if row.config == config]
    config_pairs = [row for row in pair_rows if row.config == config]
    top_dims = Counter(row.top_head_dim for row in config_models)
    top_dim_total = sum(top_dims.values())
    if not config_pairs:
        pair_summary = {
            "raw_diag_mean": None,
            "matched_mean": None,
            "random_perm_mean": None,
            "matched_minus_random_mean": None,
            "raw_role_similarity_mean": None,
            "aligned_role_similarity_mean": None,
            "random_role_similarity_mean": None,
            "raw_top_head_match_rate": None,
            "aligned_top_head_match_rate": None,
        }
    else:
        pair_summary = {
            "raw_diag_mean": None
            if config_pairs[0].raw_diag_mean is None
            else float(np.mean([row.raw_diag_mean for row in config_pairs if row.raw_diag_mean is not None])),
            "matched_mean": float(np.mean([row.matched_mean for row in config_pairs])),
            "random_perm_mean": float(np.mean([row.random_perm_mean for row in config_pairs])),
            "matched_minus_random_mean": float(np.mean([row.matched_minus_random for row in config_pairs])),
            "raw_role_similarity_mean": float(np.mean([row.raw_role_similarity for row in config_pairs])),
            "aligned_role_similarity_mean": float(np.mean([row.aligned_role_similarity for row in config_pairs])),
            "random_role_similarity_mean": float(np.mean([row.random_role_similarity for row in config_pairs])),
            "raw_top_head_match_rate": float(np.mean([row.raw_top_head_match for row in config_pairs])),
            "aligned_top_head_match_rate": float(np.mean([row.aligned_top_head_match for row in config_pairs])),
        }
    return {
        "n_models": len(config_models),
        "eval_loss_mean": float(np.mean([row.eval_loss for row in config_models])),
        "eval_accuracy_mean": float(np.mean([row.eval_accuracy for row in config_models])),
        "top_specialization_mean": float(np.mean([row.top_specialization for row in config_models])),
        "own_top_loss_delta_mean": float(np.mean([row.own_top_loss_delta for row in config_models])),
        "random_same_layer_loss_delta_mean": float(
            np.mean([row.random_same_layer_loss_delta_mean for row in config_models])
        ),
        **pair_summary,
        "top_head_dim_counts": {str(dim): count for dim, count in sorted(top_dims.items())},
        "top_head_dim_fraction": {
            str(dim): count / top_dim_total for dim, count in sorted(top_dims.items())
        },
    }


def write_config_summary(path: Path, summary_by_config: dict[str, dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "n_models",
        "eval_loss_mean",
        "eval_accuracy_mean",
        "top_specialization_mean",
        "own_top_loss_delta_mean",
        "random_same_layer_loss_delta_mean",
        "raw_diag_mean",
        "matched_mean",
        "random_perm_mean",
        "matched_minus_random_mean",
        "raw_role_similarity_mean",
        "aligned_role_similarity_mean",
        "random_role_similarity_mean",
        "raw_top_head_match_rate",
        "aligned_top_head_match_rate",
        "top_head_dim_counts_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config, values in summary_by_config.items():
            row = dict(values)
            counts = row.pop("top_head_dim_counts")
            row.pop("top_head_dim_fraction")
            writer.writerow({"config": config, **row, "top_head_dim_counts_json": json.dumps(counts)})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rng = np.random.default_rng(0)

    seq_len = 2 * args.num_pairs + 1
    vocab_size = 4 + args.key_vocab + args.value_vocab
    eval_input_ids, eval_targets, eval_key_positions = make_kv_dataset(
        rng=np.random.default_rng(12345),
        n_examples=args.eval_examples,
        num_pairs=args.num_pairs,
        key_vocab=args.key_vocab,
        value_vocab=args.value_vocab,
    )

    model_rows: list[ModelSummary] = []
    head_rows: list[HeadRoleScore] = []
    pair_rows: list[PairStability] = []
    raw_vectors_by_model: dict[tuple[str, int], list[np.ndarray]] = {}
    role_dist_by_model: dict[tuple[str, int, int], np.ndarray] = {}
    head_dims_by_config: dict[str, list[int]] = {}

    for config_name in args.configs:
        head_dims = resolve_config(config_name)
        head_dims_by_config[config_name] = head_dims
        total_head_dim = sum(head_dims)
        print(f"config={config_name} head_dims={head_dims} total_head_dim={total_head_dim}", flush=True)
        for seed in args.seeds:
            print(f"training config={config_name} seed={seed}", flush=True)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = TinyTransformer(
                vocab_size=vocab_size,
                seq_len=seq_len,
                d_model=args.d_model,
                head_dims=head_dims,
                n_layers=args.n_layers,
                mlp_dim=args.mlp_dim,
            ).to(device)

            train_model(model, np.random.default_rng(seed), args, device)
            eval_loss, eval_accuracy = evaluate_loss_accuracy(
                model,
                eval_input_ids,
                eval_targets,
                eval_key_positions,
                args.batch_size,
                device,
            )
            attention_role_scores, raw_vectors = collect_attention_metrics(
                model,
                eval_input_ids,
                eval_targets,
                eval_key_positions,
                args.batch_size,
                device,
                args.role_target,
            )
            ablation_loss_deltas, ablation_accuracy_deltas = evaluate_all_single_head_ablations(
                model,
                eval_input_ids,
                eval_targets,
                eval_key_positions,
                args.batch_size,
                device,
                eval_loss,
                eval_accuracy,
            )
            if args.role_metric == "attention":
                role_scores = attention_role_scores
                role_name = f"{args.role_target}_fetch_attention"
            else:
                role_scores = [np.maximum(layer_scores, 0.0) for layer_scores in ablation_loss_deltas]
                role_name = "single_head_ablation_loss_delta"
            raw_vectors_by_model[(config_name, seed)] = raw_vectors

            top_layer = 0
            top_head = 0
            top_score = -1.0
            top_specialization = 0.0
            for layer_idx, layer_scores in enumerate(role_scores):
                dist = specialization_distribution(layer_scores)
                role_dist_by_model[(config_name, seed, layer_idx)] = dist
                for head_idx, (score, spec) in enumerate(zip(layer_scores, dist)):
                    head_rows.append(
                        HeadRoleScore(
                            config=config_name,
                            seed=seed,
                            layer=layer_idx,
                            head=head_idx,
                            head_dim=head_dims[head_idx],
                            role=role_name,
                            attention_role_score=float(attention_role_scores[layer_idx][head_idx]),
                            ablation_accuracy_delta=float(ablation_accuracy_deltas[layer_idx][head_idx]),
                            role_score=float(score),
                            specialization=float(spec),
                        )
                    )
                    if float(score) > top_score:
                        top_layer = layer_idx
                        top_head = head_idx
                        top_score = float(score)
                        top_specialization = float(spec)

            top_loss_delta = float(ablation_loss_deltas[top_layer][top_head])
            top_accuracy_delta = float(ablation_accuracy_deltas[top_layer][top_head])
            random_loss_deltas = []
            random_accuracy_deltas = []
            random_scores = []
            candidates = [head for head in range(len(head_dims)) if head != top_head]
            if not candidates:
                candidates = [top_head]
            for _ in range(args.random_controls):
                sampled_head = int(comparison_rng.choice(candidates))
                random_loss_deltas.append(float(ablation_loss_deltas[top_layer][sampled_head]))
                random_accuracy_deltas.append(float(ablation_accuracy_deltas[top_layer][sampled_head]))
                random_scores.append(float(role_scores[top_layer][sampled_head]))

            model_rows.append(
                ModelSummary(
                    config=config_name,
                    seed=seed,
                    head_dims_json=json.dumps(head_dims),
                    eval_loss=eval_loss,
                    eval_accuracy=eval_accuracy,
                    top_layer=top_layer,
                    top_head=top_head,
                    top_head_dim=head_dims[top_head],
                    role_metric=args.role_metric,
                    top_role_score=top_score,
                    top_specialization=top_specialization,
                    random_same_layer_role_score_mean=float(np.mean(random_scores)),
                    own_top_loss_delta=top_loss_delta,
                    own_top_accuracy_delta=top_accuracy_delta,
                    random_same_layer_loss_delta_mean=float(np.mean(random_loss_deltas)),
                    random_same_layer_accuracy_delta_mean=float(np.mean(random_accuracy_deltas)),
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

                    dist_a = role_dist_by_model[(config_name, seed_a, layer_idx)]
                    dist_b = role_dist_by_model[(config_name, seed_b, layer_idx)]
                    aligned_dist_b = apply_assignment(dist_b, assignment)
                    random_role_scores = [
                        distribution_similarity(dist_a, comparison_rng.permutation(dist_b))
                        for _ in range(args.random_permutations)
                    ]

                    pair_rows.append(
                        PairStability(
                            config=config_name,
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
        "head_dims_by_config": head_dims_by_config,
        "summary_by_config": summary_by_config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary_by_config, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
