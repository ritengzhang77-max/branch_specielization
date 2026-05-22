#!/usr/bin/env python3
"""Toy induction-style head-dimension intervention.

This is a richer companion to toy_head_dim_intervention.py. Instead of a
one-step key-value lookup, it trains tiny decoder-only transformers on repeated
random token sequences and scores second-half next-token prediction:

    [x_1, ..., x_n, x_1, ..., x_n]

At the second occurrence of x_i, the model must predict x_{i+1}. This is a
minimal induction-style task and is closer to the repeat-match behavior measured
in Pythia.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    match_attention_score: float
    source_next_attention_score: float
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["uniform4", "hetero4", "uniform2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--repeat-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--random-permutations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_induction_head_dim"))
    return parser.parse_args()


def make_repeat_dataset(
    rng: np.random.Generator,
    n_examples: int,
    repeat_length: int,
    vocab_size: int,
) -> torch.Tensor:
    if vocab_size - 4 < repeat_length:
        raise ValueError("vocab_size is too small for unique repeated sequences.")

    rows = []
    token_ids = np.arange(4, vocab_size)
    for _ in range(n_examples):
        base = rng.choice(token_ids, size=repeat_length, replace=False)
        rows.append(np.concatenate([base, base]))
    return torch.tensor(np.stack(rows), dtype=torch.long)


def score_positions(repeat_length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(repeat_length, 2 * repeat_length - 1, device=device)


def repeat_loss_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repeat_length: int,
) -> tuple[torch.Tensor, float]:
    positions = score_positions(repeat_length, logits.device)
    selected_logits = logits[:, positions, :]
    targets = input_ids[:, positions + 1]
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    accuracy = float((selected_logits.argmax(dim=-1) == targets).float().mean().detach().cpu())
    return loss, accuracy


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
        input_ids = make_repeat_dataset(train_rng, args.batch_size, args.repeat_length, args.vocab_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(input_ids)
        loss, _ = repeat_loss_accuracy(logits, input_ids, args.repeat_length)
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


def evaluate_loss_accuracy(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    repeat_length: int,
    batch_size: int,
    device: torch.device,
    ablate: list[tuple[int, int]] | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total = 0
    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            logits, _, _ = model(ids, ablate=ablate)
            loss, accuracy = repeat_loss_accuracy(logits, ids, repeat_length)
            total_loss += float(loss.detach().cpu()) * ids.shape[0]
            total_accuracy += accuracy * ids.shape[0]
            total += ids.shape[0]
    return total_loss / total, total_accuracy / total


def evaluate_all_single_head_ablations(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    repeat_length: int,
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
                repeat_length,
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
    repeat_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model.eval()
    seq_len = input_ids.shape[1]
    causal_entries = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).reshape(-1)
    positions_cpu = torch.arange(repeat_length, 2 * repeat_length - 1)
    match_positions_cpu = positions_cpu - repeat_length
    source_next_positions_cpu = match_positions_cpu + 1

    match_chunks: list[list[torch.Tensor]] | None = None
    source_next_chunks: list[list[torch.Tensor]] | None = None
    raw_chunks: list[list[torch.Tensor]] | None = None

    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            _, attentions, raw_scores = model(ids, return_attentions=True)
            if match_chunks is None:
                match_chunks = [[] for _ in attentions]
                source_next_chunks = [[] for _ in attentions]
                raw_chunks = [[] for _ in raw_scores]

            positions = positions_cpu.to(device)
            match_positions = match_positions_cpu.to(device)
            source_next_positions = source_next_positions_cpu.to(device)
            for layer_idx, attention in enumerate(attentions):
                # attention: [batch, heads, seq, seq]
                match_values = attention[:, :, positions, match_positions].mean(dim=-1)
                source_next_values = attention[:, :, positions, source_next_positions].mean(dim=-1)
                match_chunks[layer_idx].append(match_values.detach().float().cpu())
                source_next_chunks[layer_idx].append(source_next_values.detach().float().cpu())

            for layer_idx, raw in enumerate(raw_scores):
                raw_flat = raw.detach().float().cpu().reshape(raw.shape[0], raw.shape[1], -1)
                raw_flat = raw_flat[:, :, causal_entries]
                raw_flat = raw_flat.permute(1, 0, 2).reshape(raw.shape[1], -1)
                raw_chunks[layer_idx].append(raw_flat)

    if match_chunks is None or source_next_chunks is None or raw_chunks is None:
        raise RuntimeError("No attention metrics collected.")

    match_scores = [torch.cat(layer, dim=0).mean(dim=0).numpy() for layer in match_chunks]
    source_next_scores = [torch.cat(layer, dim=0).mean(dim=0).numpy() for layer in source_next_chunks]
    raw_vectors = [torch.cat(layer, dim=1).numpy() for layer in raw_chunks]
    return match_scores, source_next_scores, raw_vectors


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
    if config_pairs:
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
    else:
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

    seq_len = 2 * args.repeat_length
    eval_ids = make_repeat_dataset(
        rng=np.random.default_rng(12345),
        n_examples=args.eval_examples,
        repeat_length=args.repeat_length,
        vocab_size=args.vocab_size,
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
        print(f"config={config_name} head_dims={head_dims} total_head_dim={sum(head_dims)}", flush=True)
        for seed in args.seeds:
            print(f"training config={config_name} seed={seed}", flush=True)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = TinyTransformer(
                vocab_size=args.vocab_size,
                seq_len=seq_len,
                d_model=args.d_model,
                head_dims=head_dims,
                n_layers=args.n_layers,
                mlp_dim=args.mlp_dim,
            ).to(device)

            train_model(model, np.random.default_rng(seed), args, device)
            eval_loss, eval_accuracy = evaluate_loss_accuracy(
                model,
                eval_ids,
                args.repeat_length,
                args.batch_size,
                device,
            )
            match_scores, source_next_scores, raw_vectors = collect_attention_metrics(
                model,
                eval_ids,
                args.repeat_length,
                args.batch_size,
                device,
            )
            ablation_loss_deltas, ablation_accuracy_deltas = evaluate_all_single_head_ablations(
                model,
                eval_ids,
                args.repeat_length,
                args.batch_size,
                device,
                eval_loss,
                eval_accuracy,
            )
            role_scores = [np.maximum(layer_scores, 0.0) for layer_scores in ablation_loss_deltas]
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
                            role="repeat_next_single_head_ablation_loss_delta",
                            match_attention_score=float(match_scores[layer_idx][head_idx]),
                            source_next_attention_score=float(source_next_scores[layer_idx][head_idx]),
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
                    role_metric="ablation",
                    top_role_score=top_score,
                    top_specialization=top_specialization,
                    random_same_layer_role_score_mean=float(np.mean(random_scores)),
                    own_top_loss_delta=float(ablation_loss_deltas[top_layer][top_head]),
                    own_top_accuracy_delta=float(ablation_accuracy_deltas[top_layer][top_head]),
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
