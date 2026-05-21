#!/usr/bin/env python3
"""Estimate simple attention-role specialization across seeds.

This is a first-pass functional specialization proxy. It does not yet establish
causal importance. It asks whether heads specialize in recognizable attention
roles, then tests whether those role distributions are stable by raw head index
or after an independently computed attention-score alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from attention_stability import (
    batched,
    load_model_and_tokenizer,
    load_probe_texts,
    model_name_from_template,
    resolve_device,
    resolve_dtype,
)


@dataclass
class HeadRoleScore:
    seed: str
    layer: int
    head: int
    role: str
    score: float
    specialization: float


@dataclass
class LayerRoleSummary:
    seed: str
    layer: int
    role: str
    max_specialization: float
    entropy: float
    effective_heads: float
    top_head: int
    top_score: float


@dataclass
class PairRoleConsistency:
    seed_a: str
    seed_b: str
    layer: int
    role: str
    raw_distribution_similarity: float
    aligned_distribution_similarity: float | None
    random_distribution_similarity: float
    raw_top_head_match: bool
    aligned_top_head_match: bool | None
    top_head_a: int
    top_head_b: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-template",
        default="EleutherAI/pythia-{model_size}-seed{seed}",
        help="Hugging Face model template. Available fields: {model_size}, {seed}.",
    )
    parser.add_argument("--model-size", default="14m")
    parser.add_argument("--seeds", nargs="+", default=["1", "2"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--probe-file", type=Path, default=Path("probes/phase0_probe_texts.txt"))
    parser.add_argument("--num-texts", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--synthetic-repeat-sequences", type=int, default=32)
    parser.add_argument("--synthetic-repeat-length", type=int, default=32)
    parser.add_argument("--synthetic-token-low", type=int, default=1000)
    parser.add_argument("--alignment-summary", type=Path, default=None)
    parser.add_argument("--random-permutations", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/attention_role_specialization"))
    return parser.parse_args()


def synthetic_repeated_token_ids(
    vocab_size: int,
    n_sequences: int,
    repeat_length: int,
    token_low: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    high = vocab_size - 1
    if high - token_low < repeat_length:
        raise ValueError("Vocabulary range is too small for synthetic repeated sequences.")

    rows = []
    for _ in range(n_sequences):
        base = rng.choice(np.arange(token_low, high), size=repeat_length, replace=False)
        rows.append(np.concatenate([base, base]))
    return torch.tensor(np.stack(rows), dtype=torch.long)


def empty_role_store() -> dict[str, list[np.ndarray]]:
    return {"previous_token": [], "bos": [], "repeat_match": []}


def role_scores_from_attentions(
    attentions: tuple[torch.Tensor, ...],
    lengths: list[int],
    repeat_length: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Return role -> list of per-layer arrays [n_heads]."""
    scores = empty_role_store()

    for attention in attentions:
        # [batch, heads, seq, seq]
        attention = attention.detach().float().cpu()
        batch_size, n_heads, _, _ = attention.shape
        prev_values = []
        bos_values = []
        repeat_values = []

        for batch_idx in range(batch_size):
            length = int(lengths[batch_idx])
            if length > 1:
                q_prev = torch.arange(1, length)
                k_prev = q_prev - 1
                prev_values.append(attention[batch_idx, :, q_prev, k_prev])
                bos_values.append(attention[batch_idx, :, 1:length, 0])

            if repeat_length is not None and length >= 2 * repeat_length:
                q_repeat = torch.arange(repeat_length, 2 * repeat_length)
                k_repeat = q_repeat - repeat_length
                repeat_values.append(attention[batch_idx, :, q_repeat, k_repeat])

        if prev_values:
            prev = torch.cat(prev_values, dim=1).mean(dim=1).numpy()
            bos = torch.cat(bos_values, dim=1).mean(dim=1).numpy()
        else:
            prev = np.zeros(n_heads, dtype=np.float32)
            bos = np.zeros(n_heads, dtype=np.float32)

        if repeat_values:
            repeat = torch.cat(repeat_values, dim=1).mean(dim=1).numpy()
        else:
            repeat = np.zeros(n_heads, dtype=np.float32)

        scores["previous_token"].append(prev)
        scores["bos"].append(bos)
        scores["repeat_match"].append(repeat)

    return scores


def add_role_scores(
    total: dict[str, list[np.ndarray]] | None,
    batch_scores: dict[str, list[np.ndarray]],
    weight: int,
    weights: dict[str, int],
) -> dict[str, list[np.ndarray]]:
    if total is None:
        total = {role: [np.zeros_like(layer) for layer in layers] for role, layers in batch_scores.items()}
    for role, layers in batch_scores.items():
        if weight == 0:
            continue
        weights[role] += weight
        for layer_idx, layer_scores in enumerate(layers):
            total[role][layer_idx] += layer_scores * weight
    return total


def normalize_role_scores(
    total: dict[str, list[np.ndarray]],
    weights: dict[str, int],
) -> dict[str, list[np.ndarray]]:
    normalized = {}
    for role, layers in total.items():
        if weights[role] == 0:
            continue
        normalized[role] = [layer / weights[role] for layer in layers]
    return normalized


def extract_seed_role_scores(
    model,
    tokenizer,
    natural_texts: list[str],
    synthetic_ids: torch.Tensor,
    max_length: int,
    batch_size: int,
    repeat_length: int,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    total: dict[str, list[np.ndarray]] | None = None
    weights: dict[str, int] = defaultdict(int)

    for text_batch in batched(natural_texts, batch_size):
        encoded = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        batch_scores = role_scores_from_attentions(outputs.attentions, lengths)
        total = add_role_scores(total, batch_scores, len(text_batch), weights)

    for start in range(0, synthetic_ids.shape[0], batch_size):
        ids = synthetic_ids[start : start + batch_size].to(device)
        mask = torch.ones_like(ids)
        lengths = [ids.shape[1]] * ids.shape[0]
        with torch.inference_mode():
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        batch_scores = role_scores_from_attentions(outputs.attentions, lengths, repeat_length=repeat_length)
        # Synthetic repeated sequences are only used for repeat_match.
        synthetic_only = {
            "previous_token": [np.zeros_like(layer) for layer in batch_scores["previous_token"]],
            "bos": [np.zeros_like(layer) for layer in batch_scores["bos"]],
            "repeat_match": batch_scores["repeat_match"],
        }
        total = add_role_scores(total, synthetic_only, ids.shape[0], weights)

    if total is None:
        raise RuntimeError("No role scores were extracted.")
    return normalize_role_scores(total, weights)


def specialization_distribution(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.maximum(scores.astype(np.float64), 0.0)
    total = clipped.sum()
    if total <= eps:
        return np.ones_like(clipped) / clipped.size
    return clipped / total


def entropy(distribution: np.ndarray, eps: float = 1e-12) -> float:
    p = np.maximum(distribution, eps)
    return float(-(p * np.log(p)).sum())


def distribution_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # For probability distributions, 1 - total variation distance.
    return float(1.0 - 0.5 * np.abs(a - b).sum())


def load_alignment(path: Path | None) -> dict[tuple[str, str, int], list[tuple[int, int]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    alignments = {}
    for item in payload["layer_metrics"]:
        key = (str(item["seed_a"]), str(item["seed_b"]), int(item["layer"]))
        alignments[key] = [(int(a), int(b)) for a, b in item["best_assignment"]]
    return alignments


def apply_assignment(vector: np.ndarray, assignment: list[tuple[int, int]], direction: str) -> np.ndarray:
    aligned = np.zeros_like(vector)
    if direction == "a_to_b":
        for head_a, head_b in assignment:
            aligned[head_a] = vector[head_b]
    elif direction == "b_to_a":
        for head_a, head_b in assignment:
            aligned[head_b] = vector[head_a]
    else:
        raise ValueError(f"Unknown assignment direction: {direction}")
    return aligned


def find_assignment(
    alignments: dict[tuple[str, str, int], list[tuple[int, int]]],
    seed_a: str,
    seed_b: str,
    layer: int,
) -> tuple[list[tuple[int, int]], bool] | None:
    direct = alignments.get((seed_a, seed_b, layer))
    if direct is not None:
        return direct, False
    reverse = alignments.get((seed_b, seed_a, layer))
    if reverse is not None:
        return reverse, True
    return None


def write_head_scores(path: Path, rows: list[HeadRoleScore]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_layer_summaries(path: Path, rows: list[LayerRoleSummary]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_pair_consistency(path: Path, rows: list[PairRoleConsistency]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_role_summary(path: Path, summary: dict[str, dict[str, float]]) -> None:
    fieldnames = [
        "role",
        "n_layer_pairs",
        "raw_distribution_similarity_mean",
        "aligned_distribution_similarity_mean",
        "random_distribution_similarity_mean",
        "raw_top_head_match_rate",
        "aligned_top_head_match_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for role, values in sorted(summary.items()):
            writer.writerow({"role": role, **values})


def write_role_layer_consistency_summary(path: Path, rows: list[PairRoleConsistency]) -> None:
    fieldnames = [
        "role",
        "layer",
        "n_seed_pairs",
        "raw_distribution_similarity_mean",
        "aligned_distribution_similarity_mean",
        "random_distribution_similarity_mean",
        "raw_top_head_match_rate",
        "aligned_top_head_match_rate",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.role, row.layer)].append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (role, layer), group in sorted(grouped.items()):
            aligned = [
                item.aligned_distribution_similarity
                for item in group
                if item.aligned_distribution_similarity is not None
            ]
            aligned_top = [
                item.aligned_top_head_match
                for item in group
                if item.aligned_top_head_match is not None
            ]
            writer.writerow(
                {
                    "role": role,
                    "layer": layer,
                    "n_seed_pairs": len(group),
                    "raw_distribution_similarity_mean": float(
                        np.mean([item.raw_distribution_similarity for item in group])
                    ),
                    "aligned_distribution_similarity_mean": None if not aligned else float(np.mean(aligned)),
                    "random_distribution_similarity_mean": float(
                        np.mean([item.random_distribution_similarity for item in group])
                    ),
                    "raw_top_head_match_rate": float(np.mean([item.raw_top_head_match for item in group])),
                    "aligned_top_head_match_rate": None if not aligned_top else float(np.mean(aligned_top)),
                }
            )


def write_layer_role_specialization_summary(path: Path, rows: list[LayerRoleSummary]) -> None:
    fieldnames = [
        "role",
        "layer",
        "n_seeds",
        "max_specialization_mean",
        "effective_heads_mean",
        "top_score_mean",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.role, row.layer)].append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (role, layer), group in sorted(grouped.items()):
            writer.writerow(
                {
                    "role": role,
                    "layer": layer,
                    "n_seeds": len(group),
                    "max_specialization_mean": float(np.mean([item.max_specialization for item in group])),
                    "effective_heads_mean": float(np.mean([item.effective_heads for item in group])),
                    "top_score_mean": float(np.mean([item.top_score for item in group])),
                }
            )


def summarize_pair_consistency(rows: list[PairRoleConsistency]) -> dict[str, dict[str, float]]:
    by_role = defaultdict(list)
    for row in rows:
        by_role[row.role].append(row)

    summary = {}
    for role, role_rows in by_role.items():
        aligned = [
            row.aligned_distribution_similarity
            for row in role_rows
            if row.aligned_distribution_similarity is not None
        ]
        aligned_top = [
            row.aligned_top_head_match
            for row in role_rows
            if row.aligned_top_head_match is not None
        ]
        summary[role] = {
            "n_layer_pairs": float(len(role_rows)),
            "raw_distribution_similarity_mean": float(np.mean([r.raw_distribution_similarity for r in role_rows])),
            "aligned_distribution_similarity_mean": None if not aligned else float(np.mean(aligned)),
            "random_distribution_similarity_mean": float(np.mean([r.random_distribution_similarity for r in role_rows])),
            "raw_top_head_match_rate": float(np.mean([r.raw_top_head_match for r in role_rows])),
            "aligned_top_head_match_rate": None if not aligned_top else float(np.mean(aligned_top)),
        }
    return summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(0)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    natural_texts = load_probe_texts(args.probe_file, args.num_texts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignments = load_alignment(args.alignment_summary)

    seed_role_scores = {}
    model_names = {}

    for seed in args.seeds:
        model_name = model_name_from_template(args.model_template, args.model_size, seed)
        model_names[seed] = model_name
        print(f"loading seed={seed} model={model_name} revision={args.revision} device={device}", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        synthetic_ids = synthetic_repeated_token_ids(
            vocab_size=len(tokenizer),
            n_sequences=args.synthetic_repeat_sequences,
            repeat_length=args.synthetic_repeat_length,
            token_low=args.synthetic_token_low,
            rng=rng,
        )
        seed_role_scores[seed] = extract_seed_role_scores(
            model=model,
            tokenizer=tokenizer,
            natural_texts=natural_texts,
            synthetic_ids=synthetic_ids,
            max_length=args.max_length,
            batch_size=args.batch_size,
            repeat_length=args.synthetic_repeat_length,
            device=device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    head_rows: list[HeadRoleScore] = []
    layer_rows: list[LayerRoleSummary] = []
    distributions: dict[tuple[str, int, str], np.ndarray] = {}

    for seed, role_scores in seed_role_scores.items():
        for role, layers in role_scores.items():
            for layer_idx, scores in enumerate(layers):
                dist = specialization_distribution(scores)
                distributions[(seed, layer_idx, role)] = dist
                top_head = int(np.argmax(dist))
                layer_rows.append(
                    LayerRoleSummary(
                        seed=seed,
                        layer=layer_idx,
                        role=role,
                        max_specialization=float(dist[top_head]),
                        entropy=entropy(dist),
                        effective_heads=float(np.exp(entropy(dist))),
                        top_head=top_head,
                        top_score=float(scores[top_head]),
                    )
                )
                for head_idx, (score, spec) in enumerate(zip(scores, dist)):
                    head_rows.append(
                        HeadRoleScore(
                            seed=seed,
                            layer=layer_idx,
                            head=head_idx,
                            role=role,
                            score=float(score),
                            specialization=float(spec),
                        )
                    )

    pair_rows: list[PairRoleConsistency] = []
    seeds = [str(seed) for seed in args.seeds]
    roles = sorted({role for _, _, role in distributions})

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            for role in roles:
                n_layers = len(seed_role_scores[seed_a][role])
                for layer_idx in range(n_layers):
                    dist_a = distributions[(seed_a, layer_idx, role)]
                    dist_b = distributions[(seed_b, layer_idx, role)]
                    raw_similarity = distribution_similarity(dist_a, dist_b)

                    random_scores = []
                    for _ in range(args.random_permutations):
                        random_scores.append(distribution_similarity(dist_a, rng.permutation(dist_b)))

                    assignment_result = find_assignment(alignments, seed_a, seed_b, layer_idx)
                    aligned_similarity = None
                    aligned_top_match = None
                    if assignment_result is not None:
                        assignment, reversed_order = assignment_result
                        if reversed_order:
                            aligned_b = apply_assignment(dist_b, assignment, "b_to_a")
                        else:
                            aligned_b = apply_assignment(dist_b, assignment, "a_to_b")
                        aligned_similarity = distribution_similarity(dist_a, aligned_b)
                        aligned_top_match = bool(np.argmax(dist_a) == np.argmax(aligned_b))

                    pair_rows.append(
                        PairRoleConsistency(
                            seed_a=seed_a,
                            seed_b=seed_b,
                            layer=layer_idx,
                            role=role,
                            raw_distribution_similarity=raw_similarity,
                            aligned_distribution_similarity=aligned_similarity,
                            random_distribution_similarity=float(np.mean(random_scores)),
                            raw_top_head_match=bool(np.argmax(dist_a) == np.argmax(dist_b)),
                            aligned_top_head_match=aligned_top_match,
                            top_head_a=int(np.argmax(dist_a)),
                            top_head_b=int(np.argmax(dist_b)),
                        )
                    )

    write_head_scores(args.output_dir / "head_role_scores.csv", head_rows)
    write_layer_summaries(args.output_dir / "layer_role_summary.csv", layer_rows)
    write_pair_consistency(args.output_dir / "pair_role_consistency.csv", pair_rows)

    summary_by_role = summarize_pair_consistency(pair_rows)
    write_role_summary(args.output_dir / "role_consistency_summary.csv", summary_by_role)
    write_role_layer_consistency_summary(args.output_dir / "role_layer_consistency_summary.csv", pair_rows)
    write_layer_role_specialization_summary(args.output_dir / "role_layer_specialization_summary.csv", layer_rows)

    summary = {
        "args": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "probe_file": str(args.probe_file) if args.probe_file else None,
            "alignment_summary": str(args.alignment_summary) if args.alignment_summary else None,
        },
        "model_names": model_names,
        "summary_by_role": summary_by_role,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["summary_by_role"], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
