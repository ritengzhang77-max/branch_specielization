#!/usr/bin/env python3
"""Compute cross-seed attention-head similarity.

This is the Phase 0 calibration script. It extracts attention matrices for a
fixed probe text set, compares heads across seeds, and reports raw head-index
similarity, Hungarian-matched similarity, and a random-permutation baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROBE_TEXTS = [
    "The lawyer gave the client a contract because the client asked for legal advice.",
    "When Mary and John went to the store, John gave a book to Mary before dinner.",
    "A B A C A B A",
    "Paris is in France. Berlin is in Germany. Rome is in Italy.",
    "The quick brown fox jumped over the lazy dog and then ran back to the yard.",
    "If the sequence is red blue red green red blue, the next repeated color is",
    "The teacher praised the student because the student solved the difficult problem.",
    "In 2019 the company grew quickly, but in 2020 the market changed sharply.",
]


@dataclass
class LayerPairMetrics:
    seed_a: str
    seed_b: str
    layer: int
    n_heads_a: int
    n_heads_b: int
    raw_diag_mean: float | None
    raw_diag_std: float | None
    matched_mean: float
    matched_std: float
    random_perm_mean: float
    random_perm_std: float
    matched_minus_random: float
    best_assignment: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-template",
        default="EleutherAI/pythia-{model_size}-seed{seed}",
        help="Hugging Face model template. Available fields: {model_size}, {seed}.",
    )
    parser.add_argument("--model-size", default="14m", help="Model size used in the template.")
    parser.add_argument("--seeds", nargs="+", default=["1", "2"], help="Seed identifiers.")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision, e.g. main or step143000 for Pythia checkpoints.",
    )
    parser.add_argument("--probe-file", type=Path, default=None, help="Optional newline-delimited probe texts.")
    parser.add_argument("--num-texts", type=int, default=None, help="Limit probe texts for a smoke run.")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype. Use float32 on CPU.",
    )
    parser.add_argument("--random-permutations", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase0_attention_stability"))
    parser.add_argument("--save-similarity-npz", action="store_true")
    return parser.parse_args()


def load_probe_texts(path: Path | None, num_texts: int | None) -> list[str]:
    if path is None:
        texts = list(DEFAULT_PROBE_TEXTS)
    else:
        texts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if num_texts is not None:
        texts = texts[:num_texts]
    if not texts:
        raise ValueError("No probe texts provided.")
    return texts


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def model_name_from_template(template: str, model_size: str, seed: str) -> str:
    return template.format(model_size=model_size, seed=seed)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_model_and_tokenizer(
    model_name: str,
    revision: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def extract_attention_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Return one array per layer with shape [n_heads, flattened_probe_entries]."""
    layer_chunks: list[list[torch.Tensor]] | None = None

    for text_batch in batched(texts, batch_size):
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

        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Try an eager attention implementation.")

        if layer_chunks is None:
            layer_chunks = [[] for _ in outputs.attentions]

        for layer_idx, attention in enumerate(outputs.attentions):
            # attention: [batch, heads, seq, seq]
            attention = attention.detach().float().cpu()
            n_heads = attention.shape[1]
            head_vectors = []
            for head_idx in range(n_heads):
                pieces = []
                for batch_idx, length in enumerate(lengths):
                    pieces.append(attention[batch_idx, head_idx, :length, :length].reshape(-1))
                head_vectors.append(torch.cat(pieces))
            layer_chunks[layer_idx].append(torch.stack(head_vectors, dim=0))

    if layer_chunks is None:
        raise RuntimeError("No attention chunks were collected.")

    return [torch.cat(chunks, dim=1).numpy() for chunks in layer_chunks]


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


def compare_layer(
    seed_a: str,
    seed_b: str,
    layer: int,
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    random_permutations: int,
    rng: np.random.Generator,
) -> tuple[LayerPairMetrics, np.ndarray]:
    similarity = cosine_similarity_matrix(vecs_a, vecs_b)

    raw_diag = None
    if similarity.shape[0] == similarity.shape[1]:
        raw_diag = np.diag(similarity)

    rows, cols = linear_sum_assignment(-similarity)
    matched = similarity[rows, cols]
    random_scores = random_permutation_scores(similarity, random_permutations, rng)

    metrics = LayerPairMetrics(
        seed_a=seed_a,
        seed_b=seed_b,
        layer=layer,
        n_heads_a=int(similarity.shape[0]),
        n_heads_b=int(similarity.shape[1]),
        raw_diag_mean=None if raw_diag is None else float(raw_diag.mean()),
        raw_diag_std=None if raw_diag is None else float(raw_diag.std()),
        matched_mean=float(matched.mean()),
        matched_std=float(matched.std()),
        random_perm_mean=float(random_scores.mean()),
        random_perm_std=float(random_scores.std()),
        matched_minus_random=float(matched.mean() - random_scores.mean()),
        best_assignment=[[int(row), int(col)] for row, col in zip(rows, cols)],
    )
    return metrics, similarity


def write_metrics_csv(path: Path, metrics: list[LayerPairMetrics]) -> None:
    fieldnames = [
        "seed_a",
        "seed_b",
        "layer",
        "n_heads_a",
        "n_heads_b",
        "raw_diag_mean",
        "raw_diag_std",
        "matched_mean",
        "matched_std",
        "random_perm_mean",
        "random_perm_std",
        "matched_minus_random",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            row = asdict(item)
            row.pop("best_assignment")
            writer.writerow(row)


def write_layer_summary_csv(path: Path, metrics: list[LayerPairMetrics]) -> None:
    fieldnames = [
        "layer",
        "n_seed_pairs",
        "raw_diag_mean",
        "matched_mean",
        "random_perm_mean",
        "matched_minus_random_mean",
        "matched_minus_random_min",
        "matched_minus_random_max",
    ]
    layers = sorted({item.layer for item in metrics})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in layers:
            layer_items = [item for item in metrics if item.layer == layer]
            raw_values = [item.raw_diag_mean for item in layer_items if item.raw_diag_mean is not None]
            gaps = [item.matched_minus_random for item in layer_items]
            writer.writerow(
                {
                    "layer": layer,
                    "n_seed_pairs": len(layer_items),
                    "raw_diag_mean": None if not raw_values else float(np.mean(raw_values)),
                    "matched_mean": float(np.mean([item.matched_mean for item in layer_items])),
                    "random_perm_mean": float(np.mean([item.random_perm_mean for item in layer_items])),
                    "matched_minus_random_mean": float(np.mean(gaps)),
                    "matched_minus_random_min": float(np.min(gaps)),
                    "matched_minus_random_max": float(np.max(gaps)),
                }
            )


def summarize(metrics: list[LayerPairMetrics]) -> dict:
    raw_values = [m.raw_diag_mean for m in metrics if m.raw_diag_mean is not None]
    matched_values = [m.matched_mean for m in metrics]
    gap_values = [m.matched_minus_random for m in metrics]
    return {
        "n_layer_pairs": len(metrics),
        "raw_diag_mean_over_layers": None if not raw_values else float(np.mean(raw_values)),
        "matched_mean_over_layers": float(np.mean(matched_values)),
        "matched_minus_random_mean_over_layers": float(np.mean(gap_values)),
        "min_layer_gap": float(np.min(gap_values)),
        "max_layer_gap": float(np.max(gap_values)),
    }


def main() -> None:
    args = parse_args()
    texts = load_probe_texts(args.probe_file, args.num_texts)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    rng = np.random.default_rng(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_vectors: dict[str, list[np.ndarray]] = {}
    model_names: dict[str, str] = {}

    for seed in args.seeds:
        model_name = model_name_from_template(args.model_template, args.model_size, seed)
        model_names[seed] = model_name
        print(f"loading seed={seed} model={model_name} revision={args.revision} device={device}")
        model, tokenizer = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        seed_vectors[seed] = extract_attention_vectors(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_metrics: list[LayerPairMetrics] = []
    similarity_payload = {}
    seeds = list(args.seeds)

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            layers_a = seed_vectors[seed_a]
            layers_b = seed_vectors[seed_b]
            if len(layers_a) != len(layers_b):
                raise ValueError(f"Layer count mismatch for {seed_a} vs {seed_b}.")
            for layer_idx, (vecs_a, vecs_b) in enumerate(zip(layers_a, layers_b)):
                metrics, similarity = compare_layer(
                    seed_a=seed_a,
                    seed_b=seed_b,
                    layer=layer_idx,
                    vecs_a=vecs_a,
                    vecs_b=vecs_b,
                    random_permutations=args.random_permutations,
                    rng=rng,
                )
                all_metrics.append(metrics)
                if args.save_similarity_npz:
                    similarity_payload[f"{seed_a}_vs_{seed_b}_layer_{layer_idx}"] = similarity

    write_metrics_csv(args.output_dir / "layer_pair_metrics.csv", all_metrics)
    write_layer_summary_csv(args.output_dir / "layer_summary.csv", all_metrics)
    summary = {
        "args": vars(args) | {"output_dir": str(args.output_dir), "probe_file": str(args.probe_file) if args.probe_file else None},
        "model_names": model_names,
        "n_probe_texts": len(texts),
        "summary": summarize(all_metrics),
        "layer_metrics": [asdict(item) for item in all_metrics],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.save_similarity_npz:
        np.savez_compressed(args.output_dir / "similarity_matrices.npz", **similarity_payload)

    print(json.dumps(summary["summary"], indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
