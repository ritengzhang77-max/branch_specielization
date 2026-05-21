#!/usr/bin/env python3
"""Causally test repeat-match heads by head-output ablation.

The script uses the role-specialization output to identify top repeat-match
heads, then measures how much zeroing those heads changes repeated-token
next-token prediction loss. It compares own top heads, random same-layer heads,
and heads transferred from other seeds by raw-score Hungarian alignment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.gpt_neox import modeling_gpt_neox

from attention_stability import (
    load_model_and_tokenizer,
    model_name_from_template,
    resolve_device,
    resolve_dtype,
)
from attention_role_specialization import synthetic_repeated_token_ids


@dataclass
class AblationResult:
    target_seed: str
    condition: str
    source_seed: str | None
    control_id: int | None
    heads_json: str
    baseline_loss: float
    ablated_loss: float
    loss_delta: float
    baseline_target_logit: float
    ablated_target_logit: float
    target_logit_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-template",
        default="EleutherAI/pythia-{model_size}-seed{seed}",
        help="Hugging Face model template. Available fields: {model_size}, {seed}.",
    )
    parser.add_argument("--model-size", default="160m")
    parser.add_argument("--seeds", nargs="+", default=[str(i) for i in range(1, 10)])
    parser.add_argument("--revision", default="step143000")
    parser.add_argument(
        "--role-scores",
        type=Path,
        default=Path("results/phase1_pythia160m_attention_role_specialization/head_role_scores.csv"),
    )
    parser.add_argument(
        "--alignment-summary",
        type=Path,
        default=Path("results/phase0_pythia160m_seed1_to_9_step143000_raw_scores/summary.json"),
    )
    parser.add_argument("--role", default="repeat_match")
    parser.add_argument("--layers", default="0,1", help="Comma-separated layers to test.")
    parser.add_argument("--top-k-per-layer", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--repeat-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-token-low", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1_repeat_match_ablation"))
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(item.strip()) for item in layers.split(",") if item.strip()]


def load_top_heads(
    path: Path,
    role: str,
    layers: list[int],
    top_k_per_layer: int,
) -> tuple[dict[tuple[str, int], list[int]], int]:
    by_seed_layer = defaultdict(list)
    max_head = -1
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["role"] != role:
                continue
            layer = int(row["layer"])
            if layer not in layers:
                continue
            seed = str(row["seed"])
            head = int(row["head"])
            max_head = max(max_head, head)
            by_seed_layer[(seed, layer)].append((float(row["specialization"]), head))

    top_heads = {}
    for key, values in by_seed_layer.items():
        values = sorted(values, reverse=True)
        top_heads[key] = [head for _, head in values[:top_k_per_layer]]

    if max_head < 0:
        raise ValueError(f"No heads found for role={role} layers={layers} in {path}.")
    return top_heads, max_head + 1


def load_alignment(path: Path) -> dict[tuple[str, str, int], list[tuple[int, int]]]:
    payload = json.loads(path.read_text())
    alignments = {}
    for item in payload["layer_metrics"]:
        key = (str(item["seed_a"]), str(item["seed_b"]), int(item["layer"]))
        alignments[key] = [(int(a), int(b)) for a, b in item["best_assignment"]]
    return alignments


def map_head_between_seeds(
    alignments: dict[tuple[str, str, int], list[tuple[int, int]]],
    source_seed: str,
    target_seed: str,
    layer: int,
    source_head: int,
) -> int:
    direct = alignments.get((source_seed, target_seed, layer))
    if direct is not None:
        mapping = {head_a: head_b for head_a, head_b in direct}
        return mapping[source_head]

    reverse = alignments.get((target_seed, source_seed, layer))
    if reverse is not None:
        mapping = {head_b: head_a for head_a, head_b in reverse}
        return mapping[source_head]

    raise KeyError(f"No alignment found for {source_seed}->{target_seed} layer {layer}.")


def own_top_heads(seed: str, layers: list[int], top_heads: dict[tuple[str, int], list[int]]) -> list[tuple[int, int]]:
    heads = []
    for layer in layers:
        for head in top_heads[(seed, layer)]:
            heads.append((layer, head))
    return heads


def random_heads(
    seed: str,
    layers: list[int],
    top_heads: dict[tuple[str, int], list[int]],
    n_heads: int,
    top_k_per_layer: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    heads = []
    for layer in layers:
        excluded = set(top_heads[(seed, layer)])
        candidates = [head for head in range(n_heads) if head not in excluded]
        sampled = rng.choice(candidates, size=top_k_per_layer, replace=False)
        heads.extend((layer, int(head)) for head in sampled)
    return heads


def same_index_transfer(
    source_seed: str,
    layers: list[int],
    top_heads: dict[tuple[str, int], list[int]],
) -> list[tuple[int, int]]:
    return own_top_heads(source_seed, layers, top_heads)


def aligned_transfer(
    source_seed: str,
    target_seed: str,
    layers: list[int],
    top_heads: dict[tuple[str, int], list[int]],
    alignments: dict[tuple[str, str, int], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    heads = []
    for layer in layers:
        for source_head in top_heads[(source_seed, layer)]:
            target_head = map_head_between_seeds(alignments, source_seed, target_seed, layer, source_head)
            heads.append((layer, target_head))
    return heads


@contextlib.contextmanager
def ablate_gpt_neox_heads(heads: list[tuple[int, int]] | None):
    """Zero selected per-head attention outputs before the output projection."""
    if not heads:
        yield
        return

    by_layer = defaultdict(list)
    for layer, head in heads:
        by_layer[int(layer)].append(int(head))

    original_forward = modeling_gpt_neox.eager_attention_forward

    def ablation_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        head_mask=None,
        **kwargs,
    ):
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

        attn_output = torch.matmul(attn_weights, value)
        layer_heads = by_layer.get(int(module.layer_idx), [])
        if layer_heads:
            attn_output[:, layer_heads, :, :] = 0
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    modeling_gpt_neox.eager_attention_forward = ablation_attention_forward
    try:
        yield
    finally:
        modeling_gpt_neox.eager_attention_forward = original_forward


def repeat_position_loss_and_logit(logits: torch.Tensor, input_ids: torch.Tensor, repeat_length: int) -> tuple[float, float]:
    # Logits at position j predict input_ids[j + 1]. Score only second-half
    # continuation positions: x_i(second occurrence) -> x_{i+1}(second occurrence).
    positions = torch.arange(repeat_length, 2 * repeat_length - 1, device=logits.device)
    selected_logits = logits[:, positions, :]
    targets = input_ids[:, positions + 1]
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    target_logits = selected_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(loss.detach().cpu()), float(target_logits.mean().detach().cpu())


def evaluate_condition(
    model,
    input_ids: torch.Tensor,
    repeat_length: int,
    batch_size: int,
    device: torch.device,
    heads: list[tuple[int, int]] | None,
) -> tuple[float, float]:
    losses = []
    logits = []
    with ablate_gpt_neox_heads(heads):
        for start in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
            loss, target_logit = repeat_position_loss_and_logit(outputs.logits, ids, repeat_length)
            losses.append(loss * ids.shape[0])
            logits.append(target_logit * ids.shape[0])
    n = input_ids.shape[0]
    return sum(losses) / n, sum(logits) / n


def write_results(path: Path, rows: list[AblationResult]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[AblationResult]) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)

    summary = {}
    fieldnames = [
        "condition",
        "n",
        "loss_delta_mean",
        "loss_delta_std",
        "target_logit_delta_mean",
        "target_logit_delta_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, group in sorted(grouped.items()):
            loss_delta = np.asarray([row.loss_delta for row in group], dtype=np.float64)
            logit_delta = np.asarray([row.target_logit_delta for row in group], dtype=np.float64)
            item = {
                "n": float(len(group)),
                "loss_delta_mean": float(loss_delta.mean()),
                "loss_delta_std": float(loss_delta.std()),
                "target_logit_delta_mean": float(logit_delta.mean()),
                "target_logit_delta_std": float(logit_delta.std()),
            }
            summary[condition] = item
            writer.writerow({"condition": condition, **item})
    return summary


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    rng = np.random.default_rng(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    top_heads, n_heads = load_top_heads(args.role_scores, args.role, layers, args.top_k_per_layer)
    alignments = load_alignment(args.alignment_summary)
    rows: list[AblationResult] = []

    model_names = {}
    eval_ids = None
    eval_vocab_size = None
    for target_seed in [str(seed) for seed in args.seeds]:
        model_name = model_name_from_template(args.model_template, args.model_size, target_seed)
        model_names[target_seed] = model_name
        print(f"loading seed={target_seed} model={model_name} revision={args.revision} device={device}", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        if eval_ids is None:
            eval_vocab_size = len(tokenizer)
            eval_ids = synthetic_repeated_token_ids(
                vocab_size=eval_vocab_size,
                n_sequences=args.eval_sequences,
                repeat_length=args.repeat_length,
                token_low=args.synthetic_token_low,
                rng=rng,
            )
        elif len(tokenizer) != eval_vocab_size:
            raise ValueError("Tokenizer vocabulary size changed across seeds; cannot reuse synthetic eval ids.")

        baseline_loss, baseline_logit = evaluate_condition(
            model=model,
            input_ids=eval_ids,
            repeat_length=args.repeat_length,
            batch_size=args.batch_size,
            device=device,
            heads=None,
        )

        condition_heads = [("own_top", None, None, own_top_heads(target_seed, layers, top_heads))]
        for control_id in range(args.random_controls):
            condition_heads.append(
                (
                    "own_random",
                    None,
                    control_id,
                    random_heads(target_seed, layers, top_heads, n_heads, args.top_k_per_layer, rng),
                )
            )
        for source_seed in [str(seed) for seed in args.seeds]:
            if source_seed == target_seed:
                continue
            condition_heads.append(
                ("source_same_index", source_seed, None, same_index_transfer(source_seed, layers, top_heads))
            )
            condition_heads.append(
                (
                    "source_aligned",
                    source_seed,
                    None,
                    aligned_transfer(source_seed, target_seed, layers, top_heads, alignments),
                )
            )

        for condition, source_seed, control_id, heads in condition_heads:
            ablated_loss, ablated_logit = evaluate_condition(
                model=model,
                input_ids=eval_ids,
                repeat_length=args.repeat_length,
                batch_size=args.batch_size,
                device=device,
                heads=heads,
            )
            rows.append(
                AblationResult(
                    target_seed=target_seed,
                    condition=condition,
                    source_seed=source_seed,
                    control_id=control_id,
                    heads_json=json.dumps(heads),
                    baseline_loss=baseline_loss,
                    ablated_loss=ablated_loss,
                    loss_delta=ablated_loss - baseline_loss,
                    baseline_target_logit=baseline_logit,
                    ablated_target_logit=ablated_logit,
                    target_logit_delta=ablated_logit - baseline_logit,
                )
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_results(args.output_dir / "ablation_results.csv", rows)
    summary = write_summary(args.output_dir / "condition_summary.csv", rows)
    payload = {
        "args": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "role_scores": str(args.role_scores),
            "alignment_summary": str(args.alignment_summary),
        },
        "model_names": model_names,
        "summary_by_condition": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
