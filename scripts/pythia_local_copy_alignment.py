#!/usr/bin/env python3
"""Test cross-seed transfer of local-copy / previous-token heads in Pythia."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention_role_specialization import specialization_distribution
from attention_stability import (
    compare_layer,
    extract_attention_vectors,
    load_model_and_tokenizer,
    load_probe_texts,
    model_name_from_template,
    resolve_device,
    resolve_dtype,
)
from pythia_repeat_match_alignment_trajectory import (
    aligned_transfer,
    build_alignment_maps,
    same_index_transfer,
    write_condition_summary,
    write_revision_summary,
)
from pythia_repeat_match_checkpoint_trajectory import random_heads
from repeat_match_ablation import ablate_gpt_neox_heads


@dataclass
class ProbeHeadScore:
    model_size: str
    seed: str
    revision: str
    revision_index: int
    layer: int
    head: int
    local_copy_score: float
    specialization: float
    selected_for_ablation: bool
    layer_rank: int


@dataclass
class AblationRow:
    model_size: str
    target_seed: str
    revision: str
    revision_index: int
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


@dataclass
class RevisionSeedSummary:
    model_size: str
    seed: str
    revision: str
    revision_index: int
    selected_heads_json: str
    selected_specialization_mean: float
    selected_specialization_max: float
    baseline_loss: float
    own_top_loss_delta: float
    random_loss_delta_mean: float
    own_top_excess_over_random: float
    source_same_index_loss_delta_mean: float
    source_aligned_loss_delta_mean: float
    aligned_minus_same_index_loss_delta_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-template", default="EleutherAI/pythia-{model_size}-seed{seed}")
    parser.add_argument("--model-size", default="160m")
    parser.add_argument("--seeds", nargs="+", default=[str(i) for i in range(1, 10)])
    parser.add_argument("--target-seeds", nargs="+", default=None)
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--layers", default="3", help="Comma-separated layers to test.")
    parser.add_argument("--top-k-per-layer", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--probe-sequences", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--n-pairs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-token-low", type=int, default=1000)
    parser.add_argument("--separator-token-id", type=int, default=None)
    parser.add_argument("--alignment-probe-file", type=Path, default=Path("probes/phase0_probe_texts.txt"))
    parser.add_argument("--alignment-num-texts", type=int, default=8)
    parser.add_argument("--alignment-max-length", type=int, default=64)
    parser.add_argument("--alignment-batch-size", type=int, default=2)
    parser.add_argument("--random-permutations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_local_copy_alignment"),
    )
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(item.strip()) for item in layers.split(",") if item.strip()]


def synthetic_local_copy_token_ids(
    vocab_size: int,
    n_sequences: int,
    n_pairs: int,
    token_low: int,
    separator_token_id: int | None,
    rng: np.random.Generator,
) -> torch.Tensor:
    sep = token_low if separator_token_id is None else int(separator_token_id)
    low = token_low + 1 if separator_token_id is None else token_low
    high = vocab_size - 1
    if high - low < n_pairs:
        raise ValueError("Vocabulary range is too small for synthetic local-copy sequences.")

    rows = []
    for _ in range(n_sequences):
        values = rng.choice(np.arange(low, high), size=n_pairs, replace=False)
        row = []
        for value in values:
            row.extend([int(value), sep, int(value)])
        rows.append(row)
    return torch.tensor(np.asarray(rows), dtype=torch.long)


def local_copy_positions(n_pairs: int, device: torch.device) -> torch.Tensor:
    return torch.arange(1, 3 * n_pairs, 3, device=device)


def extract_local_copy_scores(
    model,
    input_ids: torch.Tensor,
    n_pairs: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    totals: list[np.ndarray] | None = None
    total_examples = 0
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = torch.ones_like(ids)
        positions = local_copy_positions(n_pairs, device)
        with torch.inference_mode():
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        layer_scores = []
        for attention in outputs.attentions:
            attention = attention.detach().float()
            scores = attention[:, :, positions, positions - 1].mean(dim=(0, 2)).cpu().numpy()
            layer_scores.append(scores)
        if totals is None:
            totals = [np.zeros_like(scores, dtype=np.float64) for scores in layer_scores]
        for layer_idx, scores in enumerate(layer_scores):
            totals[layer_idx] += scores.astype(np.float64) * ids.shape[0]
        total_examples += ids.shape[0]
    if totals is None or total_examples == 0:
        raise RuntimeError("No local-copy probe batches were evaluated.")
    return [scores / total_examples for scores in totals]


def selected_heads_from_scores(
    scores_by_layer: list[np.ndarray],
    layers: list[int],
    top_k_per_layer: int,
) -> tuple[list[tuple[int, int]], list[ProbeHeadScore]]:
    selected = []
    selected_set = set()
    ranks_by_layer = {}
    for layer in layers:
        order = list(np.argsort(-scores_by_layer[layer]))
        ranks_by_layer[layer] = {head: rank + 1 for rank, head in enumerate(order)}
        for head in order[:top_k_per_layer]:
            selected.append((layer, int(head)))
            selected_set.add((layer, int(head)))

    rows = []
    for layer in layers:
        distribution = specialization_distribution(scores_by_layer[layer])
        for head, score in enumerate(scores_by_layer[layer]):
            rows.append(
                ProbeHeadScore(
                    model_size="",
                    seed="",
                    revision="",
                    revision_index=-1,
                    layer=layer,
                    head=head,
                    local_copy_score=float(score),
                    specialization=float(distribution[head]),
                    selected_for_ablation=(layer, head) in selected_set,
                    layer_rank=int(ranks_by_layer[layer][head]),
                )
            )
    return selected, rows


def local_copy_loss_and_logit(logits: torch.Tensor, input_ids: torch.Tensor, n_pairs: int) -> tuple[float, float]:
    positions = local_copy_positions(n_pairs, logits.device)
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
    n_pairs: int,
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
                outputs = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            loss, target_logit = local_copy_loss_and_logit(outputs.logits, ids, n_pairs)
            losses.append(loss * ids.shape[0])
            logits.append(target_logit * ids.shape[0])
    n = input_ids.shape[0]
    return sum(losses) / n, sum(logits) / n


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def maybe_write_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [str(row["revision"]) for row in rows]
    x = np.arange(len(rows))
    metrics = [
        ("selected_specialization_mean_mean", "local-copy specialization"),
        ("own_top_excess_over_random_mean", "own top minus random"),
        ("source_same_index_loss_delta_mean_mean", "same-index transfer"),
        ("source_aligned_loss_delta_mean_mean", "aligned transfer"),
        ("aligned_minus_same_index_loss_delta_mean_mean", "aligned minus same-index"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.0))
    axes_flat = axes.ravel()
    for ax, (field, title) in zip(axes_flat, metrics):
        ax.plot(x, [float(row[field]) for row in rows], marker="o")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(alpha=0.25)
    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "local_copy_alignment.png", dpi=180)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    probe_rows: list[ProbeHeadScore],
    alignment_rows: list[object],
    ablation_rows: list[AblationRow],
    seed_summary_rows: list[RevisionSeedSummary],
    args: argparse.Namespace,
    model_names: dict[str, str],
) -> None:
    write_dataclass_csv(output_dir / "probe_head_scores.csv", probe_rows)
    write_dataclass_csv(output_dir / "alignment_rows.csv", alignment_rows)
    write_dataclass_csv(output_dir / "ablation_results.csv", ablation_rows)
    write_dataclass_csv(output_dir / "revision_seed_summary.csv", seed_summary_rows)
    condition_summary = write_condition_summary(output_dir / "condition_summary.csv", ablation_rows)
    revision_summary = write_revision_summary(output_dir / "revision_summary.csv", seed_summary_rows)
    maybe_write_plot(output_dir, revision_summary)
    payload = {
        "args": vars(args)
        | {
            "output_dir": str(output_dir),
            "alignment_probe_file": str(args.alignment_probe_file),
        },
        "model_names": model_names,
        "condition_summary": condition_summary,
        "revision_summary": revision_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_texts = load_probe_texts(args.alignment_probe_file, args.alignment_num_texts)
    rng = np.random.default_rng(0)

    probe_rows: list[ProbeHeadScore] = []
    alignment_rows = []
    ablation_rows: list[AblationRow] = []
    seed_summary_rows: list[RevisionSeedSummary] = []
    model_names = {
        str(seed): model_name_from_template(args.model_template, args.model_size, str(seed))
        for seed in args.seeds
    }
    target_seeds = [str(seed) for seed in (args.target_seeds or args.seeds)]
    probe_ids = None
    eval_ids = None
    vocab_size = None

    print(f"revision={args.revision}", flush=True)
    seed_scores: dict[str, list[np.ndarray]] = {}
    seed_selected_heads: dict[str, list[tuple[int, int]]] = {}
    seed_vectors: dict[str, list[np.ndarray]] = {}

    for seed in [str(seed) for seed in args.seeds]:
        model_name = model_names[seed]
        print(f"  probing seed={seed} model={model_name} revision={args.revision}", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        if probe_ids is None:
            vocab_size = len(tokenizer)
            probe_ids = synthetic_local_copy_token_ids(
                vocab_size=vocab_size,
                n_sequences=args.probe_sequences,
                n_pairs=args.n_pairs,
                token_low=args.synthetic_token_low,
                separator_token_id=args.separator_token_id,
                rng=np.random.default_rng(123),
            )
            eval_ids = synthetic_local_copy_token_ids(
                vocab_size=vocab_size,
                n_sequences=args.eval_sequences,
                n_pairs=args.n_pairs,
                token_low=args.synthetic_token_low,
                separator_token_id=args.separator_token_id,
                rng=np.random.default_rng(456),
            )
        elif len(tokenizer) != vocab_size:
            raise ValueError("Tokenizer vocabulary size changed across seeds.")

        assert probe_ids is not None
        scores_by_layer = extract_local_copy_scores(model, probe_ids, args.n_pairs, args.batch_size, device)
        seed_scores[seed] = scores_by_layer
        selected_heads, partial_rows = selected_heads_from_scores(scores_by_layer, layers, args.top_k_per_layer)
        seed_selected_heads[seed] = selected_heads
        for row in partial_rows:
            probe_rows.append(
                ProbeHeadScore(
                    model_size=args.model_size,
                    seed=seed,
                    revision=args.revision,
                    revision_index=0,
                    layer=row.layer,
                    head=row.head,
                    local_copy_score=row.local_copy_score,
                    specialization=row.specialization,
                    selected_for_ablation=row.selected_for_ablation,
                    layer_rank=row.layer_rank,
                )
            )

        seed_vectors[seed] = extract_attention_vectors(
            model=model,
            tokenizer=tokenizer,
            texts=alignment_texts,
            max_length=args.alignment_max_length,
            batch_size=args.alignment_batch_size,
            device=device,
            attention_representation="raw_scores",
            entry_mask="causal",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    alignment_maps, revision_alignment_rows = build_alignment_maps(
        model_size=args.model_size,
        revision=args.revision,
        revision_index=0,
        seed_vectors=seed_vectors,
        layers=layers,
        random_permutations=args.random_permutations,
        rng=rng,
    )
    alignment_rows.extend(revision_alignment_rows)

    assert eval_ids is not None
    write_outputs(args.output_dir, probe_rows, alignment_rows, ablation_rows, seed_summary_rows, args, model_names)

    for target_seed in target_seeds:
        model_name = model_names[target_seed]
        print(f"  ablating target_seed={target_seed} revision={args.revision}", flush=True)
        model, _ = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        baseline_loss, baseline_logit = evaluate_condition(
            model=model,
            input_ids=eval_ids,
            n_pairs=args.n_pairs,
            batch_size=args.batch_size,
            device=device,
            heads=None,
        )
        conditions: list[tuple[str, str | None, int | None, list[tuple[int, int]]]] = [
            ("own_top", None, None, seed_selected_heads[target_seed])
        ]
        for control_id in range(args.random_controls):
            conditions.append(
                (
                    "own_random",
                    None,
                    control_id,
                    random_heads(
                        seed_selected_heads[target_seed],
                        seed_scores[target_seed],
                        layers,
                        args.top_k_per_layer,
                        rng,
                    ),
                )
            )
        for source_seed in [str(seed) for seed in args.seeds]:
            if source_seed == target_seed:
                continue
            source_heads = seed_selected_heads[source_seed]
            conditions.append(("source_same_index", source_seed, None, same_index_transfer(source_heads)))
            conditions.append(
                (
                    "source_aligned",
                    source_seed,
                    None,
                    aligned_transfer(source_seed, target_seed, source_heads, alignment_maps),
                )
            )

        target_rows = []
        for condition, source_seed, control_id, heads in conditions:
            ablated_loss, ablated_logit = evaluate_condition(
                model=model,
                input_ids=eval_ids,
                n_pairs=args.n_pairs,
                batch_size=args.batch_size,
                device=device,
                heads=heads,
            )
            row = AblationRow(
                model_size=args.model_size,
                target_seed=target_seed,
                revision=args.revision,
                revision_index=0,
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
            ablation_rows.append(row)
            target_rows.append(row)

        own_top = [row for row in target_rows if row.condition == "own_top"][0]
        random_loss = np.asarray([row.loss_delta for row in target_rows if row.condition == "own_random"])
        same_loss = np.asarray([row.loss_delta for row in target_rows if row.condition == "source_same_index"])
        aligned_loss = np.asarray([row.loss_delta for row in target_rows if row.condition == "source_aligned"])
        selected_probe_rows = [
            row
            for row in probe_rows
            if row.seed == target_seed and row.selected_for_ablation
        ]
        seed_summary_rows.append(
            RevisionSeedSummary(
                model_size=args.model_size,
                seed=target_seed,
                revision=args.revision,
                revision_index=0,
                selected_heads_json=json.dumps(seed_selected_heads[target_seed]),
                selected_specialization_mean=float(np.mean([row.specialization for row in selected_probe_rows])),
                selected_specialization_max=float(np.max([row.specialization for row in selected_probe_rows])),
                baseline_loss=baseline_loss,
                own_top_loss_delta=own_top.loss_delta,
                random_loss_delta_mean=float(random_loss.mean()),
                own_top_excess_over_random=float(own_top.loss_delta - random_loss.mean()),
                source_same_index_loss_delta_mean=float(same_loss.mean()),
                source_aligned_loss_delta_mean=float(aligned_loss.mean()),
                aligned_minus_same_index_loss_delta_mean=float(aligned_loss.mean() - same_loss.mean()),
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        write_outputs(args.output_dir, probe_rows, alignment_rows, ablation_rows, seed_summary_rows, args, model_names)

    revision_summary = write_revision_summary(args.output_dir / "revision_summary.csv", seed_summary_rows)
    print(json.dumps(revision_summary, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
