#!/usr/bin/env python3
"""Track repeat-match head specialization, alignment, and causal transfer across checkpoints."""

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
    compare_layer,
    extract_attention_vectors,
    load_model_and_tokenizer,
    load_probe_texts,
    model_name_from_template,
    resolve_device,
    resolve_dtype,
)
from pythia_repeat_match_checkpoint_trajectory import (
    extract_repeat_match_scores,
    random_heads,
    selected_heads_from_scores,
)
from repeat_match_ablation import evaluate_condition, synthetic_repeated_token_ids


@dataclass
class AlignmentRow:
    model_size: str
    revision: str
    revision_index: int
    seed_a: str
    seed_b: str
    layer: int
    raw_diag_mean: float | None
    matched_mean: float
    random_perm_mean: float
    matched_minus_random: float
    best_assignment_json: str


@dataclass
class ProbeHeadScore:
    model_size: str
    seed: str
    revision: str
    revision_index: int
    layer: int
    head: int
    repeat_score: float
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
    parser.add_argument(
        "--model-template",
        default="EleutherAI/pythia-{model_size}-seed{seed}",
        help="Hugging Face model template. Available fields: {model_size}, {seed}.",
    )
    parser.add_argument("--model-size", default="160m")
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    parser.add_argument(
        "--revisions",
        nargs="+",
        default=["step0", "step1000", "step4000", "step16000", "step64000", "step143000"],
    )
    parser.add_argument("--layers", default="0,1", help="Comma-separated layers to test.")
    parser.add_argument("--top-k-per-layer", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--probe-sequences", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--repeat-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-token-low", type=int, default=1000)
    parser.add_argument("--alignment-probe-file", type=Path, default=Path("probes/phase0_probe_texts.txt"))
    parser.add_argument("--alignment-num-texts", type=int, default=4)
    parser.add_argument("--alignment-max-length", type=int, default=64)
    parser.add_argument("--alignment-batch-size", type=int, default=2)
    parser.add_argument("--random-permutations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_repeat_match_alignment_trajectory"),
    )
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(item.strip()) for item in layers.split(",") if item.strip()]


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_alignment_maps(
    model_size: str,
    revision: str,
    revision_index: int,
    seed_vectors: dict[str, list[np.ndarray]],
    layers: list[int],
    random_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str, int], dict[int, int]], list[AlignmentRow]]:
    maps: dict[tuple[str, str, int], dict[int, int]] = {}
    rows = []
    seeds = list(seed_vectors)
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            for layer in layers:
                metrics, _ = compare_layer(
                    seed_a=seed_a,
                    seed_b=seed_b,
                    layer=layer,
                    vecs_a=seed_vectors[seed_a][layer],
                    vecs_b=seed_vectors[seed_b][layer],
                    random_permutations=random_permutations,
                    rng=rng,
                )
                direct = {int(a): int(b) for a, b in metrics.best_assignment}
                reverse = {int(b): int(a) for a, b in metrics.best_assignment}
                maps[(seed_a, seed_b, layer)] = direct
                maps[(seed_b, seed_a, layer)] = reverse
                rows.append(
                    AlignmentRow(
                        model_size=model_size,
                        revision=revision,
                        revision_index=revision_index,
                        seed_a=seed_a,
                        seed_b=seed_b,
                        layer=layer,
                        raw_diag_mean=metrics.raw_diag_mean,
                        matched_mean=metrics.matched_mean,
                        random_perm_mean=metrics.random_perm_mean,
                        matched_minus_random=metrics.matched_minus_random,
                        best_assignment_json=json.dumps(metrics.best_assignment),
                    )
                )
    return maps, rows


def same_index_transfer(source_heads: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(layer), int(head)) for layer, head in source_heads]


def aligned_transfer(
    source_seed: str,
    target_seed: str,
    source_heads: list[tuple[int, int]],
    alignment_maps: dict[tuple[str, str, int], dict[int, int]],
) -> list[tuple[int, int]]:
    heads = []
    for layer, source_head in source_heads:
        target_head = alignment_maps[(source_seed, target_seed, int(layer))][int(source_head)]
        heads.append((int(layer), int(target_head)))
    return heads


def write_condition_summary(path: Path, rows: list[AblationRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[AblationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.revision, row.revision_index, row.condition)].append(row)

    output = []
    for (revision, revision_index, condition), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2])):
        loss = np.asarray([row.loss_delta for row in group], dtype=np.float64)
        logit = np.asarray([row.target_logit_delta for row in group], dtype=np.float64)
        output.append(
            {
                "revision": revision,
                "revision_index": revision_index,
                "condition": condition,
                "n": len(group),
                "loss_delta_mean": float(loss.mean()),
                "loss_delta_std": float(loss.std()),
                "target_logit_delta_mean": float(logit.mean()),
                "target_logit_delta_std": float(logit.std()),
            }
        )

    if output:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output)
    return output


def write_revision_summary(path: Path, rows: list[RevisionSeedSummary]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[RevisionSeedSummary]] = defaultdict(list)
    for row in rows:
        grouped[(row.revision, row.revision_index)].append(row)

    fields = [
        "selected_specialization_mean",
        "selected_specialization_max",
        "baseline_loss",
        "own_top_loss_delta",
        "random_loss_delta_mean",
        "own_top_excess_over_random",
        "source_same_index_loss_delta_mean",
        "source_aligned_loss_delta_mean",
        "aligned_minus_same_index_loss_delta_mean",
    ]
    output = []
    for (revision, revision_index), group in sorted(grouped.items(), key=lambda item: item[0][1]):
        item: dict[str, object] = {
            "revision": revision,
            "revision_index": revision_index,
            "n_seeds": len(group),
        }
        for field in fields:
            values = np.asarray([getattr(row, field) for row in group], dtype=np.float64)
            item[f"{field}_mean"] = float(values.mean())
            item[f"{field}_std"] = float(values.std())
        output.append(item)

    if output:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output)
    return output


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
        ("selected_specialization_mean_mean", "repeat-match specialization"),
        ("own_top_excess_over_random_mean", "own top minus random"),
        ("source_same_index_loss_delta_mean_mean", "same-index transfer loss delta"),
        ("source_aligned_loss_delta_mean_mean", "aligned transfer loss delta"),
        ("aligned_minus_same_index_loss_delta_mean_mean", "aligned minus same-index"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.0))
    axes_flat = axes.ravel()
    for ax, (field, title) in zip(axes_flat, metrics):
        ax.plot(x, [float(row[field]) for row in rows], marker="o")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(alpha=0.25)
    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "repeat_match_alignment_trajectory.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_texts = load_probe_texts(args.alignment_probe_file, args.alignment_num_texts)
    rng = np.random.default_rng(0)

    probe_rows: list[ProbeHeadScore] = []
    alignment_rows: list[AlignmentRow] = []
    ablation_rows: list[AblationRow] = []
    seed_summary_rows: list[RevisionSeedSummary] = []
    model_names = {
        str(seed): model_name_from_template(args.model_template, args.model_size, str(seed))
        for seed in args.seeds
    }
    probe_ids = None
    eval_ids = None
    vocab_size = None

    for revision_index, revision in enumerate(args.revisions):
        print(f"revision={revision}", flush=True)
        seed_scores: dict[str, list[np.ndarray]] = {}
        seed_selected_heads: dict[str, list[tuple[int, int]]] = {}
        seed_vectors: dict[str, list[np.ndarray]] = {}

        for seed in [str(seed) for seed in args.seeds]:
            model_name = model_names[seed]
            print(f"  probing seed={seed} model={model_name} revision={revision}", flush=True)
            model, tokenizer = load_model_and_tokenizer(model_name, revision, device, dtype)
            if probe_ids is None:
                vocab_size = len(tokenizer)
                probe_ids = synthetic_repeated_token_ids(
                    vocab_size=vocab_size,
                    n_sequences=args.probe_sequences,
                    repeat_length=args.repeat_length,
                    token_low=args.synthetic_token_low,
                    rng=np.random.default_rng(123),
                )
                eval_ids = synthetic_repeated_token_ids(
                    vocab_size=vocab_size,
                    n_sequences=args.eval_sequences,
                    repeat_length=args.repeat_length,
                    token_low=args.synthetic_token_low,
                    rng=np.random.default_rng(456),
                )
            elif len(tokenizer) != vocab_size:
                raise ValueError("Tokenizer vocabulary size changed across checkpoints.")

            assert probe_ids is not None
            scores_by_layer = extract_repeat_match_scores(
                model,
                probe_ids,
                args.repeat_length,
                args.batch_size,
                device,
            )
            seed_scores[seed] = scores_by_layer
            selected_heads, partial_rows = selected_heads_from_scores(
                scores_by_layer,
                layers,
                args.top_k_per_layer,
            )
            seed_selected_heads[seed] = selected_heads
            for row in partial_rows:
                probe_rows.append(
                    ProbeHeadScore(
                        model_size=args.model_size,
                        seed=seed,
                        revision=revision,
                        revision_index=revision_index,
                        layer=row.layer,
                        head=row.head,
                        repeat_score=row.repeat_score,
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
            revision=revision,
            revision_index=revision_index,
            seed_vectors=seed_vectors,
            layers=layers,
            random_permutations=args.random_permutations,
            rng=rng,
        )
        alignment_rows.extend(revision_alignment_rows)

        assert eval_ids is not None
        for target_seed in [str(seed) for seed in args.seeds]:
            model_name = model_names[target_seed]
            print(f"  ablating target_seed={target_seed} revision={revision}", flush=True)
            model, _ = load_model_and_tokenizer(model_name, revision, device, dtype)
            baseline_loss, baseline_logit = evaluate_condition(
                model=model,
                input_ids=eval_ids,
                repeat_length=args.repeat_length,
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

            target_rows: list[AblationRow] = []
            for condition, source_seed, control_id, heads in conditions:
                ablated_loss, ablated_logit = evaluate_condition(
                    model=model,
                    input_ids=eval_ids,
                    repeat_length=args.repeat_length,
                    batch_size=args.batch_size,
                    device=device,
                    heads=heads,
                )
                row = AblationRow(
                    model_size=args.model_size,
                    target_seed=target_seed,
                    revision=revision,
                    revision_index=revision_index,
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
            random_rows = [row for row in target_rows if row.condition == "own_random"]
            same_rows = [row for row in target_rows if row.condition == "source_same_index"]
            aligned_rows = [row for row in target_rows if row.condition == "source_aligned"]
            random_loss = np.asarray([row.loss_delta for row in random_rows], dtype=np.float64)
            same_loss = np.asarray([row.loss_delta for row in same_rows], dtype=np.float64)
            aligned_loss = np.asarray([row.loss_delta for row in aligned_rows], dtype=np.float64)
            selected_probe_rows = [
                row
                for row in probe_rows
                if row.revision == revision and row.seed == target_seed and row.selected_for_ablation
            ]
            seed_summary_rows.append(
                RevisionSeedSummary(
                    model_size=args.model_size,
                    seed=target_seed,
                    revision=revision,
                    revision_index=revision_index,
                    selected_heads_json=json.dumps(seed_selected_heads[target_seed]),
                    selected_specialization_mean=float(
                        np.mean([row.specialization for row in selected_probe_rows])
                    ),
                    selected_specialization_max=float(
                        np.max([row.specialization for row in selected_probe_rows])
                    ),
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

    write_dataclass_csv(args.output_dir / "probe_head_scores.csv", probe_rows)
    write_dataclass_csv(args.output_dir / "alignment_rows.csv", alignment_rows)
    write_dataclass_csv(args.output_dir / "ablation_results.csv", ablation_rows)
    write_dataclass_csv(args.output_dir / "revision_seed_summary.csv", seed_summary_rows)
    condition_summary = write_condition_summary(args.output_dir / "condition_summary.csv", ablation_rows)
    revision_summary = write_revision_summary(args.output_dir / "revision_summary.csv", seed_summary_rows)
    maybe_write_plot(args.output_dir, revision_summary)
    payload = {
        "args": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "alignment_probe_file": str(args.alignment_probe_file),
        },
        "model_names": model_names,
        "condition_summary": condition_summary,
        "revision_summary": revision_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(revision_summary, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
