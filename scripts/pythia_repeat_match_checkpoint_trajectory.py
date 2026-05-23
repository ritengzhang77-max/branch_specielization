#!/usr/bin/env python3
"""Track repeat-match attention probes and causal head ablations across Pythia checkpoints.

This is a small real-transformer follow-up to the toy router trajectory result.
It asks whether an attention-role probe for repeat-match heads appears before,
after, or at the same time as a causal head-output ablation effect on repeated
token continuation.
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

from attention_role_specialization import role_scores_from_attentions, specialization_distribution
from attention_stability import (
    load_model_and_tokenizer,
    model_name_from_template,
    resolve_device,
    resolve_dtype,
)
from repeat_match_ablation import evaluate_condition, synthetic_repeated_token_ids


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
    seed: str
    revision: str
    revision_index: int
    condition: str
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
    selected_repeat_score_mean: float
    selected_specialization_mean: float
    selected_specialization_max: float
    baseline_loss: float
    baseline_target_logit: float
    own_top_loss_delta: float
    own_top_target_logit_delta: float
    random_loss_delta_mean: float
    random_loss_delta_std: float
    random_target_logit_delta_mean: float
    random_target_logit_delta_std: float
    excess_loss_delta_over_random: float
    excess_target_logit_drop_over_random: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-template",
        default="EleutherAI/pythia-{model_size}-seed{seed}",
        help="Hugging Face model template. Available fields: {model_size}, {seed}.",
    )
    parser.add_argument("--model-size", default="14m")
    parser.add_argument("--seeds", nargs="+", default=["1"])
    parser.add_argument(
        "--revisions",
        nargs="+",
        default=["step0", "step64", "step256", "step1000", "step4000", "step16000", "step143000"],
    )
    parser.add_argument("--layers", default="0,1", help="Comma-separated layers to test.")
    parser.add_argument("--top-k-per-layer", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--probe-sequences", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--repeat-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-token-low", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia_repeat_match_checkpoint_trajectory"),
    )
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(item.strip()) for item in layers.split(",") if item.strip()]


def extract_repeat_match_scores(
    model,
    input_ids: torch.Tensor,
    repeat_length: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    totals: list[np.ndarray] | None = None
    total_examples = 0

    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start : start + batch_size].to(device)
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
        scores_by_role = role_scores_from_attentions(outputs.attentions, lengths, repeat_length=repeat_length)
        repeat_scores = scores_by_role["repeat_match"]
        if totals is None:
            totals = [np.zeros_like(layer_scores, dtype=np.float64) for layer_scores in repeat_scores]
        for layer_idx, layer_scores in enumerate(repeat_scores):
            totals[layer_idx] += layer_scores.astype(np.float64) * ids.shape[0]
        total_examples += ids.shape[0]

    if totals is None or total_examples == 0:
        raise RuntimeError("No repeat-match probe batches were evaluated.")
    return [layer_scores / total_examples for layer_scores in totals]


def selected_heads_from_scores(
    scores_by_layer: list[np.ndarray],
    layers: list[int],
    top_k_per_layer: int,
) -> tuple[list[tuple[int, int]], list[ProbeHeadScore]]:
    selected = []
    partial_rows = []
    selected_set = set()
    ranks_by_layer = {}

    for layer in layers:
        scores = scores_by_layer[layer]
        order = list(np.argsort(-scores))
        ranks_by_layer[layer] = {head: rank + 1 for rank, head in enumerate(order)}
        for head in order[:top_k_per_layer]:
            selected.append((layer, int(head)))
            selected_set.add((layer, int(head)))

    for layer in layers:
        scores = scores_by_layer[layer]
        distribution = specialization_distribution(scores)
        for head, score in enumerate(scores):
            partial_rows.append(
                ProbeHeadScore(
                    model_size="",
                    seed="",
                    revision="",
                    revision_index=-1,
                    layer=layer,
                    head=head,
                    repeat_score=float(score),
                    specialization=float(distribution[head]),
                    selected_for_ablation=(layer, head) in selected_set,
                    layer_rank=int(ranks_by_layer[layer][head]),
                )
            )
    return selected, partial_rows


def random_heads(
    selected_heads: list[tuple[int, int]],
    scores_by_layer: list[np.ndarray],
    layers: list[int],
    top_k_per_layer: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    selected_by_layer: dict[int, set[int]] = defaultdict(set)
    for layer, head in selected_heads:
        selected_by_layer[layer].add(head)

    heads = []
    for layer in layers:
        n_heads = len(scores_by_layer[layer])
        candidates = [head for head in range(n_heads) if head not in selected_by_layer[layer]]
        sampled = rng.choice(candidates, size=top_k_per_layer, replace=False)
        heads.extend((layer, int(head)) for head in sampled)
    return heads


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_aggregate_summary(path: Path, rows: list[RevisionSeedSummary]) -> list[dict[str, object]]:
    by_revision: dict[tuple[str, int], list[RevisionSeedSummary]] = defaultdict(list)
    for row in rows:
        by_revision[(row.revision, row.revision_index)].append(row)

    output = []
    fields = [
        "selected_repeat_score_mean",
        "selected_specialization_mean",
        "selected_specialization_max",
        "baseline_loss",
        "baseline_target_logit",
        "own_top_loss_delta",
        "own_top_target_logit_delta",
        "random_loss_delta_mean",
        "random_target_logit_delta_mean",
        "excess_loss_delta_over_random",
        "excess_target_logit_drop_over_random",
    ]
    for (revision, revision_index), group in sorted(by_revision.items(), key=lambda item: item[0][1]):
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
        fieldnames = list(output[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
        ("own_top_loss_delta_mean", "top-head ablation loss delta"),
        ("random_loss_delta_mean_mean", "random ablation loss delta"),
        ("excess_loss_delta_over_random_mean", "top minus random loss delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0))
    for ax, (field, title) in zip(axes.ravel(), metrics):
        ax.plot(x, [float(row[field]) for row in rows], marker="o")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "repeat_match_checkpoint_trajectory.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    rng = np.random.default_rng(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probe_rows: list[ProbeHeadScore] = []
    ablation_rows: list[AblationRow] = []
    summary_rows: list[RevisionSeedSummary] = []
    model_names = {}
    probe_ids = None
    eval_ids = None
    vocab_size = None

    for seed in [str(seed) for seed in args.seeds]:
        model_name = model_name_from_template(args.model_template, args.model_size, seed)
        model_names[seed] = model_name
        for revision_index, revision in enumerate(args.revisions):
            print(
                f"loading model={model_name} revision={revision} seed={seed} device={device}",
                flush=True,
            )
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

            assert probe_ids is not None and eval_ids is not None
            scores_by_layer = extract_repeat_match_scores(
                model,
                probe_ids,
                args.repeat_length,
                args.batch_size,
                device,
            )
            selected_heads, partial_probe_rows = selected_heads_from_scores(
                scores_by_layer,
                layers,
                args.top_k_per_layer,
            )
            for row in partial_probe_rows:
                row.model_size = args.model_size
                row.seed = seed
                row.revision = revision
                row.revision_index = revision_index
                probe_rows.append(row)

            baseline_loss, baseline_logit = evaluate_condition(
                model=model,
                input_ids=eval_ids,
                repeat_length=args.repeat_length,
                batch_size=args.batch_size,
                device=device,
                heads=None,
            )

            conditions = [("own_top", None, selected_heads)]
            for control_id in range(args.random_controls):
                conditions.append(
                    (
                        "own_random",
                        control_id,
                        random_heads(
                            selected_heads,
                            scores_by_layer,
                            layers,
                            args.top_k_per_layer,
                            rng,
                        ),
                    )
                )

            own_top_row = None
            random_loss_deltas = []
            random_logit_deltas = []
            for condition, control_id, heads in conditions:
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
                    seed=seed,
                    revision=revision,
                    revision_index=revision_index,
                    condition=condition,
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
                if condition == "own_top":
                    own_top_row = row
                else:
                    random_loss_deltas.append(row.loss_delta)
                    random_logit_deltas.append(row.target_logit_delta)

            if own_top_row is None:
                raise RuntimeError("own_top ablation row was not recorded.")

            selected_probe_rows = [row for row in partial_probe_rows if row.selected_for_ablation]
            random_loss = np.asarray(random_loss_deltas, dtype=np.float64)
            random_logit = np.asarray(random_logit_deltas, dtype=np.float64)
            summary_rows.append(
                RevisionSeedSummary(
                    model_size=args.model_size,
                    seed=seed,
                    revision=revision,
                    revision_index=revision_index,
                    selected_heads_json=json.dumps(selected_heads),
                    selected_repeat_score_mean=float(
                        np.mean([row.repeat_score for row in selected_probe_rows])
                    ),
                    selected_specialization_mean=float(
                        np.mean([row.specialization for row in selected_probe_rows])
                    ),
                    selected_specialization_max=float(
                        np.max([row.specialization for row in selected_probe_rows])
                    ),
                    baseline_loss=baseline_loss,
                    baseline_target_logit=baseline_logit,
                    own_top_loss_delta=own_top_row.loss_delta,
                    own_top_target_logit_delta=own_top_row.target_logit_delta,
                    random_loss_delta_mean=float(random_loss.mean()),
                    random_loss_delta_std=float(random_loss.std()),
                    random_target_logit_delta_mean=float(random_logit.mean()),
                    random_target_logit_delta_std=float(random_logit.std()),
                    excess_loss_delta_over_random=float(own_top_row.loss_delta - random_loss.mean()),
                    excess_target_logit_drop_over_random=float(
                        random_logit.mean() - own_top_row.target_logit_delta
                    ),
                )
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_dataclass_csv(args.output_dir / "probe_head_scores.csv", probe_rows)
    write_dataclass_csv(args.output_dir / "ablation_results.csv", ablation_rows)
    write_dataclass_csv(args.output_dir / "revision_seed_summary.csv", summary_rows)
    aggregate_rows = write_aggregate_summary(args.output_dir / "revision_summary.csv", summary_rows)
    maybe_write_plot(args.output_dir, aggregate_rows)
    payload = {
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "model_names": model_names,
        "aggregate_summary": aggregate_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(aggregate_rows, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
