#!/usr/bin/env python3
"""Cross-layer candidate-pool alignment on repeated natural text spans."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention_role_specialization import specialization_distribution
from attention_stability import (
    extract_attention_vectors,
    load_model_and_tokenizer,
    load_probe_texts,
    model_name_from_template,
    resolve_device,
    resolve_dtype,
)
from pythia_local_copy_alignment import AblationRow, RevisionSeedSummary
from pythia_local_copy_candidate_pool_alignment import (
    CandidateAlignmentRow,
    CandidateProbeHeadScore,
    aligned_candidate_transfer,
    build_candidate_alignment_maps,
    candidate_keys,
    random_candidate_heads,
    select_candidate_heads,
    write_outputs,
)
from repeat_match_ablation import ablate_gpt_neox_heads


@dataclass
class NaturalSpanExample:
    example_id: int
    source_start: int
    prefix_text: str
    span_text: str
    distractor_text: str
    sequence_text: str
    first_span_start: int
    second_span_start: int
    span_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-template", default="EleutherAI/pythia-{model_size}-seed{seed}")
    parser.add_argument("--model-size", default="160m")
    parser.add_argument("--seeds", nargs="+", default=[str(i) for i in range(1, 10)])
    parser.add_argument("--target-seeds", nargs="+", default=None)
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--candidate-layers", default="2,3,4")
    parser.add_argument("--top-k-total", type=int, default=2)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--probe-sequences", type=int, default=128)
    parser.add_argument("--eval-sequences", type=int, default=128)
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--span-length", type=int, default=12)
    parser.add_argument("--distractor-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--dataset-rows", type=int, default=4000)
    parser.add_argument("--min-token-stream", type=int, default=50000)
    parser.add_argument("--alignment-probe-file", type=Path, default=Path("probes/phase0_probe_texts.txt"))
    parser.add_argument("--alignment-num-texts", type=int, default=8)
    parser.add_argument("--alignment-max-length", type=int, default=64)
    parser.add_argument("--alignment-batch-size", type=int, default=2)
    parser.add_argument(
        "--alignment-source",
        default="phase0",
        choices=["phase0", "task_span"],
        help="Use generic Phase 0 texts or repeated-span probe attention vectors for cross-seed matching.",
    )
    parser.add_argument("--random-permutations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_naturalistic_span_candidate_pool"),
    )
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(item.strip()) for item in layers.split(",") if item.strip()]


def load_token_stream(tokenizer, args: argparse.Namespace) -> list[int]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    token_stream: list[int] = []
    n_rows = min(args.dataset_rows, len(dataset))
    eos = tokenizer.eos_token_id
    for item in dataset.select(range(n_rows)):
        text = str(item.get(args.dataset_text_column, "")).strip()
        if not text or text.startswith("="):
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 8:
            continue
        token_stream.extend(int(token_id) for token_id in ids)
        if eos is not None:
            token_stream.append(int(eos))
        if len(token_stream) >= args.min_token_stream:
            break
    needed = args.prefix_length + args.span_length + args.distractor_length
    if len(token_stream) < max(args.min_token_stream // 4, needed + 1):
        raise RuntimeError(
            f"Only collected {len(token_stream)} tokens from {args.dataset_name}; "
            "increase --dataset-rows or check dataset loading."
        )
    return token_stream


def natural_span_positions(
    prefix_length: int,
    span_length: int,
    distractor_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_start = prefix_length
    second_start = prefix_length + span_length + distractor_length
    offsets = torch.arange(0, span_length - 1, device=device)
    return second_start + offsets, first_start + offsets


def build_repeated_span_token_ids(
    token_stream: list[int],
    tokenizer,
    n_sequences: int,
    prefix_length: int,
    span_length: int,
    distractor_length: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    source_length = prefix_length + span_length + distractor_length
    max_start = len(token_stream) - source_length - 1
    if max_start <= 0:
        raise ValueError("Token stream is too short for requested repeated-span examples.")
    eos = tokenizer.eos_token_id
    valid_starts = []
    for start in range(max_start):
        window = token_stream[start : start + source_length]
        if eos is not None and int(eos) in window:
            continue
        first_span_token = token_stream[start + prefix_length]
        first_span_piece = tokenizer.decode([first_span_token])
        if not first_span_piece.startswith((" ", "\n")):
            continue
        valid_starts.append(start)
    if not valid_starts:
        raise ValueError("No valid repeated-span starts found after boundary filtering.")
    starts = rng.choice(
        np.asarray(valid_starts, dtype=np.int64),
        size=n_sequences,
        replace=len(valid_starts) < n_sequences,
    )
    rows = []
    metadata = []
    for example_id, start in enumerate(starts):
        prefix = token_stream[start : start + prefix_length]
        span = token_stream[start + prefix_length : start + prefix_length + span_length]
        distractor = token_stream[
            start + prefix_length + span_length : start + prefix_length + span_length + distractor_length
        ]
        rows.append(prefix + span + distractor + span)
        metadata.append(
            {
                "example_id": int(example_id),
                "source_start": int(start),
                "first_span_start": int(prefix_length),
                "second_span_start": int(prefix_length + span_length + distractor_length),
                "span_length": int(span_length),
            }
        )
    return torch.tensor(np.asarray(rows), dtype=torch.long), metadata


def example_rows(
    tokenizer,
    input_ids: torch.Tensor,
    metadata: list[dict[str, int]],
    args: argparse.Namespace,
    max_examples: int = 32,
) -> list[NaturalSpanExample]:
    rows = []
    for item in metadata[:max_examples]:
        ids = input_ids[item["example_id"]].tolist()
        prefix = ids[: args.prefix_length]
        span = ids[args.prefix_length : args.prefix_length + args.span_length]
        distractor = ids[
            args.prefix_length + args.span_length : args.prefix_length + args.span_length + args.distractor_length
        ]
        rows.append(
            NaturalSpanExample(
                example_id=item["example_id"],
                source_start=item["source_start"],
                prefix_text=tokenizer.decode(prefix),
                span_text=tokenizer.decode(span),
                distractor_text=tokenizer.decode(distractor),
                sequence_text=tokenizer.decode(ids),
                first_span_start=item["first_span_start"],
                second_span_start=item["second_span_start"],
                span_length=item["span_length"],
            )
        )
    return rows


def write_example_rows(path: Path, rows: list[NaturalSpanExample]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def extract_natural_span_scores(
    model,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    totals: list[np.ndarray] | None = None
    total_examples = 0
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = torch.ones_like(ids)
        query_positions, key_positions = natural_span_positions(
            args.prefix_length,
            args.span_length,
            args.distractor_length,
            device,
        )
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
            scores = attention[:, :, query_positions, key_positions].mean(dim=(0, 2)).cpu().numpy()
            layer_scores.append(scores)
        if totals is None:
            totals = [np.zeros_like(scores, dtype=np.float64) for scores in layer_scores]
        for layer_idx, scores in enumerate(layer_scores):
            totals[layer_idx] += scores.astype(np.float64) * ids.shape[0]
        total_examples += ids.shape[0]
    if totals is None or total_examples == 0:
        raise RuntimeError("No natural-span probe batches were evaluated.")
    return [scores / total_examples for scores in totals]


def extract_natural_span_alignment_vectors(
    model,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    chunks_by_layer: list[list[np.ndarray]] | None = None
    query_positions, key_positions = natural_span_positions(
        args.prefix_length,
        args.span_length,
        args.distractor_length,
        device,
    )
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        batch_chunks = []
        for attention in outputs.attentions:
            attention = attention.detach().float()
            values = attention[:, :, query_positions, key_positions]
            batch_chunks.append(values.permute(1, 0, 2).reshape(values.shape[1], -1).cpu().numpy())
        if chunks_by_layer is None:
            chunks_by_layer = [[] for _ in batch_chunks]
        for layer_idx, chunk in enumerate(batch_chunks):
            chunks_by_layer[layer_idx].append(chunk)
    if chunks_by_layer is None:
        raise RuntimeError("No natural-span alignment batches were evaluated.")
    return [np.concatenate(layer_chunks, axis=1) for layer_chunks in chunks_by_layer]


def natural_span_loss_and_logit(logits: torch.Tensor, input_ids: torch.Tensor, args: argparse.Namespace) -> tuple[float, float]:
    query_positions, _ = natural_span_positions(
        args.prefix_length,
        args.span_length,
        args.distractor_length,
        logits.device,
    )
    selected_logits = logits[:, query_positions, :]
    targets = input_ids[:, query_positions + 1]
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
    args: argparse.Namespace,
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
            loss, target_logit = natural_span_loss_and_logit(outputs.logits, ids, args)
            losses.append(loss * ids.shape[0])
            logits.append(target_logit * ids.shape[0])
    n = input_ids.shape[0]
    return sum(losses) / n, sum(logits) / n


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.candidate_layers)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_texts = load_probe_texts(args.alignment_probe_file, args.alignment_num_texts)
    rng = np.random.default_rng(0)

    probe_rows: list[CandidateProbeHeadScore] = []
    alignment_rows: list[CandidateAlignmentRow] = []
    ablation_rows: list[AblationRow] = []
    seed_summary_rows: list[RevisionSeedSummary] = []
    model_names = {
        str(seed): model_name_from_template(args.model_template, args.model_size, str(seed))
        for seed in args.seeds
    }
    target_seeds = [str(seed) for seed in (args.target_seeds or args.seeds)]
    probe_ids = None
    eval_ids = None
    example_preview_rows: list[NaturalSpanExample] = []
    vocab_size = None
    seed_selected_heads: dict[str, list[tuple[int, int]]] = {}
    seed_vectors: dict[str, list[np.ndarray]] = {}
    candidate_pool: list[tuple[int, int]] | None = None

    print(f"revision={args.revision}", flush=True)
    for seed in [str(seed) for seed in args.seeds]:
        model_name = model_names[seed]
        print(f"  probing seed={seed} model={model_name} revision={args.revision}", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        if probe_ids is None:
            vocab_size = len(tokenizer)
            token_stream = load_token_stream(tokenizer, args)
            probe_ids, probe_metadata = build_repeated_span_token_ids(
                token_stream=token_stream,
                tokenizer=tokenizer,
                n_sequences=args.probe_sequences,
                prefix_length=args.prefix_length,
                span_length=args.span_length,
                distractor_length=args.distractor_length,
                rng=np.random.default_rng(123),
            )
            eval_ids, eval_metadata = build_repeated_span_token_ids(
                token_stream=token_stream,
                tokenizer=tokenizer,
                n_sequences=args.eval_sequences,
                prefix_length=args.prefix_length,
                span_length=args.span_length,
                distractor_length=args.distractor_length,
                rng=np.random.default_rng(456),
            )
            example_preview_rows = example_rows(tokenizer, eval_ids, eval_metadata, args)
            write_example_rows(args.output_dir / "example_rows.csv", example_preview_rows)
            (args.output_dir / "dataset_provenance.json").write_text(
                json.dumps(
                    {
                        "dataset_name": args.dataset_name,
                        "dataset_config": args.dataset_config,
                        "dataset_split": args.dataset_split,
                        "dataset_text_column": args.dataset_text_column,
                        "dataset_rows": args.dataset_rows,
                        "token_stream_length": len(token_stream),
                        "probe_sequences": args.probe_sequences,
                        "eval_sequences": args.eval_sequences,
                        "prefix_length": args.prefix_length,
                        "span_length": args.span_length,
                        "distractor_length": args.distractor_length,
                    },
                    indent=2,
                )
                + "\n"
            )
        elif len(tokenizer) != vocab_size:
            raise ValueError("Tokenizer vocabulary size changed across seeds.")

        assert probe_ids is not None
        scores_by_layer = extract_natural_span_scores(model, probe_ids, args, args.batch_size, device)
        selected_heads, partial_rows = select_candidate_heads(scores_by_layer, layers, args.top_k_total)
        seed_selected_heads[seed] = selected_heads
        if candidate_pool is None:
            candidate_pool = candidate_keys(scores_by_layer, layers)
        for row in partial_rows:
            probe_rows.append(
                CandidateProbeHeadScore(
                    model_size=args.model_size,
                    seed=seed,
                    revision=args.revision,
                    revision_index=0,
                    layer=row.layer,
                    head=row.head,
                    local_copy_score=row.local_copy_score,
                    candidate_specialization=row.candidate_specialization,
                    selected_for_ablation=row.selected_for_ablation,
                    candidate_rank=row.candidate_rank,
                )
            )

        if args.alignment_source == "task_span":
            seed_vectors[seed] = extract_natural_span_alignment_vectors(
                model=model,
                input_ids=probe_ids,
                args=args,
                batch_size=args.alignment_batch_size,
                device=device,
            )
        else:
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

    alignment_maps, alignment_rows = build_candidate_alignment_maps(
        model_size=args.model_size,
        revision=args.revision,
        revision_index=0,
        seed_vectors=seed_vectors,
        layers=layers,
        random_permutations=args.random_permutations,
        rng=rng,
    )

    assert eval_ids is not None
    assert candidate_pool is not None
    write_outputs(args.output_dir, probe_rows, alignment_rows, ablation_rows, seed_summary_rows, args, model_names)
    write_example_rows(args.output_dir / "example_rows.csv", example_preview_rows)

    for target_seed in target_seeds:
        model_name = model_names[target_seed]
        print(f"  ablating target_seed={target_seed} revision={args.revision}", flush=True)
        model, _ = load_model_and_tokenizer(model_name, args.revision, device, dtype)
        baseline_loss, baseline_logit = evaluate_condition(
            model=model,
            input_ids=eval_ids,
            args=args,
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
                    random_candidate_heads(seed_selected_heads[target_seed], candidate_pool, rng),
                )
            )
        for source_seed in [str(seed) for seed in args.seeds]:
            if source_seed == target_seed:
                continue
            source_heads = seed_selected_heads[source_seed]
            conditions.append(("source_same_index", source_seed, None, source_heads))
            conditions.append(
                (
                    "source_aligned",
                    source_seed,
                    None,
                    aligned_candidate_transfer(source_seed, target_seed, source_heads, alignment_maps),
                )
            )

        target_rows = []
        for condition, source_seed, control_id, heads in conditions:
            ablated_loss, ablated_logit = evaluate_condition(
                model=model,
                input_ids=eval_ids,
                args=args,
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
                selected_specialization_mean=float(np.mean([row.candidate_specialization for row in selected_probe_rows])),
                selected_specialization_max=float(np.max([row.candidate_specialization for row in selected_probe_rows])),
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
        write_example_rows(args.output_dir / "example_rows.csv", example_preview_rows)

    print(json.dumps([asdict(row) for row in seed_summary_rows], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
