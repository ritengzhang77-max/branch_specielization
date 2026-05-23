#!/usr/bin/env python3
"""Cross-layer candidate-pool alignment for Pythia local-copy heads."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from attention_role_specialization import specialization_distribution
from attention_stability import (
    cosine_similarity_matrix,
    extract_attention_vectors,
    load_model_and_tokenizer,
    load_probe_texts,
    model_name_from_template,
    random_permutation_scores,
    resolve_device,
    resolve_dtype,
)
from pythia_local_copy_alignment import (
    AblationRow,
    RevisionSeedSummary,
    evaluate_condition,
    extract_local_copy_scores,
    local_copy_positions,
    synthetic_local_copy_token_ids,
)
from pythia_repeat_match_alignment_trajectory import write_condition_summary, write_revision_summary


@dataclass
class CandidateProbeHeadScore:
    model_size: str
    seed: str
    revision: str
    revision_index: int
    layer: int
    head: int
    local_copy_score: float
    candidate_specialization: float
    selected_for_ablation: bool
    candidate_rank: int


@dataclass
class CandidateAlignmentRow:
    model_size: str
    revision: str
    revision_index: int
    seed_a: str
    seed_b: str
    candidate_layers_json: str
    n_candidates: int
    matched_mean: float
    random_perm_mean: float
    matched_minus_random: float
    best_assignment_json: str


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
    parser.add_argument(
        "--alignment-source",
        default="phase0",
        choices=["phase0", "task_local_copy"],
        help="Use generic Phase 0 texts or synthetic local-copy probe attention vectors for cross-seed matching.",
    )
    parser.add_argument("--random-permutations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_local_copy_candidate_pool_alignment"),
    )
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(item.strip()) for item in layers.split(",") if item.strip()]


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def candidate_keys(seed_vectors: list[np.ndarray], layers: list[int]) -> list[tuple[int, int]]:
    keys = []
    for layer in layers:
        for head in range(seed_vectors[layer].shape[0]):
            keys.append((int(layer), int(head)))
    return keys


def flatten_candidate_vectors(seed_vectors: list[np.ndarray], layers: list[int]) -> tuple[list[tuple[int, int]], np.ndarray]:
    keys = candidate_keys(seed_vectors, layers)
    vectors = [seed_vectors[layer][head] for layer, head in keys]
    return keys, np.asarray(vectors, dtype=np.float64)


def select_candidate_heads(
    scores_by_layer: list[np.ndarray],
    layers: list[int],
    top_k_total: int,
) -> tuple[list[tuple[int, int]], list[CandidateProbeHeadScore]]:
    candidates = []
    for layer in layers:
        for head, score in enumerate(scores_by_layer[layer]):
            candidates.append((int(layer), int(head), float(score)))
    candidates.sort(key=lambda item: item[2], reverse=True)
    scores = np.asarray([item[2] for item in candidates], dtype=np.float64)
    specialization = specialization_distribution(scores)
    selected = [(layer, head) for layer, head, _ in candidates[:top_k_total]]
    selected_set = set(selected)
    rows = []
    for rank, ((layer, head, score), spec) in enumerate(zip(candidates, specialization), start=1):
        rows.append(
            CandidateProbeHeadScore(
                model_size="",
                seed="",
                revision="",
                revision_index=-1,
                layer=layer,
                head=head,
                local_copy_score=score,
                candidate_specialization=float(spec),
                selected_for_ablation=(layer, head) in selected_set,
                candidate_rank=rank,
            )
        )
    return selected, rows


def random_candidate_heads(
    selected_heads: list[tuple[int, int]],
    candidate_pool: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    selected_set = set(selected_heads)
    candidates = [head for head in candidate_pool if head not in selected_set]
    indices = rng.choice(len(candidates), size=len(selected_heads), replace=False)
    return [candidates[int(index)] for index in indices]


def build_candidate_alignment_maps(
    model_size: str,
    revision: str,
    revision_index: int,
    seed_vectors: dict[str, list[np.ndarray]],
    layers: list[int],
    random_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str], dict[tuple[int, int], tuple[int, int]]], list[CandidateAlignmentRow]]:
    maps: dict[tuple[str, str], dict[tuple[int, int], tuple[int, int]]] = {}
    rows = []
    seeds = list(seed_vectors)
    flattened = {
        seed: flatten_candidate_vectors(vectors, layers)
        for seed, vectors in seed_vectors.items()
    }
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            keys_a, vectors_a = flattened[seed_a]
            keys_b, vectors_b = flattened[seed_b]
            similarity = cosine_similarity_matrix(vectors_a, vectors_b)
            assignment_rows, assignment_cols = linear_sum_assignment(-similarity)
            matched = similarity[assignment_rows, assignment_cols]
            random_scores = random_permutation_scores(similarity, random_permutations, rng)
            direct = {
                keys_a[int(row)]: keys_b[int(col)]
                for row, col in zip(assignment_rows, assignment_cols)
            }
            reverse = {
                keys_b[int(col)]: keys_a[int(row)]
                for row, col in zip(assignment_rows, assignment_cols)
            }
            maps[(seed_a, seed_b)] = direct
            maps[(seed_b, seed_a)] = reverse
            rows.append(
                CandidateAlignmentRow(
                    model_size=model_size,
                    revision=revision,
                    revision_index=revision_index,
                    seed_a=seed_a,
                    seed_b=seed_b,
                    candidate_layers_json=json.dumps(layers),
                    n_candidates=len(keys_a),
                    matched_mean=float(matched.mean()),
                    random_perm_mean=float(random_scores.mean()),
                    matched_minus_random=float(matched.mean() - random_scores.mean()),
                    best_assignment_json=json.dumps(
                        [
                            {
                                "source": list(keys_a[int(row)]),
                                "target": list(keys_b[int(col)]),
                            }
                            for row, col in zip(assignment_rows, assignment_cols)
                        ]
                    ),
                )
            )
    return maps, rows


def aligned_candidate_transfer(
    source_seed: str,
    target_seed: str,
    source_heads: list[tuple[int, int]],
    alignment_maps: dict[tuple[str, str], dict[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    mapping = alignment_maps[(source_seed, target_seed)]
    return [mapping[(int(layer), int(head))] for layer, head in source_heads]


def extract_local_copy_alignment_vectors(
    model,
    input_ids: torch.Tensor,
    n_pairs: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    chunks_by_layer: list[list[np.ndarray]] | None = None
    positions = local_copy_positions(n_pairs, device)
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
            values = attention[:, :, positions, positions - 1]
            batch_chunks.append(values.permute(1, 0, 2).reshape(values.shape[1], -1).cpu().numpy())
        if chunks_by_layer is None:
            chunks_by_layer = [[] for _ in batch_chunks]
        for layer_idx, chunk in enumerate(batch_chunks):
            chunks_by_layer[layer_idx].append(chunk)
    if chunks_by_layer is None:
        raise RuntimeError("No local-copy alignment batches were evaluated.")
    return [np.concatenate(layer_chunks, axis=1) for layer_chunks in chunks_by_layer]


def write_outputs(
    output_dir: Path,
    probe_rows: list[CandidateProbeHeadScore],
    alignment_rows: list[CandidateAlignmentRow],
    ablation_rows: list[AblationRow],
    seed_summary_rows: list[RevisionSeedSummary],
    args: argparse.Namespace,
    model_names: dict[str, str],
) -> None:
    write_dataclass_csv(output_dir / "probe_head_scores.csv", probe_rows)
    write_dataclass_csv(output_dir / "candidate_alignment_rows.csv", alignment_rows)
    write_dataclass_csv(output_dir / "ablation_results.csv", ablation_rows)
    write_dataclass_csv(output_dir / "revision_seed_summary.csv", seed_summary_rows)
    condition_summary = write_condition_summary(output_dir / "condition_summary.csv", ablation_rows)
    revision_summary = write_revision_summary(output_dir / "revision_summary.csv", seed_summary_rows)
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
    vocab_size = None
    seed_scores: dict[str, list[np.ndarray]] = {}
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

        if args.alignment_source == "task_local_copy":
            seed_vectors[seed] = extract_local_copy_alignment_vectors(
                model=model,
                input_ids=probe_ids,
                n_pairs=args.n_pairs,
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

    revision_summary = write_revision_summary(args.output_dir / "revision_summary.csv", seed_summary_rows)
    print(json.dumps(revision_summary, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
