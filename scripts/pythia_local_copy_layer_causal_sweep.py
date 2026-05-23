#!/usr/bin/env python3
"""Sweep local-copy probe heads by layer and test own-seed causal importance."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from attention_role_specialization import specialization_distribution
from attention_stability import load_model_and_tokenizer, model_name_from_template, resolve_device, resolve_dtype
from pythia_local_copy_alignment import (
    evaluate_condition,
    extract_local_copy_scores,
    synthetic_local_copy_token_ids,
)


@dataclass
class LayerSweepRow:
    model_size: str
    seed: str
    revision: str
    layer: int
    head: int
    local_copy_score: float
    specialization: float
    baseline_loss: float
    own_top_loss_delta: float
    random_loss_delta_mean: float
    random_loss_delta_std: float
    own_top_excess_over_random: float


@dataclass
class SeedBestRow:
    model_size: str
    seed: str
    revision: str
    best_layer: int
    best_head: int
    best_own_top_excess_over_random: float
    best_specialization: float
    layer3_head: int | None
    layer3_own_top_excess_over_random: float | None
    layer3_specialization: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-template", default="EleutherAI/pythia-{model_size}-seed{seed}")
    parser.add_argument("--model-size", default="160m")
    parser.add_argument("--seeds", nargs="+", default=["4", "5", "6", "8"])
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--layers", default="all", help="Comma-separated layers or 'all'.")
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--probe-sequences", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--n-pairs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-token-low", type=int, default=1000)
    parser.add_argument("--separator-token-id", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_pythia160m_local_copy_layer_sweep_weak_targets"),
    )
    return parser.parse_args()


def parse_layers(raw_layers: str, n_layers: int) -> list[int]:
    if raw_layers.strip().lower() == "all":
        return list(range(n_layers))
    return [int(item.strip()) for item in raw_layers.split(",") if item.strip()]


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_outputs(output_dir: Path, layer_rows: list[LayerSweepRow], best_rows: list[SeedBestRow], args) -> None:
    write_dataclass_csv(output_dir / "layer_causal_sweep.csv", layer_rows)
    write_dataclass_csv(output_dir / "seed_best_layer_summary.csv", best_rows)
    payload = {
        "args": vars(args)
        | {
            "output_dir": str(output_dir),
        },
        "seed_best_layer_summary": [asdict(row) for row in best_rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def random_single_head(layer: int, selected_head: int, n_heads: int, rng: np.random.Generator) -> tuple[int, int]:
    candidates = [head for head in range(n_heads) if head != selected_head]
    return layer, int(rng.choice(candidates))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    probe_ids = None
    eval_ids = None
    vocab_size = None
    layer_rows: list[LayerSweepRow] = []
    best_rows: list[SeedBestRow] = []

    for seed in [str(seed) for seed in args.seeds]:
        model_name = model_name_from_template(args.model_template, args.model_size, seed)
        print(f"seed={seed} model={model_name} revision={args.revision}", flush=True)
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
        assert eval_ids is not None
        scores_by_layer = extract_local_copy_scores(model, probe_ids, args.n_pairs, args.batch_size, device)
        layers = parse_layers(args.layers, len(scores_by_layer))
        baseline_loss, _ = evaluate_condition(
            model=model,
            input_ids=eval_ids,
            n_pairs=args.n_pairs,
            batch_size=args.batch_size,
            device=device,
            heads=None,
        )

        seed_rows = []
        for layer in layers:
            scores = scores_by_layer[layer]
            distribution = specialization_distribution(scores)
            head = int(np.argmax(scores))
            own_loss, _ = evaluate_condition(
                model=model,
                input_ids=eval_ids,
                n_pairs=args.n_pairs,
                batch_size=args.batch_size,
                device=device,
                heads=[(layer, head)],
            )
            random_deltas = []
            for _ in range(args.random_controls):
                random_head = random_single_head(layer, head, len(scores), rng)
                random_loss, _ = evaluate_condition(
                    model=model,
                    input_ids=eval_ids,
                    n_pairs=args.n_pairs,
                    batch_size=args.batch_size,
                    device=device,
                    heads=[random_head],
                )
                random_deltas.append(random_loss - baseline_loss)
            random_array = np.asarray(random_deltas, dtype=np.float64)
            row = LayerSweepRow(
                model_size=args.model_size,
                seed=seed,
                revision=args.revision,
                layer=layer,
                head=head,
                local_copy_score=float(scores[head]),
                specialization=float(distribution[head]),
                baseline_loss=baseline_loss,
                own_top_loss_delta=own_loss - baseline_loss,
                random_loss_delta_mean=float(random_array.mean()),
                random_loss_delta_std=float(random_array.std()),
                own_top_excess_over_random=float((own_loss - baseline_loss) - random_array.mean()),
            )
            layer_rows.append(row)
            seed_rows.append(row)

        best = max(seed_rows, key=lambda row: row.own_top_excess_over_random)
        layer3 = next((row for row in seed_rows if row.layer == 3), None)
        best_rows.append(
            SeedBestRow(
                model_size=args.model_size,
                seed=seed,
                revision=args.revision,
                best_layer=best.layer,
                best_head=best.head,
                best_own_top_excess_over_random=best.own_top_excess_over_random,
                best_specialization=best.specialization,
                layer3_head=None if layer3 is None else layer3.head,
                layer3_own_top_excess_over_random=None
                if layer3 is None
                else layer3.own_top_excess_over_random,
                layer3_specialization=None if layer3 is None else layer3.specialization,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_outputs(args.output_dir, layer_rows, best_rows, args)

    print(json.dumps([asdict(row) for row in best_rows], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
