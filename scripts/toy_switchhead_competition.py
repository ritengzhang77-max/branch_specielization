#!/usr/bin/env python3
"""Toy SwitchHead routed-attention competition experiment.

This is a bridge between the hand-built two-branch router toy and a less
hand-designed routed-attention module. It uses the official plug-in SwitchHead
implementation when available at --switchhead-path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_branch_isolation_intervention import build_layout, make_task_dataset
from toy_competition_head_dim_intervention import combined_loss_accuracy, iter_batches
from toy_head_dim_intervention import resolve_device, specialization_distribution


@dataclass
class ModelSummary:
    config: str
    seed: int
    eval_loss: float
    local_loss: float
    induction_loss: float
    local_accuracy: float
    induction_accuracy: float
    local_top_component: str
    induction_top_component: str
    same_top_component: bool
    same_top_expert: bool
    routed_expert_match: bool
    expert_distribution_distance: float
    gate_distribution_distance: float
    gate_local_top_expert: int
    gate_induction_top_expert: int
    gate_same_top_expert: bool
    local_top_loss_delta: float
    induction_top_loss_delta: float
    n_layers: int
    n_heads: int
    n_experts: int
    moe_k: int


@dataclass
class ExpertScore:
    config: str
    seed: int
    layer: int
    expert: int
    local_loss_delta: float
    induction_loss_delta: float
    local_specialization: float
    induction_specialization: float
    gate_local_mean: float
    gate_induction_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switchhead-path", type=Path, default=Path(".tools/switchhead"))
    parser.add_argument("--config", default="switchhead_rope")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--local-pairs", type=int, default=8)
    parser.add_argument("--repeat-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=160)
    parser.add_argument("--task-variant", default="bidirectional_lookup", choices=["standard", "bidirectional_lookup"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-experts", type=int, default=2)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--moe-k", type=int, default=1)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--expert-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--induction-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_switchhead_competition"))
    return parser.parse_args()


def import_switchhead(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"SwitchHead path {path} does not exist. Clone https://github.com/RobertCsordas/switchhead "
            "or pass --switchhead-path."
        )
    sys.path.insert(0, str(path.resolve()))
    import switchhead  # type: ignore

    return switchhead


@contextlib.contextmanager
def zero_switchhead_expert(attn: nn.Module, expert_idx: int):
    n_experts = int(getattr(attn, "n_experts"))
    n_heads = int(getattr(attn, "n_heads"))
    if n_experts <= 1:
        yield
        return
    device = attn.v.device
    row_idx = torch.arange(n_heads, device=device, dtype=torch.long) * n_experts + int(expert_idx)
    with torch.no_grad():
        v_backup = attn.v[row_idx].detach().clone()
        o_backup = attn.o[row_idx].detach().clone()
        attn.v.index_fill_(0, row_idx, 0.0)
        attn.o.index_fill_(0, row_idx, 0.0)
    try:
        yield
    finally:
        with torch.no_grad():
            attn.v.index_copy_(0, row_idx, v_backup)
            attn.o.index_copy_(0, row_idx, o_backup)


class SwitchHeadBlock(nn.Module):
    def __init__(self, switchhead, args: argparse.Namespace) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(args.d_model)
        self.attn = switchhead.SwitchHeadRope(
            args.d_model,
            n_heads=args.n_heads,
            n_experts=args.n_experts,
            dropout=args.dropout,
            d_head=args.d_head,
            expert_dropout=args.expert_dropout,
            moe_k=args.moe_k,
            init_scale=1.0 / math.sqrt(max(args.n_layers, 1)),
        )
        self.ln2 = nn.LayerNorm(args.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(args.d_model, args.mlp_dim),
            nn.GELU(),
            nn.Linear(args.mlp_dim, args.d_model),
        )

    def forward(self, x: torch.Tensor, ablate_expert: int | None = None) -> torch.Tensor:
        xn = self.ln1(x)
        if ablate_expert is None:
            attn_out, _ = self.attn(xn, xn, xn, mask=None)
        else:
            with zero_switchhead_expert(self.attn, ablate_expert):
                attn_out, _ = self.attn(xn, xn, xn, mask=None)
        x = x + attn_out
        return x + self.mlp(self.ln2(x))

    def output_gate_distribution(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.ln1(x)
        raw = F.linear(xn, self.attn.sel_o).float()
        raw = raw.view(*raw.shape[:-1], self.attn.n_heads, self.attn.n_experts)
        probs = torch.sigmoid(raw)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class TinySwitchHeadTransformer(nn.Module):
    def __init__(self, switchhead, args: argparse.Namespace, seq_len: int) -> None:
        super().__init__()
        self.args = args
        self.token_embed = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, args.d_model))
        self.blocks = nn.ModuleList([SwitchHeadBlock(switchhead, args) for _ in range(args.n_layers)])
        self.final_ln = nn.LayerNorm(args.d_model)
        self.unembed = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embed(input_ids) + self.pos_embed[: input_ids.shape[1]]

    def forward(self, input_ids: torch.Tensor, ablate: tuple[int, int] | None = None) -> torch.Tensor:
        x = self.embed_input(input_ids)
        for layer_idx, block in enumerate(self.blocks):
            expert = ablate[1] if ablate is not None and ablate[0] == layer_idx else None
            x = block(x, ablate_expert=expert)
        return self.unembed(self.final_ln(x))

    def collect_gate_distributions(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        x = self.embed_input(input_ids)
        dists = []
        for block in self.blocks:
            dists.append(block.output_gate_distribution(x))
            x = block(x)
        return dists


def train_model(
    model: TinySwitchHeadTransformer,
    train_rng: np.random.Generator,
    args: argparse.Namespace,
    layout: dict[str, torch.Tensor | int],
    device: torch.device,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for step in range(args.steps):
        input_ids = make_task_dataset(train_rng, args.batch_size, args).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss, _ = combined_loss_accuracy(logits, input_ids, layout, args.local_weight, args.induction_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"  step={step + 1} train_loss={last_loss:.4f}", flush=True)
    return last_loss


def evaluate(
    model: TinySwitchHeadTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    args: argparse.Namespace,
    device: torch.device,
    ablate: tuple[int, int] | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total = 0
    with torch.inference_mode():
        for ids in iter_batches(input_ids, args.batch_size):
            ids = ids.to(device)
            logits = model(ids, ablate=ablate)
            loss, metrics = combined_loss_accuracy(logits, ids, layout, args.local_weight, args.induction_weight)
            n = ids.shape[0]
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * n
            total += n
    return {key: value / total for key, value in totals.items()}


def collect_gate_metrics(
    model: TinySwitchHeadTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    local_positions = layout["local_positions"].to(device)
    induction_positions = layout["induction_positions"].to(device)
    local_totals = []
    induction_totals = []
    model.eval()
    with torch.inference_mode():
        for ids in iter_batches(input_ids, args.batch_size):
            ids = ids.to(device)
            for dist in model.collect_gate_distributions(ids):
                # dist: batch, seq, heads, experts
                local_totals.append(dist[:, local_positions].mean(dim=(0, 1, 2)).detach().cpu().numpy())
                induction_totals.append(dist[:, induction_positions].mean(dim=(0, 1, 2)).detach().cpu().numpy())
    return np.mean(local_totals, axis=0), np.mean(induction_totals, axis=0)


def distribution_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def analyze_model(
    model: TinySwitchHeadTransformer,
    eval_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[ModelSummary, list[ExpertScore]]:
    baseline = evaluate(model, eval_ids, layout, args, device)
    local_deltas = np.zeros((args.n_layers, args.n_experts), dtype=np.float64)
    induction_deltas = np.zeros((args.n_layers, args.n_experts), dtype=np.float64)
    for layer_idx in range(args.n_layers):
        for expert_idx in range(args.n_experts):
            metrics = evaluate(model, eval_ids, layout, args, device, ablate=(layer_idx, expert_idx))
            local_deltas[layer_idx, expert_idx] = metrics["local_loss"] - baseline["local_loss"]
            induction_deltas[layer_idx, expert_idx] = metrics["induction_loss"] - baseline["induction_loss"]

    local_spec = specialization_distribution(local_deltas.flatten()).reshape(local_deltas.shape)
    induction_spec = specialization_distribution(induction_deltas.flatten()).reshape(induction_deltas.shape)
    local_top_flat = int(np.argmax(local_spec))
    induction_top_flat = int(np.argmax(induction_spec))
    local_top = np.unravel_index(local_top_flat, local_spec.shape)
    induction_top = np.unravel_index(induction_top_flat, induction_spec.shape)
    expert_dist = distribution_distance(local_spec.flatten(), induction_spec.flatten())

    gate_local, gate_induction = collect_gate_metrics(model, eval_ids, layout, args, device)
    gate_local_top = int(np.argmax(gate_local))
    gate_induction_top = int(np.argmax(gate_induction))
    gate_dist = distribution_distance(gate_local, gate_induction)

    expert_scores = []
    for layer_idx in range(args.n_layers):
        for expert_idx in range(args.n_experts):
            expert_scores.append(
                ExpertScore(
                    config=args.config,
                    seed=seed,
                    layer=layer_idx,
                    expert=expert_idx,
                    local_loss_delta=float(local_deltas[layer_idx, expert_idx]),
                    induction_loss_delta=float(induction_deltas[layer_idx, expert_idx]),
                    local_specialization=float(local_spec[layer_idx, expert_idx]),
                    induction_specialization=float(induction_spec[layer_idx, expert_idx]),
                    gate_local_mean=float(gate_local[expert_idx]),
                    gate_induction_mean=float(gate_induction[expert_idx]),
                )
            )

    summary = ModelSummary(
        config=args.config,
        seed=seed,
        eval_loss=baseline["loss"],
        local_loss=baseline["local_loss"],
        induction_loss=baseline["induction_loss"],
        local_accuracy=baseline["local_accuracy"],
        induction_accuracy=baseline["induction_accuracy"],
        local_top_component=f"L{local_top[0]}E{local_top[1]}",
        induction_top_component=f"L{induction_top[0]}E{induction_top[1]}",
        same_top_component=local_top == induction_top,
        same_top_expert=int(local_top[1]) == int(induction_top[1]),
        routed_expert_match=int(local_top[1]) == 0 and int(induction_top[1]) == 1,
        expert_distribution_distance=expert_dist,
        gate_distribution_distance=gate_dist,
        gate_local_top_expert=gate_local_top,
        gate_induction_top_expert=gate_induction_top,
        gate_same_top_expert=gate_local_top == gate_induction_top,
        local_top_loss_delta=float(local_deltas[local_top]),
        induction_top_loss_delta=float(induction_deltas[induction_top]),
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_experts=args.n_experts,
        moe_k=args.moe_k,
    )
    return summary, expert_scores


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[ModelSummary]) -> dict[str, object]:
    return {
        "n_models": len(rows),
        "local_accuracy_mean": float(np.mean([row.local_accuracy for row in rows])),
        "induction_accuracy_mean": float(np.mean([row.induction_accuracy for row in rows])),
        "same_top_component_rate": float(np.mean([row.same_top_component for row in rows])),
        "same_top_expert_rate": float(np.mean([row.same_top_expert for row in rows])),
        "routed_expert_match_rate": float(np.mean([row.routed_expert_match for row in rows])),
        "expert_distribution_distance_mean": float(np.mean([row.expert_distribution_distance for row in rows])),
        "gate_distribution_distance_mean": float(np.mean([row.gate_distribution_distance for row in rows])),
        "gate_same_top_expert_rate": float(np.mean([row.gate_same_top_expert for row in rows])),
        "local_top_loss_delta_mean": float(np.mean([row.local_top_loss_delta for row in rows])),
        "induction_top_loss_delta_mean": float(np.mean([row.induction_top_loss_delta for row in rows])),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        raise ValueError("SwitchHead uses Triton CVMM kernels and requires CUDA for n_experts > 1.")
    switchhead = import_switchhead(args.switchhead_path)
    device = resolve_device(args.device)
    layout = build_layout(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_rows: list[ModelSummary] = []
    expert_rows: list[ExpertScore] = []
    for seed in args.seeds:
        print(f"training config={args.config} seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_rng = np.random.default_rng(10_000 + seed)
        eval_rng = np.random.default_rng(20_000 + seed)
        model = TinySwitchHeadTransformer(switchhead, args, int(layout["seq_len"])).to(device)
        train_model(model, train_rng, args, layout, device)
        eval_ids = make_task_dataset(eval_rng, args.eval_examples, args)
        summary, scores = analyze_model(model, eval_ids, layout, args, device, seed)
        model_rows.append(summary)
        expert_rows.extend(scores)

    write_csv(args.output_dir / "model_summary.csv", [asdict(row) for row in model_rows])
    write_csv(args.output_dir / "expert_scores.csv", [asdict(row) for row in expert_rows])
    payload = {
        "args": vars(args) | {"switchhead_path": str(args.switchhead_path), "output_dir": str(args.output_dir)},
        "summary": summarize(model_rows),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
