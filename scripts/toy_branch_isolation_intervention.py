#!/usr/bin/env python3
"""Toy explicit-branch intervention for functional modularity.

This script follows the head-dimension toy experiments with a stronger
architectural intervention: two separate attention/MLP towers over a shared
embedding and unembedding.

Modes:

- branch_sum: both branch residual updates contribute at every position.
- oracle_route: local scored positions use branch 0, induction scored positions
  use branch 1, and unscored positions use both branches.
- learned_position_router: branch weights are learned per sequence position.
- learned_token_router: branch weights are learned from the token/position
  representation at each position.
- weak_position_router: same as learned_position_router, with a weak auxiliary
  loss that encourages local scored positions to use branch 0 and induction
  scored positions to use branch 1.
- weak_token_router: same as learned_token_router, with the same weak auxiliary
  scored-position routing loss.

The goal is to test whether explicit separation/routing can turn functional
specialization into branch-level modularity.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from toy_competition_head_dim_intervention import (
    combined_loss_accuracy,
    iter_batches,
    make_competition_dataset,
    sequence_layout,
)
from toy_head_dim_intervention import TransformerBlock, resolve_device, specialization_distribution


VALID_MODES = {
    "branch_sum",
    "oracle_route",
    "learned_position_router",
    "learned_token_router",
    "weak_position_router",
    "weak_token_router",
}
POSITION_ROUTER_MODES = {"learned_position_router", "weak_position_router"}
TOKEN_ROUTER_MODES = {"learned_token_router", "weak_token_router"}
WEAK_SUPERVISION_MODES = {"weak_position_router", "weak_token_router"}


@dataclass
class ModelSummary:
    config: str
    seed: int
    branch_head_dims_json: str
    eval_loss: float
    local_loss: float
    induction_loss: float
    local_accuracy: float
    induction_accuracy: float
    local_branch0_loss_delta: float
    local_branch1_loss_delta: float
    induction_branch0_loss_delta: float
    induction_branch1_loss_delta: float
    local_branch0_specialization: float
    local_branch1_specialization: float
    induction_branch0_specialization: float
    induction_branch1_specialization: float
    local_top_branch: int
    induction_top_branch: int
    same_top_branch: bool
    routed_role_match: bool
    branch_distribution_distance: float
    local_gate_branch0_mean: float
    local_gate_branch1_mean: float
    induction_gate_branch0_mean: float
    induction_gate_branch1_mean: float
    local_gate_entropy_mean: float
    induction_gate_entropy_mean: float
    global_gate_entropy_mean: float
    global_gate_branch0_mean: float
    global_gate_branch1_mean: float
    global_gate_balance_error_mean: float
    gate_distribution_distance: float
    gate_routed_role_match: bool
    gate_target_nll_mean: float


@dataclass
class BranchScore:
    config: str
    seed: int
    branch: int
    branch_head_dims_json: str
    local_loss_delta: float
    induction_loss_delta: float
    local_specialization: float
    induction_specialization: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "branch_sum",
            "learned_position_router",
            "learned_token_router",
            "weak_position_router",
            "weak_token_router",
            "oracle_route",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--local-pairs", type=int, default=8)
    parser.add_argument("--repeat-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=160)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--branch-head-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-supervision-weight", type=float, default=0.05)
    parser.add_argument("--router-entropy-weight", type=float, default=0.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.0)
    parser.add_argument("--local-weight", type=float, default=0.25)
    parser.add_argument("--induction-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_branch_isolation"))
    return parser.parse_args()


class BranchTower(nn.Module):
    def __init__(self, d_model: int, head_dims: list[int], n_layers: int, mlp_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model, head_dims, mlp_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h, _, _ = block(h, causal_mask, None)
        return h - x


class TwoBranchTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        branch_head_dims: list[int],
        n_layers: int,
        mlp_dim: int,
        mode: str,
        local_positions: torch.Tensor,
        induction_positions: torch.Tensor,
        router_temperature: float,
    ) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown branch mode: {mode}")
        self.mode = mode
        self.router_temperature = router_temperature
        self.branch_head_dims = list(branch_head_dims)
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model))
        self.branches = nn.ModuleList(
            [BranchTower(d_model, branch_head_dims, n_layers, mlp_dim) for _ in range(2)]
        )
        self.position_router_logits = nn.Parameter(torch.zeros(seq_len, 2))
        self.token_router = nn.Linear(d_model, 2)
        self.final_ln = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("local_positions", local_positions.clone().long(), persistent=False)
        self.register_buffer("induction_positions", induction_positions.clone().long(), persistent=False)

    def embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        return self.token_embed(input_ids) + self.pos_embed[:seq_len]

    def route_weights(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        weights = torch.ones(batch_size, 2, seq_len, device=device, dtype=dtype)
        if self.mode == "oracle_route":
            weights[:, :, self.local_positions.to(device)] = torch.tensor(
                [1.0, 0.0], device=device, dtype=dtype
            ).view(1, 2, 1)
            weights[:, :, self.induction_positions.to(device)] = torch.tensor(
                [0.0, 1.0], device=device, dtype=dtype
            ).view(1, 2, 1)
        elif self.mode in POSITION_ROUTER_MODES:
            logits = self.position_router_logits[:seq_len].to(device=device, dtype=dtype)
            probs = torch.softmax(logits / self.router_temperature, dim=-1)
            weights = probs.T.unsqueeze(0).expand(batch_size, -1, -1)
        elif self.mode in TOKEN_ROUTER_MODES:
            logits = self.token_router(x)
            probs = torch.softmax(logits / self.router_temperature, dim=-1)
            weights = probs.permute(0, 2, 1)
        return weights

    def forward(self, input_ids: torch.Tensor, ablate_branch: int | None = None) -> torch.Tensor:
        device = input_ids.device
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        x = self.embed_input(input_ids)
        branch_deltas = []
        for branch_idx, branch in enumerate(self.branches):
            delta = branch(x, causal_mask)
            if ablate_branch is not None and branch_idx == ablate_branch:
                delta = torch.zeros_like(delta)
            branch_deltas.append(delta)

        weights = self.route_weights(x)
        h = x
        for branch_idx, delta in enumerate(branch_deltas):
            h = h + weights[:, branch_idx, :].unsqueeze(-1) * delta
        return self.unembed(self.final_ln(h))


def routing_target_nll(
    weights: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
) -> torch.Tensor:
    distribution = normalized_gate_distribution(weights)
    local_positions = layout["local_positions"].to(weights.device)
    induction_positions = layout["induction_positions"].to(weights.device)
    local_nll = -distribution[:, 0, local_positions].clamp_min(1e-12).log().mean()
    induction_nll = -distribution[:, 1, induction_positions].clamp_min(1e-12).log().mean()
    return 0.5 * (local_nll + induction_nll)


def routing_regularization_loss(
    weights: torch.Tensor,
    entropy_weight: float,
    balance_weight: float,
) -> torch.Tensor:
    distribution = normalized_gate_distribution(weights)
    loss = weights.new_tensor(0.0)
    if entropy_weight > 0.0:
        loss = loss + entropy_weight * gate_entropy(distribution).mean()
    if balance_weight > 0.0:
        branch_usage = distribution.mean(dim=(0, 2))
        target = torch.full_like(branch_usage, 1.0 / branch_usage.numel())
        loss = loss + balance_weight * torch.square(branch_usage - target).sum()
    return loss


def train_model(
    model: TwoBranchTransformer,
    train_rng: np.random.Generator,
    args: argparse.Namespace,
    layout: dict[str, torch.Tensor | int],
    device: torch.device,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for step in range(args.steps):
        input_ids = make_competition_dataset(
            train_rng,
            args.batch_size,
            args.local_pairs,
            args.repeat_length,
            args.vocab_size,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss, _ = combined_loss_accuracy(logits, input_ids, layout, args.local_weight, args.induction_weight)
        if (
            model.mode in WEAK_SUPERVISION_MODES
            and args.router_supervision_weight > 0.0
        ) or args.router_entropy_weight > 0.0 or args.router_balance_weight > 0.0:
            weights = model.route_weights(model.embed_input(input_ids))
        if model.mode in WEAK_SUPERVISION_MODES and args.router_supervision_weight > 0.0:
            loss = loss + args.router_supervision_weight * routing_target_nll(weights, layout)
        if args.router_entropy_weight > 0.0 or args.router_balance_weight > 0.0:
            loss = loss + routing_regularization_loss(
                weights,
                args.router_entropy_weight,
                args.router_balance_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"  step={step + 1} train_loss={last_loss:.4f}", flush=True)
    return last_loss


def evaluate(
    model: TwoBranchTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    local_weight: float,
    induction_weight: float,
    batch_size: int,
    device: torch.device,
    ablate_branch: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    total = 0
    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            logits = model(ids, ablate_branch=ablate_branch)
            loss, metrics = combined_loss_accuracy(logits, ids, layout, local_weight, induction_weight)
            batch_size_actual = ids.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size_actual
            for key, value in metrics.items():
                totals[key] += value * batch_size_actual
            total += batch_size_actual
    return {key: value / total for key, value in totals.items()}


def branch_distribution(scores: np.ndarray) -> np.ndarray:
    return specialization_distribution(np.maximum(scores, 0.0))


def distribution_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def normalized_gate_distribution(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def gate_entropy(distribution: torch.Tensor) -> torch.Tensor:
    return -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=1)


def collect_route_metrics(
    model: TwoBranchTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, torch.Tensor | int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    total = 0
    local_positions_cpu = layout["local_positions"]
    induction_positions_cpu = layout["induction_positions"]
    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            x = model.embed_input(ids)
            weights = model.route_weights(x)
            distribution = normalized_gate_distribution(weights)
            entropy = gate_entropy(distribution)
            local_positions = local_positions_cpu.to(device)
            induction_positions = induction_positions_cpu.to(device)
            local_dist = distribution[:, :, local_positions].mean(dim=(0, 2))
            induction_dist = distribution[:, :, induction_positions].mean(dim=(0, 2))
            global_dist = distribution.mean(dim=(0, 2))
            global_balance_error = torch.abs(global_dist - 0.5).sum()
            local_entropy = entropy[:, local_positions].mean()
            induction_entropy = entropy[:, induction_positions].mean()
            global_entropy = entropy.mean()
            batch_size_actual = ids.shape[0]
            totals["local_gate_branch0"] += float(local_dist[0].detach().cpu()) * batch_size_actual
            totals["local_gate_branch1"] += float(local_dist[1].detach().cpu()) * batch_size_actual
            totals["induction_gate_branch0"] += float(induction_dist[0].detach().cpu()) * batch_size_actual
            totals["induction_gate_branch1"] += float(induction_dist[1].detach().cpu()) * batch_size_actual
            totals["local_gate_entropy"] += float(local_entropy.detach().cpu()) * batch_size_actual
            totals["induction_gate_entropy"] += float(induction_entropy.detach().cpu()) * batch_size_actual
            totals["global_gate_entropy"] += float(global_entropy.detach().cpu()) * batch_size_actual
            totals["global_gate_branch0"] += float(global_dist[0].detach().cpu()) * batch_size_actual
            totals["global_gate_branch1"] += float(global_dist[1].detach().cpu()) * batch_size_actual
            totals["global_gate_balance_error"] += (
                float(global_balance_error.detach().cpu()) * batch_size_actual
            )
            totals["gate_target_nll"] += float(routing_target_nll(weights, layout).detach().cpu()) * batch_size_actual
            total += batch_size_actual

    metrics = {key: value / total for key, value in totals.items()}
    local_gate = np.asarray([metrics["local_gate_branch0"], metrics["local_gate_branch1"]])
    induction_gate = np.asarray([metrics["induction_gate_branch0"], metrics["induction_gate_branch1"]])
    metrics["gate_distribution_distance"] = distribution_distance(local_gate, induction_gate)
    metrics["gate_routed_role_match"] = bool(np.argmax(local_gate) == 0 and np.argmax(induction_gate) == 1)
    return metrics


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_config(config: str, model_rows: list[ModelSummary]) -> dict[str, object]:
    rows = [row for row in model_rows if row.config == config]
    local_top = Counter(row.local_top_branch for row in rows)
    induction_top = Counter(row.induction_top_branch for row in rows)
    return {
        "n_models": len(rows),
        "eval_loss_mean": float(np.mean([row.eval_loss for row in rows])),
        "local_loss_mean": float(np.mean([row.local_loss for row in rows])),
        "induction_loss_mean": float(np.mean([row.induction_loss for row in rows])),
        "local_accuracy_mean": float(np.mean([row.local_accuracy for row in rows])),
        "induction_accuracy_mean": float(np.mean([row.induction_accuracy for row in rows])),
        "same_top_branch_rate": float(np.mean([row.same_top_branch for row in rows])),
        "routed_role_match_rate": float(np.mean([row.routed_role_match for row in rows])),
        "branch_distribution_distance_mean": float(
            np.mean([row.branch_distribution_distance for row in rows])
        ),
        "gate_routed_role_match_rate": float(np.mean([row.gate_routed_role_match for row in rows])),
        "gate_distribution_distance_mean": float(np.mean([row.gate_distribution_distance for row in rows])),
        "local_gate_entropy_mean": float(np.mean([row.local_gate_entropy_mean for row in rows])),
        "induction_gate_entropy_mean": float(np.mean([row.induction_gate_entropy_mean for row in rows])),
        "global_gate_entropy_mean": float(np.mean([row.global_gate_entropy_mean for row in rows])),
        "global_gate_branch0_mean": float(np.mean([row.global_gate_branch0_mean for row in rows])),
        "global_gate_branch1_mean": float(np.mean([row.global_gate_branch1_mean for row in rows])),
        "global_gate_balance_error_mean": float(
            np.mean([row.global_gate_balance_error_mean for row in rows])
        ),
        "local_gate_branch0_mean": float(np.mean([row.local_gate_branch0_mean for row in rows])),
        "local_gate_branch1_mean": float(np.mean([row.local_gate_branch1_mean for row in rows])),
        "induction_gate_branch0_mean": float(np.mean([row.induction_gate_branch0_mean for row in rows])),
        "induction_gate_branch1_mean": float(np.mean([row.induction_gate_branch1_mean for row in rows])),
        "gate_target_nll_mean": float(np.mean([row.gate_target_nll_mean for row in rows])),
        "local_branch0_loss_delta_mean": float(np.mean([row.local_branch0_loss_delta for row in rows])),
        "local_branch1_loss_delta_mean": float(np.mean([row.local_branch1_loss_delta for row in rows])),
        "induction_branch0_loss_delta_mean": float(
            np.mean([row.induction_branch0_loss_delta for row in rows])
        ),
        "induction_branch1_loss_delta_mean": float(
            np.mean([row.induction_branch1_loss_delta for row in rows])
        ),
        "local_top_branch_counts": {str(branch): count for branch, count in sorted(local_top.items())},
        "induction_top_branch_counts": {
            str(branch): count for branch, count in sorted(induction_top.items())
        },
    }


def write_config_summary(path: Path, summary_by_config: dict[str, dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "n_models",
        "eval_loss_mean",
        "local_loss_mean",
        "induction_loss_mean",
        "local_accuracy_mean",
        "induction_accuracy_mean",
        "same_top_branch_rate",
        "routed_role_match_rate",
        "branch_distribution_distance_mean",
        "gate_routed_role_match_rate",
        "gate_distribution_distance_mean",
        "local_gate_entropy_mean",
        "induction_gate_entropy_mean",
        "global_gate_entropy_mean",
        "global_gate_branch0_mean",
        "global_gate_branch1_mean",
        "global_gate_balance_error_mean",
        "local_gate_branch0_mean",
        "local_gate_branch1_mean",
        "induction_gate_branch0_mean",
        "induction_gate_branch1_mean",
        "gate_target_nll_mean",
        "local_branch0_loss_delta_mean",
        "local_branch1_loss_delta_mean",
        "induction_branch0_loss_delta_mean",
        "induction_branch1_loss_delta_mean",
        "local_top_branch_counts_json",
        "induction_top_branch_counts_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config, values in summary_by_config.items():
            row = dict(values)
            local_counts = row.pop("local_top_branch_counts")
            induction_counts = row.pop("induction_top_branch_counts")
            writer.writerow(
                {
                    "config": config,
                    **row,
                    "local_top_branch_counts_json": json.dumps(local_counts),
                    "induction_top_branch_counts_json": json.dumps(induction_counts),
                }
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    layout = sequence_layout(args.local_pairs, args.repeat_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = make_competition_dataset(
        np.random.default_rng(12345),
        args.eval_examples,
        args.local_pairs,
        args.repeat_length,
        args.vocab_size,
    )

    model_rows: list[ModelSummary] = []
    branch_rows: list[BranchScore] = []

    for config in args.configs:
        print(f"config={config} branch_head_dims={args.branch_head_dims}", flush=True)
        for seed in args.seeds:
            print(f"training config={config} seed={seed}", flush=True)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = TwoBranchTransformer(
                vocab_size=args.vocab_size,
                seq_len=int(layout["seq_len"]),
                d_model=args.d_model,
                branch_head_dims=args.branch_head_dims,
                n_layers=args.n_layers,
                mlp_dim=args.mlp_dim,
                mode=config,
                local_positions=layout["local_positions"],
                induction_positions=layout["induction_positions"],
                router_temperature=args.router_temperature,
            ).to(device)
            train_model(model, np.random.default_rng(seed), args, layout, device)
            baseline = evaluate(
                model,
                eval_ids,
                layout,
                args.local_weight,
                args.induction_weight,
                args.batch_size,
                device,
            )
            route_metrics = collect_route_metrics(model, eval_ids, layout, args.batch_size, device)
            ablations = [
                evaluate(
                    model,
                    eval_ids,
                    layout,
                    args.local_weight,
                    args.induction_weight,
                    args.batch_size,
                    device,
                    ablate_branch=branch,
                )
                for branch in range(2)
            ]
            local_scores = np.asarray(
                [max(metrics["local_loss"] - baseline["local_loss"], 0.0) for metrics in ablations],
                dtype=np.float64,
            )
            induction_scores = np.asarray(
                [max(metrics["induction_loss"] - baseline["induction_loss"], 0.0) for metrics in ablations],
                dtype=np.float64,
            )
            local_dist = branch_distribution(local_scores)
            induction_dist = branch_distribution(induction_scores)
            local_top = int(np.argmax(local_scores))
            induction_top = int(np.argmax(induction_scores))

            for branch in range(2):
                branch_rows.append(
                    BranchScore(
                        config=config,
                        seed=seed,
                        branch=branch,
                        branch_head_dims_json=json.dumps(args.branch_head_dims),
                        local_loss_delta=float(local_scores[branch]),
                        induction_loss_delta=float(induction_scores[branch]),
                        local_specialization=float(local_dist[branch]),
                        induction_specialization=float(induction_dist[branch]),
                    )
                )

            model_rows.append(
                ModelSummary(
                    config=config,
                    seed=seed,
                    branch_head_dims_json=json.dumps(args.branch_head_dims),
                    eval_loss=baseline["loss"],
                    local_loss=baseline["local_loss"],
                    induction_loss=baseline["induction_loss"],
                    local_accuracy=baseline["local_accuracy"],
                    induction_accuracy=baseline["induction_accuracy"],
                    local_branch0_loss_delta=float(local_scores[0]),
                    local_branch1_loss_delta=float(local_scores[1]),
                    induction_branch0_loss_delta=float(induction_scores[0]),
                    induction_branch1_loss_delta=float(induction_scores[1]),
                    local_branch0_specialization=float(local_dist[0]),
                    local_branch1_specialization=float(local_dist[1]),
                    induction_branch0_specialization=float(induction_dist[0]),
                    induction_branch1_specialization=float(induction_dist[1]),
                    local_top_branch=local_top,
                    induction_top_branch=induction_top,
                    same_top_branch=local_top == induction_top,
                    routed_role_match=local_top == 0 and induction_top == 1,
                    branch_distribution_distance=distribution_distance(local_dist, induction_dist),
                    local_gate_branch0_mean=float(route_metrics["local_gate_branch0"]),
                    local_gate_branch1_mean=float(route_metrics["local_gate_branch1"]),
                    induction_gate_branch0_mean=float(route_metrics["induction_gate_branch0"]),
                    induction_gate_branch1_mean=float(route_metrics["induction_gate_branch1"]),
                    local_gate_entropy_mean=float(route_metrics["local_gate_entropy"]),
                    induction_gate_entropy_mean=float(route_metrics["induction_gate_entropy"]),
                    global_gate_entropy_mean=float(route_metrics["global_gate_entropy"]),
                    global_gate_branch0_mean=float(route_metrics["global_gate_branch0"]),
                    global_gate_branch1_mean=float(route_metrics["global_gate_branch1"]),
                    global_gate_balance_error_mean=float(route_metrics["global_gate_balance_error"]),
                    gate_distribution_distance=float(route_metrics["gate_distribution_distance"]),
                    gate_routed_role_match=bool(route_metrics["gate_routed_role_match"]),
                    gate_target_nll_mean=float(route_metrics["gate_target_nll"]),
                )
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_by_config = {config: summarize_config(config, model_rows) for config in args.configs}
    write_csv(args.output_dir / "model_summary.csv", model_rows)
    write_csv(args.output_dir / "branch_scores.csv", branch_rows)
    write_config_summary(args.output_dir / "config_summary.csv", summary_by_config)
    payload = {
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "layout": {
            key: int(value) if isinstance(value, int) else [int(item) for item in value.tolist()]
            for key, value in layout.items()
        },
        "summary_by_config": summary_by_config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary_by_config, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
