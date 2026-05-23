#!/usr/bin/env python3
"""Parameter-pair diagnostics for saved SwitchHead toy checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


PARAM_RE = re.compile(r"blocks\.(?P<layer>\d+)\.attn\.(?P<param>v|o|sel_v|sel_o)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-experts", type=int, default=2)
    parser.add_argument("--expert-a", type=int, default=0)
    parser.add_argument("--expert-b", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = 0.5 * (a.norm() + b.norm()).clamp_min(1e-12)
    return float(((a - b).norm() / denom).item())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        checkpoint = torch.load(
            args.checkpoint_dir / f"model_seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint["model_state_dict"]
        for key, tensor in state.items():
            match = PARAM_RE.fullmatch(key)
            if match is None:
                continue
            layer = int(match.group("layer"))
            param = match.group("param")
            if tensor.shape[0] % args.n_experts != 0:
                raise ValueError(f"{key} first dimension is not divisible by n_experts={args.n_experts}")
            n_heads = tensor.shape[0] // args.n_experts
            for head in range(n_heads):
                row_a = head * args.n_experts + args.expert_a
                row_b = head * args.n_experts + args.expert_b
                a = tensor[row_a].float()
                b = tensor[row_b].float()
                rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "param": param,
                        "head": head,
                        "cosine": cosine(a, b),
                        "relative_l2": relative_l2(a, b),
                        "norm_a": float(a.norm().item()),
                        "norm_b": float(b.norm().item()),
                    }
                )

    summary_rows = []
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), str(row["param"]))].append(row)
    for (layer, param), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "layer": layer,
                "param": param,
                "n": len(group),
                "cosine_mean": sum(float(row["cosine"]) for row in group) / len(group),
                "relative_l2_mean": sum(float(row["relative_l2"]) for row in group) / len(group),
                "norm_a_mean": sum(float(row["norm_a"]) for row in group) / len(group),
                "norm_b_mean": sum(float(row["norm_b"]) for row in group) / len(group),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "parameter_pair_metrics.csv", rows)
    write_csv(args.output_dir / "parameter_pair_summary.csv", summary_rows)
    for row in summary_rows:
        print(
            f"layer={row['layer']} param={row['param']} "
            f"cos={row['cosine_mean']:.4f} rel_l2={row['relative_l2_mean']:.4f}"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
