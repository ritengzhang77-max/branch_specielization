#!/usr/bin/env python3
"""Toy Ontology v2 attention-head dimension intervention.

The unit of analysis is an ordinary attention head. The dataset is a single
synthetic next-token LM sequence containing many role scenes. Each role produces
one row in a role x head causal matrix via single-head ablation.

The script tests three questions:

1. structural role affinity: which head dimension/type a role chooses;
2. functional specialization: how concentrated each role is over heads;
3. functional modularity: whether related roles have similar head distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from toy_head_dim_intervention import TinyTransformer, resolve_device, specialization_distribution


CONFIG_PRESETS = {
    "uniform4": [32, 32, 32, 32],
    "uniform2": [64, 64],
    "hetero4_unique_mild": [16, 24, 40, 48],
    "hetero4_unique_64": [8, 16, 40, 64],
    "hetero4_unique_extreme": [8, 16, 24, 80],
    "hetero2_unique_mild": [48, 80],
    "hetero2_unique_mid": [32, 96],
    "hetero2_unique_extreme": [16, 112],
}


SPECIAL = {
    "local_copy": 1,
    "previous_token": 2,
    "kv_lookup": 3,
    "duplicate_token": 4,
    "induction_short": 5,
    "induction_long": 6,
    "induction_ngram": 7,
    "false_induction_control": 8,
    "bos_sink": 9,
    "sep_sink": 10,
    "fixed_offset_prev": 11,
    "punctuation_boundary": 12,
    "distractor_suppression": 13,
    "anti_copy": 14,
    "wrong_key_suppression": 15,
    "recency_conflict": 16,
    "repeated_name_detection": 17,
    "pronoun_antecedent": 18,
    "simple_ioi_name_mover": 19,
    "negative_name_control": 20,
    "punct": 21,
    "pronoun": 22,
}
TOKEN_LOW = 32


ROLE_ORDER = [
    "local_copy",
    "previous_token",
    "kv_lookup",
    "duplicate_token",
    "induction_short",
    "induction_long",
    "induction_ngram",
    "false_induction_control",
    "bos_sink",
    "sep_sink",
    "fixed_offset_prev",
    "punctuation_boundary",
    "distractor_suppression",
    "anti_copy",
    "wrong_key_suppression",
    "recency_conflict",
    "repeated_name_detection",
    "pronoun_antecedent",
    "simple_ioi_name_mover",
    "negative_name_control",
]


ROLE_FAMILIES = {
    "local_copy": "copy_transport",
    "previous_token": "copy_transport",
    "kv_lookup": "copy_transport",
    "duplicate_token": "copy_transport",
    "induction_short": "induction",
    "induction_long": "induction",
    "induction_ngram": "induction",
    "false_induction_control": "induction",
    "bos_sink": "position_boundary",
    "sep_sink": "position_boundary",
    "fixed_offset_prev": "position_boundary",
    "punctuation_boundary": "position_boundary",
    "distractor_suppression": "suppression_conflict",
    "anti_copy": "suppression_conflict",
    "wrong_key_suppression": "suppression_conflict",
    "recency_conflict": "suppression_conflict",
    "repeated_name_detection": "entity_coreference",
    "pronoun_antecedent": "entity_coreference",
    "simple_ioi_name_mover": "entity_coreference",
    "negative_name_control": "entity_coreference",
}


ROLE_METADATA = {
    "local_copy": {
        "scene": "[x, LOCAL_SEP, x]",
        "target": "At LOCAL_SEP, predict the token just before the separator.",
        "control": "The target token is random per example, so a constant separator shortcut fails.",
    },
    "previous_token": {
        "scene": "[x, y, PREV_SEP, y]",
        "target": "At PREV_SEP, copy the immediately previous token y.",
        "control": "The target is not fixed and changes independently of the separator.",
    },
    "kv_lookup": {
        "scene": "[KV_SEP, k1,v1,...,kq,vq]",
        "target": "At the query key kq, predict its paired value vq.",
        "control": "Keys and values are resampled; the query index varies per example.",
    },
    "duplicate_token": {
        "scene": "[DUP_SEP, a,b,c,a,DUP_QUERY,a]",
        "target": "At DUP_QUERY, predict the repeated token a.",
        "control": "Distractor tokens b and c appear once.",
    },
    "induction_short": {
        "scene": "[IND_SHORT_SEP, x1,...,x6,x1,...,x6]",
        "target": "On the second copy, predict the token that followed the first occurrence.",
        "control": "The base sequence is random per example.",
    },
    "induction_long": {
        "scene": "[IND_LONG_SEP, x1,...,x12,x1,...,x12]",
        "target": "Same induction rule over a longer context.",
        "control": "Length differs from the short induction role.",
    },
    "induction_ngram": {
        "scene": "[IND_NGRAM_SEP, x1,...,x9,x1,...,x9]",
        "target": "Repeated n-gram continuation over an intermediate length.",
        "control": "Uses a distinct marker and length from short/long induction.",
    },
    "false_induction_control": {
        "scene": "[FALSE_IND_SEP, a,b,c,a,c]",
        "target": "At the second a, predict c, not the previous continuation b.",
        "control": "The earlier a->b continuation is a deliberate induction trap.",
    },
    "bos_sink": {
        "scene": "[anchor,n1,n2,n3,BOS_QUERY,anchor]",
        "target": "At BOS_QUERY, retrieve the first token of the segment.",
        "control": "The first token is random, so this is not constant BOS prediction.",
    },
    "sep_sink": {
        "scene": "[SEP_ANCHOR,value,n1,n2,SEP_QUERY,value]",
        "target": "At SEP_QUERY, retrieve the token after the earlier separator.",
        "control": "Noise tokens intervene between separator and query.",
    },
    "fixed_offset_prev": {
        "scene": "[a,b,c,d,OFFSET_QUERY,b]",
        "target": "At OFFSET_QUERY, retrieve the token three positions back.",
        "control": "Nearby tokens a/c/d are distractors.",
    },
    "punctuation_boundary": {
        "scene": "[n1,PUNCT,value,n2,PUNCT_QUERY,value]",
        "target": "At PUNCT_QUERY, retrieve the token after punctuation.",
        "control": "The punctuation value changes each example.",
    },
    "distractor_suppression": {
        "scene": "[SUPPRESS_SEP,target,distractor,distractor,SUPPRESS_QUERY,target]",
        "target": "At SUPPRESS_QUERY, predict target while ignoring the repeated distractor.",
        "control": "The repeated distractor is closer and more frequent than the target.",
    },
    "anti_copy": {
        "scene": "[a,b,c,a,ANTI_COPY_QUERY,c]",
        "target": "At ANTI_COPY_QUERY, predict c and ignore the misleading a->b continuation.",
        "control": "Combines a repeated-token trap with a non-induction target.",
    },
    "wrong_key_suppression": {
        "scene": "[WK_SEP,key,right,wrong,wrong_value,wrong,trap,key,right]",
        "target": "At the final key, predict right while ignoring wrong-key decoys.",
        "control": "Wrong-key tokens and values are repeated near the query.",
    },
    "recency_conflict": {
        "scene": "[RECENCY_SEP,key,old,key,recent,key,old]",
        "target": "At the final key, predict the older value rather than the recent value.",
        "control": "The recent value is closer and shares the same key.",
    },
    "repeated_name_detection": {
        "scene": "[REP_NAME_SEP,A,B,A,REP_NAME_QUERY,A]",
        "target": "At REP_NAME_QUERY, identify the repeated name token A.",
        "control": "The other name B appears only once.",
    },
    "pronoun_antecedent": {
        "scene": "[name,PRONOUN,n1,n2,PRONOUN_QUERY,name]",
        "target": "At PRONOUN_QUERY, retrieve the name before the pronoun marker.",
        "control": "Noise tokens intervene between pronoun and query.",
    },
    "simple_ioi_name_mover": {
        "scene": "[A,B,A,IOI_QUERY,B]",
        "target": "At IOI_QUERY, predict the non-repeated name B.",
        "control": "The repeated name A is the distractor.",
    },
    "negative_name_control": {
        "scene": "[A,B,B,NEG_NAME_QUERY,A]",
        "target": "At NEG_NAME_QUERY, predict the non-repeated name A.",
        "control": "The repeated name B is the distractor.",
    },
}


@dataclass
class ModelSummary:
    config: str
    seed: int
    role_set: str
    head_dims_json: str
    eval_loss_mean: float
    role_accuracy_mean: float
    role_accuracy_min: float
    role_specialization_mean: float
    role_effective_heads_mean: float
    role_top_dim_mass_mean: float
    within_family_similarity_mean: float
    between_family_similarity_mean: float
    family_gap: float
    family_cluster_ari: float


@dataclass
class RoleSummary:
    config: str
    seed: int
    role_set: str
    role: str
    family: str
    target_count: int
    loss: float
    accuracy: float
    top_layer: int
    top_head: int
    top_head_dim: int
    top_role_score: float
    global_top_specialization: float
    effective_heads: float
    top_dim: int
    top_dim_mass: float
    head_dim_affinity_json: str


@dataclass
class HeadRoleScore:
    config: str
    seed: int
    role_set: str
    role: str
    family: str
    layer: int
    head: int
    head_dim: int
    role_score: float
    role_mass: float
    ablation_accuracy_delta: float


@dataclass
class RolePairSummary:
    config: str
    seed: int
    role_set: str
    role_a: str
    family_a: str
    role_b: str
    family_b: str
    same_family: bool
    tv_distance: float
    similarity: float


@dataclass
class FamilySummary:
    config: str
    seed: int
    role_set: str
    family: str
    n_roles: int
    accuracy_mean: float
    specialization_mean: float
    effective_heads_mean: float
    top_dim_mass_mean: float
    within_family_similarity_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["uniform4", "uniform2", "hetero4_unique_mild", "hetero4_unique_64", "hetero2_unique_mild"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--role-set", default="v2_full", choices=["v2_synthetic", "v2_full"])
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_toy_role_ontology_v2_head_dim"))
    return parser.parse_args()


def resolve_head_dims(config_name: str) -> list[int]:
    if config_name in CONFIG_PRESETS:
        return CONFIG_PRESETS[config_name]
    try:
        dims = [int(item) for item in config_name.split(",") if item]
    except ValueError as exc:
        raise ValueError(f"Unknown config preset or head-dim list: {config_name}") from exc
    if not dims or min(dims) <= 0:
        raise ValueError(f"Invalid head-dim list: {config_name}")
    return dims


def roles_for_set(role_set: str) -> list[str]:
    if role_set == "v2_synthetic":
        return ROLE_ORDER[:16]
    if role_set == "v2_full":
        return list(ROLE_ORDER)
    raise ValueError(f"Unknown role set: {role_set}")


def choose(rng: np.random.Generator, token_ids: np.ndarray, n: int) -> list[int]:
    return [int(item) for item in rng.choice(token_ids, size=n, replace=False)]


def append_local_copy(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    x = choose(rng, token_ids, 1)[0]
    start = len(row)
    row.extend([x, SPECIAL["local_copy"], x])
    return [start + 1]


def append_previous_token(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    x, y = choose(rng, token_ids, 2)
    start = len(row)
    row.extend([x, y, SPECIAL["previous_token"], y])
    return [start + 2]


def append_kv_lookup(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    keys = choose(rng, token_ids, 4)
    values = choose(rng, token_ids, 4)
    query_idx = int(rng.integers(0, 4))
    start = len(row)
    row.append(SPECIAL["kv_lookup"])
    for key, value in zip(keys, values):
        row.extend([key, value])
    row.extend([keys[query_idx], values[query_idx]])
    return [start + 1 + 2 * len(keys)]


def append_duplicate_token(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b, c = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([SPECIAL["duplicate_token"], a, b, c, a, SPECIAL["duplicate_token"], a])
    return [start + 5]


def append_induction(row: list[int], rng: np.random.Generator, token_ids: np.ndarray, role: str, length: int) -> list[int]:
    base = choose(rng, token_ids, length)
    start = len(row)
    row.append(SPECIAL[role])
    row.extend(base)
    second_start = len(row)
    row.extend(base)
    return list(range(second_start, second_start + length - 1))


def append_false_induction_control(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b, c = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([SPECIAL["false_induction_control"], a, b, c, a, c])
    return [start + 4]


def append_bos_sink(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    anchor, n1, n2, n3 = choose(rng, token_ids, 4)
    start = len(row)
    row.extend([anchor, n1, n2, n3, SPECIAL["bos_sink"], anchor])
    return [start + 4]


def append_sep_sink(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    value, n1, n2 = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([SPECIAL["sep_sink"], value, n1, n2, SPECIAL["sep_sink"], value])
    return [start + 4]


def append_fixed_offset_prev(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b, c, d = choose(rng, token_ids, 4)
    start = len(row)
    row.extend([a, b, c, d, SPECIAL["fixed_offset_prev"], b])
    return [start + 4]


def append_punctuation_boundary(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    n1, value, n2 = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([n1, SPECIAL["punct"], value, n2, SPECIAL["punctuation_boundary"], value])
    return [start + 4]


def append_distractor_suppression(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    target, distractor = choose(rng, token_ids, 2)
    start = len(row)
    row.extend([SPECIAL["distractor_suppression"], target, distractor, distractor, SPECIAL["distractor_suppression"], target])
    return [start + 4]


def append_anti_copy(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b, c = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([a, b, c, a, SPECIAL["anti_copy"], c])
    return [start + 4]


def append_wrong_key_suppression(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    key, right, wrong, wrong_value, trap = choose(rng, token_ids, 5)
    start = len(row)
    row.extend([SPECIAL["wrong_key_suppression"], key, right, wrong, wrong_value, wrong, trap, key, right])
    return [start + 7]


def append_recency_conflict(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    key, old, recent = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([SPECIAL["recency_conflict"], key, old, key, recent, key, old])
    return [start + 5]


def append_repeated_name_detection(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b = choose(rng, token_ids, 2)
    start = len(row)
    row.extend([SPECIAL["repeated_name_detection"], a, b, a, SPECIAL["repeated_name_detection"], a])
    return [start + 4]


def append_pronoun_antecedent(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    name, n1, n2 = choose(rng, token_ids, 3)
    start = len(row)
    row.extend([name, SPECIAL["pronoun"], n1, n2, SPECIAL["pronoun_antecedent"], name])
    return [start + 4]


def append_simple_ioi_name_mover(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b = choose(rng, token_ids, 2)
    start = len(row)
    row.extend([a, b, a, SPECIAL["simple_ioi_name_mover"], b])
    return [start + 3]


def append_negative_name_control(row: list[int], rng: np.random.Generator, token_ids: np.ndarray) -> list[int]:
    a, b = choose(rng, token_ids, 2)
    start = len(row)
    row.extend([a, b, b, SPECIAL["negative_name_control"], a])
    return [start + 3]


ROLE_APPENDERS = {
    "local_copy": append_local_copy,
    "previous_token": append_previous_token,
    "kv_lookup": append_kv_lookup,
    "duplicate_token": append_duplicate_token,
    "induction_short": lambda row, rng, token_ids: append_induction(row, rng, token_ids, "induction_short", 6),
    "induction_long": lambda row, rng, token_ids: append_induction(row, rng, token_ids, "induction_long", 12),
    "induction_ngram": lambda row, rng, token_ids: append_induction(row, rng, token_ids, "induction_ngram", 9),
    "false_induction_control": append_false_induction_control,
    "bos_sink": append_bos_sink,
    "sep_sink": append_sep_sink,
    "fixed_offset_prev": append_fixed_offset_prev,
    "punctuation_boundary": append_punctuation_boundary,
    "distractor_suppression": append_distractor_suppression,
    "anti_copy": append_anti_copy,
    "wrong_key_suppression": append_wrong_key_suppression,
    "recency_conflict": append_recency_conflict,
    "repeated_name_detection": append_repeated_name_detection,
    "pronoun_antecedent": append_pronoun_antecedent,
    "simple_ioi_name_mover": append_simple_ioi_name_mover,
    "negative_name_control": append_negative_name_control,
}


def make_example(
    rng: np.random.Generator,
    token_ids: np.ndarray,
    role_names: list[str],
) -> tuple[list[int], dict[str, list[int]], dict[str, list[int]]]:
    row: list[int] = []
    role_positions: dict[str, list[int]] = {}
    role_spans: dict[str, list[int]] = {}
    for role in role_names:
        start = len(row)
        role_positions[role] = ROLE_APPENDERS[role](row, rng, token_ids)
        role_spans[role] = [start, len(row)]
    return row, role_positions, role_spans


def build_layout(role_names: list[str], vocab_size: int) -> dict[str, object]:
    token_ids = np.arange(TOKEN_LOW, vocab_size)
    row, role_positions, role_spans = make_example(np.random.default_rng(0), token_ids, role_names)
    return {
        "seq_len": len(row),
        "role_positions": {
            role: torch.tensor(positions, dtype=torch.long)
            for role, positions in role_positions.items()
        },
        "role_spans": role_spans,
    }


def make_ontology_dataset(
    rng: np.random.Generator,
    n_examples: int,
    role_names: list[str],
    vocab_size: int,
    seq_len: int,
) -> torch.Tensor:
    if vocab_size <= TOKEN_LOW + 32:
        raise ValueError("vocab_size is too small for the requested ontology task.")
    token_ids = np.arange(TOKEN_LOW, vocab_size)
    rows = []
    for _ in range(n_examples):
        row, _, _ = make_example(rng, token_ids, role_names)
        if len(row) != seq_len:
            raise RuntimeError(f"Bad generated row length: {len(row)} != {seq_len}")
        rows.append(row)
    return torch.tensor(np.asarray(rows), dtype=torch.long)


def objective_loss_accuracy(logits: torch.Tensor, input_ids: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, float]:
    positions = positions.to(logits.device)
    selected_logits = logits[:, positions, :]
    targets = input_ids[:, positions + 1]
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    accuracy = float((selected_logits.argmax(dim=-1) == targets).float().mean().detach().cpu())
    return loss, accuracy


def combined_loss_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    layout: dict[str, object],
    role_names: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    role_positions: dict[str, torch.Tensor] = layout["role_positions"]  # type: ignore[assignment]
    losses = []
    metrics = {}
    for role in role_names:
        role_loss, role_accuracy = objective_loss_accuracy(logits, input_ids, role_positions[role])
        losses.append(role_loss)
        metrics[f"{role}_loss"] = float(role_loss.detach().cpu())
        metrics[f"{role}_accuracy"] = role_accuracy
    loss = torch.stack(losses).mean()
    metrics["loss_mean"] = float(loss.detach().cpu())
    metrics["accuracy_mean"] = float(np.mean([metrics[f"{role}_accuracy"] for role in role_names]))
    return loss, metrics


def iter_batches(input_ids: torch.Tensor, batch_size: int):
    for start in range(0, input_ids.shape[0], batch_size):
        yield input_ids[start : start + batch_size]


def train_model(
    model: TinyTransformer,
    train_rng: np.random.Generator,
    args: argparse.Namespace,
    layout: dict[str, object],
    role_names: list[str],
    device: torch.device,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    seq_len = int(layout["seq_len"])
    for step in range(args.steps):
        input_ids = make_ontology_dataset(
            train_rng,
            args.batch_size,
            role_names,
            args.vocab_size,
            seq_len,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(input_ids)
        loss, _ = combined_loss_accuracy(logits, input_ids, layout, role_names)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % max(args.steps // 4, 1) == 0:
            print(f"  step={step + 1} train_loss={last_loss:.4f}", flush=True)
    return last_loss


def evaluate(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, object],
    role_names: list[str],
    batch_size: int,
    device: torch.device,
    ablate: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    total = 0
    with torch.inference_mode():
        for ids in iter_batches(input_ids, batch_size):
            ids = ids.to(device)
            logits, _, _ = model(ids, ablate=ablate)
            _, metrics = combined_loss_accuracy(logits, ids, layout, role_names)
            batch_size_actual = ids.shape[0]
            for key, value in metrics.items():
                totals[key] += value * batch_size_actual
            total += batch_size_actual
    return {key: value / total for key, value in totals.items()}


def evaluate_all_single_head_ablations(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    layout: dict[str, object],
    role_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    baseline: dict[str, float],
) -> dict[str, dict[str, list[np.ndarray]]]:
    role_scores = {role: [] for role in role_names}
    role_accuracy_deltas = {role: [] for role in role_names}
    for layer_idx in range(len(model.blocks)):
        layer_scores = {role: [] for role in role_names}
        layer_accuracy_deltas = {role: [] for role in role_names}
        for head_idx in range(len(model.head_dims)):
            metrics = evaluate(model, input_ids, layout, role_names, args.batch_size, device, ablate=[(layer_idx, head_idx)])
            for role in role_names:
                layer_scores[role].append(metrics[f"{role}_loss"] - baseline[f"{role}_loss"])
                layer_accuracy_deltas[role].append(metrics[f"{role}_accuracy"] - baseline[f"{role}_accuracy"])
        for role in role_names:
            role_scores[role].append(np.asarray(layer_scores[role], dtype=np.float64))
            role_accuracy_deltas[role].append(np.asarray(layer_accuracy_deltas[role], dtype=np.float64))
    return {
        "role_scores": role_scores,
        "role_accuracy_deltas": role_accuracy_deltas,
    }


def flatten_role_scores(role_scores: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.maximum(layer, 0.0) for layer in role_scores]).astype(np.float64)


def entropy(distribution: np.ndarray, eps: float = 1e-12) -> float:
    clipped = np.clip(distribution.astype(np.float64), eps, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def distribution_tv_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def head_dim_affinity(distribution: np.ndarray, head_dims: list[int]) -> dict[int, float]:
    affinity: dict[int, float] = defaultdict(float)
    n_heads = len(head_dims)
    for flat_idx, mass in enumerate(distribution):
        head_idx = flat_idx % n_heads
        affinity[int(head_dims[head_idx])] += float(mass)
    return dict(sorted(affinity.items()))


def top_role_summary(
    distribution: np.ndarray,
    raw_scores: np.ndarray,
    head_dims: list[int],
) -> dict[str, float | int | dict[int, float]]:
    n_heads = len(head_dims)
    top_idx = int(np.argmax(distribution))
    top_layer = top_idx // n_heads
    top_head = top_idx % n_heads
    affinity = head_dim_affinity(distribution, head_dims)
    top_dim = max(affinity.items(), key=lambda item: item[1])[0]
    return {
        "top_layer": top_layer,
        "top_head": top_head,
        "top_head_dim": int(head_dims[top_head]),
        "top_role_score": float(raw_scores[top_idx]),
        "global_top_specialization": float(distribution[top_idx]),
        "effective_heads": float(math.exp(entropy(distribution))),
        "top_dim": int(top_dim),
        "top_dim_mass": float(affinity[top_dim]),
        "head_dim_affinity": affinity,
    }


def adjusted_rand_index(true_labels: list[str], pred_labels: list[int]) -> float:
    def comb2(n: int) -> float:
        return n * (n - 1) / 2.0

    n = len(true_labels)
    if n < 2:
        return 0.0

    contingency = Counter(zip(true_labels, pred_labels))
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    sum_comb = sum(comb2(count) for count in contingency.values())
    sum_true = sum(comb2(count) for count in true_counts.values())
    sum_pred = sum(comb2(count) for count in pred_counts.values())
    total = comb2(n)
    expected = sum_true * sum_pred / total if total else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((sum_comb - expected) / denom)


def cluster_roles(
    role_names: list[str],
    role_families: dict[str, str],
    role_distributions: dict[str, np.ndarray],
) -> tuple[list[int], float]:
    n_roles = len(role_names)
    distance = np.zeros((n_roles, n_roles), dtype=np.float64)
    for i, role_a in enumerate(role_names):
        for j, role_b in enumerate(role_names):
            distance[i, j] = distribution_tv_distance(role_distributions[role_a], role_distributions[role_b])
    condensed = squareform(distance, checks=False)
    n_families = len(set(role_families[role] for role in role_names))
    clusters = fcluster(linkage(condensed, method="average"), t=n_families, criterion="maxclust")
    ari = adjusted_rand_index([role_families[role] for role in role_names], [int(item) for item in clusters])
    return [int(item) for item in clusters], ari


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_config(
    config: str,
    role_names: list[str],
    model_rows: list[ModelSummary],
    role_rows: list[RoleSummary],
) -> dict[str, object]:
    config_models = [row for row in model_rows if row.config == config]
    config_roles = [row for row in role_rows if row.config == config]
    role_top_dim_counts: dict[str, dict[str, int]] = {}
    role_top_head_counts: dict[str, dict[str, int]] = {}
    for role in role_names:
        dim_counts = Counter(row.top_dim for row in config_roles if row.role == role)
        role_top_dim_counts[role] = {str(dim): count for dim, count in sorted(dim_counts.items())}
        head_counts = Counter(f"L{row.top_layer}H{row.top_head}" for row in config_roles if row.role == role)
        role_top_head_counts[role] = {str(head): count for head, count in sorted(head_counts.items())}

    return {
        "n_models": len(config_models),
        "eval_loss_mean": float(np.mean([row.eval_loss_mean for row in config_models])),
        "role_accuracy_mean": float(np.mean([row.role_accuracy_mean for row in config_models])),
        "role_accuracy_min": float(np.min([row.role_accuracy_min for row in config_models])),
        "role_specialization_mean": float(np.mean([row.role_specialization_mean for row in config_models])),
        "role_effective_heads_mean": float(np.mean([row.role_effective_heads_mean for row in config_models])),
        "role_top_dim_mass_mean": float(np.mean([row.role_top_dim_mass_mean for row in config_models])),
        "within_family_similarity_mean": float(np.mean([row.within_family_similarity_mean for row in config_models])),
        "between_family_similarity_mean": float(np.mean([row.between_family_similarity_mean for row in config_models])),
        "family_gap_mean": float(np.mean([row.family_gap for row in config_models])),
        "family_cluster_ari_mean": float(np.mean([row.family_cluster_ari for row in config_models])),
        "role_top_dim_counts": role_top_dim_counts,
        "role_top_head_counts": role_top_head_counts,
    }


def write_config_summary(path: Path, summary_by_config: dict[str, dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "n_models",
        "eval_loss_mean",
        "role_accuracy_mean",
        "role_accuracy_min",
        "role_specialization_mean",
        "role_effective_heads_mean",
        "role_top_dim_mass_mean",
        "within_family_similarity_mean",
        "between_family_similarity_mean",
        "family_gap_mean",
        "family_cluster_ari_mean",
        "role_top_dim_counts_json",
        "role_top_head_counts_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for config, values in summary_by_config.items():
            row = dict(values)
            role_top_dim_counts = row.pop("role_top_dim_counts")
            role_top_head_counts = row.pop("role_top_head_counts")
            writer.writerow(
                {
                    "config": config,
                    **row,
                    "role_top_dim_counts_json": json.dumps(role_top_dim_counts),
                    "role_top_head_counts_json": json.dumps(role_top_head_counts),
                }
            )


def write_sample_examples(path: Path, role_names: list[str], layout: dict[str, object], vocab_size: int) -> None:
    token_ids = np.arange(TOKEN_LOW, vocab_size)
    row, role_positions, role_spans = make_example(np.random.default_rng(20260523), token_ids, role_names)
    payload = {
        "full_sequence": row,
        "role_examples": {},
    }
    for role in role_names:
        start, end = role_spans[role]
        positions = role_positions[role]
        payload["role_examples"][role] = {
            "family": ROLE_FAMILIES[role],
            **ROLE_METADATA[role],
            "segment_start": start,
            "segment_end_exclusive": end,
            "segment_tokens": row[start:end],
            "score_positions": positions,
            "target_tokens": [row[position + 1] for position in positions],
        }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    role_names = roles_for_set(args.role_set)
    role_families = {role: ROLE_FAMILIES[role] for role in role_names}
    layout = build_layout(role_names, args.vocab_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = make_ontology_dataset(
        np.random.default_rng(12345),
        args.eval_examples,
        role_names,
        args.vocab_size,
        int(layout["seq_len"]),
    )

    model_rows: list[ModelSummary] = []
    role_rows: list[RoleSummary] = []
    head_rows: list[HeadRoleScore] = []
    pair_rows: list[RolePairSummary] = []
    family_rows: list[FamilySummary] = []
    head_dims_by_config: dict[str, list[int]] = {}

    print(
        f"role_set={args.role_set} n_roles={len(role_names)} "
        f"n_families={len(set(role_families.values()))} seq_len={layout['seq_len']} device={device}",
        flush=True,
    )

    for config_name in args.configs:
        head_dims = resolve_head_dims(config_name)
        head_dims_by_config[config_name] = head_dims
        print(f"config={config_name} head_dims={head_dims} total_head_dim={sum(head_dims)}", flush=True)
        for seed in args.seeds:
            print(f"training config={config_name} seed={seed}", flush=True)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = TinyTransformer(
                vocab_size=args.vocab_size,
                seq_len=int(layout["seq_len"]),
                d_model=args.d_model,
                head_dims=head_dims,
                n_layers=args.n_layers,
                mlp_dim=args.mlp_dim,
            ).to(device)

            train_model(model, np.random.default_rng(seed), args, layout, role_names, device)
            baseline = evaluate(model, eval_ids, layout, role_names, args.batch_size, device)
            ablation = evaluate_all_single_head_ablations(model, eval_ids, layout, role_names, args, device, baseline)

            role_distributions: dict[str, np.ndarray] = {}
            role_specializations = []
            role_effective_heads = []
            role_top_dim_masses = []
            role_accuracies = []

            role_positions: dict[str, torch.Tensor] = layout["role_positions"]  # type: ignore[assignment]
            family_acc: dict[str, list[float]] = defaultdict(list)
            family_spec: dict[str, list[float]] = defaultdict(list)
            family_eff: dict[str, list[float]] = defaultdict(list)
            family_top_dim_mass: dict[str, list[float]] = defaultdict(list)

            for role in role_names:
                family = role_families[role]
                flat_scores = flatten_role_scores(ablation["role_scores"][role])
                distribution = specialization_distribution(flat_scores)
                role_distributions[role] = distribution
                summary = top_role_summary(distribution, flat_scores, head_dims)
                role_specializations.append(float(summary["global_top_specialization"]))
                role_effective_heads.append(float(summary["effective_heads"]))
                role_top_dim_masses.append(float(summary["top_dim_mass"]))
                role_accuracies.append(baseline[f"{role}_accuracy"])
                family_acc[family].append(baseline[f"{role}_accuracy"])
                family_spec[family].append(float(summary["global_top_specialization"]))
                family_eff[family].append(float(summary["effective_heads"]))
                family_top_dim_mass[family].append(float(summary["top_dim_mass"]))

                role_rows.append(
                    RoleSummary(
                        config=config_name,
                        seed=seed,
                        role_set=args.role_set,
                        role=role,
                        family=family,
                        target_count=int(role_positions[role].numel()),
                        loss=baseline[f"{role}_loss"],
                        accuracy=baseline[f"{role}_accuracy"],
                        top_layer=int(summary["top_layer"]),
                        top_head=int(summary["top_head"]),
                        top_head_dim=int(summary["top_head_dim"]),
                        top_role_score=float(summary["top_role_score"]),
                        global_top_specialization=float(summary["global_top_specialization"]),
                        effective_heads=float(summary["effective_heads"]),
                        top_dim=int(summary["top_dim"]),
                        top_dim_mass=float(summary["top_dim_mass"]),
                        head_dim_affinity_json=json.dumps({str(k): v for k, v in summary["head_dim_affinity"].items()}),
                    )
                )

                n_heads = len(head_dims)
                for flat_idx, role_mass in enumerate(distribution):
                    layer_idx = flat_idx // n_heads
                    head_idx = flat_idx % n_heads
                    head_rows.append(
                        HeadRoleScore(
                            config=config_name,
                            seed=seed,
                            role_set=args.role_set,
                            role=role,
                            family=family,
                            layer=layer_idx,
                            head=head_idx,
                            head_dim=head_dims[head_idx],
                            role_score=float(flat_scores[flat_idx]),
                            role_mass=float(role_mass),
                            ablation_accuracy_delta=float(ablation["role_accuracy_deltas"][role][layer_idx][head_idx]),
                        )
                    )

            within = []
            between = []
            family_pair_sims: dict[str, list[float]] = defaultdict(list)
            for role_a, role_b in combinations(role_names, 2):
                tv = distribution_tv_distance(role_distributions[role_a], role_distributions[role_b])
                similarity = 1.0 - tv
                family_a = role_families[role_a]
                family_b = role_families[role_b]
                same_family = family_a == family_b
                if same_family:
                    within.append(similarity)
                    family_pair_sims[family_a].append(similarity)
                else:
                    between.append(similarity)
                pair_rows.append(
                    RolePairSummary(
                        config=config_name,
                        seed=seed,
                        role_set=args.role_set,
                        role_a=role_a,
                        family_a=family_a,
                        role_b=role_b,
                        family_b=family_b,
                        same_family=same_family,
                        tv_distance=tv,
                        similarity=similarity,
                    )
                )

            _, ari = cluster_roles(role_names, role_families, role_distributions)
            within_mean = float(np.mean(within)) if within else 0.0
            between_mean = float(np.mean(between)) if between else 0.0
            model_rows.append(
                ModelSummary(
                    config=config_name,
                    seed=seed,
                    role_set=args.role_set,
                    head_dims_json=json.dumps(head_dims),
                    eval_loss_mean=baseline["loss_mean"],
                    role_accuracy_mean=float(np.mean(role_accuracies)),
                    role_accuracy_min=float(np.min(role_accuracies)),
                    role_specialization_mean=float(np.mean(role_specializations)),
                    role_effective_heads_mean=float(np.mean(role_effective_heads)),
                    role_top_dim_mass_mean=float(np.mean(role_top_dim_masses)),
                    within_family_similarity_mean=within_mean,
                    between_family_similarity_mean=between_mean,
                    family_gap=within_mean - between_mean,
                    family_cluster_ari=ari,
                )
            )

            for family, accuracies in sorted(family_acc.items()):
                family_rows.append(
                    FamilySummary(
                        config=config_name,
                        seed=seed,
                        role_set=args.role_set,
                        family=family,
                        n_roles=len(accuracies),
                        accuracy_mean=float(np.mean(accuracies)),
                        specialization_mean=float(np.mean(family_spec[family])),
                        effective_heads_mean=float(np.mean(family_eff[family])),
                        top_dim_mass_mean=float(np.mean(family_top_dim_mass[family])),
                        within_family_similarity_mean=float(np.mean(family_pair_sims[family]))
                        if family_pair_sims[family]
                        else 0.0,
                    )
                )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_by_config = {
        config: summarize_config(config, role_names, model_rows, role_rows)
        for config in args.configs
    }

    write_csv(args.output_dir / "model_summary.csv", model_rows)
    write_csv(args.output_dir / "role_summary.csv", role_rows)
    write_csv(args.output_dir / "head_role_scores.csv", head_rows)
    write_csv(args.output_dir / "role_pair_summary.csv", pair_rows)
    write_csv(args.output_dir / "family_summary.csv", family_rows)
    write_config_summary(args.output_dir / "config_summary.csv", summary_by_config)
    write_sample_examples(args.output_dir / "role_dataset_examples.json", role_names, layout, args.vocab_size)

    payload = {
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "layout": {
            "seq_len": int(layout["seq_len"]),
            "role_positions": {
                role: [int(item) for item in positions.tolist()]
                for role, positions in layout["role_positions"].items()  # type: ignore[union-attr]
            },
            "role_spans": layout["role_spans"],
        },
        "role_names": role_names,
        "role_families": role_families,
        "role_metadata": {role: ROLE_METADATA[role] for role in role_names},
        "head_dims_by_config": head_dims_by_config,
        "summary_by_config": summary_by_config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary_by_config, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
