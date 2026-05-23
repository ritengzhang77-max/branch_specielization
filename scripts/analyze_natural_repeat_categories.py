#!/usr/bin/env python3
"""Classify natural-repeat example spans with simple text heuristics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pythia_natural_repeat_ngram_candidate_pool_alignment import (
    build_natural_repeat_token_ids,
    example_rows,
    load_token_stream,
    split_metadata,
)


@dataclass
class CategoryRow:
    example_id: int
    span_text: str
    primary_category: str
    labels_json: str
    has_digit: bool
    has_quote: bool
    has_tokenizer_markup: bool
    capitalized_word_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example-rows", type=Path)
    source.add_argument("--result-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def classify_span(text: str) -> tuple[str, list[str], dict[str, bool | int]]:
    has_digit = any(char.isdigit() for char in text)
    has_quote = any(char in text for char in ['"', "'", "“", "”"])
    has_tokenizer_markup = "@" in text
    capitalized_words = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
    all_caps = re.findall(r"\b[A-Z]{2,}\b", text)
    labels = []
    if has_digit:
        labels.append("numeric_or_date")
    if has_quote:
        labels.append("quoted_or_title")
    if has_tokenizer_markup:
        labels.append("tokenizer_markup")
    if len(capitalized_words) >= 2:
        labels.append("proper_name_like")
    if all_caps:
        labels.append("initialism_or_all_caps")
    if not labels:
        labels.append("ordinary_phrase")

    for candidate in [
        "numeric_or_date",
        "quoted_or_title",
        "proper_name_like",
        "initialism_or_all_caps",
        "tokenizer_markup",
        "ordinary_phrase",
    ]:
        if candidate in labels:
            primary = candidate
            break
    else:
        primary = "ordinary_phrase"
    features = {
        "has_digit": has_digit,
        "has_quote": has_quote,
        "has_tokenizer_markup": has_tokenizer_markup,
        "capitalized_word_count": len(capitalized_words),
    }
    return primary, labels, features


def rows_from_example_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def rows_from_result_dir(result_dir: Path) -> list[dict[str, str]]:
    summary = json.loads((result_dir / "summary.json").read_text())
    run_args = argparse.Namespace(**summary["args"])
    first_model_name = next(iter(summary["model_names"].values()))
    tokenizer = AutoTokenizer.from_pretrained(first_model_name)
    token_stream = load_token_stream(tokenizer, run_args)
    total_sequences = run_args.probe_sequences + run_args.eval_sequences
    all_ids, all_metadata, _, _ = build_natural_repeat_token_ids(
        token_stream=token_stream,
        tokenizer=tokenizer,
        args=run_args,
        n_sequences=total_sequences,
        rng=np.random.default_rng(123),
    )
    eval_ids = all_ids[run_args.probe_sequences :]
    eval_metadata = split_metadata(all_metadata, run_args.probe_sequences, total_sequences)
    rows = example_rows(tokenizer, eval_ids, eval_metadata, max_examples=len(eval_metadata))
    return [asdict(row) for row in rows]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.result_dir is not None:
        source_rows = rows_from_result_dir(args.result_dir)
        source_name = str(args.result_dir)
    else:
        source_rows = rows_from_example_csv(args.example_rows)
        source_name = str(args.example_rows)

    rows: list[CategoryRow] = []
    for row in source_rows:
        primary, labels, features = classify_span(row["span_text"])
        rows.append(
            CategoryRow(
                example_id=int(row["example_id"]),
                span_text=row["span_text"],
                primary_category=primary,
                labels_json=json.dumps(labels),
                has_digit=bool(features["has_digit"]),
                has_quote=bool(features["has_quote"]),
                has_tokenizer_markup=bool(features["has_tokenizer_markup"]),
                capitalized_word_count=int(features["capitalized_word_count"]),
            )
        )

    counts = Counter(row.primary_category for row in rows)
    label_counts = Counter()
    for row in rows:
        label_counts.update(json.loads(row.labels_json))

    with (args.output_dir / "natural_repeat_category_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary = {
        "source": source_name,
        "n_examples": len(rows),
        "primary_category_counts": dict(counts),
        "label_counts": dict(label_counts),
    }
    (args.output_dir / "natural_repeat_category_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
