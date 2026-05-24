# Phase 0 Baseline: Pythia-160M Raw Attention Scores Across 9 Seeds

Date: 2026-05-21

This is the first Stage 1 calibration result that should be treated as
methodologically central. It compares pre-mask, pre-softmax GPT-NeoX attention
scores, i.e. scaled `QK^T`, across all nine Pythia-160M seeds.

## Command

```bash
python3 -u scripts/attention_stability.py \
  --model-size 160m \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --revision step143000 \
  --probe-file probes/phase0_probe_texts.txt \
  --max-length 64 \
  --batch-size 1 \
  --device auto \
  --dtype float32 \
  --attention-representation raw_scores \
  --entry-mask causal \
  --random-permutations 200 \
  --output-dir results/phase0_pythia160m_seed1_to_9_step143000_raw_scores
```

## Aggregate Result

Aggregate over 36 seed pairs x 12 layers = 432 layer-pair comparisons:

| Metric | Value |
|---|---:|
| Raw same-index similarity, mean over layer pairs | 0.3735 |
| Hungarian-matched similarity, mean over layer pairs | 0.6692 |
| Matched minus random-permutation baseline, mean over layer pairs | 0.2982 |
| Minimum layer-pair matched-minus-random gap | 0.0224 |
| Maximum layer-pair matched-minus-random gap | 0.7550 |

## Layer Summary

| Layer | Seed pairs | Same-index mean | Matched mean | Random mean | Matched-random gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 36 | 0.8468 | 0.8940 | 0.8438 | 0.0502 |
| 1 | 36 | 0.4747 | 0.7389 | 0.4704 | 0.2684 |
| 2 | 36 | 0.5029 | 0.6994 | 0.5078 | 0.1916 |
| 3 | 36 | 0.2677 | 0.6377 | 0.2894 | 0.3483 |
| 4 | 36 | 0.1370 | 0.6087 | 0.1469 | 0.4619 |
| 5 | 36 | 0.1725 | 0.6777 | 0.1580 | 0.5198 |
| 6 | 36 | 0.1097 | 0.5814 | 0.1156 | 0.4658 |
| 7 | 36 | 0.0907 | 0.4891 | 0.0502 | 0.4389 |
| 8 | 36 | 0.2074 | 0.5186 | 0.2236 | 0.2949 |
| 9 | 36 | 0.4491 | 0.6605 | 0.4336 | 0.2269 |
| 10 | 36 | 0.5756 | 0.7360 | 0.5692 | 0.1668 |
| 11 | 36 | 0.6482 | 0.7886 | 0.6441 | 0.1445 |

## Comparison Against Attention Probabilities

| Metric | Attention probabilities | Raw pre-softmax scores |
|---|---:|---:|
| Same-index mean | 0.7127 | 0.3735 |
| Hungarian-matched mean | 0.8127 | 0.6692 |
| Matched minus random mean | 0.0998 | 0.2982 |

Probability-space comparison made heads look much more similar under the same
index and under random permutation. Raw-score comparison is more discriminating
and gives a much larger matched-minus-random gap.

## Main Finding

The strongest current framing is:

> Pythia-160M seeds learn related attention-score roles, but those roles are not
> reliably assigned to the same head index. Cross-seed stability is much clearer
> after permutation alignment.

This is a weak-universality result. It supports the project premise and justifies
moving from attention-pattern similarity to task-specific specialization scores.

## Layer-Level Interpretation

The middle layers show the lowest same-index raw-score similarity and the largest
matched-minus-random gaps. In this probe setup, the middle-layer story is not
"no stability"; it is closer to:

```text
low same-index stability + strong relabeled role correspondence
```

That distinction matters because it separates:

- strong universality: same function in same head index;
- weak universality: same family of functions, but permuted across heads.

## Caveats

- The probe set is still small and hand-written.
- Attention-score similarity is not causal evidence of task specialization.
- The raw-score hook currently targets GPT-NeoX/Pythia only.
- The next step must compute `S(h,t)` for real task functions such as induction
  copying or previous-token behavior.

## Decision

Use raw pre-softmax attention-score similarity as the primary Phase 0 stability
metric. Keep returned attention-probability similarity as a secondary sanity
check.

