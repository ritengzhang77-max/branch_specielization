# Phase 0 Raw-Score Pilot: Pythia-160M Seeds 1 vs 2

Date: 2026-05-21

This pilot reruns the Pythia-160M seed 1 vs seed 2 comparison using pre-mask,
pre-softmax GPT-NeoX attention scores, i.e. scaled `QK^T` scores before the
causal mask and softmax. This is closer to the roadmap's intended
"attention-score matrix" metric than comparing returned attention probabilities.

## Command

```bash
python3 -u scripts/attention_stability.py \
  --model-size 160m \
  --seeds 1 2 \
  --revision step143000 \
  --probe-file probes/phase0_probe_texts.txt \
  --max-length 64 \
  --batch-size 1 \
  --device auto \
  --dtype float32 \
  --attention-representation raw_scores \
  --entry-mask causal \
  --random-permutations 100 \
  --output-dir results/phase0_pythia160m_seed1_seed2_raw_scores
```

## Aggregate Result

| Metric | Attention probabilities | Raw pre-softmax scores |
|---|---:|---:|
| Raw same-index similarity, mean over layers | 0.7082 | 0.3342 |
| Hungarian-matched similarity, mean over layers | 0.8220 | 0.6831 |
| Matched minus random-permutation baseline, mean over layers | 0.0989 | 0.3444 |

The raw-score metric is much more discriminating. Same-index similarity drops
substantially, while Hungarian matching still recovers corresponding heads. This
is exactly the distinction the project needs: same head index is not very stable,
but corresponding roles may exist after relabeling.

## Layer Summary

| Layer | Raw same-index mean | Matched mean | Random mean | Matched-random gap |
|---:|---:|---:|---:|---:|
| 0 | 0.8569 | 0.9131 | 0.8746 | 0.0385 |
| 1 | 0.5483 | 0.7142 | 0.5865 | 0.1277 |
| 2 | 0.3923 | 0.6341 | 0.3649 | 0.2693 |
| 3 | 0.3091 | 0.6805 | 0.3316 | 0.3488 |
| 4 | -0.0067 | 0.6778 | 0.0291 | 0.6487 |
| 5 | 0.3926 | 0.7986 | 0.2071 | 0.5916 |
| 6 | 0.1801 | 0.5197 | 0.1000 | 0.4197 |
| 7 | 0.1575 | 0.4752 | 0.0133 | 0.4619 |
| 8 | 0.1489 | 0.5697 | 0.2540 | 0.3157 |
| 9 | 0.5045 | 0.6402 | 0.5053 | 0.1349 |
| 10 | 0.3830 | 0.7586 | 0.4916 | 0.2670 |
| 11 | 0.1443 | 0.8154 | 0.3068 | 0.5086 |

## Interpretation

This is the strongest early signal so far.

Probability-space comparison made many heads look moderately similar even under
random permutation. Raw-score comparison reveals much lower same-index stability,
especially in middle and late layers, while matched stability remains high.

This supports the weak-universality framing:

> Across seeds, models appear to learn related attention-score roles, but those
> roles are not reliably assigned to the same head index.

It also shows why raw-score extraction should become the primary Stage 1 metric.

## Caution

This is still only one seed pair. The all-seed raw-score run is needed before
making any layer-wise claim about the Bali-style stability dip.

