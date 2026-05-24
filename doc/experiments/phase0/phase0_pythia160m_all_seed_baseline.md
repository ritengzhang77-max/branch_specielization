# Phase 0 Baseline: Pythia-160M Seeds 1-9

Date: 2026-05-21

This run scales the Phase 0 attention-stability metric to all nine official
Pythia-160M seeds using the final `step143000` revision. It is the first useful
baseline for the project, but it still uses a small probe set and attention
probabilities rather than pre-softmax attention logits.

## Command

```bash
python3 scripts/attention_stability.py \
  --model-size 160m \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --revision step143000 \
  --probe-file probes/phase0_probe_texts.txt \
  --max-length 64 \
  --batch-size 2 \
  --device auto \
  --dtype float32 \
  --random-permutations 200 \
  --output-dir results/phase0_pythia160m_seed1_to_9_step143000
```

## Aggregate Result

Models:

- `EleutherAI/pythia-160m-seed1` through `EleutherAI/pythia-160m-seed9`
- revision: `step143000`

Probe set:

- 8 short texts
- maximum sequence length 64

Aggregate over 36 seed pairs x 12 layers = 432 layer-pair comparisons:

| Metric | Value |
|---|---:|
| Raw same-index similarity, mean over layer pairs | 0.7127 |
| Hungarian-matched similarity, mean over layer pairs | 0.8127 |
| Matched minus random-permutation baseline, mean over layer pairs | 0.0998 |
| Minimum layer-pair matched-minus-random gap | 0.0303 |
| Maximum layer-pair matched-minus-random gap | 0.2272 |

## Layer Summary

| Layer | Seed pairs | Raw same-index mean | Matched mean | Random mean | Matched-random gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 36 | 0.7135 | 0.8205 | 0.7092 | 0.1113 |
| 1 | 36 | 0.7256 | 0.8492 | 0.7300 | 0.1191 |
| 2 | 36 | 0.7303 | 0.8196 | 0.7279 | 0.0917 |
| 3 | 36 | 0.6854 | 0.7912 | 0.6797 | 0.1115 |
| 4 | 36 | 0.7028 | 0.7874 | 0.7038 | 0.0836 |
| 5 | 36 | 0.8465 | 0.9027 | 0.8452 | 0.0575 |
| 6 | 36 | 0.8287 | 0.8890 | 0.8281 | 0.0609 |
| 7 | 36 | 0.7921 | 0.8628 | 0.7892 | 0.0736 |
| 8 | 36 | 0.6889 | 0.8045 | 0.6903 | 0.1142 |
| 9 | 36 | 0.6861 | 0.7819 | 0.6848 | 0.0970 |
| 10 | 36 | 0.6720 | 0.7820 | 0.6768 | 0.1053 |
| 11 | 36 | 0.4804 | 0.6613 | 0.4895 | 0.1718 |

## Interpretation

This baseline supports three immediate conclusions.

First, the pipeline is scalable to all nine Pythia-160M seeds. The run completed
without model-loading, GPU-memory, or matching failures.

Second, the matched-over-random gap is positive for every layer. That means
Hungarian matching is not merely producing arbitrary improvements; there is
non-random cross-seed structure in attention patterns.

Third, the layer profile needs careful interpretation. Layers 5-6 have high raw
same-index similarity, high matched similarity, and high random-permutation
similarity, which yields a relatively small matched-minus-random gap. This may
mean many heads in those layers share generic attention patterns on this small
probe set. It should not yet be interpreted as confirming or refuting Bali et
al.'s reported mid-layer stability dip.

The final layer has low raw same-index similarity but the largest matched-minus-
random gap. That suggests the same broad attention roles may exist across seeds
but are less tied to the same head index in the last layer.

## Current Limitation

The script currently compares returned attention matrices from Hugging Face,
which are attention probabilities after softmax. The roadmap and Bali et al.
prior emphasize attention score matrices. The next rigor upgrade is to either:

1. capture pre-softmax attention logits with model hooks, or
2. explicitly frame this metric as attention-probability similarity and keep it
   separate from raw-score similarity.

## Next Decisive Experiments

1. Increase probe-set size and diversity to check whether the high mid-layer
   random baseline is a probe artifact.
2. Add raw pre-softmax attention-score extraction for Pythia/GPT-NeoX.
3. Add simple head-role probes for induction and previous-token behavior.
4. Compute the first task-specific specialization score `S(h,t)` for induction
   copying, then compare it across seeds with and without Hungarian matching.

