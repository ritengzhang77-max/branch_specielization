# Phase 0 Pilot: Pythia-160M Seeds 1 vs 2

Date: 2026-05-21

This pilot moves from the toy Pythia-14M smoke test to the first target-scale
model in the roadmap: Pythia-160M. It is still a calibration run, because it uses
only two seeds and eight short probe texts.

## Command

```bash
python3 scripts/attention_stability.py \
  --model-size 160m \
  --seeds 1 2 \
  --revision main \
  --probe-file probes/phase0_probe_texts.txt \
  --max-length 64 \
  --batch-size 2 \
  --device auto \
  --dtype float32 \
  --random-permutations 100 \
  --output-dir results/phase0_pythia160m_seed1_seed2
```

## Result Summary

Models:

- `EleutherAI/pythia-160m-seed1`
- `EleutherAI/pythia-160m-seed2`

Aggregate over 12 layers:

| Metric | Value |
|---|---:|
| Raw same-index similarity, mean over layers | 0.7082 |
| Hungarian-matched similarity, mean over layers | 0.8220 |
| Matched minus random-permutation baseline, mean over layers | 0.0989 |
| Minimum layer matched-minus-random gap | 0.0533 |
| Maximum layer matched-minus-random gap | 0.1475 |

Layer-level matched-minus-random gap:

| Layer | Gap |
|---:|---:|
| 0 | 0.1057 |
| 1 | 0.1185 |
| 2 | 0.0841 |
| 3 | 0.1475 |
| 4 | 0.0885 |
| 5 | 0.0533 |
| 6 | 0.0534 |
| 7 | 0.1087 |
| 8 | 0.0853 |
| 9 | 0.0959 |
| 10 | 0.1127 |
| 11 | 0.1340 |

## Interpretation

The Pythia-160M pilot confirms that the Phase 0 script works on the model scale
that the roadmap prioritizes.

The run shows:

- same-index attention patterns are already moderately similar across seeds;
- Hungarian matching gives a substantial increase over raw same-index matching;
- the matched score remains above a random-permutation baseline in every layer.

This supports the weak-universality framing: heads are not just arbitrary, but
head roles may need relabeling across seeds.

Do not treat this as evidence for or against the Bali et al. mid-layer stability
dip yet. With only one seed pair and eight short texts, the layer profile is too
small to interpret scientifically.

## Next Decisive Experiment

Scale this exact analysis to all available Pythia-160M seeds and a larger probe
set. The next report should aggregate across many seed pairs and ask:

- Does the matched-minus-random gap vary systematically by layer?
- Is there a mid-layer instability dip?
- Are late-layer similarities inflated by generic attention sinks or repeated
  prompt statistics?
- How sensitive are results to probe-set size and domain?

