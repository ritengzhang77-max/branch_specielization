# Phase 0 Smoke Report: Pythia-14M Attention Stability

Date: 2026-05-21

This is an infrastructure validation, not a paper-level result. The goal was to
confirm that the repository can load multi-seed Pythia checkpoints, extract
attention matrices, compute raw head-index similarity, compute Hungarian-matched
similarity, and compare against a random-permutation baseline.

## Command

```bash
python3 scripts/attention_stability.py \
  --model-size 14m \
  --seeds 1 2 3 \
  --revision main \
  --probe-file probes/phase0_probe_texts.txt \
  --max-length 64 \
  --batch-size 2 \
  --device auto \
  --dtype float32 \
  --random-permutations 100 \
  --output-dir results/phase0_pythia14m_seed1_seed2_seed3
```

## Result Summary

Models:

- `EleutherAI/pythia-14m-seed1`
- `EleutherAI/pythia-14m-seed2`
- `EleutherAI/pythia-14m-seed3`

Probe set:

- 8 short probe texts from `probes/phase0_probe_texts.txt`
- maximum sequence length 64

Aggregate metrics over 3 seed pairs x 6 layers = 18 layer-pair comparisons:

| Metric | Value |
|---|---:|
| Raw same-index similarity, mean over layers | 0.5813 |
| Hungarian-matched similarity, mean over layers | 0.6828 |
| Matched minus random-permutation baseline, mean over layers | 0.1115 |
| Minimum layer-pair matched-minus-random gap | 0.0278 |
| Maximum layer-pair matched-minus-random gap | 0.2424 |

## Interpretation

The pipeline is working and produces the expected distinction between:

- **raw head-index stability**: whether the same `(layer, head)` slot is similar
  across seeds;
- **matched role stability**: whether there exists a corresponding head after
  optimal relabeling;
- **null-adjusted matched stability**: whether the matched result exceeds random
  permutation baselines.

The positive matched-minus-random gap suggests that even this small smoke run has
non-random cross-seed head-role structure. However, the result should not be
overinterpreted:

- Pythia-14M is much smaller than the main target models.
- The probe set has only 8 short texts.
- Attention-pattern similarity is behavioral but not causal.
- No task-specific specialization score `S(h,t)` was computed yet.

## Next Decisive Experiment

Run the same script on Pythia-160M with at least two seeds and the same probe set.
If the script remains stable, scale to all 9 Pythia-160M seeds and generate:

- raw vs matched stability by layer;
- random-baseline-adjusted matched gap by layer;
- a first estimate of whether the mid-layer stability dip appears in this setup.

