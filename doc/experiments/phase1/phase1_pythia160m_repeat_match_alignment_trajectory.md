# Phase 1: Pythia-160M Repeat-Match Alignment Trajectory

Date: 2026-05-22

## Question

The Pythia-14M checkpoint trajectory showed that a repeat-match attention probe
can rise before selected heads have a strong causal effect under ablation. This
follow-up asks the stronger cross-seed question on Pythia-160M:

```text
When repeat-match heads become causally important, is that causal role tied to
the same layer/head index across seeds, or does it transfer after raw-score
alignment?
```

This tests the distinction between strong universality and weak, relabeled role
universality:

- **Strong universality**: the same `(layer, head)` slot carries the role across
  seeds.
- **Relabeled-role universality**: the role appears across seeds, but the slot
  must be matched by a seed-specific alignment.

## Setup

- Models: `EleutherAI/pythia-160m-seed1`, seed2, seed3.
- Checkpoints: `step0`, `step1000`, `step4000`, `step16000`, `step64000`,
  `step143000`.
- Layers tested: 0 and 1.
- Probe: 64 synthetic repeated-token sequences
  `[x_1, ..., x_32, x_1, ..., x_32]`.
- Evaluation: 64 separate repeated-token sequences.
- Selected heads: top repeat-match head in each tested layer.
- Alignment: checkpoint-specific raw attention-score Hungarian matching on four
  fixed natural probe texts from `probes/phase0_probe_texts.txt`.
- Causal intervention: zero selected head outputs before the attention output
  projection, then measure second-half continuation loss.

The run compares four ablation conditions for each target seed and checkpoint:

- `own_top`: the target seed's own selected repeat-match heads.
- `own_random`: random same-layer controls.
- `source_same_index`: source seed's selected heads transferred by same
  `(layer, head)` index into the target seed.
- `source_aligned`: source seed's selected heads transferred through the
  checkpoint-specific raw-score alignment map.

## Metrics

- **Repeat-match specialization**: normalized repeat-match attention score of
  selected heads. This is a probe score, not causal evidence.
- **Own top minus random**: `own_top` loss delta minus mean `own_random` loss
  delta. This asks whether the selected heads are more causally important than
  same-layer controls.
- **Same-index transfer**: loss delta from ablating the target seed's heads at
  the source seed's raw `(layer, head)` slots.
- **Aligned transfer**: loss delta from ablating the target seed's heads matched
  to the source seed's selected heads by raw-score Hungarian alignment.
- **Aligned minus same-index**: paired transfer advantage of role alignment over
  raw index transfer.

## Aggregate Results

Means are over seeds 1-3. Transfer means use the six ordered source-target pairs
per checkpoint.

| Revision | Probe spec. | Own top - random | Same-index transfer | Aligned transfer | Aligned - same |
|---|---:|---:|---:|---:|---:|
| step0 | 0.0859 | 0.0011 | -0.0000 | 0.0005 | 0.0005 |
| step1000 | 0.0995 | 0.0042 | -0.0304 | -0.0371 | -0.0067 |
| step4000 | 0.4794 | -0.0180 | 0.0149 | 0.0034 | -0.0114 |
| step16000 | 0.5916 | 0.2324 | 0.0952 | 0.2890 | 0.1939 |
| step64000 | 0.6778 | 0.0206 | 0.3034 | 0.8057 | 0.5023 |
| step143000 | 0.6715 | 1.2500 | 0.3046 | 1.1774 | 0.8728 |

Paired aligned-transfer comparison:

| Revision | Pairs | Aligned better | Aligned - same mean |
|---|---:|---:|---:|
| step0 | 6 | 4 | 0.0005 |
| step1000 | 6 | 2 | -0.0067 |
| step4000 | 6 | 4 | -0.0114 |
| step16000 | 6 | 5 | 0.1939 |
| step64000 | 6 | 5 | 0.5023 |
| step143000 | 6 | 6 | 0.8728 |

## Interpretation

The result strengthens Phase 1 in three ways.

1. **Probe specialization appears early.** The selected repeat-match
   specialization rises sharply by `step4000` (`0.4794`), but causal excess over
   random controls is still negative or near zero there (`-0.0180`).
2. **Causal importance appears later.** By `step16000`, selected heads are
   causally above random controls (`0.2324`). At the final checkpoint, the own
   top heads are strongly causal (`1.2500` loss-delta excess over random).
3. **Cross-seed role identity is mostly relabeled, not same-index.** At the
   final checkpoint, same-index transfer is much weaker than aligned transfer
   (`0.3046` vs `1.1774`). The aligned transfer advantage is positive in all six
   ordered seed pairs at `step143000`.

The defensible claim is therefore:

```text
In Pythia-160M layers 0/1, repeat-match functional role identity becomes
causally transferable across seeds after raw-score alignment, while raw
same-index transfer remains much weaker.
```

This is evidence for weak cross-seed universality of a causal head role. It is
not evidence by itself that ordinary Pythia attention heads are functionally
modular. Pythia has no explicit router or branch isolation, and the causal test
targets a synthetic repeated-token behavior.

## Caveats

- Only Pythia-160M seeds 1-3 were used.
- Only layers 0 and 1 were tested because repeat-match specialization is
  concentrated there in the existing Phase 1 role-probe results.
- The alignment probe used four natural text prompts. This is enough for a
  pilot, but the final claim should use a larger probe corpus.
- The task is synthetic repeated-token continuation.
- The intervention is head-output zero ablation, not full path patching.
- Random same-layer controls are noisy at `step64000`, so the aligned-vs-same
  paired transfer metric is cleaner than own-top-minus-random for that
  checkpoint.

## Files

- Script: `scripts/pythia_repeat_match_alignment_trajectory.py`.
- Paired-transfer analyzer: `scripts/analyze_pythia_alignment_trajectory.py`.
- Results directory:
  `results/phase1_pythia160m_repeat_match_alignment_trajectory/`.
- Main plot:
  `results/phase1_pythia160m_repeat_match_alignment_trajectory/repeat_match_alignment_trajectory.png`.
- Checkpoint deck:
  `presentations/2026-05-22-1902-pythia160m-alignment-trajectory/pythia160m_alignment_trajectory_checkpoint.pdf`.

## Reproduction Commands

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/pythia_repeat_match_alignment_trajectory.py \
  --model-size 160m \
  --seeds 1 2 3 \
  --revisions step0 step1000 step4000 step16000 step64000 step143000 \
  --layers 0,1 \
  --top-k-per-layer 1 \
  --random-controls 4 \
  --probe-sequences 64 \
  --eval-sequences 64 \
  --repeat-length 32 \
  --batch-size 8 \
  --alignment-num-texts 4 \
  --alignment-batch-size 2 \
  --random-permutations 100 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase1_pythia160m_repeat_match_alignment_trajectory

python3 -u scripts/analyze_pythia_alignment_trajectory.py
```

## Recommended Next Step

Scale this result before making paper-level claims:

```text
Run Pythia-160M seeds 1-9 at selected checkpoints
(`step4000`, `step16000`, `step143000`) with a larger alignment probe set.
```

This is cheaper and more decisive than a full 9-seed, all-checkpoint sweep. It
tests the transition from probe-only specialization to causal aligned transfer,
and it directly asks whether the final aligned-transfer result survives more
seed pairs.
