# Phase 1: Pythia-14M Repeat-Match Checkpoint Trajectory

Date: 2026-05-22

This checkpoint tests whether the toy router trajectory result has an analogue
in a real pretrained transformer. The toy result was:

```text
the gate/probe can become role-aligned before the branch computation becomes
causally separable.
```

For Pythia, there is no router gate. The closest real-model analogue is:

```text
an attention-role probe can identify repeat-match heads before those heads have
a large causal effect under head-output ablation.
```

## Claim Tested

This run asks whether repeat-match attention specialization and causal
repeat-token behavior emerge at the same training checkpoints.

Positive lag outcome:

```text
repeat-match attention specialization rises before top-head ablation is clearly
larger than same-layer random ablations.
```

No-lag outcome:

```text
the attention probe and causal ablation effect rise together.
```

## Setup

Model:

```text
EleutherAI/pythia-14m-seed{1,2,3}
```

Checkpoints:

```text
step0, step64, step256, step1000, step4000, step16000, step64000, step143000
```

Task:

```text
[x_1, ..., x_32, x_1, ..., x_32]
```

The probe measures attention from each second occurrence of `x_i` back to the
first occurrence of `x_i`. The causal metric measures next-token loss on
second-half continuation positions after zeroing selected head outputs.

To avoid selecting and evaluating on the exact same examples:

- 64 synthetic sequences were used for attention-probe head selection;
- 64 separate synthetic sequences were used for causal ablation evaluation.

Only layers 0 and 1 were tested, matching the earlier Pythia-160M
repeat-match-head result.

Representative command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/pythia_repeat_match_checkpoint_trajectory.py \
  --model-size 14m \
  --seeds 1 2 3 \
  --revisions step0 step64 step256 step1000 step4000 step16000 step64000 step143000 \
  --layers 0,1 \
  --top-k-per-layer 1 \
  --random-controls 8 \
  --probe-sequences 64 \
  --eval-sequences 64 \
  --repeat-length 32 \
  --batch-size 8 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase1_pythia14m_repeat_match_checkpoint_trajectory
```

## Metrics

Probe metric:

```text
selected_specialization_mean = mean normalized repeat-match score of selected
top heads in layers 0 and 1
```

Causal metric:

```text
excess_loss_delta_over_random =
  loss_delta(top repeat-match heads) - mean loss_delta(random same-layer heads)
```

The excess metric is important because some random same-layer ablations improve
loss on this synthetic task at intermediate checkpoints. A positive excess means
the probe-selected heads are more causally important than random same-layer
heads, even if the absolute ablation effect is noisy.

## Aggregate Results

Mean over 3 seeds:

| Revision | Probe specialization | Top loss delta | Random loss delta | Top minus random |
|---|---:|---:|---:|---:|
| step0 | 0.2590 | 0.0018 | -0.0007 | 0.0025 |
| step64 | 0.2600 | 0.0011 | -0.0001 | 0.0012 |
| step256 | 0.2662 | 0.0119 | 0.0021 | 0.0097 |
| step1000 | 0.2796 | 0.0527 | 0.0027 | 0.0500 |
| step4000 | 0.3675 | -0.1155 | -0.1711 | 0.0556 |
| step16000 | 0.5566 | -1.3471 | -2.6695 | 1.3224 |
| step64000 | 0.5888 | 2.5466 | -1.6040 | 4.1506 |
| step143000 | 0.6206 | 5.0833 | -2.2149 | 7.2982 |

## Interpretation

This is a real-transformer analogue of the toy warning, but not an identical
branch-modularity result.

Supported:

```text
attention-role specialization appears before strong causal importance.
```

The probe specialization rises above the early baseline by `step4000`
(`0.3675` vs `0.2590` at step0), while the causal excess over random ablations
is still small (`0.0556`). The causal excess becomes meaningfully larger only
later: `1.3224` at step16000, `4.1506` at step64000, and `7.2982` at step143000.

Not shown:

```text
Pythia has branch-level modularity.
```

This test is about ordinary attention heads, not routed branches. It supports
the broader measurement warning: probe-defined functional specialization can
precede causal functional importance, so specialization metrics and causal
ablation metrics should be reported separately.

## Caveats

- This is a small Pythia-14M pilot, not the final Pythia-160M or MultiBERTs
  analysis.
- Only 3 seeds were tested.
- Only layers 0 and 1 were tested.
- The repeated-token task is synthetic.
- The causal metric uses head-output ablation, not full path patching.
- Random same-layer ablations sometimes improve the synthetic loss, so the
  relative excess over random controls is the most interpretable causal metric
  in this run.

## Project-Level Update

The project now has the same methodological lesson from two different settings:

1. In the routed toy model, gate alignment preceded causal branch modularity.
2. In Pythia-14M, attention-role specialization preceded strong causal head
   importance.

The next paper-facing claim should not be:

```text
specialization metrics imply modularity or causal importance.
```

It should be:

```text
structural or probe-level specialization must be paired with causal tests,
because probe/gate alignment can appear before causal functional separation or
importance.
```

## Recommended Next Step

Run the same checkpoint-trajectory analysis on Pythia-160M, using the already
validated repeat-match layers and cross-seed alignment machinery, then test
whether raw-score-aligned heads become causally important earlier or later than
same-index heads across checkpoints.

## Artifacts

- Script:
  `scripts/pythia_repeat_match_checkpoint_trajectory.py`
- Output:
  `results/phase1_pythia14m_repeat_match_checkpoint_trajectory/`
- Plot:
  `results/phase1_pythia14m_repeat_match_checkpoint_trajectory/repeat_match_checkpoint_trajectory.png`
- Checkpoint deck:
  `presentations/2026-05-22-1759-pythia-repeat-trajectory/pythia_repeat_trajectory_checkpoint.pdf`
