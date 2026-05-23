# Phase 1: Pythia-160M Seed-9 Alignment Transfer at Selected Checkpoints

Date: 2026-05-22

## Question

The 3-seed Pythia-160M alignment trajectory showed that repeat-match causal
head roles transfer across seeds much better after raw-score alignment than by
same head index. This follow-up asks whether that result survives the full
official Pythia-160M seed set at the most decision-relevant checkpoints:

```text
Does raw-score alignment still transfer the repeat-match causal head role across
seeds when tested on all 9 Pythia-160M seeds?
```

## Setup

- Models: `EleutherAI/pythia-160m-seed1` through seed9.
- Checkpoints: `step4000`, `step16000`, `step143000`.
- Layers tested: 0 and 1.
- Probe/eval task: synthetic repeated-token sequences
  `[x_1, ..., x_32, x_1, ..., x_32]`.
- Selected heads: top repeat-match head per tested layer.
- Alignment: checkpoint-specific raw attention-score Hungarian matching using
  all 8 prompts in `probes/phase0_probe_texts.txt`.
- Causal test: zero selected head outputs before the attention output
  projection, then measure second-half continuation loss.
- Controls: 4 random same-layer controls per seed and checkpoint.
- Transfer comparisons: 72 ordered source-target seed pairs per checkpoint.

Because the first all-checkpoint run was interrupted by unrelated GPU contention,
the final run was executed as three one-checkpoint jobs and then combined.

## Aggregate Results

| Revision | Probe spec. | Own top - random | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|---:|---:|
| step4000 | 0.5039 | 0.1659 | -0.0049 | 0.1682 | 0.1731 | 62/72 |
| step16000 | 0.7107 | 0.3650 | 0.0785 | 0.3958 | 0.3173 | 64/72 |
| step143000 | 0.7903 | 1.2586 | 0.2541 | 1.0619 | 0.8078 | 59/72 |

## Interpretation

The main Phase 1 claim survives and becomes stronger:

```text
Raw-score alignment recovers a repeat-match causal head role across Pythia-160M
seeds much better than raw same-index transfer.
```

The 9-seed result also revises the timing story. In the 3-seed pilot, `step4000`
looked like a mostly probe-only checkpoint. With all 9 seeds and a larger
alignment probe set, `step4000` already shows measurable causal aligned transfer:

- own selected heads beat random controls by `0.1659`;
- aligned source-head transfer beats same-index transfer by `0.1731`;
- aligned transfer beats same-index in `62/72` ordered source-target pairs.

So the better conclusion is not that causal transfer is absent at `step4000`.
The cleaner conclusion is:

```text
Repeat-match specialization, causal importance, and aligned cross-seed transfer
all strengthen across training, but aligned transfer is already detectable by
step4000 in the 9-seed Pythia-160M run.
```

At the final checkpoint, the magnitude is much larger: aligned transfer loss
delta is `1.0619` versus `0.2541` for same-index transfer. Alignment still does
not win in every pair (`59/72`), so the final result is robust but heterogeneous
across source-target pairs.

## What This Means For The Project

This is the strongest real-transformer evidence so far for weak, relabeled
cross-seed role universality:

- the repeat-match role exists across seeds;
- it is causally meaningful under head-output ablation;
- it is not reliably attached to the same raw head index;
- raw-score alignment transfers the causal role substantially better.

This remains a specialization/stability result, not a branch-modularity result.
Pythia is ordinary MHA with residual-stream mixing, so it does not establish that
the role is functionally modular or isolated from other heads.

## Caveats

- Only Pythia-160M was tested in the 9-seed scale-up.
- Only layers 0 and 1 were tested.
- The task is synthetic repeated-token continuation.
- The causal intervention is head-output zero ablation, not full path patching.
- Alignment uses eight short probe texts; a larger, more diverse corpus remains
  a useful robustness check.
- The first all-checkpoint run was interrupted by unrelated GPU contention. The
  reported result comes from three completed one-checkpoint jobs with identical
  metric settings.

## Files

- Main trajectory script:
  `scripts/pythia_repeat_match_alignment_trajectory.py`.
- Paired-transfer analyzer:
  `scripts/analyze_pythia_alignment_trajectory.py`.
- Combined selected-checkpoint analyzer:
  `scripts/analyze_pythia_seed9_alignment_selected.py`.
- Combined results:
  `results/phase1_pythia160m_repeat_match_alignment_seed9_selected/`.
- Checkpoint deck:
  `presentations/phase1/2026-05-22-2032-pythia160m-seed9-alignment/outputs/pythia160m_seed9_alignment_checkpoint.pdf`.

## Reproduction Commands

Run each selected checkpoint separately:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u scripts/pythia_repeat_match_alignment_trajectory.py \
  --model-size 160m \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --revisions step4000 \
  --layers 0,1 \
  --top-k-per-layer 1 \
  --random-controls 4 \
  --probe-sequences 64 \
  --eval-sequences 64 \
  --repeat-length 32 \
  --batch-size 8 \
  --alignment-num-texts 8 \
  --alignment-batch-size 2 \
  --random-permutations 100 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step4000
```

Repeat the same command with `--revisions step16000` and
`--output-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step16000`,
then with `--revisions step143000` and
`--output-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step143000`.

Then analyze:

```bash
python3 -u scripts/analyze_pythia_alignment_trajectory.py \
  --input-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step4000
python3 -u scripts/analyze_pythia_alignment_trajectory.py \
  --input-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step16000
python3 -u scripts/analyze_pythia_alignment_trajectory.py \
  --input-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step143000
python3 -u scripts/analyze_pythia_seed9_alignment_selected.py
```

## Recommended Next Step

The next Phase 1 expansion should test whether this alignment-transfer result
is specific to repeat-match heads:

```text
Add at least one second causal head-role task, preferably previous-token or
induction-style copying with a path-patching readout, and test whether raw-score
alignment again transfers the causal role better than same index.
```

This is more informative than immediately adding more Pythia checkpoints, since
the selected-checkpoint trajectory already establishes the repeat-match result
across all official Pythia-160M seeds.
