# Phase 1: Pythia-160M Local-Copy / Previous-Token Pilot

Date: 2026-05-22

## Question

The repeat-match result showed strong alignment-based causal transfer across
Pythia-160M seeds. This pilot asks whether the same pattern appears for a second
candidate head role:

```text
Do local-copy / previous-token heads transfer causally across seeds better by
raw-score alignment than by same head index?
```

## Task Definition

The synthetic local-copy task uses repeated triples:

```text
[x, SEP, x]
```

The probe score is attention from the `SEP` position back to the immediately
previous token `x`. The causal readout is next-token loss at the `SEP` position,
where the target is the copied token `x`.

This is intended as a previous-token/local-copy contrast to repeat-match. Prior
all-seed role-probe summaries already suggested that plain previous-token
attention is more distributed than repeat-match:

- previous-token all-layer aligned top-head match rate: `0.2824`;
- repeat-match all-layer aligned top-head match rate: `0.2338`;
- previous-token layer 3 aligned top-head match rate: `0.4167`;
- previous-token layer 3 mean max specialization: about `0.2215`;
- repeat-match layers 0/1 mean max specialization: `0.7728` and `0.8045`.

So previous-token role distributions are alignable, but less branch-like by
specialization concentration.

## Completed Pilots

### Pythia-14M two-seed smoke

Command used a tiny `14m` smoke run with seeds 1-2, layer 3, 8 probe/eval
sequences, and 16 local-copy triples per sequence.

Result:

- baseline loss: `27.17`;
- own top excess over random: `-0.1005`;
- aligned-minus-same transfer: `-0.0338`.

Interpretation: the 14M model does not reliably solve this arbitrary local-copy
task. This is mostly an infrastructure smoke test, not scientific evidence.

### Pythia-160M two-seed pilot

Command used seeds 1-2, final checkpoint `step143000`, layer 3, 16 probe/eval
sequences, and 16 local-copy triples per sequence.

| Metric | Value |
|---|---:|
| selected local-copy specialization | 0.3142 |
| baseline loss | 10.3996 |
| own top loss delta | 2.0876 |
| random loss delta | -0.0732 |
| own top excess over random | 2.1609 |
| same-index source transfer | 0.0033 |
| aligned source transfer | 0.1033 |
| aligned - same | 0.0999 |
| aligned better count | 1/2 |

Interpretation:

- The selected layer-3 local-copy head is causally important in its own model.
- Same-index source transfer is essentially zero in this two-seed pilot.
- Raw-score aligned transfer is slightly better than same-index transfer, but
  still far smaller than own-head causality.

This is a useful contrast to repeat-match: local-copy may have strong within-seed
causal heads without strong cross-seed causal transfer, at least in the current
small pilot.

### Pythia-160M all-source / target-seeds 1-3 chunk

After adding partial-output writes and `--target-seeds`, a larger chunk completed
successfully:

- source seeds: 1-9;
- target seeds: 1-3;
- checkpoint: `step143000`;
- layer: 3;
- probe/eval sequences: 64 each;
- local-copy triples per sequence: 32;
- transfer comparisons: 24 ordered source-target pairs.

| Metric | Value |
|---|---:|
| selected local-copy specialization | 0.3258 |
| baseline loss | 10.0969 |
| own top loss delta | 2.8318 |
| random loss delta | -0.0250 |
| own top excess over random | 2.8567 |
| same-index source transfer | 0.1901 |
| aligned source transfer | 1.5894 |
| aligned - same | 1.3993 |
| aligned better count | 17/24 |

Interpretation:

- The weak two-seed cross-seed transfer was not representative of the larger
  source pool.
- For target seeds 1-3, raw-score alignment transfers local-copy causal heads
  much better than same index.
- This is still not a full all-target result. The completed chunk covers all
  source seeds but only one third of the target seeds.

### Pythia-160M all-source / target-seeds 4-6 chunk

The next target chunk completed under the same settings.

| Metric | Value |
|---|---:|
| selected local-copy specialization | 0.3254 |
| baseline loss | 11.0943 |
| own top loss delta | 0.1946 |
| random loss delta | 0.1527 |
| own top excess over random | 0.0420 |
| same-index source transfer | 0.1700 |
| aligned source transfer | 0.1584 |
| aligned - same | -0.0116 |
| aligned better count | 11/24 |

Per-target summaries:

| Target seed | Selected head | Own top excess | Same-index transfer | Aligned transfer | Aligned - same |
|---:|---|---:|---:|---:|---:|
| 4 | L3H9 | -0.1784 | 0.4401 | 0.0606 | -0.3796 |
| 5 | L3H11 | 0.1950 | -0.0399 | 0.0701 | 0.1100 |
| 6 | L3H10 | 0.1093 | 0.1098 | 0.3446 | 0.2348 |

Interpretation:

- The target seeds 4-6 selected heads have similar probe specialization to
  target seeds 1-3, but much weaker own-seed causal importance.
- Because the target's own selected head barely matters above random, the
  aligned source-head transfer test is not expected to show a large positive
  effect.
- This suggests local-copy may be less reliable than repeat-match as a causal
  role, or the current layer-3 probe is picking attention behavior that is not
  consistently causal across seeds.

### Pythia-160M all-source / target-seeds 7-9 chunk

The final target chunk completed under the same settings.

| Metric | Value |
|---|---:|
| selected local-copy specialization | 0.3274 |
| baseline loss | 10.2249 |
| own top loss delta | 1.8916 |
| random loss delta | -0.0314 |
| own top excess over random | 1.9231 |
| same-index source transfer | 0.5826 |
| aligned source transfer | 1.2933 |
| aligned - same | 0.7107 |
| aligned better count | 12/24 |

Per-target summaries:

| Target seed | Selected head | Own top excess | Same-index transfer | Aligned transfer | Aligned - same |
|---:|---|---:|---:|---:|---:|
| 7 | L3H5 | 3.1516 | 0.9932 | 2.3682 | 1.3749 |
| 8 | L3H10 | 0.0570 | 0.1791 | -0.0641 | -0.2433 |
| 9 | L3H5 | 2.5606 | 0.5754 | 1.5758 | 1.0004 |

### Pythia-160M all-source / all-target combined result

Merging target chunks 1-3, 4-6, and 7-9 gives the complete all-target result:

- source seeds: 1-9;
- target seeds: 1-9;
- checkpoint: `step143000`;
- layer: 3;
- transfer comparisons: 72 ordered source-target pairs.

| Metric | Value |
|---|---:|
| selected local-copy specialization | 0.3262 |
| baseline loss | 10.4720 |
| own top loss delta | 1.6393 |
| random loss delta | 0.0321 |
| own top excess over random | 1.6072 |
| same-index source transfer | 0.3142 |
| aligned source transfer | 1.0137 |
| aligned - same | 0.6995 |
| aligned better count | 40/72 |

Target-level heterogeneity is large:

| Target seed group | Own top excess | Aligned - same | Aligned better |
|---|---:|---:|---:|
| 1-3 | 2.8567 | 1.3993 | 17/24 |
| 4-6 | 0.0420 | -0.0116 | 11/24 |
| 7-9 | 1.9231 | 0.7107 | 12/24 |

Interpretation:

- Local-copy is a second positive all-target alignment-transfer result in mean
  effect size: aligned source heads produce about `3.2x` the same-index loss
  delta (`1.0137` vs `0.3142`).
- It is weaker and less uniform than repeat-match. The pair-level aligned-better
  count is only `40/72`, and the target-level sign count is `7/9`.
- The target-level correlation between own-head causal excess and
  aligned-minus-same transfer is `0.9664`, suggesting that alignment transfers
  the role primarily when the target seed actually uses the selected local-copy
  head causally. The target-level sign count is only `7/9` (`p=0.1797`,
  two-sided sign test), so the mean effect and the causal-strength correlation
  are more informative than a simple positive-target count.
- The right conclusion is not "local-copy always transfers"; it is "when the
  probed local-copy head is causally active in the target, raw-score alignment
  identifies source heads that transfer substantially better than same index."

## Infrastructure Blocker

Attempts to scale this exact local-copy experiment to all 9 Pythia-160M target
seeds were interrupted under current machine/GPU contention. The script was
updated to support:

- partial output writes after probing and after each target seed;
- `--target-seeds` chunking.

With chunking, all target chunks completed. Earlier failed attempts were local
infrastructure interruptions, not model results.

The completed chunks can be merged incrementally with
`scripts/analyze_local_copy_chunks.py`. At the current checkpoint, the combined
directory contains all target chunks.

## Files

- Script: `scripts/pythia_local_copy_alignment.py`.
- 14M smoke results: `results/debug_pythia14m_local_copy_alignment/`.
- 160M two-seed pilot results: `results/debug_pythia160m_local_copy_alignment/`.
- 160M all-source target 1-3 results:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets1_3/`.
- 160M all-source target 4-6 results:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets4_6/`.
- 160M all-source target 7-9 results:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets7_9/`.
- Incremental combined results:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_combined/`.
  This includes `target_diagnostic_summary.csv` for the target-level
  heterogeneity check.
- Chunk combiner: `scripts/analyze_local_copy_chunks.py`.
- Sleep-run log: `doc/autonomous_sleep_log_2026-05-22.md`.

## Current Interpretation

The repeat-match result now has a second-role analogue, but local-copy is more
conditional and less uniform.

Current evidence separates three cases:

1. **Repeat-match**: strong own causality and strong alignment-based causal
   transfer across all 9 seeds.
2. **Previous-token attention probe**: all-seed role distributions are somewhat
   improved by alignment, but the role is distributed and less concentrated.
3. **Local-copy causal result**: all-target mean transfer favors aligned source
   heads over same-index source heads, but the effect is concentrated in target
   seeds where the selected local-copy head is causally important.

This supports a more precise version of the alignment-transfer claim: structural
alignment can reveal functionally reusable heads, but a probe-only specialization
score is not enough to guarantee that the target seed uses that role causally.

## Layer-Selection Follow-Up

A follow-up layer sweep showed that the weak layer-3 targets do have causal
local-copy heads, just in different layers:

- seeds 1, 2, 3, 7, and 9: best causal local-copy head is in layer 3;
- seed 4: best causal local-copy head is in layer 2;
- seeds 5, 6, and 8: best causal local-copy head is in layer 4.

The fixed layers 2+4 rule rescues own-head causality for the weak targets
(`1.8528` own excess over random), but it is worse than layer 3 on the full
all-target transfer comparison (`aligned-minus-same=0.2441` vs `0.6995` for
layer 3). A cross-layer candidate-pool follow-up then fixed the structural-slot
problem: selecting the top 2 local-copy heads across layers 2-4 and matching
over the full cross-layer candidate pool produced `aligned-minus-same=1.7838`,
with aligned transfer better in `66/72` ordered pairs. Details are in
`doc/phase1_pythia160m_local_copy_layer_selection.md` and
`doc/phase1_pythia160m_local_copy_candidate_pool.md`.

## Recommended Next Step

Turn the target heterogeneity into a candidate-pool test:

```text
Can a cross-layer candidate-pool method identify causally active local-copy
heads before testing cross-seed transfer?
```

The first candidate-pool result is positive. The next step is to replicate the
same cross-layer candidate-pool test at earlier checkpoints and/or on another
role so we can distinguish a final-checkpoint phenomenon from a training-time
developmental pattern.
