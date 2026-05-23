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

## Infrastructure Blocker

Attempts to scale this exact local-copy experiment to all 9 Pythia-160M target
seeds were interrupted under current machine/GPU contention. The script was
updated to support:

- partial output writes after probing and after each target seed;
- `--target-seeds` chunking.

With chunking, target seeds 1-3 completed. A target seeds 4-6 retry was
interrupted during probing before target rows were produced. This is a local
infrastructure issue, not a model result for target seeds 4-9.

The completed chunks can be merged incrementally with
`scripts/analyze_local_copy_chunks.py`. At the current checkpoint, the combined
directory contains only the target seeds 1-3 chunk and therefore reproduces the
same partial result.

## Files

- Script: `scripts/pythia_local_copy_alignment.py`.
- 14M smoke results: `results/debug_pythia14m_local_copy_alignment/`.
- 160M two-seed pilot results: `results/debug_pythia160m_local_copy_alignment/`.
- 160M all-source target 1-3 results:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets1_3/`.
- Incremental combined results:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_combined/`.
- Chunk combiner: `scripts/analyze_local_copy_chunks.py`.
- Sleep-run log: `doc/autonomous_sleep_log_2026-05-22.md`.

## Current Interpretation

The repeat-match result now has a stronger second-role analogue, but the
local-copy result is still partial.

Current evidence separates three cases:

1. **Repeat-match**: strong own causality and strong alignment-based causal
   transfer across all 9 seeds.
2. **Previous-token attention probe**: all-seed role distributions are somewhat
   improved by alignment, but the role is distributed and less concentrated.
3. **Local-copy causal pilot**: own selected heads are strongly causal, and
   aligned source-head transfer is strong for target seeds 1-3 when using all
   source seeds. The all-target result remains incomplete.

This is a promising extension of the alignment-transfer claim, but it needs the
remaining target seeds before being treated as an all-seed result.

## Recommended Next Step

Complete the remaining target chunks when a GPU is stable:

```text
Pythia-160M, source seeds 1-9, final checkpoint, local-copy layer 3,
target seeds 4-6 and 7-9.
```

Use the new `--target-seeds` flag and keep partial outputs. If the remaining
target chunks look like target seeds 1-3, local-copy becomes a second positive
causal alignment-transfer role.
