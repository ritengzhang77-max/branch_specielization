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

## Infrastructure Blocker

Attempts to scale this exact local-copy experiment to all 9 Pythia-160M seeds
were interrupted under current machine/GPU contention before target-seed rows
could be completed. The script was updated to support:

- partial output writes after probing and after each target seed;
- `--target-seeds` chunking.

Even with chunking, the available GPUs and CPU runs were killed before a stable
all-seed local-copy result could be produced during this block. This is a local
infrastructure issue, not a model result.

## Files

- Script: `scripts/pythia_local_copy_alignment.py`.
- 14M smoke results: `results/debug_pythia14m_local_copy_alignment/`.
- 160M two-seed pilot results: `results/debug_pythia160m_local_copy_alignment/`.
- Sleep-run log: `doc/autonomous_sleep_log_2026-05-22.md`.

## Current Interpretation

The repeat-match result should not yet be generalized to all head roles.

Current evidence separates three cases:

1. **Repeat-match**: strong own causality and strong alignment-based causal
   transfer across all 9 seeds.
2. **Previous-token attention probe**: all-seed role distributions are somewhat
   improved by alignment, but the role is distributed and less concentrated.
3. **Local-copy causal pilot**: own selected heads can be strongly causal, but
   cross-seed causal transfer is weak in the two-seed pilot.

This is a promising boundary condition: alignment-based causal universality may
hold for some roles but not automatically for every attention pattern.

## Recommended Next Step

Run a more robust second-role test when a GPU is free:

```text
Pythia-160M, seeds 1-9, final checkpoint, local-copy layer 3,
target seeds chunked as 1-3, 4-6, 7-9.
```

Use the new `--target-seeds` flag and keep partial outputs. If local-copy remains
weak under aligned transfer, write it up as a contrast result rather than trying
to force the repeat-match story onto a different role.
