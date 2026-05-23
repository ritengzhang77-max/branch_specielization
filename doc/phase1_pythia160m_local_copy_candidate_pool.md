# Phase 1: Pythia-160M Local-Copy Candidate-Pool Alignment

Date: 2026-05-22

## Question

The layer-selection follow-up showed that causal local-copy heads exist in all
9 Pythia-160M seeds, but their best layer varies:

- layer 3: seeds 1, 2, 3, 7, 9;
- layer 2: seed 4;
- layer 4: seeds 5, 6, 8.

This experiment asks whether cross-seed functional transfer improves if we stop
forcing the role to stay in the same layer and instead align a cross-layer
candidate pool.

```text
Does cross-layer role alignment recover local-copy functional transfer better
than fixed layer-3 or fixed layers 2+4 alignment?
```

## Method

Candidate pool:

- model family: official Pythia-160M seeds 1-9;
- checkpoint: `step143000`;
- candidate layers: 2, 3, 4;
- candidates per seed: top 2 heads by local-copy probe score across all
  candidate layers;
- alignment representation: raw attention scores on the shared Phase 0 probe
  corpus;
- alignment method: Hungarian matching over the full 36-head candidate pool
  (`3 layers x 12 heads`), allowing source heads to map across layers;
- transfer baseline: same raw `(layer, head)` index.

Selected target heads:

| Seed | Selected heads |
|---:|---|
| 1 | L3H2, L3H10 |
| 2 | L3H4, L2H9 |
| 3 | L3H5, L3H0 |
| 4 | L2H10, L4H0 |
| 5 | L2H1, L4H6 |
| 6 | L4H4, L4H9 |
| 7 | L3H5, L2H4 |
| 8 | L4H6, L4H11 |
| 9 | L3H5, L2H0 |

## Result

All-target result over 72 ordered source-target pairs:

| Metric | Value |
|---|---:|
| own top excess over random | 2.2896 |
| same-index source transfer | 0.4876 |
| cross-layer aligned source transfer | 2.2714 |
| aligned - same | 1.7838 |
| aligned better count | 66/72 |

Per-target summary:

| Target seed | Own top excess | Same-index transfer | Aligned transfer | Aligned - same |
|---:|---:|---:|---:|---:|
| 1 | 1.4674 | 0.4682 | 1.7710 | 1.3028 |
| 2 | 3.8741 | 0.0916 | 3.7845 | 3.6930 |
| 3 | 2.5936 | 0.9861 | 2.6648 | 1.6787 |
| 4 | 1.7296 | 0.3700 | 1.1686 | 0.7986 |
| 5 | 2.1376 | 0.1807 | 1.2107 | 1.0300 |
| 6 | 2.1767 | 0.2177 | 2.1483 | 1.9306 |
| 7 | 2.3115 | 0.9451 | 3.2375 | 2.2924 |
| 8 | 1.8853 | 0.4557 | 1.9961 | 1.5404 |
| 9 | 2.4302 | 0.6733 | 2.4609 | 1.7877 |

The target-level aligned-minus-same effect is positive for `9/9` target seeds
(two-sided sign test `p=0.0039`). The pair-level aligned-better count is `66/72`
(two-sided sign test about `7.3e-14`).

## Comparison

| Selection/alignment rule | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|---:|
| Fixed layer 3 | 1.6072 | 0.3142 | 1.0137 | 0.6995 | 40/72 |
| Fixed layers 2+4 | 0.8974 | 0.3710 | 0.6151 | 0.2441 | 40/72 |
| Cross-layer pool 2-4, top 2 | 2.2896 | 0.4876 | 2.2714 | 1.7838 | 66/72 |

The candidate-pool result is much stronger than either fixed-slot rule. It
raises own-head causal importance and makes cross-seed aligned transfer
consistent across nearly all source-target pairs.

## Training Trajectory

The same candidate-pool experiment was run at selected Pythia-160M checkpoints:

| Checkpoint | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|---:|
| step0 | -0.0004 | 0.0001 | -0.0003 | -0.0004 | 34/72 |
| step4000 | 0.4339 | 0.0631 | 0.4822 | 0.4191 | 66/72 |
| step16000 | 1.9006 | 0.3281 | 1.5318 | 1.2037 | 66/72 |
| step143000 | 2.2896 | 0.4876 | 2.2714 | 1.7838 | 66/72 |

Interpretation:

- The effect is absent at initialization.
- Cross-layer aligned causal transfer is already detectable by `step4000`.
- Magnitude grows substantially by `step16000` and again by the final checkpoint.
- The pairwise aligned-better count saturates early (`66/72` from `step4000`
  onward), while the causal effect size keeps increasing.

## Model-Size Checks

I also ran final-checkpoint Pythia-70M and Pythia-410M checks with the same
candidate-pool framework.

| Model / candidate pool | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|---:|
| 70M layers 1-3, top 2 | 0.0508 | 0.1463 | 0.1115 | -0.0348 | 35/72 |
| 70M all layers 0-5, top 2 | 0.2692 | 0.1043 | 0.1854 | 0.0810 | 41/72 |
| 160M layers 2-4, top 2 | 2.2896 | 0.4876 | 2.2714 | 1.7838 | 66/72 |
| 410M layers 2-6, top 2 | 4.1723 | 0.2562 | 1.9116 | 1.6554 | 49/72 |

The 70M result is weak even when all layers are allowed. This should be treated
as a capacity/model-size caveat rather than a failure of alignment: the selected
70M heads are only weakly causal for the synthetic local-copy task, so there is
little functional module to transfer.

The 410M result restores a strong positive effect: aligned-minus-same is
`1.6554`, target-level aligned-minus-same is positive for `9/9` target seeds,
and the pair-level aligned-better count is `49/72` (`p ~= 0.0029`, two-sided
sign test). Pair-level consistency is lower than 160M, but the mean functional
transfer effect remains large.

The Pythia-410M selected-checkpoint trajectory is not monotonic:

| Checkpoint | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|---:|
| step4000 | 1.3363 | 0.0624 | 1.2686 | 1.2062 | 72/72 |
| step16000 | 4.1083 | 0.1679 | 3.5737 | 3.4057 | 71/72 |
| step143000 | 4.1723 | 0.2562 | 1.9116 | 1.6554 | 49/72 |

The local-copy role is already strong by `step4000`, peaks by this metric at
`step16000`, and remains positive but less clean by the final checkpoint. This
may indicate later redistribution of the synthetic local-copy behavior or a
candidate-window mismatch at final.

Significance summaries:

| Model | Pair mean CI | Pair sign p | Target mean CI | Target sign p |
|---|---:|---:|---:|---:|
| 70M all layers | [-0.1165, 0.2619] | 0.2888 | [-0.1332, 0.2989] | 0.1797 |
| 160M layers 2-4 | [1.5049, 2.0556] | 7.3e-14 | [1.3341, 2.3715] | 0.0039 |
| 410M layers 2-6 | [1.0527, 2.2303] | 0.0029 | [1.0261, 2.2362] | 0.0039 |

These bootstrap intervals are over ordered source-target pairs for the pair
column and over target-seed means for the target column.

## Interpretation

This is the clearest current local-copy result:

```text
Functional specialization is stable across seeds after role-level relabeling,
but the structural slot carrying the role can move across nearby layers.
```

So the right framing is not "does the same layer/head become functionally
modular?" It is:

```text
Does structural/role specialization, after the correct cross-seed relabeling,
lead to functional specialization?
```

For local-copy, the answer is currently yes under a cross-layer candidate pool.
The weaker layer-3 result was not evidence that the role failed to transfer; it
was evidence that fixed structural slots are too brittle.

## Files

- Script: `scripts/pythia_local_copy_candidate_pool_alignment.py`.
- Significance script: `scripts/analyze_transfer_significance.py`.
- Result directory:
  `results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2/`.
- Trajectory combiner: `scripts/analyze_candidate_pool_trajectory.py`.
- Trajectory summary:
  `results/phase1_pythia160m_local_copy_candidate_pool_trajectory/`.
- Pythia-70M layers 1-3 check:
  `results/phase1_pythia70m_local_copy_candidate_pool_layers1_3_top2/`.
- Pythia-70M all-layer check:
  `results/phase1_pythia70m_local_copy_candidate_pool_all_layers_top2/`.
- Pythia-410M layers 2-6 check:
  `results/phase1_pythia410m_local_copy_candidate_pool_layers2_6_top2/`.
- Pythia-410M trajectory:
  `results/phase1_pythia410m_local_copy_candidate_pool_trajectory/`.
- Prior layer-selection memo:
  `doc/phase1_pythia160m_local_copy_layer_selection.md`.
