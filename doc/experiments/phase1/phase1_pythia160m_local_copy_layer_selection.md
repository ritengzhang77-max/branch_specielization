# Phase 1: Pythia-160M Local-Copy Layer Selection Follow-Up

Date: 2026-05-22

## Question

The layer-3 local-copy all-target run was positive on average, but target seeds
4-6 and 8 had weak own-head causality. This follow-up asks:

```text
Did those seeds lack a causal local-copy head, or did the layer-3 probe select
the wrong structural slot?
```

## Layer Causal Sweep

For each seed, the script selected the top local-copy probe head in every layer,
then ablated each selected head against same-layer random-head controls.

| Seed | Best causal head | Best excess | Layer-3 head | Layer-3 excess |
|---:|---|---:|---|---:|
| 1 | L3H2 | 1.9411 | L3H2 | 1.9411 |
| 2 | L3H4 | 3.8196 | L3H4 | 3.8196 |
| 3 | L3H5 | 2.6416 | L3H5 | 2.6416 |
| 4 | L2H10 | 1.4051 | L3H9 | -0.3294 |
| 5 | L4H6 | 2.0489 | L3H11 | 0.1554 |
| 6 | L4H4 | 1.8105 | L3H10 | -0.0029 |
| 7 | L3H5 | 2.9073 | L3H5 | 2.9073 |
| 8 | L4H6 | 1.8033 | L3H10 | -0.1622 |
| 9 | L3H5 | 2.4525 | L3H5 | 2.4525 |

Summary:

- best layer counts: layer 2: `1`, layer 3: `5`, layer 4: `3`;
- layer 3 is genuinely optimal for seeds `1, 2, 3, 7, 9`;
- seeds `4, 5, 6, 8` do have causal local-copy heads, but not in layer 3;
- best-layer causal excess averages `2.3144`, versus `1.4914` for layer 3 in
  the sweep.

This explains the weak layer-3 targets: they are not missing the local-copy
function; the probe picked a structural slot whose attention behavior did not
match causal importance.

## Layers 2+4 Transfer Test

To test whether a simple fixed multi-layer selection rule repairs the weak
targets, I reran cross-seed transfer with one top local-copy head from layer 2
and one from layer 4 in each seed.

### Weak targets only: seeds 4, 5, 6, 8

| Metric | Value |
|---|---:|
| own top excess over random | 1.8528 |
| same-index source transfer | 0.4560 |
| aligned source transfer | 0.7782 |
| aligned - same | 0.3222 |
| aligned better count | 19/32 |

Per-target result:

| Target seed | Selected heads | Own top excess | Aligned - same |
|---:|---|---:|---:|
| 4 | L2H10, L4H0 | 1.7508 | 1.0321 |
| 5 | L2H1, L4H6 | 2.1201 | 0.1300 |
| 6 | L2H2, L4H4 | 1.6541 | -0.0070 |
| 8 | L2H4, L4H6 | 1.8861 | 0.1339 |

Layers 2+4 mostly fixes own-target causality in the previously weak seeds, but
cross-seed aligned transfer improves only modestly.

### All targets: layers 2+4 combined

| Metric | Layer 3 only | Layers 2+4 |
|---|---:|---:|
| selected specialization | 0.3262 | 0.2759 |
| own top excess over random | 1.6072 | 0.8974 |
| same-index source transfer | 0.3142 | 0.3710 |
| aligned source transfer | 1.0137 | 0.6151 |
| aligned - same | 0.6995 | 0.2441 |
| aligned better count | 40/72 | 40/72 |

Layers 2+4 is not a better global rule. It helps seeds 4, 5, 6, and 8, but it
damages the seeds where layer 3 was already the causal local-copy layer.

## Interpretation

This strengthens the project's central distinction:

```text
Probe-defined structural specialization can fail to imply functional
specialization at a fixed structural slot.
```

For local-copy, structural role location is seed-dependent. Five seeds put the
causal role in layer 3, one in layer 2, and three in layer 4. Raw-score alignment
still transfers function better than same index in the layer-3 all-target run,
but the result is conditional on selecting a causally active target slot.

The clean next question is not whether local-copy transfers at one fixed layer.
It is whether a candidate-pool method can identify causally active local-copy
heads across layers before testing cross-seed transfer.

## Candidate-Pool Update

That next test was positive. Selecting the top 2 local-copy heads across layers
2-4 and running Hungarian alignment over the full 36-head cross-layer candidate
pool produced:

- own top excess over random: `2.2896`;
- same-index source transfer: `0.4876`;
- cross-layer aligned source transfer: `2.2714`;
- aligned-minus-same: `1.7838`;
- aligned better count: `66/72`.

This is substantially stronger than either fixed layer 3 or fixed layers 2+4.
The detailed memo is
`doc/experiments/phase1/phase1_pythia160m_local_copy_candidate_pool.md`.

## Files

- Layer sweep script: `scripts/pythia_local_copy_layer_causal_sweep.py`.
- Layer sweep combiner: `scripts/analyze_local_copy_layer_sweeps.py`.
- Weak-target layer sweep:
  `results/phase1_pythia160m_local_copy_layer_sweep_weak_targets/`.
- Other-target layer sweep:
  `results/phase1_pythia160m_local_copy_layer_sweep_other_targets/`.
- Combined layer sweep:
  `results/phase1_pythia160m_local_copy_layer_sweep_combined/`.
- Layers 2+4 weak-target transfer:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layers2_4_weak_targets/`.
- Layers 2+4 other-target transfer:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layers2_4_other_targets/`.
- Layers 2+4 combined transfer:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layers2_4_combined/`.
