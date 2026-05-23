# Autonomous Sleep Checkpoint Summary

Date: 2026-05-22

## Main Conclusion So Far

The strongest current framing is:

```text
Structural/role specialization can lead to functional specialization, but the
stable unit is not necessarily a fixed raw layer/head slot. For local-copy, the
stable unit is recovered by cross-layer role-level relabeling.
```

This directly addresses the project's central question:

```text
Does structural modularity/specialization lead to functional
modularity/specialization?
```

Current evidence says yes for repeat-match and local-copy in Pythia-160M, with
an important qualifier: the local-copy role can move across nearby layers across
seeds, so fixed-slot comparisons are too brittle.

## Key Results

### Repeat-match, Pythia-160M

All 9 official seeds, selected checkpoints, 72 ordered source-target pairs:

| Checkpoint | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|
| step4000 | -0.0049 | 0.1682 | 0.1731 | 62/72 |
| step16000 | 0.0785 | 0.3958 | 0.3174 | 64/72 |
| step143000 | 0.2541 | 1.0619 | 0.8078 | 59/72 |

Interpretation: repeat-match functional roles are much more stable after
raw-score alignment than by same head index, and the causal transfer effect grows
through training.

### Local-copy, fixed layer 3, Pythia-160M

All 9 seeds, final checkpoint:

| Metric | Value |
|---|---:|
| own top excess over random | 1.6072 |
| same-index transfer | 0.3142 |
| aligned transfer | 1.0137 |
| aligned - same | 0.6995 |
| aligned better | 40/72 |

Interpretation: positive on average but heterogeneous. It failed mainly where
layer 3 was not the causal local-copy layer.

### Local-copy layer-selection follow-up

Best causal local-copy layer by seed:

| Best layer | Seeds |
|---|---|
| layer 2 | 4 |
| layer 3 | 1, 2, 3, 7, 9 |
| layer 4 | 5, 6, 8 |

Interpretation: all seeds have causal local-copy heads, but the role is not
locked to the same layer.

### Local-copy cross-layer candidate pool, Pythia-160M

Candidate layers 2-4, top 2 local-copy probe heads per seed, Hungarian matching
over the full 36-head cross-layer candidate pool:

| Metric | Value |
|---|---:|
| own top excess over random | 2.2896 |
| same-index transfer | 0.4876 |
| cross-layer aligned transfer | 2.2714 |
| aligned - same | 1.7838 |
| aligned better | 66/72 |

Target-level aligned-minus-same is positive for 9/9 target seeds. Bootstrap CI
over target means: `[1.3341, 2.3715]`.

Interpretation: this is the strongest local-copy result. The earlier layer-3
weakness was a fixed-slot artifact.

### Local-copy candidate-pool trajectory, Pythia-160M

| Checkpoint | Own top excess | Aligned - same | Aligned better |
|---|---:|---:|---:|
| step0 | -0.0004 | -0.0004 | 34/72 |
| step4000 | 0.4339 | 0.4191 | 66/72 |
| step16000 | 1.9006 | 1.2037 | 66/72 |
| step143000 | 2.2896 | 1.7838 | 66/72 |

Interpretation: the effect is absent at initialization, appears by step4000, and
grows in causal magnitude through training.

### Model-size checks

| Model | Candidate pool | Own top excess | Aligned - same | Target CI |
|---|---|---:|---:|---:|
| Pythia-70M | all layers 0-5 | 0.2692 | 0.0810 | [-0.1332, 0.2989] |
| Pythia-160M | layers 2-4 | 2.2896 | 1.7838 | [1.3341, 2.3715] |
| Pythia-410M | layers 2-6 | 4.1723 | 1.6554 | [1.0261, 2.2362] |

Interpretation: 70M does not robustly implement the synthetic local-copy role;
160M and 410M do. This looks like a capacity/task threshold, not a failure of
the alignment method.

410M trajectory nuance:

| Checkpoint | Own top excess | Aligned - same | Aligned better |
|---|---:|---:|---:|
| step0 | -0.0009 | -0.0007 | 23/72 |
| step4000 | 1.3363 | 1.2062 | 72/72 |
| step16000 | 4.1083 | 3.4057 | 71/72 |
| step143000 | 4.1723 | 1.6554 | 49/72 |

The 410M result is absent at initialization and strong but nonmonotonic under
the current layers 2-6 candidate window; the best transfer point is `step16000`,
not final.

## Files Added In This Sleep Block

- `scripts/pythia_local_copy_alignment.py`
- `scripts/analyze_local_copy_chunks.py`
- `scripts/pythia_local_copy_layer_causal_sweep.py`
- `scripts/analyze_local_copy_layer_sweeps.py`
- `scripts/pythia_local_copy_candidate_pool_alignment.py`
- `scripts/analyze_candidate_pool_trajectory.py`
- `scripts/analyze_transfer_significance.py`
- `doc/phase1_pythia160m_local_copy_pilot.md`
- `doc/phase1_pythia160m_local_copy_layer_selection.md`
- `doc/phase1_pythia160m_local_copy_candidate_pool.md`

## Best Next Step

The next useful experiment is a naturalistic local-copy/induction probe:

```text
Does the cross-layer candidate-pool result persist when the repeated-token
behavior is measured on natural text or more induction-like synthetic sequences,
not only arbitrary [x, SEP, x] triples?
```

This is the right next step because it tests whether the current result is a
general attention-head role or an artifact of the synthetic local-copy task.
