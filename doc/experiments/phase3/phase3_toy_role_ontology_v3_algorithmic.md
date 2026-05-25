# Phase 3: Toy Ontology v3 Algorithmic-Family Experiment

Date: 2026-05-24

## Why This Was Run

The previous Toy Ontology v2 modularity result was mixed, and the main weakness
was conceptual:

```text
The modularity metric depends on whether the ontology families are good
functional clusters.
```

Toy Ontology v3 fixes that by making every family a set of repeated variants of
one algorithmic primitive, rather than broad literature-inspired buckets.

## Quick Answer About The Earlier Task-Primitive Ontology

The task-primitive ontology result in v2 was **suggestive but weak**.

Clean 16-slot v2 result:

| Config | Task-Primitive Ontology Alignment | Gap vs `uniform8` | Shuffle p |
|---|---:|---:|---:|
| `uniform8` | 0.004 | 0.000 | 0.453 |
| `hetero8_unique_spread` | 0.034 | +0.030 | 0.495 |
| `hetero8_unique_extreme` | 0.063 | +0.059 | 0.221 |

So hetero beat uniform under that ontology, but the absolute effect was small
and not statistically convincing. It was a clue for designing v3, not a result
strong enough to claim.

## V3 Ontology

Each family has four variants of one primitive.

| Family | Roles |
|---|---|
| `local_offset` | `v3_offset_prev1`, `v3_offset_prev2`, `v3_offset_prev3`, `v3_offset_prev4` |
| `key_value_lookup` | `v3_kv_lookup_2pair`, `v3_kv_lookup_4pair`, `v3_wrong_key_control`, `v3_recency_key_conflict` |
| `sequence_induction` | `v3_induction_len4`, `v3_induction_len8`, `v3_induction_len12`, `v3_induction_len16` |
| `boundary_anchor` | `v3_bos_anchor`, `v3_sep_anchor`, `v3_punctuation_anchor`, `v3_newline_anchor` |
| `conflict_suppression` | `v3_distractor_suppression`, `v3_anti_copy`, `v3_false_induction`, `v3_negative_name_control` |

This gives:

```text
5 families x 4 roles = 20 roles
```

## Experiment Setting

```text
ordinary attention heads
8 heads/layer x 2 layers = 16 head slots
5 seeds
2000 training steps
matched total attention dimension = 384
```

Configs:

| Config | Head Dims |
|---|---|
| `uniform8` | `[48,48,48,48,48,48,48,48]` |
| `hetero8_unique_spread` | `[16,24,32,40,48,56,72,96]` |
| `hetero8_unique_extreme` | `[8,16,24,32,40,48,64,152]` |

Artifacts:

```text
scripts/toy_role_ontology_v2_head_dim_intervention.py
scripts/analyze_role_ontology_v2.py
scripts/analyze_ontology_refinement.py
results/phase3_toy_role_ontology_v3_main_2000_20260524
```

## Main Table

Accuracy is listed first because specialization or modularity is not meaningful
if a model fails the task.

| Config | Mean Acc | Min Acc | Specialization | Effective Heads | Ontology Align | Shuffle p | Sep-Adjusted Clusterability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform8` | 0.9999 | 0.9983 | 0.660 | 3.586 | 0.139 | 0.044 | 0.417 |
| `hetero8_unique_spread` | 1.0000 | 0.9995 | 0.670 | 3.174 | 0.142 | 0.059 | 0.450 |
| `hetero8_unique_extreme` | 0.9999 | 0.9986 | 0.783 | 2.073 | 0.193 | 0.016 | 0.437 |

Head-dimension top counts are still saved in raw analysis files as a structural
affinity diagnostic, but they are not a headline metric. The main project claim
is not that the largest head is always best; it is that head structure affects
function assignment, specialization, and possibly modularity.

## Seed-Level Comparisons Against `uniform8`

| Metric | `hetero8_unique_spread` | `hetero8_unique_extreme` |
|---|---:|---:|
| Ontology alignment mean diff | +0.003 | +0.055 |
| Ontology alignment wins | 2/5 seeds | 4/5 seeds |
| Separation-adjusted clusterability mean diff | +0.033 | +0.020 |
| Separation-adjusted clusterability wins | 5/5 seeds | 2/5 seeds |
| Specialization mean diff | +0.010 | +0.122 |
| Specialization wins | 4/5 seeds | 3/5 seeds |

Interpretation:

- Extreme hetero gives the clearest ontology-alignment improvement.
- Spread hetero is basically tied with uniform on ontology alignment.
- Spread hetero gives the cleanest separation-adjusted clusterability
  improvement.

## Label-Free Clusterability

This ignores the ontology labels and asks whether the role geometry itself is
clustered.

| Config | Mean Pair TV | Silhouette k=5 | Best Silhouette | Separation-Adjusted |
|---|---:|---:|---:|---:|
| `uniform8` | 0.662 | 0.649 | 0.760 | 0.417 |
| `hetero8_unique_spread` | 0.700 | 0.646 | 0.713 | 0.450 |
| `hetero8_unique_extreme` | 0.671 | 0.652 | 0.726 | 0.437 |

Interpretation:

- Raw silhouette is similar across configs.
- Separation-adjusted clusterability improves for both hetero configs.
- Spread hetero wins this metric in 5/5 seeds.
- This is a real improvement over v2, where hetero often looked clustered only
  because overall role separation shrank.

## Per-Family Result

| Config | Family | Acc | Specialization | Effective Heads | Within-Family Similarity |
|---|---|---:|---:|---:|---:|
| `uniform8` | `local_offset` | 1.000 | 0.507 | 4.895 | 0.190 |
| `uniform8` | `key_value_lookup` | 1.000 | 0.564 | 4.866 | 0.297 |
| `uniform8` | `sequence_induction` | 1.000 | 0.673 | 2.662 | 0.808 |
| `uniform8` | `boundary_anchor` | 1.000 | 0.826 | 2.117 | 0.501 |
| `uniform8` | `conflict_suppression` | 1.000 | 0.732 | 3.392 | 0.524 |
| `hetero8_unique_extreme` | `local_offset` | 1.000 | 0.802 | 2.112 | 0.268 |
| `hetero8_unique_extreme` | `key_value_lookup` | 1.000 | 0.790 | 2.034 | 0.323 |
| `hetero8_unique_extreme` | `sequence_induction` | 1.000 | 0.674 | 2.722 | 0.705 |
| `hetero8_unique_extreme` | `boundary_anchor` | 1.000 | 0.811 | 1.757 | 0.529 |
| `hetero8_unique_extreme` | `conflict_suppression` | 1.000 | 0.835 | 1.742 | 0.474 |

Family-level interpretation:

- `local_offset` and `key_value_lookup` improve on within-family similarity
  under extreme hetero.
- `boundary_anchor` is roughly preserved/slightly improved.
- `sequence_induction` and `conflict_suppression` lose within-family similarity,
  despite strong specialization.
- So the modularity gain is not uniform across all families.

## Conclusion

Toy Ontology v3 is the strongest modularity evidence so far.

Supported:

```text
Heterogeneous head dimensions preserve accuracy.
Extreme heterogeneity substantially increases role-level specialization.
Extreme heterogeneity improves ontology alignment on this better predeclared
algorithmic ontology.
Heterogeneous configs improve separation-adjusted label-free clusterability.
```

Not yet fully supported:

```text
All heterogeneous configs robustly beat uniform on every modularity metric.
Every functional family becomes cleaner under heterogeneity.
```

Best current claim:

```text
When the ontology is defined as repeated variants of algorithmic primitives,
non-uniform attention-head dimensions show the first meaningful positive
evidence for improved functional modularity, especially in the extreme
heterogeneous layout. The effect is promising but still needs replication with
more layouts and seeds.
```

## Robustness Follow-Up

The next decisive follow-up was a v3 robustness sweep:

```text
more seeds, e.g. 10
layout permutations of the extreme hetero vector
moderate hetero layouts between spread and extreme
possibly 32-slot v3 control
```

The key check is whether the ontology-alignment improvement follows the
structural head type rather than a lucky head index or one seed-specific
training path.

That follow-up is recorded in:

```text
doc/experiments/phase3/phase3_toy_role_ontology_v3_robustness.md
```
