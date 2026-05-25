# Phase 3: Toy Ontology v3 Robustness Sweep

Date: 2026-05-24

## Question

This is the follow-up to the Toy Ontology v3 algorithmic-family experiment.

The unit is still:

```text
ordinary attention head
```

The intervention is still:

```text
matched-capacity uniform head dimensions
vs
matched-capacity non-uniform head dimensions
```

The three questions are kept separate:

1. Does the model still solve the task?
2. Do roles become more concentrated into fewer heads?
3. Does the head-usage geometry become more modular under the predeclared
   algorithmic ontology or under label-free clustering?

## V3 Setting

Toy Ontology v3 has 20 roles:

```text
5 families x 4 variants per family
```

Families:

| Family | Roles |
|---|---|
| `local_offset` | `v3_offset_prev1`, `v3_offset_prev2`, `v3_offset_prev3`, `v3_offset_prev4` |
| `key_value_lookup` | `v3_kv_lookup_2pair`, `v3_kv_lookup_4pair`, `v3_wrong_key_control`, `v3_recency_key_conflict` |
| `sequence_induction` | `v3_induction_len4`, `v3_induction_len8`, `v3_induction_len12`, `v3_induction_len16` |
| `boundary_anchor` | `v3_bos_anchor`, `v3_sep_anchor`, `v3_punctuation_anchor`, `v3_newline_anchor` |
| `conflict_suppression` | `v3_distractor_suppression`, `v3_anti_copy`, `v3_false_induction`, `v3_negative_name_control` |

The important design choice is that a family is not a broad literature bucket.
Each family is a repeated variant of one algorithmic primitive.

## Artifacts

Scripts:

```text
scripts/toy_role_ontology_v2_head_dim_intervention.py
scripts/analyze_role_ontology_v2.py
scripts/analyze_ontology_refinement.py
```

Result roots:

```text
results/phase3_toy_role_ontology_v3_main_2000_20260524
results/phase3_toy_role_ontology_v3_main_10seed_2000_20260524
results/phase3_toy_role_ontology_v3_layout_moderate_2000_20260524
```

## 10-Seed Main Replication

This combines the original 5 seeds with seeds 6 through 10 for the original
three configs:

| Config | Mean Acc | Min Acc | Specialization | Effective Heads | Ontology Align | Shuffle p | Sep-Adjusted Clusterability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform8` | 0.9999 | 0.9988 | 0.630 | 3.958 | 0.184 | 0.027 | 0.415 |
| `hetero8_unique_spread` | 1.0000 | 0.9995 | 0.670 | 3.249 | 0.182 | 0.035 | 0.464 |
| `hetero8_unique_extreme` | 0.9999 | 0.9993 | 0.784 | 2.080 | 0.218 | 0.026 | 0.445 |

Seed-level comparison against `uniform8`:

| Metric | `hetero8_unique_spread` | `hetero8_unique_extreme` |
|---|---:|---:|
| Specialization mean diff | +0.041, wins 9/10 | +0.154, wins 8/10 |
| Effective heads mean diff | -0.708, wins 9/10 | -1.878, wins 10/10 |
| Ontology alignment mean diff | -0.002, wins 5/10 | +0.034, wins 7/10 |
| Shuffled-label p mean diff | +0.008, lower p in 5/10 | -0.001, lower p in 6/10 |
| Sep-adjusted clusterability mean diff | +0.049, wins 9/10 | +0.030, wins 6/10 |

Interpretation:

- Accuracy is matched across all configs.
- Specialization is robust, especially in `hetero8_unique_extreme`.
- Functional modularity is positive but not uniform:
  - `hetero8_unique_extreme` is better on ontology alignment;
  - `hetero8_unique_spread` is better on separation-adjusted clusterability.

## Layout And Moderate Heterogeneity Sweep

This run asks whether the v3 result survives:

```text
moderate heterogeneity between spread and extreme
moving the 152-dim head to different indices
```

All configs use:

```text
8 heads per layer
2 layers
16 total head slots
matched total attention dimension = 384
5 seeds
2000 training steps
```

Main table, compared against the same 5-seed `uniform8` baseline:

| Config | Mean Acc | Min Acc | Specialization | Delta Spec / Wins | Eff Heads | Delta Eff / Wins | Ont Align | Delta Align / Wins | Shuffle p | Sep-Adjusted | Delta Sep / Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform8` | 0.9999 | 0.9983 | 0.660 | +0.000 / - | 3.586 | +0.000 / - | 0.139 | +0.000 / - | 0.042 | 0.417 | +0.000 / - |
| `hetero8_unique_spread` | 1.0000 | 0.9995 | 0.670 | +0.010 / 4/5 | 3.174 | -0.412 / 4/5 | 0.142 | +0.003 / 2/5 | 0.051 | 0.450 | +0.033 / 5/5 |
| `hetero8_unique_extreme` | 0.9999 | 0.9986 | 0.783 | +0.122 / 3/5 | 2.073 | -1.513 / 5/5 | 0.193 | +0.055 / 4/5 | 0.022 | 0.437 | +0.020 / 2/5 |
| `hetero8_unique_mid_104` | 0.9999 | 0.9992 | 0.699 | +0.039 / 4/5 | 2.880 | -0.707 / 4/5 | 0.188 | +0.049 / 3/5 | 0.036 | 0.424 | +0.007 / 2/5 |
| `hetero8_unique_mid_120` | 0.9998 | 0.9979 | 0.677 | +0.017 / 2/5 | 2.938 | -0.649 / 4/5 | 0.225 | +0.086 / 4/5 | 0.006 | 0.405 | -0.012 / 1/5 |
| `hetero8_extreme_152_first` | 0.9999 | 0.9984 | 0.789 | +0.129 / 4/5 | 2.038 | -1.549 / 5/5 | 0.181 | +0.042 / 2/5 | 0.046 | 0.418 | +0.001 / 3/5 |
| `hetero8_extreme_152_second` | 1.0000 | 0.9995 | 0.785 | +0.124 / 5/5 | 2.115 | -1.472 / 5/5 | 0.155 | +0.016 / 3/5 | 0.034 | 0.433 | +0.016 / 2/5 |
| `hetero8_extreme_152_middle` | 0.9999 | 0.9988 | 0.796 | +0.135 / 4/5 | 2.041 | -1.546 / 4/5 | 0.142 | +0.003 / 3/5 | 0.108 | 0.413 | -0.004 / 3/5 |
| `hetero8_extreme_152_seventh` | 0.9999 | 0.9984 | 0.817 | +0.156 / 5/5 | 1.994 | -1.593 / 5/5 | 0.160 | +0.022 / 3/5 | 0.026 | 0.428 | +0.011 / 4/5 |

Interpretation:

- Accuracy remains solved for all new layouts.
- Specialization is the cleanest result:
  - every new hetero config has higher mean specialization than `uniform8`;
  - every new hetero config has lower effective-head count than `uniform8`.
- Ontology alignment is positive on average for every new hetero config, but
  the seed wins are not uniformly strong.
- Separation-adjusted clusterability is mixed:
  - original `hetero8_unique_spread` remains the cleanest result on this metric;
  - layout-permuted extreme configs are only slightly above baseline or tied;
  - `hetero8_unique_mid_120` improves ontology alignment but loses slightly on
    separation-adjusted clusterability.

## Structural Affinity Diagnostic

Largest-head rate is not a headline metric, but the layout sweep is still useful
as a diagnostic for structural role affinity.

For three roles, the top head index moved exactly with the 152-dim head:

| Config | 152 Index | Role | Top Head Index Counts |
|---|---:|---|---|
| `hetero8_extreme_152_first` | 0 | `v3_offset_prev1` | 0:5 |
| `hetero8_extreme_152_first` | 0 | `v3_kv_lookup_2pair` | 0:5 |
| `hetero8_extreme_152_first` | 0 | `v3_kv_lookup_4pair` | 0:5 |
| `hetero8_extreme_152_second` | 1 | `v3_offset_prev1` | 1:5 |
| `hetero8_extreme_152_second` | 1 | `v3_kv_lookup_2pair` | 1:5 |
| `hetero8_extreme_152_second` | 1 | `v3_kv_lookup_4pair` | 1:5 |
| `hetero8_extreme_152_middle` | 3 | `v3_offset_prev1` | 3:5 |
| `hetero8_extreme_152_middle` | 3 | `v3_kv_lookup_2pair` | 3:5 |
| `hetero8_extreme_152_middle` | 3 | `v3_kv_lookup_4pair` | 3:5 |
| `hetero8_extreme_152_seventh` | 6 | `v3_offset_prev1` | 6:5 |
| `hetero8_extreme_152_seventh` | 6 | `v3_kv_lookup_2pair` | 6:5 |
| `hetero8_extreme_152_seventh` | 6 | `v3_kv_lookup_4pair` | 6:5 |

This supports a structural-type story for those roles: they are not just
choosing a fixed head index.

However, not every role chooses the largest head. In the extreme layout
permutations:

| Role | Family | 152 Top Count / 20 | Mode Dim |
|---|---|---:|---:|
| `v3_kv_lookup_2pair` | `key_value_lookup` | 20/20 | 152 |
| `v3_kv_lookup_4pair` | `key_value_lookup` | 20/20 | 152 |
| `v3_offset_prev1` | `local_offset` | 20/20 | 152 |
| `v3_false_induction` | `conflict_suppression` | 16/20 | 152 |
| `v3_newline_anchor` | `boundary_anchor` | 15/20 | 152 |
| `v3_induction_len8` | `sequence_induction` | 2/20 | 64 |
| `v3_induction_len12` | `sequence_induction` | 2/20 | 64 |
| `v3_induction_len16` | `sequence_induction` | 3/20 | 64 |

This is useful because the diagnostic is not simply "everything picks the
largest head." Some role families, especially induction variants, often prefer
the 64-dim head instead.

## Answer By Question

### 1. Does structural heterogeneity create stable role affinity?

Current answer:

```text
Yes for several roles, strongly in the extreme-layout diagnostic.
```

The clearest examples are `v3_offset_prev1`, `v3_kv_lookup_2pair`, and
`v3_kv_lookup_4pair`: when the 152-dim head moves from index 0 to 1 to 3 to 6,
the top head for those roles moves with it in 20/20 cases per role.

This is a diagnostic, not the main modularity metric.

### 2. Does heterogeneity increase role-level specialization?

Current answer:

```text
Yes, robustly.
```

Evidence:

- In the 10-seed main run, `hetero8_unique_extreme` increases specialization
  from `0.630` to `0.784` and reduces effective heads from `3.958` to `2.080`.
- In the layout/moderate sweep, every new hetero config has higher mean
  specialization and lower mean effective-head count than `uniform8`.

This is currently the strongest result.

### 3. Does heterogeneity improve functional modularity?

Current answer:

```text
Promising, but not fully robust yet.
```

Evidence for:

- In the 10-seed main run, `hetero8_unique_extreme` improves ontology alignment
  from `0.184` to `0.218`.
- In the layout/moderate sweep, every new hetero config improves mean ontology
  alignment over the 5-seed `uniform8` baseline.
- `hetero8_unique_mid_120` gives the strongest ontology-alignment value so far:
  `0.225`, with shuffled-label p `0.006`.

Evidence against a final strong claim:

- Separation-adjusted clusterability is not consistently improved across the
  layout/moderate sweep.
- Some layout configs only barely beat the baseline on ontology alignment.
- Some family-level behavior remains uneven.

So the correct wording is:

```text
Non-uniform head dimensions produce robust specialization and give positive
evidence for functional modularity under the v3 algorithmic ontology, but the
modularity claim is still layout- and metric-sensitive.
```

## Current Best Claim

The strongest defensible claim after this sweep is:

```text
In matched-capacity toy transformers where the unit is an ordinary attention
head, non-uniform head dimensions preserve task performance and robustly
increase role-level functional specialization. On a predeclared algorithmic
role ontology, they also produce positive evidence for functional modularity,
especially in ontology alignment, but the modularity effect is not yet as robust
as the specialization effect.
```

## Follow-Up

The next toy experiment was a larger-head v3 control:

```text
same v3 ontology
more ordinary attention heads
matched total attention dimension
uniform baseline vs non-uniform configs
same metrics as this memo
```

Reason:

```text
If modularity is sparse, the current 16 head slots may be too few to see clean
family-level clusters. A larger-head control tests whether the mixed modularity
result is an architectural limitation or a real weakness of the claim.
```

That follow-up is recorded in:

```text
doc/experiments/phase3/phase3_toy_role_ontology_v3_larger_head_control.md
```

The result did not rescue the modularity claim. After that larger-head control,
the next phase should return to real-model validation on ordinary pretrained
heads.
