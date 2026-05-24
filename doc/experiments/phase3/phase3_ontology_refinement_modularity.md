# Phase 3: Ontology Refinement For Functional Modularity

Date: 2026-05-24

## Question

The previous modularity result was mixed. The user pointed out the main reason
this can happen:

```text
Any modularity metric depends on whether the ontology families are a good
definition of functional clusters.
```

That is correct. A low ontology-alignment score can mean either:

1. the model is not modular;
2. the model is modular along another axis;
3. the hand-written ontology is too broad or wrong.

This pass tests whether a more defensible ontology view makes non-uniform
attention heads look more modular.

## Scope

- Unit remains ordinary attention heads.
- No new training in this pass.
- Inputs are the existing role-pair head-usage matrices.
- This is not allowed to relabel roles just to make hetero win.

New analysis script:

```text
scripts/analyze_ontology_refinement.py
```

Generated local outputs:

```text
results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523/analysis/ontology_refinement
results/phase3_toy_role_ontology_v2_large_heads_1000_20260523/analysis/ontology_refinement
results/phase3_toy_role_ontology_v2_full_1600_20260523/analysis/ontology_refinement
```

## Candidate Ontologies Tested

### 1. Original v2 Family Ontology

The existing 5-family ontology:

```text
copy_transport
induction
position_boundary
suppression_conflict
entity_coreference
```

This is the baseline ontology.

### 2. Task-Primitive 3-Way Ontology

A coarser ontology based on the immediate task primitive:

```text
direct_pointer:
  local_copy, previous_token, kv_lookup, duplicate_token,
  bos_sink, sep_sink, fixed_offset_prev, punctuation_boundary,
  repeated_name_detection, pronoun_antecedent

sequence_repeat:
  induction_short, induction_long, induction_ngram,
  false_induction_control, anti_copy

conflict_choice:
  distractor_suppression, wrong_key_suppression, recency_conflict,
  simple_ioi_name_mover, negative_name_control
```

This is not a discovered clustering result. It is a predeclared semantic
alternative: direct pointer retrieval vs repeated-sequence continuation vs
conflict/distractor choice.

### 3. Mechanism Group Ontology

A finer single-label ontology:

```text
local_offset
key_value_lookup
repeat_detection
induction
boundary_anchor
suppression_conflict
entity_coreference
```

### 4. Mechanism Multilabel Ontology

Each role receives multiple task attributes, such as:

```text
copy
local_offset
key_value_lookup
repeat_pattern
conflict
entity
boundary_anchor
```

Pairwise ontology similarity is Jaccard similarity over these attribute sets.

## Metrics

For each candidate ontology:

```text
ontology alignment =
  Spearman correlation between head-usage similarity and ontology similarity
```

The script also reports a shuffled-label null with 1000 permutations.

For label-free geometry:

```text
silhouette_k5
best_silhouette over k=2..8
mean pairwise TV distance
clusterability_x_separation = silhouette_k5 * mean_pair_tv
```

This distinguishes two cases:

```text
high silhouette alone:
  roles form crisp clusters, but they might collapse into a few similar heads

high silhouette x separation:
  roles form crisp clusters and those clusters are genuinely separated
```

## Main Result: Clean 16-Slot Run

Setting:

```text
8 heads/layer x 2 layers = 16 ordinary head slots
5 seeds
20 roles
2000 training steps
```

### Ontology Alignment

| Config | Original v2 | Task Primitive 3-Way | Mechanism Group | Multilabel |
|---|---:|---:|---:|---:|
| `uniform8` | 0.130 | 0.004 | 0.072 | 0.040 |
| `hetero8_unique_spread` | 0.087 | 0.034 | 0.023 | -0.026 |
| `hetero8_unique_extreme` | 0.087 | 0.063 | 0.038 | -0.085 |

Interpretation:

- Under the original v2 ontology, uniform still aligns better.
- Under the coarse task-primitive ontology, hetero improves over uniform.
- The effect is small and not enough by itself for a strong claim.
- The multilabel ontology does not help hetero.

### Label-Free Clusterability

| Config | Mean Pair TV | Silhouette k=5 | Best Silhouette | Separation-Adjusted |
|---|---:|---:|---:|---:|
| `uniform8` | 0.642 | 0.647 | 0.693 | 0.417 |
| `hetero8_unique_spread` | 0.567 | 0.674 | 0.730 | 0.381 |
| `hetero8_unique_extreme` | 0.493 | 0.693 | 0.784 | 0.337 |

Interpretation:

- Hetero creates sharper clusters by silhouette.
- But hetero also reduces overall pairwise role separation.
- Once clusterability is penalized by separation, uniform remains stronger.
- This suggests some hetero clustering may be capacity collapse around large
  heads rather than clean functional modularity.

## 32-Slot Control

Setting:

```text
8 heads/layer x 4 layers = 32 ordinary head slots
5 seeds
1000 training steps
```

### Ontology Alignment

| Config | Original v2 | Task Primitive 3-Way | Mechanism Group | Multilabel |
|---|---:|---:|---:|---:|
| `uniform8` | 0.050 | 0.121 | 0.064 | 0.141 |
| `hetero8_unique_spread` | 0.061 | 0.087 | 0.094 | 0.118 |
| `hetero8_unique_extreme` | 0.045 | 0.070 | 0.049 | 0.125 |

Interpretation:

- Hetero spread is slightly better under original v2 and mechanism group.
- Uniform is better under task-primitive and multilabel.
- This does not support a general hetero-modularity claim.

### Label-Free Clusterability

| Config | Mean Pair TV | Silhouette k=5 | Best Silhouette | Separation-Adjusted |
|---|---:|---:|---:|---:|
| `uniform8` | 0.720 | 0.492 | 0.564 | 0.353 |
| `hetero8_unique_spread` | 0.664 | 0.475 | 0.549 | 0.316 |
| `hetero8_unique_extreme` | 0.647 | 0.500 | 0.582 | 0.319 |

Interpretation:

- The 32-slot setting is mixed.
- Hetero extreme has slightly higher silhouette than uniform.
- Uniform has higher separation-adjusted clusterability.

## Earlier Full v2 Sweep

The earlier sweep is useful because it includes 2-head and 4-head configs.

### Original v2 Ontology Alignment

| Config | Alignment |
|---|---:|
| `uniform4` | 0.153 |
| `uniform2` | 0.139 |
| `hetero4_unique_mild` | 0.152 |
| `hetero4_unique_64` | 0.145 |
| `hetero4_unique_extreme` | 0.171 |
| `hetero2_unique_mild` | 0.162 |
| `hetero2_unique_mid` | 0.197 |
| `hetero2_unique_extreme` | 0.100 |

Interpretation:

- Some hetero configs beat uniform under the original ontology.
- The best was `hetero2_unique_mid`.
- Extreme imbalance can hurt modularity.

### Label-Free Clusterability

| Config | Mean Pair TV | Silhouette k=5 | Best Silhouette | Separation-Adjusted |
|---|---:|---:|---:|---:|
| `uniform4` | 0.585 | 0.733 | 0.797 | 0.427 |
| `uniform2` | 0.539 | 0.609 | 0.699 | 0.329 |
| `hetero4_unique_mild` | 0.622 | 0.663 | 0.749 | 0.413 |
| `hetero4_unique_64` | 0.582 | 0.677 | 0.756 | 0.393 |
| `hetero4_unique_extreme` | 0.395 | 0.765 | 0.842 | 0.299 |
| `hetero2_unique_mild` | 0.469 | 0.679 | 0.766 | 0.322 |
| `hetero2_unique_mid` | 0.436 | 0.703 | 0.773 | 0.308 |
| `hetero2_unique_extreme` | 0.322 | 0.687 | 0.777 | 0.215 |

Interpretation:

- Hetero extremes often produce high raw silhouette.
- But they also reduce pairwise separation, consistent with large-head
  attraction or collapse.
- The separation-adjusted metric does not clearly favor hetero.

## Conclusion

This refinement pass does **not** prove that non-uniform attention heads have
better functional modularity.

It does show a more nuanced picture:

```text
Non-uniform heads can increase raw cluster sharpness and can look better under
some defensible ontology views, especially coarse task-primitives.
```

But:

```text
The positive modularity signal is not robust across ontology choices, head
counts, and separation-adjusted clusterability controls.
```

The honest current claim remains:

```text
Heterogeneous attention-head dimensions robustly increase structural role
affinity and role-level specialization. Evidence for improved functional
modularity is suggestive in some settings but not yet strong.
```

## What Would Actually Fix The Ontology Issue

Post-hoc relabeling cannot prove the claim. The right next experiment is a Toy
Ontology v3 where the families are designed as repeated variants of the same
underlying computation, not broad literature-inspired buckets.

Recommended v3 families:

```text
local_offset:
  previous_token_1, previous_token_2, previous_token_3, marker_local_copy

key_value_lookup:
  kv_lookup_2pair, kv_lookup_4pair, wrong_key_control, recency_key_conflict

sequence_induction:
  induction_short, induction_mid, induction_long, induction_ngram

boundary_anchor:
  bos_anchor, sep_anchor, punctuation_anchor, newline_anchor

conflict_suppression:
  distractor_suppression, anti_copy, false_induction, negative_name_control
```

This would make the ontology much more defensible because each family would be
several variants of one algorithmic primitive. If hetero beats uniform there,
the modularity claim becomes much stronger.

Recommended v3 success criterion:

```text
Hetero must match or beat uniform on accuracy,
beat uniform on specialization,
beat uniform on original predeclared v3 ontology alignment,
and beat uniform on separation-adjusted clusterability.
```

Do not claim improved functional modularity before that.
