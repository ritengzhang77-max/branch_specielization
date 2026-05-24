# Ontology Alignment Metric For Functional Modularity

Date: 2026-05-24

## Why This Metric Exists

The project should not assume that a hand-written ontology family is guaranteed
to become one discovered cluster of attention heads. A family label such as
`copy_transport` is a research hypothesis about related roles, not a clustering
result.

The modularity question is therefore better phrased as:

```text
Are roles that are closer in the ontology also closer in attention-head usage?
```

This replaces ARI and Family Gap as the main ontology-based modularity
diagnostic. ARI and Family Gap can still be computed for debugging or appendix
checks, but they are too easy to misread as primary evidence.

## Inputs

For each role `r`, measure the positive causal contribution of every ordinary
attention head `h`:

```text
score(r, h) = max(performance drop after ablating h on role r, 0)
```

Normalize those scores into a head-usage distribution:

```text
p_r(h) = score(r, h) / sum_h score(r, h)
```

The unit is still the ordinary attention head. This is not an MoE-expert metric.

## Head-Usage Similarity

For two roles `i` and `j`, compare their normalized head-usage distributions
with total variation distance:

```text
TV(i, j) = 0.5 * sum_h |p_i(h) - p_j(h)|
```

Interpretation:

| TV distance | Meaning |
|---:|---|
| `0` | the two roles use heads identically |
| `1` | the two roles use disjoint heads |

Convert distance to similarity:

```text
S_head(i, j) = 1 - TV(i, j)
```

Interpretation:

| Head similarity | Meaning |
|---:|---|
| `1` | same head usage |
| `0` | completely different head usage |

## Ontology Similarity

For the current Toy Ontology v2, the implemented ontology is binary:

```text
S_ontology(i, j) = 1 if roles i and j are in the same family
S_ontology(i, j) = 0 otherwise
```

If future ontologies add subfamilies, this should become hierarchical:

```text
same subfamily:                  1.0
same family, different subfamily: 0.5
different family:                0.0
```

The current data has family labels but not validated subfamily labels, so the
binary version is the honest version for now.

## Ontology Alignment Score

Compute all role pairs in one trained model, then calculate:

```text
Ontology Alignment Score =
  Spearman correlation between S_head(i, j) and S_ontology(i, j)
```

Interpretation:

| Score | Meaning |
|---:|---|
| high positive | same-family roles tend to use more similar heads |
| near zero | ontology family labels do not explain head usage |
| negative | same-family roles are less similar than unrelated roles |

The reported table also includes a shuffled-label baseline:

```text
shuffle the family labels across roles;
recompute the alignment score;
repeat 1000 times.
```

This gives a null mean, a z-score, and a one-sided permutation p-value. The null
baseline is important because a small positive score can occur just from the
number and size of families.

## Family Gap, Appendix Only

Family Gap is the simplest effect-size version of the same binary-family idea:

```text
Family Gap =
  average S_head(i, j) for same-family pairs
  -
  average S_head(i, j) for different-family pairs
```

Interpretation:

| Family Gap | Meaning |
|---:|---|
| positive and large | same-family roles are meaningfully closer |
| near zero | weak or no ontology-level modularity |
| negative | same-family roles are less coherent than unrelated roles |

Family Gap and Ontology Alignment are not independent evidence in the current
binary-family setting. They both ask whether same-family role pairs have higher
head-usage similarity than different-family pairs.

The difference is:

| Metric | What It Does |
|---|---|
| Family Gap | Raw mean difference: same-family similarity minus different-family similarity. |
| Ontology Alignment | Correlation between ontology similarity and head-usage similarity, with a shuffled-label null. |

Because they measure the same underlying signal, Family Gap should not be a
main modularity metric. Keep it as an appendix/simple-effect-size check only.

## Separation-Adjusted Clusterability

This is the label-free modularity check. It does not use ontology labels.

First, cluster roles using only their pairwise head-usage distances:

```text
TV(i, j) = 0.5 * sum_h |p_i(h) - p_j(h)|
```

Then compute silhouette score:

```text
silhouette = how much closer each role is to its discovered cluster than to the
nearest other discovered cluster
```

Silhouette alone can be misleading if all roles collapse into a narrow part of
head space. So the reported label-free metric is:

```text
separation_adjusted_clusterability =
  silhouette_k5 * mean_pairwise_TV_distance
```

This rewards discovered clusters that are both internally coherent and
meaningfully separated.

## Per-Role Neighbor Margin

For every role, compute:

```text
role_margin(r) =
  average similarity from r to same-family roles
  -
  average similarity from r to different-family roles
```

This shows which roles support or weaken the modularity result.

Also report:

```text
top1_same_family_rate = fraction of seeds where the nearest role is same-family
top3_same_family_rate = fraction of top-3 nearest neighbors that are same-family
```

This is useful because a global family score can hide one family doing well and
another family failing.

## What To Report In Main Tables

Accuracy must come first. Specialization only matters if task performance is
matched or better than the uniform baseline.

Recommended table order:

1. `accuracy`
2. `specialization`
3. `effective heads`
4. `ontology alignment score`
5. `shuffled-label p-value`
6. `separation-adjusted clusterability`

Per-role appendices should report:

```text
accuracy
specialization
effective heads
same-family neighbor similarity
different-family neighbor similarity
role margin
top-1 / top-3 same-family neighbor rates
```

Head-dimension top counts and largest-dimension top rate may remain in raw
analysis files as structural-affinity diagnostics, but they should not be
headline metrics because they can imply the wrong claim: "bigger head is better."
The project claim is about non-random structure-function assignment, not a
universal preference for the largest head.

## Current Artifacts

The metric is implemented in:

```text
scripts/analyze_role_ontology_v2.py
```

Current output files include:

```text
results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523/analysis/ontology_alignment_table.csv
results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523/analysis/role_ontology_neighbor_table.csv
results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523/analysis/dimension_ontology_alignment_table.csv
```

## Current Interpretation

Toy Ontology v3 currently supports:

```text
heterogeneous attention-head dimensions preserve task accuracy,
increase role-level specialization, and give the first meaningful positive
evidence for improved ontology-level modularity under a cleaner predeclared
algorithmic ontology.
```

It does not yet support:

```text
heterogeneous attention-head dimensions always improve modularity across all
ontology choices, layouts, and families.
```

The reason is that same-family head-usage similarity is positive but uneven
across families, and the best uniform baseline remains competitive on the
ontology-alignment metrics.
