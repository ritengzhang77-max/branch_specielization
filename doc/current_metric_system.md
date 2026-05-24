# Current Metric System

Date: 2026-05-24

This is the current agreed metric system for the branch-specialization /
attention-head-heterogeneity project.

## Scope

The unit of analysis is:

```text
ordinary attention head
```

The current core intervention is:

```text
uniform attention-head dimensions
vs
non-uniform / heterogeneous attention-head dimensions
```

The three questions remain separate:

1. Does the model solve the task?
2. Do roles become more specialized into fewer attention heads?
3. Do roles become more modular in head-usage geometry?

Structural-affinity diagnostics can still be saved, but they should not be
headline metrics.

## 0. Accuracy Gate

Accuracy is always reported first.

Reason:

```text
Specialization or modularity is only meaningful if the model still solves the
task at matched or better performance.
```

Main accuracy fields:

```text
mean role accuracy
minimum role accuracy
```

Interpretation:

| Result | Meaning |
|---|---|
| hetero accuracy >= uniform accuracy | safe to interpret specialization/modularity |
| hetero accuracy lower than uniform | specialization/modularity may be capacity damage |

## 1. Specialization Metrics

For each role `r`, compute positive causal head scores:

```text
score(r, h) = max(performance drop after ablating head h on role r, 0)
```

Normalize into a distribution over heads:

```text
p_r(h) = score(r, h) / sum_h score(r, h)
```

### Top-Head Specialization

```text
specialization(r) = max_h p_r(h)
```

Meaning:

| Value | Meaning |
|---:|---|
| high | one head carries most of the role |
| low | the role is spread across many heads |

### Effective Heads

```text
effective_heads(r) = exp(entropy(p_r))
```

Meaning:

| Value | Meaning |
|---:|---|
| `1` | role is effectively carried by one head |
| `2` | role uses about two heads |
| `5` | role is distributed across about five heads |

Report these averaged across roles and seeds.

## 2. Main Modularity Metrics

There are two headline modularity metrics.

### 2.1 Ontology Alignment

This is the main ontology-based modularity metric.

For every pair of roles `(i, j)`, compute head-usage similarity:

```text
TV(i, j) = 0.5 * sum_h |p_i(h) - p_j(h)|
S_head(i, j) = 1 - TV(i, j)
```

Also compute ontology similarity:

```text
S_ontology(i, j) = 1 if same predeclared family
S_ontology(i, j) = 0 otherwise
```

For future hierarchical ontologies, this can become:

```text
same subfamily:                   1.0
same family, different subfamily: 0.5
different family:                 0.0
```

Then:

```text
ontology_alignment =
  Spearman correlation between S_head(i, j) and S_ontology(i, j)
```

Interpretation:

| Value | Meaning |
|---:|---|
| high positive | related roles use similar heads |
| near zero | ontology does not explain head usage |
| negative | same-family roles are less similar than unrelated roles |

### 2.2 Shuffled-Label p

This is the significance check for ontology alignment.

Procedure:

1. Compute the real ontology alignment.
2. Randomly shuffle family labels across roles.
3. Recompute ontology alignment.
4. Repeat 1000 times.
5. Report:

```text
shuffled_label_p =
  fraction of shuffled scores >= real score
```

Interpretation:

| Pair | Meaning |
|---|---|
| high alignment + low p | strong modularity evidence |
| low alignment + high p | weak/no modularity evidence |
| high alignment + high p | large but unstable/not clearly better than random labels |
| low alignment + low p | statistically real but likely small practical effect |

### 2.3 Separation-Adjusted Clusterability

This is the main label-free modularity sanity check.

It does **not** use ontology labels.

Steps:

1. Compute pairwise role distances using TV distance over head-usage
   distributions.
2. Cluster roles using only those distances.
3. Compute silhouette score for the discovered clusters.
4. Multiply by average pairwise role distance:

```text
separation_adjusted_clusterability =
  silhouette_k5 * mean_pairwise_TV_distance
```

Interpretation:

| Component | Meaning |
|---|---|
| silhouette | discovered clusters are sharp |
| mean pairwise TV | roles are meaningfully separated |
| product | clusters are both sharp and separated |

This avoids over-crediting apparent clusters caused by role collapse into a
narrow part of head space.

## 3. Appendix / Diagnostic Metrics

These may be saved but should not be headline metrics.

### Family Gap

```text
family_gap =
  average same-family head similarity
  -
  average different-family head similarity
```

Status:

```text
appendix only
```

Reason:

Family Gap and Ontology Alignment measure the same underlying binary-family
signal. Ontology Alignment is preferred because it supports correlation-style
reporting, shuffled-label baselines, and future hierarchical ontologies.

### Top-Dimension / Largest-Head Rate

This asks whether roles choose a specific head dimension, especially the largest
dimension.

Status:

```text
raw diagnostic only
```

Reason:

It can mislead readers into thinking the claim is:

```text
bigger heads are better
```

That is not the project claim. The real structural question is whether
head-structure differences affect function assignment, not whether every role
should prefer the largest head.

### ARI

Adjusted Rand Index over discovered clusters and family labels is no longer a
main metric.

Status:

```text
avoid in main text; optional appendix only
```

Reason:

ARI assumes ontology families should appear as exact discovered clusters. That
is too brittle for this project.

## Main Reporting Table

Future main tables should use this order:

| Config | Mean Acc | Min Acc | Specialization | Effective Heads | Ontology Align | Shuffled p | Sep-Adjusted Clusterability |
|---|---:|---:|---:|---:|---:|---:|---:|

The interpretation order should be:

1. Did accuracy match?
2. Did specialization increase?
3. Did ontology alignment increase?
4. Is ontology alignment significant against shuffled labels?
5. Does label-free clusterability support the same direction?

## Current Best Result: Toy Ontology v3

| Config | Mean Acc | Min Acc | Specialization | Effective Heads | Ontology Align | Shuffled p | Sep-Adjusted Clusterability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform8` | 0.9999 | 0.9983 | 0.660 | 3.586 | 0.139 | 0.044 | 0.417 |
| `hetero8_unique_spread` | 1.0000 | 0.9995 | 0.670 | 3.174 | 0.142 | 0.059 | 0.450 |
| `hetero8_unique_extreme` | 0.9999 | 0.9986 | 0.783 | 2.073 | 0.193 | 0.016 | 0.437 |

Current interpretation:

```text
Toy Ontology v3 gives the strongest evidence so far that non-uniform ordinary
attention-head dimensions can improve functional modularity under a cleaner
predeclared algorithmic ontology.
```

But the claim is not final yet:

```text
The v3 positive result needs robustness sweeps over more seeds, layout
permutations, and moderate heterogeneity settings.
```
