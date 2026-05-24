# Baseline Comparisons For The Three Questions

Date: 2026-05-23

## Purpose

This document presents the results in the form they should be interpreted:

```text
baseline result -> heterogeneous result -> what changed
```

The three questions are:

1. Structural role affinity: which type/index of head gets a role?
2. Functional specialization: how concentrated is each role over heads?
3. Functional modularity: do related role families cluster together?

All results here use ordinary attention heads.

## Experiment Context

Model:

```text
d_model = 128
n_layers = 2
ordinary attention heads
total attention head dimension = 128
```

Role ontology:

```text
local_copy: local_a, local_b
kv_lookup:  kv_a, kv_b
induction:  induction_short, induction_long
```

Main layouts:

```text
uniform4:          [32, 32, 32, 32]
hetero4:           [16, 16, 32, 64]
hetero4_64first:   [64, 16, 16, 32]
hetero4_64second:  [16, 64, 16, 32]
hetero4_64third:   [16, 32, 64, 16]
uniform2:          [64, 64]
```

## Question 1: Structural Role Affinity

Question:

```text
Do non-uniform models make a role consistently attach to a particular head type?
```

Here the relevant roles are the four local/KV roles:

```text
local_a, local_b, kv_a, kv_b
```

There are 5 seeds, so each layout has:

```text
5 seeds x 4 roles = 20 local/KV role cases
```

### Baseline: `uniform4`

Layout:

```text
[32, 32, 32, 32]
```

All heads are 32-dim, so dimension affinity cannot be tested. The meaningful
baseline is which equal-width head index gets the roles.

Local/KV top head-index counts:

| Head index | Count |
|---|---:|
| H0 | 2/20 |
| H1 | 9/20 |
| H2 | 5/20 |
| H3 | 4/20 |

Interpretation:

```text
With equal-width four-head attention, local/KV roles do not all go to one
structural dimension. They are spread across equal-width head indices.
```

### Baseline: `uniform2`

Layout:

```text
[64, 64]
```

All heads are 64-dim, so `top_dim = 64` is trivial. The meaningful baseline is
head-index concentration under fewer/wider uniform heads.

Local/KV top head-index counts:

| Head index | Count |
|---|---:|
| H0 | 16/20 |
| H1 | 4/20 |

Interpretation:

```text
Uniform2 already has a strong head-index preference. This is why it is a strong
capacity/head-count baseline.
```

### Heterogeneous Four-Head Layouts

Each hetero4 layout has one 64-dim head and three smaller heads.

Local/KV result:

| Layout | 64-dim head position | Local/KV roles choosing 64-dim type |
|---|---:|---:|
| `[16,16,32,64]` | H3 | 20/20 |
| `[64,16,16,32]` | H0 | 20/20 |
| `[16,64,16,32]` | H1 | 20/20 |
| `[16,32,64,16]` | H2 | 20/20 |

Combined:

```text
80/80 local/KV role cases chose the 64-dim structural type.
```

Baseline comparison:

```text
uniform4: equal-width roles spread over head indices; no dimension type exists.
uniform2: strong head-index concentration, but all heads are 64-dim.
hetero4: when exactly one 64-dim type exists among smaller heads, local/KV roles
         follow that 64-dim type across all moved positions.
```

What changed:

```text
hetero4-style layouts create a clean role-to-head-type assignment for local/KV
roles. The role follows the 64-dim structural type, not one fixed head index.
```

Caveat:

```text
This shows strong large-head affinity inside heterogeneous four-head layouts.
It does not yet prove heterogeneity beats uniform2, because uniform2 is also
strongly organized by head index and capacity.
```

## Question 2: Functional Specialization

Question:

```text
How concentrated is each role's causal mass over heads?
```

Higher specialization and lower effective-head count mean the role is carried by
fewer heads.

### Baselines

| Layout | Specialization | Effective heads |
|---|---:|---:|
| `uniform4 [32,32,32,32]` | 0.449 | 4.15 |
| `uniform2 [64,64]` | 0.636 | 2.25 |

Interpretation:

```text
uniform4 spreads roles across more heads.
uniform2 is already more concentrated because it has fewer/wider heads.
```

### Heterogeneous Layouts

| Layout | Specialization | Effective heads |
|---|---:|---:|
| `hetero4 [16,16,32,64]` | 0.663 | 2.30 |
| `hetero4_64first [64,16,16,32]` | 0.695 | 2.28 |
| `hetero4_64second [16,64,16,32]` | 0.723 | 2.08 |
| `hetero4_64third [16,32,64,16]` | 0.723 | 2.09 |

Baseline comparison:

```text
versus uniform4: all hetero4 layouts are much more specialized
                 (0.663-0.723 vs 0.449).

versus uniform2: hetero4 layouts are similar or somewhat stronger
                 (0.663-0.723 vs 0.636), but not separated enough to ignore
                 the capacity/head-count explanation.
```

What changed:

```text
heterogeneous four-head layouts make roles more concentrated than uniform4.
```

Caveat:

```text
uniform2 is also concentrated, so specialization is not uniquely explained by
heterogeneity. It may partly come from fewer/wider heads.
```

## Question 3: Functional Modularity

Question:

```text
Do related role families cluster together across heads more than unrelated
families?
```

Metrics:

```text
family_gap = within_family_similarity - between_family_similarity
ARI = clustering agreement with role-family labels
```

### Baselines

| Layout | Family gap | ARI |
|---|---:|---:|
| `uniform4 [32,32,32,32]` | 0.511 | 0.586 |
| `uniform2 [64,64]` | 0.653 | 1.000 |

Interpretation:

```text
uniform4 has some role-family structure, but clustering is imperfect.
uniform2 has very strong role-family clustering.
```

### Heterogeneous Layouts

| Layout | Family gap | ARI |
|---|---:|---:|
| `hetero4 [16,16,32,64]` | 0.505 | 0.889 |
| `hetero4_64first [64,16,16,32]` | 0.557 | 1.000 |
| `hetero4_64second [16,64,16,32]` | 0.607 | 0.889 |
| `hetero4_64third [16,32,64,16]` | 0.587 | 0.889 |

Baseline comparison:

```text
versus uniform4: most hetero layouts improve ARI and several improve family gap.

versus uniform2: no hetero4 layout beats uniform2's family gap, and only
                 hetero4_64first ties uniform2's ARI.
```

What changed:

```text
heterogeneity can improve family clustering over the four-head uniform baseline,
but it does not beat the two-head wide uniform baseline.
```

Conclusion for modularity:

```text
mixed. Hetero helps relative to uniform4, but uniform2 is stronger.
```

## Summary Table

| Question | Uniform4 baseline | Uniform2 baseline | Hetero4-style result | Current conclusion |
|---|---|---|---|---|
| Structural role affinity | local/KV head index spread: H0 2/20, H1 9/20, H2 5/20, H3 4/20 | H0 16/20, H1 4/20, but both heads are 64-dim | local/KV choose 64-dim type in 80/80 | strong vs uniform4; need hetero2 vs uniform2 |
| Specialization | spec 0.449, eff heads 4.15 | spec 0.636, eff heads 2.25 | spec 0.663-0.723, eff heads 2.08-2.30 | strong vs uniform4; capacity confound remains |
| Modularity | family gap 0.511, ARI 0.586 | family gap 0.653, ARI 1.000 | family gap 0.505-0.607, ARI 0.889-1.000 | mixed; uniform2 strongest |

## Next Missing Baseline

The comparison originally missing was:

```text
uniform2 [64,64] vs hetero2 [32,96] or [48,80]
```

This has now been run:

```text
doc/experiments/phase3/phase3_toy_role_ontology_hetero2.md
results/phase3_toy_role_ontology_hetero2_20260523
```

Result:

| Question | `uniform2 [64,64]` | Best hetero2 result | Interpretation |
|---|---:|---:|---|
| Structural role affinity | local/KV top head index H0 `16/20`, H1 `4/20`; no dimension contrast | local/KV choose larger hetero2 head in `60/60` | hetero2 strengthens role-to-head-type affinity |
| Specialization | `0.636` | `[48,80]`: `0.756` | hetero2 beats uniform2 on concentration |
| Modularity | family gap `0.653`, ARI `1.000` | `[48,80]`: gap `0.601`, ARI `1.000` | hetero2 does not beat uniform2 on modularity |

Updated conclusion:

```text
unequal two-head structure improves structural role affinity and specialization,
but not ontology-level functional modularity over the strong uniform2 baseline.
```
