# Three-Question Framework For Attention-Head Structure

Date: 2026-05-23

## Scope

The primary unit is:

```text
ordinary attention head
```

This framework is not about SwitchHead experts, MoE experts, or branch towers.

The core intervention is:

```text
make attention heads structurally different, especially by giving heads
different dimensions.
```

## The Three Questions

### 1. Structural Role Affinity

Question:

```text
In non-uniform models, across seeds and layout permutations, is the same role
more likely to be assigned to the same kind of attention head?
```

Small correction:

```text
not necessarily the same exact head index, but the same structural head type.
```

Example:

```text
[16,16,32,64] -> role goes to the 64-dim head at H3
[64,16,16,32] -> role goes to the 64-dim head at H0
[16,64,16,32] -> role goes to the 64-dim head at H1
[16,32,64,16] -> role goes to the 64-dim head at H2
```

So the role follows:

```text
head dimension = 64
```

not:

```text
fixed head index H3
```

Short definition:

```text
Affinity asks: which type of head gets a role?
```

### 2. Functional Specialization

Question:

```text
Once a role is learned, is it concentrated in one/few heads or spread across
many heads?
```

This is different from affinity.

Affinity asks:

```text
which kind of head gets the role?
```

Specialization asks:

```text
how concentrated is the role?
```

Example of strong affinity but weaker specialization:

```text
local role mostly prefers large heads, but several heads share causal mass.
```

Example of strong specialization:

```text
local role has 90% of its causal mass on one head.
```

Metrics:

```text
S_r(h) = max(loss_r(model with h ablated) - loss_r(base model), 0)
p_r(h) = S_r(h) / sum_h S_r(h)

max_h p_r(h)
entropy(p_r)
effective_num_heads = exp(entropy(p_r))
```

Short definition:

```text
Specialization asks: how concentrated is each role?
```

### 3. Functional Modularity

Question:

```text
Across many roles and subroles, do related roles cluster together across heads,
while unrelated roles separate?
```

Example ontology:

```text
local_copy:
  local_a
  local_b

kv_lookup:
  kv_a
  kv_b

induction:
  induction_short
  induction_long
```

A modular result would mean:

```text
local_a and local_b use similar heads;
kv_a and kv_b use similar heads;
induction_short and induction_long use similar heads;
different families use different head patterns.
```

Old first-pass metrics:

```text
family_gap = within_family_similarity - between_family_similarity
family_cluster_ari = clustering agreement with role-family labels
```

Current preferred metric:

```text
ontology_alignment_score =
  Spearman correlation between pairwise head-usage similarity and pairwise
  ontology similarity
```

For Toy Ontology v2, ontology similarity is currently binary: `1` for
same-family pairs and `0` for different-family pairs. Future ontologies can make
this hierarchical with subfamily labels. ARI should not be used as the main
metric because the ontology families are hypotheses, not guaranteed discovered
clusters.

Short definition:

```text
Modularity asks: does the geometry of role usage over ordinary attention heads
respect the predefined role ontology better than chance?
```

## Why These Are Different

The three questions can dissociate.

High affinity without strong specialization:

```text
a role prefers 64-dim heads, but its mass is spread across several 64-dim or
large-capacity heads.
```

High specialization without modularity:

```text
each role concentrates in one head, but unrelated roles may concentrate in the
same head.
```

Modularity without unique affinity:

```text
related role families cluster cleanly, but the exact structural head type may
vary across layouts.
```

The project should therefore report all three:

```text
Affinity:       which type of head gets a role?
Specialization: how concentrated is each role?
Modularity:     do related roles form stable groups across heads?
```

## Evidence So Far

### Experiment Settings

The ordinary-head toy experiments used tiny decoder-only transformers:

```text
d_model = 128
n_layers = 2
ordinary attention heads
matched total attention dimension = 128
```

Main head-dimension layouts:

```text
uniform4:          [32, 32, 32, 32]
hetero4:           [16, 16, 32, 64]
hetero4_64first:   [64, 16, 16, 32]
hetero4_64second:  [16, 64, 16, 32]
hetero4_64third:   [16, 32, 64, 16]
uniform2:          [64, 64]
```

The role-ontology experiment used six roles:

```text
local_copy: local_a, local_b
kv_lookup:  kv_a, kv_b
induction:  induction_short, induction_long
```

For every trained model, every role, and every head, the script performed
single-head causal ablation and measured role-specific loss increase.

### Baselines First

The baseline results must be stated before the heterogeneous result.

#### Uniform4 Baseline

Configuration:

```text
uniform4 = [32, 32, 32, 32]
```

There are four heads per layer, all with the same dimension. Therefore this
baseline cannot show dimension affinity, because every head has the same
structural dimension.

For local-copy and KV-lookup roles:

```text
20 role cases = 5 seeds x 4 roles
top_dim = 32 in 20/20 cases
```

This is trivial because all heads are 32-dim.

The useful baseline is the top head-index distribution:

```text
H0: 2/20
H1: 9/20
H2: 5/20
H3: 4/20
```

So in `uniform4`, local/KV roles are not all forced into one structural head
type, because there is no dimension difference to select. The roles are spread
over ordinary equal-width head indices.

Uniform4 role-ontology metrics:

```text
specialization:    0.449
effective heads:   4.15
family gap:        0.511
cluster ARI:       0.586
```

This is the main four-head uniform baseline.

#### Uniform2 Baseline

Configuration:

```text
uniform2 = [64, 64]
```

There are two heads per layer, both 64-dim. Therefore:

```text
top_dim = 64
```

is also trivial in this baseline, because every head is 64-dim.

For local-copy and KV-lookup roles:

```text
top head index H0: 16/20
top head index H1:  4/20
```

Uniform2 role-ontology metrics:

```text
specialization:    0.636
effective heads:   2.25
family gap:        0.653
cluster ARI:       1.000
```

This is the strongest capacity/head-count baseline so far. It shows that fewer,
wider uniform heads can already create high specialization and strong
role-family clustering.

Therefore any heterogeneity claim must beat or explain these two baselines:

```text
uniform4 tests equal-width four-head MHA.
uniform2 tests fewer/wider equal-width heads.
hetero layouts test unequal head types at matched total head dimension.
```

### Why Structural Role Affinity Is Strong

The key test moved the 64-dim head across positions:

```text
[16,16,32,64] -> 64 at H3
[64,16,16,32] -> 64 at H0
[16,64,16,32] -> 64 at H1
[16,32,64,16] -> 64 at H2
```

Across these four one-64 layouts:

```text
4 layouts x 5 seeds = 20 trained models
```

For four local/KV roles:

```text
local_a
local_b
kv_a
kv_b
```

that gives:

```text
20 models x 4 roles = 80 role cases
```

Observed result:

```text
local_a -> 64-dim top structural type in 20/20
local_b -> 64-dim top structural type in 20/20
kv_a    -> 64-dim top structural type in 20/20
kv_b    -> 64-dim top structural type in 20/20
```

Combined:

```text
local-copy and KV-lookup roles chose the 64-dim structural type in 80/80 cases.
```

Why this is strong:

```text
causal: based on ablation loss, not just attention pattern;
cross-seed: repeated across 5 random seeds;
layout-controlled: the 64-dim head moved from H3 to H0/H1/H2;
role-repeated: four related roles showed the same preference.
```

Baseline intuition:

```text
In each hetero4 layout, the 64-dim type is only 1 of 4 heads per layer.
If top-role assignment were random over head types, rough chance would be 25%.
Observed local/KV result was 100%.
```

Baseline comparison:

```text
uniform4 has no dimension type to select and local/KV roles are spread over
head indices: H0 2/20, H1 9/20, H2 5/20, H3 4/20.

hetero4-style layouts have one 64-dim type, and local/KV roles select that
64-dim type in 80/80 cases across layout permutations.
```

Caveat:

```text
This proves strong large-head affinity in one-64 heterogeneous layouts. It does
not yet prove heterogeneity beats every capacity baseline.
```

The missing control is:

```text
hetero2, such as [32, 96] or [48, 80], compared against uniform2 [64, 64].
```

### Why Functional Specialization Is Strong

Specialization was measured by how concentrated each role's causal distribution
was over heads.

In the six-role ontology experiment:

```text
uniform4 specialization:          0.449
uniform4 effective heads:         4.15

uniform2 specialization:          0.636
uniform2 effective heads:         2.25
```

Compared with those baselines:

```text
uniform4 specialization:          0.449
hetero4 specialization:           0.663
hetero4_64first specialization:   0.695
hetero4_64second specialization:  0.723
hetero4_64third specialization:   0.723
uniform2 specialization:          0.636
```

The effective number of heads dropped:

```text
uniform4 effective heads:          4.15
hetero4 effective heads:           2.30
hetero4_64first effective heads:   2.28
hetero4_64second effective heads:  2.08
hetero4_64third effective heads:   2.09
uniform2 effective heads:          2.25
```

Interpretation:

```text
heterogeneous one-64 layouts made roles more concentrated than uniform4.
```

This is strong because the result appears across several heterogeneous layouts,
not just one configuration.

Caveat:

```text
uniform2 is also concentrated, so some of the specialization gain may come from
fewer/wider heads rather than heterogeneity alone.
```

### Why Functional Modularity Is Mixed

Functional modularity was tested with the six-role ontology.

The question was:

```text
do same-family roles have more similar head distributions than different-family
roles?
```

Results:

```text
uniform4:          family gap 0.511, ARI 0.586
hetero4:           family gap 0.505, ARI 0.889
hetero4_64first:   family gap 0.557, ARI 1.000
hetero4_64second:  family gap 0.607, ARI 0.889
hetero4_64third:   family gap 0.587, ARI 0.889
uniform2:          family gap 0.653, ARI 1.000
```

Interpretation:

```text
some hetero layouts improve role-family clustering over uniform4, but uniform2
is the strongest clustering control.
```

So the modularity claim is mixed:

```text
heterogeneity can help family clustering relative to uniform4, but it is not yet
shown to beat fewer/wider uniform-head baselines.
```

## Current Best Framing

The strongest current claims are:

```text
1. Heterogeneous head dimensions create structural role affinity.
2. Heterogeneous head dimensions increase functional specialization over
   uniform4.
```

The still-open claim is:

```text
3. Heterogeneous head dimensions improve ontology-level functional modularity
   beyond strong capacity/head-count baselines.
```

The next decisive experiment is:

```text
compare uniform2 [64,64] against hetero2 layouts such as [32,96] and [48,80].
```
