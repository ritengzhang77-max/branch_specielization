# Project Plan v2: Attention-Head Structure, Roles, and Modularity

Date: 2026-05-23

## Scope

The primary unit is still:

```text
ordinary attention head
```

Do not replace this with SwitchHead experts, MoE experts, or branch towers unless
the user explicitly approves a side experiment.

## Main Question

The revised project question is:

```text
Do heterogeneous attention-head structures make functional roles more
predictably assigned, more specialized, and more modular than uniform
attention-head baselines?
```

In the current toy intervention, "heterogeneous structure" mainly means
different per-head dimensions inside the same attention layer, for example:

```text
uniform4: [32, 32, 32, 32]
hetero4:  [16, 16, 32, 64]
```

## Four Quantities To Keep Separate

### 1. Structural Role Affinity

This is the central vocabulary for the user's intended claim.

Question:

```text
Given different-shaped heads, does a role reliably prefer a particular head type?
```

Example:

```text
local copy -> 64-dim head
```

Formalization:

Let `tau(h)` be a structural type of head `h`, such as:

```text
head_dim = 16, 32, 48, 64
head position = H0, H1, H2, H3
layer = L0, L1
```

Let:

```text
S_r(h) = max(loss_r(model with h ablated) - loss_r(base model), 0)
p_r(h) = S_r(h) / sum_h S_r(h)
```

Define structural affinity:

```text
A_r(tau) = sum_{h: tau(h)=tau} p_r(h)
```

Evidence for structural role affinity:

```text
argmax_tau A_r(tau) is stable across seeds.
```

Stronger evidence:

```text
If the 64-dim head moves from H3 to H0/H1/H2, the role follows the 64-dim head
rather than staying at the original head index.
```

Current strongest evidence:

```text
In one-64 heterogeneous layouts, the local role follows the 64-dim head in 40/40
toy models.
```

This is not exactly specialization and not exactly modularity. It is a
structure-to-function assignment bias.

### 2. Functional Specialization

Question:

```text
For one role, is causal importance concentrated in a small number of heads?
```

Metrics:

```text
top_specialization = max_h p_r(h)
entropy(p_r)
effective_num_heads = exp(entropy(p_r))
```

Interpretation:

```text
High specialization means the role is carried mostly by one/few heads.
```

### 3. Pairwise Separability

Question:

```text
Do two measured roles use different heads?
```

Metric:

```text
TV(r1, r2) = 0.5 * sum_h |p_r1(h) - p_r2(h)|
```

Status:

```text
This is valid as a small diagnostic, but it is not a standalone project claim.
```

It is useful because if even two roles cannot separate, ontology-level modularity
is unlikely in that setting. But if two roles do separate, that still does not
prove full modularity.

### 4. Ontology-Level Functional Modularity

Question:

```text
Across many roles and subroles, do related functions cluster together across
heads while unrelated functions separate?
```

This requires a role ontology, for example:

```text
local-copy family:
  local_sep_a
  local_sep_b
  previous-token variants

induction family:
  induction_len8
  induction_len16
  repeated-ngram variants

key-value lookup family:
  kv_lookup_a
  kv_lookup_b
  value-copy variants

position/sink family:
  BOS/SEP attention
  position-marker probes

suppression/distractor family:
  ignore distractor
  suppress repeated wrong token
```

Metrics:

```text
role x head causal matrix P[r, h]
within-family role similarity
between-family role similarity
family_gap = mean(within-family similarity) - mean(between-family similarity)
clustering adjusted Rand index against role-family labels
head-group purity / role-family purity
```

Evidence for functional modularity:

```text
related subroles have similar head distributions;
unrelated role families have separated head distributions;
the clustering is stable across seeds and stronger under heterogeneous heads
than under uniform baselines.
```

## What The Current Results Mean

Current ordinary-head toy evidence supports:

```text
heterogeneous head dimensions can create structural role affinity and strong
specialization.
```

The clean result is:

```text
local copy follows the 64-dim head across layout permutations.
```

Current evidence does not yet prove:

```text
heterogeneous head dimensions create ontology-level functional modularity.
```

The local-vs-induction experiment only tested pairwise separability. That is a
useful diagnostic but too small to carry the word "modularity" by itself.

The first toy role-ontology experiment was run on 2026-05-23:

```text
doc/phase3_toy_role_ontology_head_dim.md
results/phase3_toy_role_ontology_head_dim_20260523
```

It tested six roles in three families:

```text
local_copy: local_a, local_b
kv_lookup:  kv_a, kv_b
induction:  induction_short, induction_long
```

Main update:

```text
local-copy and KV-lookup subroles choose the 64-dim structural type in 80/80
one-64 heterogeneous cases.
```

Induction does not simply follow the 64-dim head:

```text
induction_short -> 64 in 9/20, 32 in 8/20, 16 in 3/20
induction_long  -> 64 in 2/20, 32 in 12/20, 16 in 6/20
```

Ontology-level clustering improved over `uniform4` in several hetero layouts,
but `uniform2` remained the strongest clustering control:

```text
uniform4 family gap:        0.511, ARI 0.586
hetero4_64second family gap: 0.607, ARI 0.889
uniform2 family gap:        0.653, ARI 1.000
```

Interpretation:

```text
structural role affinity and specialization are strongly supported; full
functional modularity is partially supported but not uniquely caused by
heterogeneity, because fewer/wider uniform heads are a strong baseline.
```

## Revised Research Claims To Test

### Claim A: Structural Role Affinity

```text
Heterogeneous head dimensions make certain roles reliably attach to particular
head types across seeds and layout permutations.
```

This is currently the strongest claim.

### Claim B: Functional Specialization

```text
Heterogeneous head dimensions increase causal concentration of roles into fewer
heads relative to matched uniform-head baselines.
```

This is currently supported in key-value, induction, and local-copy toy settings.

### Claim C: Ontology-Level Functional Modularity

```text
Heterogeneous head dimensions make related role/subrole families cluster more
cleanly across heads than matched uniform-head baselines.
```

This is not answered yet and is the next experiment.

### Claim D: Pairwise Separability

```text
Some role pairs separate more under heterogeneous heads than under uniform heads.
```

This is a secondary diagnostic. It should not be the main paper claim.

## Next Experiment: Toy Role Ontology

Build one ordinary-attention-head toy setting with several supervised role
families in the same model:

```text
local-copy subroles:
  local_a
  local_b

key-value lookup subroles:
  kv_a
  kv_b

induction subroles:
  induction_short
  induction_long
```

For each model, compute:

```text
S_r(h) for every role r and head h
p_r(h) over flattened (layer, head) slots
top structural type per role
specialization per role
within-family vs between-family role similarity
family_gap
```

Compare:

```text
uniform4 vs hetero4 vs 64-position controls vs uniform2
```

Decision criteria:

```text
Structural role affinity:
  roles consistently prefer the same head type across seeds and follow that
  head type when it moves.

Functional specialization:
  heterogeneous layouts reduce effective_num_heads or increase max_h p_r(h).

Ontology-level modularity:
  within-family role similarity exceeds between-family role similarity, and the
  family_gap is larger in heterogeneous layouts than matched uniform baselines.
```

## Process Rule

Before autonomous work, state:

```text
unit = ordinary attention head
claim = structural role affinity / specialization / ontology-level modularity
experiment family = toy ordinary-head or real-model head-level validation
```

If a proposed experiment changes the unit of analysis, stop and ask.
