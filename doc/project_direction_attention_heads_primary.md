# Project Direction Correction: Attention Heads Are The Primary Unit

Date: 2026-05-23

## Why This Exists

The user corrected an important scope drift on 2026-05-23:

```text
The main project is about ordinary attention heads / attention-head structure.
Do not silently replace the unit with SwitchHead experts, MoE experts, or branch
towers.
```

The SwitchHead work is related but not the core proof. It must be labeled as a
secondary routed-attention-expert case, not as evidence about ordinary attention
heads.

## Correct Main Question

The project should be stated as:

```text
Do structural changes in attention heads or attention-head branches lead to
stable functional specialization or functional modularity?
```

More concretely:

```text
If we change attention-head structure, especially heterogeneous head dimensions,
do functions land in more stable causal head slots across seeds, and do different
functions separate across heads?
```

## Primary Unit

The primary unit is:

```text
ordinary attention head
```

Allowed primary interventions:

- heterogeneous attention-head dimensions;
- moving the large/small head slots to test whether functions follow structure;
- matched-parameter uniform-head controls;
- ordinary multi-head attention models with head-level ablations and head-level
  role metrics.

Not primary unless the user explicitly approves:

- SwitchHead attention experts;
- MoE/MLP experts;
- separate branch towers;
- generic routers that are not ordinary attention heads.

## What Current Evidence Supports

### Strong Core Evidence: Attention-Head Specialization

Two ordinary attention-head toy experiments support the core direction.

1. Toy key-value recall:

```text
task: [k1, v1, ..., k8, v8, query_key] -> query_value
unit: ordinary attention head
metric: single-head ablation loss delta, top_specialization
hetero4 [16,16,32,64] top_specialization = 0.9741
uniform4 [32,32,32,32] top_specialization = 0.4414
64-dim head was top causal head in 5/5 seeds
moving the 64-dim head moved the role in 5/5 seeds
```

Memo:

```text
doc/experiments/phase3/phase3_toy_head_dim_intervention.md
```

2. Toy induction / repeated-sequence continuation:

```text
task: [x1, ..., x16, x1, ..., x16] -> next token
unit: ordinary attention head
metric: single-head ablation loss delta, top_specialization
hetero4 [16,16,32,64] top_specialization = 0.9830
uniform4 [32,32,32,32] top_specialization = 0.5796
64-dim head was top causal head in 5/5 seeds
moving the 64-dim head moved the role in 5/5 seeds
```

Memo:

```text
doc/experiments/phase3/phase3_toy_induction_head_dim_intervention.md
```

These are the strongest current results for the user's intended project.

### Latest Evidence: Head-Level Specialization vs Pairwise Separability

A larger ordinary-attention-head local-vs-induction sweep was run on
2026-05-23:

```text
results/phase3_toy_competition_head_dim_modularity_sweep_20260523
doc/experiments/phase3/phase3_attention_head_specialization_modularity_sweep.md
```

The experiment used 10 seeds across nine matched-budget head-dimension layouts.

Strong result:

```text
In one-64 heterogeneous layouts, the local causal role follows the 64-dim head
slot in 40/40 models.
```

Concretely:

```text
hetero4            [16,16,32,64] -> local top slot L1H3 in 10/10
hetero4_64first    [64,16,16,32] -> local top slot L1H0 in 10/10
hetero4_64second   [16,64,16,32] -> local top slot L1H1 in 10/10
hetero4_64third    [16,32,64,16] -> local top slot L1H2 in 10/10
```

This supports:

```text
structural head-dimension heterogeneity can create stable functional
specialization slots in ordinary attention heads.
```

The local-vs-induction pairwise separability result is mixed:

```text
hetero4 improves local-vs-induction role separation over uniform4
(TV distance 0.528 vs 0.398), but uniform2 also has high role separation
(TV distance 0.511).
```

This should not be described as full ontology-level modularity. It compares two
roles, local copy and induction copy, across a small number of ordinary head
slots. A full modularity claim needs many roles/subroles and a clustering or
partition analysis over the resulting head-role matrix.

This means the project should keep two questions separate:

```text
Q1: Does structure make role-to-head assignment stable?  Current toy answer: yes.
Q2: Does structure make two measured roles separate across heads? Current toy
    answer: sometimes for local-vs-induction, but not automatically and not
    uniquely because of heterogeneity.
Q3: Does structure produce ontology-level functional modularity across many
    related roles/subroles? Current answer: not tested yet.
```

### Latest v2 Term: Structural Role Affinity

The cleanest term for the user's core intended claim is:

```text
structural role affinity
```

Definition:

```text
given different-shaped attention heads, a role reliably prefers a particular
structural head type, such as the 64-dim head, across seeds and layout
permutations.
```

The first ordinary-head role-ontology experiment supports this strongly:

```text
local-copy and KV-lookup subroles choose the 64-dim structural type in 80/80
one-64 heterogeneous cases.
```

Induction subroles do not simply follow the 64-dim type:

```text
induction_short -> 64 in 9/20, 32 in 8/20, 16 in 3/20
induction_long  -> 64 in 2/20, 32 in 12/20, 16 in 6/20
```

This supports a sharper claim than "modularity":

```text
different-shaped attention heads bias which functional roles attach to which
head types.
```

The same experiment partially supports ontology-level modularity but does not
settle it:

```text
hetero4_64second family gap 0.607, ARI 0.889
uniform4 family gap 0.511, ARI 0.586
uniform2 family gap 0.653, ARI 1.000
```

So functional modularity remains a controlled question against fewer/wider
uniform-head baselines.

## What Must Be Reframed

SwitchHead findings should not be presented as if they were ordinary
attention-head findings.

Correct label:

```text
SwitchHead attention-expert side case
```

Incorrect label:

```text
attention-head proof
```

Branch tower experiments should also be treated as secondary architecture
controls, not the main evidence for head-level claims.

## Immediate Next Experiment Direction

The next experiment should stay on ordinary attention heads.

Recommended next question:

```text
Does heterogeneous attention-head dimension produce functional modularity across
ordinary heads when two roles compete in the same model?
```

Use the existing local-vs-induction competition task, but keep the unit as
ordinary attention head.

### Proposed Conditions

Matched total attention dimension:

```text
uniform4:          [32, 32, 32, 32]
hetero4:           [16, 16, 32, 64]
hetero4_64first:   [64, 16, 16, 32]
hetero4_64middle:  [16, 64, 16, 32]
two_large:         [16, 16, 48, 48] or [32, 32, 64, 64] with matched budget
uniform2:          [64, 64]
```

Seeds:

```text
at least 10 if cheap; otherwise 5 first, then expand
```

### Metrics

Task metrics:

```text
local accuracy
induction accuracy
local loss
induction loss
```

Head-level causal metrics:

```text
single-head ablation loss delta for local role
single-head ablation loss delta for induction role
local top head
induction top head
same_top_head
same_top_head_dim
role_distribution_distance
top_specialization per role
```

Cross-seed stability metrics:

```text
does the top local role follow the same structural head slot?
does the top induction role follow the same structural head slot?
does moving the 64-dim head move the role?
does heterogeneity reduce top-head variance across seeds?
```

Pairwise separability metrics:

```text
same_top_head lower is more pairwise separated
role_distribution_distance higher is more separated
local top specialization high and induction top specialization high, with
different top heads, is stronger pairwise separability
```

Full functional modularity metrics require a broader role ontology and should
test whether related subroles cluster together while unrelated roles separate.

### Decision Criteria

Evidence for attention-head specialization:

```text
one role repeatedly follows the 64-dim head across seeds and layout
permutations.
```

Evidence for local-vs-induction pairwise separability:

```text
local and induction reliably choose different heads, with high
role_distribution_distance and low same_top_head.
```

Evidence for ontology-level functional modularity:

```text
many roles/subroles form stable, interpretable clusters over head slots, with
related subroles sharing groups and unrelated roles separating.
```

Possible outcomes:

1. Heterogeneity gives specialization but not pairwise separability.
2. Heterogeneity gives both specialization and pairwise separability.
3. Heterogeneity only gives capacity dominance.
4. Uniform controls already separate the measured roles, weakening the
   heterogeneity-specific separability claim.
5. Broader role ontology does or does not support full functional modularity.

All four outcomes are informative, but they must be reported as ordinary
attention-head results.

## Process Rule Going Forward

Before any autonomous work:

```text
state the unit of analysis;
state the claim being tested;
state whether the experiment is primary or secondary;
ask the user if the unit changes.
```

If unsure:

```text
stop and ask.
```

Do not continue into related but different architectures without explicit
approval.
