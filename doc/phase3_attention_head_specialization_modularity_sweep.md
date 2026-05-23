# Phase 3 Attention-Head Sweep: Specialization vs Modularity

Date: 2026-05-23

## Scope Lock

This memo is about ordinary attention heads only.

The unit of analysis is:

```text
ordinary attention head slot = (layer, head)
```

This is not a SwitchHead expert, MoE expert, routed branch, or separate branch
tower result.

## Formal Distinction

Let `r` be a role/task, such as local copy or induction copy. Let `h` be an
ordinary attention head slot. Define the causal role score:

```text
S_r(h) = max(loss_r(model with h ablated) - loss_r(base model), 0)
```

Normalize scores into a distribution over heads:

```text
p_r(h) = S_r(h) / sum_h S_r(h)
```

### Functional Specialization

Specialization asks:

```text
For one role r, is p_r concentrated on a small number of heads?
```

In this script, `top_specialization` is the normalized causal mass of the top
head within its layer. A higher number means the role depends much more on one
head than on peer heads in that layer.

### Pairwise Functional Separability

The experiment in this memo does not measure full ontology-level modularity. It
measures a narrower pairwise proxy:

```text
For two roles r1 and r2, are p_r1 and p_r2 separated across heads?
```

This sweep adds two explicit head-level separability metrics over flattened
`(layer, head)` slots:

```text
role_distribution_tv_distance = 0.5 * sum_h |p_local(h) - p_induction(h)|
role_distribution_overlap     = sum_h min(p_local(h), p_induction(h))
```

Interpretation:

```text
TV distance = 0: local and induction use the same head distribution.
TV distance = 1: local and induction use disjoint head distributions.
```

It also reports:

```text
same_top_slot_rate = fraction of seeds where local and induction have the same
                     top causal head slot.
```

This is useful as a first test, but it is not the final modularity definition.
Full functional modularity would require a role ontology with many functions,
subfunctions, and task variants, then ask whether related subfunctions cluster
inside the same heads or head groups while unrelated functions separate.

### Why The Difference Matters

Specialization and pairwise separability are related but not equivalent.

High specialization without modularity:

```text
local p(h)     = [0.90, 0.10, 0.00, 0.00]
induction p(h) = [0.90, 0.10, 0.00, 0.00]
```

Both roles are concentrated, but they use the same head.

Modularity with weaker specialization:

```text
local p(h)     = [0.45, 0.45, 0.05, 0.05]
induction p(h) = [0.05, 0.05, 0.45, 0.45]
```

Neither role is one-head-clean, but the two roles are separated.

For this project, the terminology should be:

```text
specialization          = "does a role concentrate into a head?"
pairwise separability   = "do two measured roles separate across heads?"
functional modularity   = "do many related roles/subroles form a stable,
                           interpretable partition across heads?"
```

So the local-vs-induction experiment can support or weaken a modularity story,
but it cannot by itself prove full modularity.

## Experiment

Task:

```text
local copy + induction copy in the same model
```

Local region:

```text
[x, SEP, x]
```

At `SEP`, predict the immediately previous token `x`.

Induction region:

```text
[y_1, ..., y_16, y_1, ..., y_16]
```

At the second occurrence of `y_i`, predict `y_{i+1}`.

Command:

```bash
python3 -u scripts/toy_competition_head_dim_intervention.py \
  --configs uniform4 hetero4 hetero4_64first hetero4_64second \
            hetero4_64third hetero4_two48_center hetero4_two48_skip \
            hetero4_two48_front uniform2 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --steps 1600 \
  --eval-examples 1024 \
  --device cuda \
  --output-dir results/phase3_toy_competition_head_dim_modularity_sweep_20260523
```

Artifacts:

```text
results/phase3_toy_competition_head_dim_modularity_sweep_20260523/config_summary.csv
results/phase3_toy_competition_head_dim_modularity_sweep_20260523/model_summary.csv
results/phase3_toy_competition_head_dim_modularity_sweep_20260523/head_role_scores.csv
results/phase3_toy_competition_head_dim_modularity_sweep_20260523/pair_stability.csv
results/phase3_toy_competition_head_dim_modularity_sweep_20260523/summary.json
```

## Results

All models learned the task well enough for the head-level analysis. Every
config had minimum held-out local and induction accuracy above `0.99` across the
10 seeds.

| Config | Local acc. | Induction acc. | Local top spec. | Induction top spec. | Same top slot | Pairwise TV | Local top dims | Induction top dims |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `uniform4` | 1.0000 | 0.9994 | 0.562 | 0.579 | 0.30 | 0.398 | `{"32": 10}` | `{"32": 10}` |
| `hetero4` | 1.0000 | 1.0000 | 0.925 | 0.819 | 0.20 | 0.528 | `{"64": 10}` | `{"16": 5, "64": 5}` |
| `hetero4_64first` | 1.0000 | 0.9999 | 0.933 | 0.763 | 0.30 | 0.490 | `{"64": 10}` | `{"16": 6, "64": 4}` |
| `hetero4_64second` | 1.0000 | 1.0000 | 0.945 | 0.780 | 0.30 | 0.475 | `{"64": 10}` | `{"16": 6, "64": 4}` |
| `hetero4_64third` | 1.0000 | 1.0000 | 0.957 | 0.763 | 0.40 | 0.414 | `{"64": 10}` | `{"16": 4, "64": 6}` |
| `hetero4_two48_center` | 1.0000 | 1.0000 | 0.593 | 0.618 | 0.30 | 0.356 | `{"16": 2, "48": 8}` | `{"16": 3, "48": 7}` |
| `hetero4_two48_skip` | 1.0000 | 0.9987 | 0.630 | 0.600 | 0.20 | 0.419 | `{"16": 1, "48": 9}` | `{"16": 2, "48": 8}` |
| `hetero4_two48_front` | 1.0000 | 1.0000 | 0.652 | 0.647 | 0.50 | 0.380 | `{"16": 1, "48": 9}` | `{"16": 1, "48": 9}` |
| `uniform2` | 1.0000 | 0.9994 | 0.639 | 0.697 | 0.20 | 0.511 | `{"64": 10}` | `{"64": 10}` |

## Interpretation

### Q1: Does heterogeneous head dimension create stable specialization?

Yes, strongly for the local role in the one-64-head layouts.

Across all four one-64 heterogeneous layouts:

```text
local top head dimension = 64 in 40/40 trained models
```

The top local slot also follows the 64-dim head's structural position:

```text
hetero4            [16,16,32,64] -> local top slot L1H3 in 10/10
hetero4_64first    [64,16,16,32] -> local top slot L1H0 in 10/10
hetero4_64second   [16,64,16,32] -> local top slot L1H1 in 10/10
hetero4_64third    [16,32,64,16] -> local top slot L1H2 in 10/10
```

This is the cleanest ordinary-attention-head evidence so far:

```text
moving the large head moves the local causal role.
```

Compared with `uniform4`, one-64 heterogeneous layouts also greatly increase
local specialization:

```text
uniform4 local top specialization:       0.562
one-64 hetero local top specialization:  0.925 to 0.957
```

Induction specialization also increases relative to `uniform4`, but the top
induction role does not always choose the 64-dim head. It splits between 16-dim
and 64-dim heads depending on seed and layout.

### Q2: Does heterogeneous head dimension create functional modularity?

This experiment only answers a narrower question:

```text
Does heterogeneous head dimension increase pairwise causal separability between
the local-copy role and the induction-copy role?
```

For that narrower question, the evidence is positive but mixed.

The original one-64 `hetero4` layout improves role separation relative to
`uniform4`:

```text
uniform4 TV distance: 0.398, same top slot: 0.30
hetero4  TV distance: 0.528, same top slot: 0.20
```

This means local and induction are more separated in causal head dependence under
`hetero4` than under the equal-four-head baseline.

But this is not a clean universal modularity result:

```text
hetero4_64first  TV distance: 0.490
hetero4_64second TV distance: 0.475
hetero4_64third  TV distance: 0.414
```

The effect depends on layout.

The strongest control is `uniform2`:

```text
uniform2 TV distance: 0.511, same top slot: 0.20
```

`uniform2` has no within-layer head-dimension heterogeneity, yet it gets
role-separation comparable to `hetero4`. So even the pairwise separability claim
cannot be:

```text
heterogeneity alone is necessary for local-vs-induction head separability.
```

The better current claim is:

```text
heterogeneous head dimensions strongly stabilize which structural head slot
carries a role; pairwise functional separability can appear, but it is a
separate outcome that also depends on capacity layout and task pressure.
```

To claim full functional modularity, the project needs a broader role ontology:
for example previous-token/local-copy variants, induction variants, key-value
lookup variants, suppression/distractor roles, positional roles, and possibly
syntax/name-mover-style probes in real models.

## Current Bottom Line

The original project direction is still alive and stronger on the specialization
axis:

```text
structural head-dimension heterogeneity -> stable functional specialization slots
```

The modularity axis is not yet proven:

```text
structural head-dimension heterogeneity -> functional modularity
```

is only weakly and indirectly supported. The toy evidence says heterogeneity can
improve local-vs-induction pairwise separation in some layouts, but a
non-heterogeneous wide-head control can also separate the two roles. This is not
enough to claim ontology-level modularity.

This is not a failure. It means the paper should keep the two questions separate:

1. Does structure make role-to-head assignment stable?
2. Does structure make different roles causally separable across heads?
3. Do whole families of related roles/subroles form stable head groups?

The answer so far is:

```text
Q1: yes, strongly in toy ordinary-head models.
Q2: sometimes for local-vs-induction, but not automatically and not uniquely
    because of heterogeneity.
Q3: not answered yet.
```

## Next Step

Stay with ordinary attention heads and strengthen the modularity test by moving
from pairwise separability to a small role ontology.

Recommended next experiments:

1. Add global specialization metrics over flattened `(layer, head)` slots:
   `max_h p_r(h)`, entropy, and effective number of heads. This avoids relying
   only on the current layer-local `top_specialization`.
2. Add role-selectivity metrics for top heads:
   `S_local(h) / (S_induction(h) + eps)` and
   `S_induction(h) / (S_local(h) + eps)`.
3. Run more task pairs, not just local-vs-induction:
   local copy vs key-value recall, previous-token vs induction, and local copy
   vs suppression-style distractor.
4. Define a role/subrole ontology and evaluate clustering:
   previous-token/local-copy variants, induction variants, key-value lookup
   variants, positional/BOS-style probes, and distractor/suppression probes.
5. Add harder competition where both roles must use similar token positions or
   the same query token, while still using ordinary attention heads.
6. Validate the head-level metrics on real pretrained models by measuring local
   copy and induction-style roles in Pythia/MultiBERT heads.

Do not switch the unit of analysis unless explicitly approved.
