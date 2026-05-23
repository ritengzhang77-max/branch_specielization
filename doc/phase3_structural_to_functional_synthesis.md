# Phase 3 Synthesis: Does Structure Become Function?

Date: 2026-05-23

## Question

The project framing should be:

```text
Does structural branch design or structural heterogeneity lead to stable
functional specialization or functional modularity?
```

Use the terms carefully:

- **Structural heterogeneity**: components differ before training, such as mixed
  head dimensions.
- **Structural modularity**: components are separated or weakly coupled by
  architecture, such as separate branches or routed towers.
- **Functional specialization**: one component disproportionately supports a
  role or task.
- **Functional modularity**: different roles are causally separable across
  components, not merely identifiable by probes.

The current evidence does not support a simple claim that structure alone causes
functional modularity. It supports a more precise conditional claim:

```text
Structural heterogeneity can create stable functional specialization slots.
Structural branch/routing design can support functional modularity, but reliable
functional modularity appears only when the training signal is role-informative
for long enough to reshape branch computations.
```

## Evidence Summary

| Test | Main result | Interpretation |
|---|---|---|
| Heterogeneous head dimensions on toy key-value recall | `hetero4 [16,16,32,64]` put the top causal role in the 64-dim head in 5/5 seeds; top specialization `0.9741` vs `0.4414` for `uniform4`. | Structural heterogeneity can break permutation symmetry and make a stable functional slot. |
| Heterogeneous head dimensions on toy induction | 64-dim head was the top causal induction head in 5/5 seeds; `hetero4` top specialization `0.9830` vs `0.5796` for `uniform4`. | The stable-slot effect is not limited to key-value recall. |
| Local-vs-induction competition layouts | Across four placements of the 64-dim head, local used the 64-dim slot in 19/20 models, while induction used it in 10/20. | Head dimension is a capacity attractor, not a fixed semantic role label. Task pressure and layout decide role allocation. |
| Local-weight sweep | Induction used the 64-dim head in 19/20 models when local pressure was absent/tiny, but only 5/20 models at local weight `0.25`. | Functional role allocation is pressure-sensitive, not determined by structure alone. |
| Two-attractor layouts | Two 48-dim heads and `uniform2 [64,64]` did not produce clean local/induction separation; same top slot was `0.80` in both two-48 and uniform2 controls. | More attractive capacity slots do not automatically create functional modularity. |
| Separate branch towers without routing | `branch_sum` solved the task but local and induction branch ablations remained entangled; same top branch `0.60` to `0.80` across related runs. | Structural separation alone is not enough. |
| Oracle routing | Local routed to branch 0 and induction to branch 1 produced routed role match `1.00` and branch distance `1.00`. | Correct routing is sufficient for branch-level functional modularity in the toy setup. |
| Unconstrained learned routing | Learned position/token routers solved the task but had same top branch `1.00` and routed role match `0.00`. | Learned gates do not spontaneously discover the role split in this setup. |
| Weak role-supervised routing | Weak token router at supervision weight `0.05` reached same top branch `0.00`, routed role match `1.00`, branch distance `0.8957`. | A small role-informative signal is sufficient for near-oracle functional modularity. |
| Weak-router supervision sweep | Gate routed match reached `1.00` by weight `0.02`, but causal routed role match reached `1.00` only at weight `0.05`. | Gate compliance is not the same as causal modularity. |
| Unlabeled entropy/balance regularization | Entropy and load balancing changed gate statistics but did not reliably produce causal role separation; best unlabeled routed role match was `0.40`. | Sharp or balanced gates are not sufficient for functional modularity. |
| Bottlenecked branches | 16-dim branches still failed under unlabeled regularizers, while oracle routing still achieved routed role match `1.00`. | Capacity bottlenecks alone are not the missing ingredient. |
| Conflict-heavy bidirectional lookup | Even direct predecessor-vs-successor role conflict did not make unlabeled routers split roles; weak labels and oracle routing both reached `1.00`. | Task conflict alone did not rescue spontaneous modularity in this pilot. |
| Annealed weak labels | Labels through step 400 failed despite solved task and role-aligned gates; labels through step 800 partially persisted; labels through step 1200 persisted more strongly. | Role-informative pressure must last beyond early symmetry breaking. |
| Router trajectory | At step 400, gate match was `1.00` but causal routed role match was `0.00`; with continued labels, causal routed role match reached `1.00` by step 600. | Gate alignment appears before causal branch separation. |
| Pythia repeat/copy follow-up | Pythia heads show cross-seed functional role stability after alignment, and causal transfer strengthens through training. | Real transformers support the role-stability part, but do not by themselves establish branch modularity. |

## Current Answer

### Does structural heterogeneity lead to functional specialization?

Yes, in the toy interventions tested so far. Mixed head dimensions repeatedly
create stable high-capacity slots. The strongest evidence is that the role
follows the 64-dim head when its position is moved, rather than staying at a raw
head index.

This should be stated as:

```text
Structural heterogeneity can stabilize function-to-slot mappings.
```

It should not be stated as:

```text
Large heads always learn global/induction roles, or small heads always learn
local roles.
```

The competition and weight-sweep results show that role allocation is mediated by
task pressure and optimization basin.

### Does structural modularity lead to functional modularity?

Not by itself in the current toy evidence. Separate branch towers and generic
learned routers can solve the task while remaining functionally entangled.
Adding more capacity slots, bottlenecking branches, entropy regularization, load
balancing, or direct role conflict did not reliably produce role-aligned causal
branch separation.

This should be stated as:

```text
Structural modularity provides a substrate for functional modularity, but does
not guarantee it.
```

### What does produce functional modularity?

The reliable mechanism observed so far is role-informative routing pressure:

- oracle routing is a clean upper bound;
- weak scored-position routing labels are sufficient;
- late removal of weak labels can preserve a weakened split;
- early removal fails even when the task is already solved and the gate is
  already role-aligned.

The mechanism is therefore not just "the router points to different branches."
The branch computations need time under role-aligned routing before causal
modularity consolidates.

## Metrics That Matter Most

- `top_specialization`: concentration of a role's ablation effect in the top
  component. This measures functional specialization, not modularity.
- `same_top_branch`: whether two roles depend most on the same branch. Lower is
  better evidence for modularity when the roles are intended to separate.
- `routed_role_match`: whether local depends most on branch 0 and induction on
  branch 1 in the labeled routing setup. This is a branch-level modularity
  readout.
- `branch_distribution_distance`: distance between the local and induction
  branch-ablation distributions. This measures separation strength, not just
  top-branch identity.
- `gate_routed_role_match` and `gate_distribution_distance`: router behavior.
  These are necessary diagnostics, but the supervision sweep and trajectory show
  that they can overstate modularity if reported without branch ablations.
- `own_top_excess`, `aligned_transfer`, and `aligned_minus_same`: Phase 1
  cross-seed role-stability metrics. These measure functional specialization and
  relabeled role universality, not branch modularity.

## Paper-Level Framing

The clean paper claim is not:

```text
Structural modularity causes functional modularity.
```

The clean paper claim is:

```text
Structural design changes which functional specialization slots are stable
across seeds. Functional modularity is a stricter causal property: it emerges
reliably in these toy branch models only when role-informative routing pressure
is supplied for long enough to consolidate branch computations.
```

This framing makes positive and negative results both useful:

- positive specialization result: heterogeneity stabilizes slots;
- negative spontaneous-modularity result: branches/gates/regularizers are
  insufficient without role information;
- positive conditional-modularity result: weak role routing creates modularity;
- developmental result: gate alignment precedes causal branch separation.

## Next Narrow Experiment

The existing trajectory has a coarse jump between step 400 and step 600:

```text
step 400: gate match 1.00, causal routed role match 0.00
step 600: gate match 1.00, causal routed role match 1.00
```

The next smallest useful experiment is a denser trajectory in that window for
the `end800` weak-token-router condition:

```text
steps: 400, 450, 500, 550, 600, 650, 700, 750, 800
```

Goal:

```text
locate the causal consolidation window after gate alignment but before full
branch modularity.
```

This is more informative than another broad entropy/balance sweep because the
mechanism has already narrowed to timing under role-informative routing pressure.
