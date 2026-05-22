# Research Questions and Project Phases

## Core Framing

The cleanest framing is:

> Does **structural branch design** in transformer-style architectures induce
> **stable functional specialization** or **functional modularity** across random
> seeds?

This is an empirical question, not a design desideratum. The project should not
assume that functions ought to be separated, or that separation is always better.
The goal is to measure which outcomes actually occur:

- no reliable specialization;
- specialization without modularity;
- modularity without role specialization;
- specialization and modularity together.

Positive, negative, and mixed outcomes are all potentially informative.

Use the terms as follows:

- **Structural heterogeneity**: architectural differences among components before
  training, such as heterogeneous head dimensions, routed experts, attention-vs-SSM
  branches, or MoE attention experts.
- **Structural modularity**: weak coupling by construction or by learned weights,
  such as sparse routing, separate branches, or low cross-cluster edge weight.
- **Functional specialization**: one component disproportionately supports a
  task/function, such as induction copying, previous-token behavior, BOS attention,
  syntactic relations, or name moving.
- **Functional modularity**: specialized components are also functionally separable,
  meaning their computations are not strongly entangled with other components.

Avoid the phrase **structural specialization** as the main term. A component can be
structurally different or structurally isolated, but it is only specialized after
we observe what function it performs.

## Main Research Questions

### RQ1: Do Attention Heads Have Stable Functional Roles Across Seeds?

Given the same architecture and training recipe, do independently trained seeds
learn the same head functions?

Operational tests:

- raw head-index attention-pattern similarity;
- Hungarian-matched attention-pattern similarity;
- head-level specialization score `S(h, t)`;
- cross-seed consistency `C(h, t)`.

This is the Stage 1 baseline and should be tested first on Pythia and MultiBERTs.

### RQ2: Is Stability a Head-Index Property or a Relabeled-Role Property?

If the same function appears across seeds, does it appear in the same `(layer,
head)` slot, or only after optimal relabeling?

Operational tests:

- raw similarity vs Hungarian-matched similarity;
- matched similarity gap over random permutation baseline;
- optional Git-Re-Basin-style weight alignment as a robustness check.

This question separates strong universality from weak universality.

### RQ3: Does Specialization Coincide With Modularity?

When a head or branch is specialized for a function, is it also functionally
separable from other heads or branches? Or do multiple functions cohabit the same
component despite specialization?

Operational tests:

- specialization score `S(h, t)`;
- graph modularity / clusterability of head-head interaction graphs;
- Csordas-style mask IoU / IoMin;
- conditional mutual information between head outputs, if estimation is reliable;
- causal path-patching separability.

This is a core measurement axis, not a pass/fail criterion. Specialization
without modularity is possible and scientifically important, especially in
transformer residual-stream architectures.

### RQ4: Do Explicit Branch Architectures Change Specialization or Modularity Relative to Vanilla MHA?

Do MoE experts, routed attention experts, or SwitchHead-style attention experts
show different specialization or modularity patterns than ordinary attention
heads?

Operational tests:

- router entropy;
- expert co-activation;
- domain and vocabulary specialization;
- expert ablation effects;
- cross-seed or cross-checkpoint stability where seeds are unavailable.

This tests whether "branch" is a more natural unit for some functions than
ordinary attention heads, and whether branch structure changes specialization,
modularity, both, or neither.

### RQ5: Does Structural Heterogeneity Change Functional Specialization or Modularity?

Does an intervention such as heterogeneous per-head dimension make functional
roles more consistent across seeds? Separately, does it affect functional
modularity, or only specialization?

Candidate intervention:

```text
uniform baseline:       [64, 64, 64, 64, 64, 64, 64, 64]
heterogeneous variant:  [32, 32, 64, 64, 128, 128, 256, 256]
```

Primary outcomes:

- lower cross-seed variance of `S(h, t)`;
- larger Hungarian-matched similarity gap;
- change in clusterability/modularity, without assuming the direction in advance;
- stronger reuse of head-dimension slots for the same function.

Required controls:

- matched validation loss;
- matched total Q/K/V dimension or parameter budget;
- uniform-but-wider/fewer-head baseline to address the Differential Transformer
  follow-up warning.

## Phases

### Phase 0: Calibration

Reproduce a minimal attention-head stability result on Pythia.

Deliverables:

- fixed probe text set;
- attention extraction script;
- raw cross-seed similarity matrices;
- Hungarian-matched similarity report;
- layer-wise stability summary.

Decision criterion:

- If even a two-seed Pythia smoke test cannot produce stable attention matrices
  and sensible matched-vs-random gaps, debug infrastructure before expanding.

### Phase 1: Existing Multi-Seed Models

Analyze Pythia and MultiBERTs without retraining.

Deliverables:

- full seed-pair similarity tables;
- head-role probe results;
- ablation/path-patching results for a small task set;
- stability heatmaps.

Decision criterion:

- If cross-seed stability is already very high, focus on unstable mid-layer heads.
- If it is low but improves after matching, frame as weak universality.
- If it is low even after matching, the project becomes a negative result about
  head-role non-universality.

### Phase 2: Explicit Branch Architectures

Analyze OLMoE, SwitchHead, and other routed/expert architectures.

Deliverables:

- expert specialization and routing metrics;
- comparison against vanilla MHA head stability;
- within-model or cross-checkpoint proxy analysis where multi-seed checkpoints do
  not exist.

### Phase 3: Architectural Intervention

Train uniform, heterogeneous-head-dimension, and explicitly routed/separated
toy or small-scale models across multiple seeds.

Deliverables:

- matched-loss training curves;
- specialization/stability comparison;
- modularity comparison, including the possibility of specialization without
  modularity;
- control for uniform wider/fewer heads.

Decision criterion:

- If heterogeneous head dimensions or routing do not produce a practically
  meaningful change in specialization, modularity, or their dissociation in
  pilot experiments, reframe the architectural-intervention claim.

Current toy-pilot status as of 2026-05-22:

- Heterogeneous head dimensions have produced stable capacity-attractor slots for
  functional specialization in toy key-value recall and induction-style tasks.
- In the local-vs-induction competition task, heterogeneous capacity affects
  which role occupies which slot, but does not by itself guarantee role-specific
  functional modularity.
- Separate branch towers without routing solve the task but remain causally
  entangled across roles.
- Oracle routing can produce clean branch-level functional modularity.
- Unconstrained learned routing, tested with position-based and token-based
  gates, solved the task but did not recover the oracle role split.

The next Phase 3 question is therefore:

```text
What routing signal, regularizer, or task pressure is sufficient for learned
functional modularity to emerge, if any?
```

### Phase 4: Mechanistic Interpretation

If Phase 3 shows a signal, explain it mechanistically.

Deliverables:

- path-patching circuits for the key tasks;
- evidence about whether any structural slot consistently hosts any role, without
  assuming that small heads should host local roles or large heads should host
  global/induction roles;
- comparison of specialization and modularity axes.
