# Research Questions and Project Phases

## Core Framing

The cleanest framing is:

> Does **structural branch design** in transformer-style architectures induce
> **stable functional specialization** or **functional modularity** across random
> seeds?

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

When a head is specialized for a function, is it also functionally separable from
other heads?

Operational tests:

- specialization score `S(h, t)`;
- graph modularity / clusterability of head-head interaction graphs;
- Csordas-style mask IoU / IoMin;
- conditional mutual information between head outputs, if estimation is reliable;
- causal path-patching separability.

This is the conceptual hazard in the project: specialization without modularity is
possible and likely in transformer residual-stream architectures.

### RQ4: Do Explicit Branch Architectures Produce Cleaner Specialization Than Vanilla MHA?

Do MoE experts, routed attention experts, or SwitchHead-style attention experts
show more stable specialization than ordinary attention heads?

Operational tests:

- router entropy;
- expert co-activation;
- domain and vocabulary specialization;
- expert ablation effects;
- cross-seed or cross-checkpoint stability where seeds are unavailable.

This tests whether "branch" is a more natural unit for experts than for vanilla
attention heads.

### RQ5: Does Structural Heterogeneity Cause More Stable Functional Specialization?

Does an intervention such as heterogeneous per-head dimension make functional
roles more consistent across seeds?

Candidate intervention:

```text
uniform baseline:       [64, 64, 64, 64, 64, 64, 64, 64]
heterogeneous variant:  [32, 32, 64, 64, 128, 128, 256, 256]
```

Primary outcomes:

- lower cross-seed variance of `S(h, t)`;
- larger Hungarian-matched similarity gap;
- higher clusterability/modularity;
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

Train uniform and heterogeneous-head-dimension models across multiple seeds.

Deliverables:

- matched-loss training curves;
- specialization/stability comparison;
- modularity comparison;
- control for uniform wider/fewer heads.

Decision criterion:

- If heterogeneous head dimensions do not reduce cross-seed variance in pilot
  experiments by a practically meaningful amount, abandon or reframe the
  architectural-intervention claim.

### Phase 4: Mechanistic Interpretation

If Phase 3 shows a signal, explain it mechanistically.

Deliverables:

- path-patching circuits for the key tasks;
- evidence about whether small heads consistently host local/positional roles and
  large heads host global/induction/name-moving roles;
- comparison of specialization and modularity axes.

