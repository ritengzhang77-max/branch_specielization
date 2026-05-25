# Recent Chat

Date: 2026-05-24

## What We Clarified

The recent experiments are still toy experiments, not real large-language-model
experiments.

Toy experiment means:

```text
small transformer trained from scratch
synthetic integer-token sequences
roles are known because we generate the data
ordinary attention heads are ablated after training
```

Real LLM experiment means:

```text
pretrained model weights, e.g. Pythia or MultiBERTs
real tokenizer and learned embeddings
natural or semi-natural text/probe data
roles must be inferred from probes and causal ablations
```

The toy setting is useful because it gives clean labels and controlled
architectures. The real setting is harder, but necessary before claiming the
result holds in actual pretrained language models.

## Larger-Head V3 Control

We ran the decisive next toy step:

```text
same Toy Ontology v3
more ordinary attention-head slots
matched total attention dimension
same metric system
uniform baseline vs non-uniform head dimensions
```

The purpose was to test whether the mixed modularity result was caused by having
too few head slots in the earlier 8-head-per-layer setup.

## Important Training Detail

The 16-head-per-layer control failed the accuracy gate at total attention
dimension 384, so it was not used for the final claim.

The viable larger-head control was:

```text
12 heads per layer
2 layers
24 ordinary head slots total
total attention dimension = 384
5 seeds
2250 training steps
```

The 2250-step schedule was used because `uniform12` seed 4 solved around step
2250 but degraded by step 3000. The shorter schedule gave the clean, fair
baseline.

## Main Result

| Config | Accuracy | Specialization | Ontology Alignment | Sep-Adjusted Clusterability |
|---|---:|---:|---:|---:|
| `uniform12` | 1.0000 | 0.585 | 0.179 | 0.432 |
| `hetero12_unique_moderate` | 1.0000 | 0.524 | 0.201 | 0.419 |
| `hetero12_unique_spread` | 1.0000 | 0.643 | 0.116 | 0.389 |

## Interpretation

The larger-head control did not rescue the functional-modularity claim.

- `hetero12_unique_spread` improved specialization, but worsened both modularity
  metrics.
- `hetero12_unique_moderate` slightly improved ontology alignment, but worsened
  specialization and separation-adjusted clusterability.
- More head slots do not make heterogeneous head dimensions reliably beat a
  matched uniform baseline on functional modularity.

Current safest framing:

```text
Non-uniform attention-head dimensions can create structural role affinity and
can increase role-level specialization when the heterogeneity is sufficiently
spread. Functional modularity is separate and harder: the current evidence is
positive in some toy settings but not robust across larger-head controls.
```

## Current Project Direction

The next best step is real-model validation on ordinary pretrained heads, not
more small toy variations.

Priority probes:

```text
Pythia local-copy probes
repeat-match probes
induction-like probes
careful causal head ablations
same specialization metrics where possible
```

## Artifacts From This Chat

Main document:

```text
doc/experiments/phase3/phase3_toy_role_ontology_v3_larger_head_control.md
```

Clean result root:

```text
results/phase3_toy_role_ontology_v3_large_heads_12h_2250_20260524
```

Latest pushed commit:

```text
13aa902 Add toy ontology v3 larger-head control
```
