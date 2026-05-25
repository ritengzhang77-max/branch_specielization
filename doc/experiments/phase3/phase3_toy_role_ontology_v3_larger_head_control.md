# Phase 3: Toy Ontology v3 Larger-Head Control

Date: 2026-05-24

## Toy Versus Real LLM

These runs are toy experiments, not real large-language-model experiments.

Toy setting:

```text
small transformer trained from scratch
synthetic integer-token sequences
roles are known because we generated them
ordinary attention heads are ablated after training
```

Real LLM setting:

```text
pretrained model weights, e.g. Pythia or MultiBERTs
real tokenizer and learned embeddings
natural or semi-natural text/probe data
roles must be inferred from probes and causal ablations
```

The toy setting is useful because it gives clean labels and controlled
architectures. The real setting is harder but necessary before making a claim
about actual pretrained language models.

## Question

The previous v3 robustness sweep used:

```text
8 heads per layer x 2 layers = 16 ordinary head slots
total attention dimension per layer = 384
```

The concern was:

```text
Maybe functional modularity looks mixed only because 16 slots are too few for a
sparse role-family geometry.
```

This control increases the head count while keeping total attention dimension
matched within the comparison.

## Configs

The first attempted larger control used 16 heads per layer:

| Config | Head Dims | Result |
|---|---|---|
| `uniform16` | `[24 x 16]` | not a valid baseline at this schedule; some roles failed |
| `hetero16_unique_moderate` | 16 unique dims summing to 384 | preset added, not used for final claim |
| `hetero16_unique_extreme` | 16 unique dims summing to 384 | preset added, not used for final claim |

`uniform16` made each head too small at this total capacity and failed the
accuracy gate in calibration, so the fair larger-head control used 12 heads per
layer:

| Config | Head Dims | Total Dim | Head Slots |
|---|---|---:|---:|
| `uniform12` | `[32 x 12]` | 384 | 24 |
| `hetero12_unique_moderate` | `[26,27,28,29,30,31,32,33,34,35,36,43]` | 384 | 24 |
| `hetero12_unique_spread` | `[8,12,16,20,24,28,32,36,40,44,48,76]` | 384 | 24 |

All non-uniform 12-head configs use all-distinct head dimensions.

## Training Schedule Note

A 3000-step 12-head run exposed an optimizer/schedule artifact:

```text
uniform12 seed 4 solved around step 2250 but degraded by step 3000
```

The fair final run therefore uses:

```text
2250 steps
same schedule for all configs
5 seeds
same v3 role ontology
same metrics
```

Clean result root:

```text
results/phase3_toy_role_ontology_v3_large_heads_12h_2250_20260524
```

Raw diagnostic roots:

```text
results/phase3_toy_role_ontology_v3_large_heads_calibration_5000_20260524
results/phase3_toy_role_ontology_v3_large_heads_calibration_12h_3000_20260524
results/phase3_toy_role_ontology_v3_large_heads_12h_3000_20260524
results/phase3_toy_role_ontology_v3_large_heads_12h_uniform12_seed4_2250_20260524
```

## Main Result

Accuracy is first because specialization/modularity only matters after the task
is solved.

| Config | Mean Acc | Min Acc | Specialization | Delta Spec / Wins | Eff Heads | Delta Eff / Wins | Ont Align | Delta Align / Wins | Shuffle p | Delta p / Wins | Sep-Adjusted | Delta Sep / Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform12` | 1.0000 | 0.9995 | 0.585 | +0.000 / - | 5.721 | +0.000 / - | 0.179 | +0.000 / - | 0.038 | +0.000 / - | 0.432 | +0.000 / - |
| `hetero12_unique_moderate` | 1.0000 | 0.9992 | 0.524 | -0.061 / 2/5 | 6.685 | +0.964 / 2/5 | 0.201 | +0.022 / 3/5 | 0.016 | -0.021 / 3/5 | 0.419 | -0.013 / 3/5 |
| `hetero12_unique_spread` | 1.0000 | 0.9997 | 0.643 | +0.057 / 3/5 | 4.227 | -1.493 / 3/5 | 0.116 | -0.063 / 2/5 | 0.088 | +0.050 / 2/5 | 0.389 | -0.043 / 2/5 |

## Interpretation By Question

### 1. Accuracy

All final 12-head configs pass the accuracy gate.

### 2. Specialization

The result is mixed:

- `hetero12_unique_spread` improves specialization over `uniform12`;
- `hetero12_unique_moderate` is less specialized than `uniform12`.

So the refined claim is not:

```text
any heterogeneity increases specialization
```

The better claim is:

```text
sufficiently spread non-uniform head dimensions can increase specialization,
but mild heterogeneity need not.
```

### 3. Functional Modularity

The larger-head control does not rescue the modularity claim.

- `hetero12_unique_moderate` slightly improves ontology alignment
  (`0.201` vs `0.179`) and has lower shuffled-label p, but loses on
  specialization and separation-adjusted clusterability.
- `hetero12_unique_spread` improves specialization, but loses on ontology
  alignment and separation-adjusted clusterability.

This means the v3 modularity result remains:

```text
positive in some settings, but not robust across head-count and layout controls
```

## Current Conclusion

The larger-head control changes the paper framing:

```text
More ordinary head slots do not make heterogeneous head dimensions reliably beat
a matched uniform baseline on functional modularity.
```

The safest current scientific claim is:

```text
In controlled toy transformers, non-uniform attention-head dimensions can create
stable structural role affinity and can increase role-level specialization when
the heterogeneity is sufficiently spread. Functional modularity is a separate,
harder property: the current evidence is positive in some v3 settings but does
not survive all larger-head controls.
```

## Next Step

The next step should move away from trying to force the toy modularity claim and
return to real-model validation:

```text
ordinary pretrained heads
Pythia role probes
same specialization metrics where possible
careful causal ablation for local copy / repeat match / induction-like probes
```

Toy work should continue only if it introduces a clearly new ontology or a
better training protocol, not as more small variations of the same head-count
sweep.
