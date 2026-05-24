# Phase 3 Toy Role Ontology: Hetero2 Control

Date: 2026-05-23

## Purpose

This experiment tests the missing baseline:

```text
uniform2 [64,64] vs two-head non-uniform layouts
```

The goal is to separate:

```text
head count effect:     2 heads vs 4 heads
capacity effect:       wide heads
heterogeneity effect:  unequal head dimensions with the same number of heads
```

The unit is still ordinary attention heads.

## Setup

Model:

```text
d_model = 128
n_layers = 2
total attention head dimension = 128
ordinary attention heads
```

Role ontology:

```text
local_copy: local_a, local_b
kv_lookup:  kv_a, kv_b
induction:  induction_short, induction_long
```

Command:

```bash
python3 -u scripts/toy_role_ontology_head_dim_intervention.py \
  --configs uniform2 32,96 48,80 16,112 \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --device cuda \
  --output-dir results/phase3_toy_role_ontology_hetero2_20260523
```

Artifacts:

```text
results/phase3_toy_role_ontology_hetero2_20260523/config_summary.csv
results/phase3_toy_role_ontology_hetero2_20260523/role_summary.csv
results/phase3_toy_role_ontology_hetero2_20260523/head_role_scores.csv
results/phase3_toy_role_ontology_hetero2_20260523/role_pair_summary.csv
results/phase3_toy_role_ontology_hetero2_20260523/summary.json
```

All configs solved the task:

```text
role accuracy mean >= 0.9993
role accuracy min  >= 0.9922
```

## Results: Baseline vs Hetero2

| Question | `uniform2 [64,64]` baseline | Hetero2 result | Interpretation |
|---|---|---|---|
| Structural role affinity | local/KV top head index: H0 `16/20`, H1 `4/20`; all heads are 64-dim, so top dimension is trivial | `[32,96]`: local/KV choose 96 in `20/20`; `[48,80]`: choose 80 in `20/20`; `[16,112]`: choose 112 in `20/20` | hetero2 creates strong large-head affinity for local/KV roles |
| Specialization | spec `0.636`, effective heads `2.25` | `[32,96]`: spec `0.695`, eff `1.86`; `[48,80]`: spec `0.756`, eff `1.91`; `[16,112]`: spec `0.702`, eff `1.76` | hetero2 beats uniform2 on concentration |
| Functional modularity | family gap `0.653`, ARI `1.000` | `[32,96]`: gap `0.480`, ARI `0.889`; `[48,80]`: gap `0.601`, ARI `1.000`; `[16,112]`: gap `0.327`, ARI `0.667` | hetero2 does not beat uniform2 on modularity |

## Structural Role Affinity

The local-copy and KV-lookup roles strongly prefer the larger head in every
hetero2 condition:

| Layout | Larger head | local/KV roles choosing larger head |
|---|---:|---:|
| `[32,96]` | 96 | 20/20 |
| `[48,80]` | 80 | 20/20 |
| `[16,112]` | 112 | 20/20 |

Combined:

```text
local/KV roles choose the larger hetero2 head in 60/60 cases.
```

Baseline comparison:

```text
uniform2 has a head-index skew, H0 16/20, but both heads are the same dimension.
hetero2 has a structural dimension difference, and local/KV roles always choose
the larger head.
```

Conclusion:

```text
structural role affinity remains strong in two-head heterogeneous models.
```

## Induction Roles

The induction roles show a more informative split.

| Layout | induction_short top dims | induction_long top dims |
|---|---|---|
| `[32,96]` | 96 in `3/5`, 32 in `2/5` | 96 in `3/5`, 32 in `2/5` |
| `[48,80]` | 48 in `3/5`, 80 in `2/5` | 48 in `4/5`, 80 in `1/5` |
| `[16,112]` | 112 in `5/5` | 112 in `5/5` |

Interpretation:

```text
moderate heterogeneity lets induction roles prefer smaller/intermediate heads,
but extreme heterogeneity [16,112] collapses all roles onto the huge head.
```

This supports the idea that head dimension creates role affinity, but the
mapping is role- and layout-dependent.

## Specialization

Baseline:

```text
uniform2 specialization: 0.636
uniform2 effective heads: 2.25
```

Hetero2:

```text
[32,96]  specialization: 0.695, effective heads: 1.86
[48,80]  specialization: 0.756, effective heads: 1.91
[16,112] specialization: 0.702, effective heads: 1.76
```

Conclusion:

```text
hetero2 increases functional specialization beyond uniform2.
```

This is the cleanest specialization result so far because head count and total
head dimension are both matched.

## Functional Modularity

Baseline:

```text
uniform2 family gap: 0.653
uniform2 ARI:        1.000
```

Hetero2:

```text
[32,96]  family gap: 0.480, ARI: 0.889
[48,80]  family gap: 0.601, ARI: 1.000
[16,112] family gap: 0.327, ARI: 0.667
```

Conclusion:

```text
hetero2 does not improve ontology-level modularity over uniform2.
```

The best hetero2 modularity setting is `[48,80]`, which ties `uniform2` on ARI
but still has a lower family gap:

```text
uniform2 gap: 0.653
[48,80] gap:  0.601
```

The extreme `[16,112]` condition is worse:

```text
family gap: 0.327
ARI:        0.667
```

This suggests too much capacity imbalance may collapse multiple role families
onto the huge head, hurting modularity.

## Current Interpretation

The hetero2 control sharpens the project:

```text
structural role affinity: strong
functional specialization: strong, now even against uniform2
functional modularity: not supported against uniform2
```

The best current claim is:

```text
unequal head dimensions reliably bias which head type carries which roles and
increase causal concentration, but they do not automatically improve
ontology-level functional modularity over strong fewer/wider uniform-head
baselines.
```

## Next Step

Do not chase modularity with only capacity imbalance. Next tests should ask
whether a larger role ontology or less toy-like tasks change the picture:

1. Expand role families beyond local/KV/induction.
2. Add suppression/distractor and positional/BOS/SEP tasks.
3. Add more induction variants, because induction is the family that resists the
   simple large-head-affinity pattern.
4. Move the strongest affinity/specialization probes into real pretrained
   ordinary-head models.
