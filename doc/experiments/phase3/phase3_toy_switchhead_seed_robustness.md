# Phase 3 Toy Pilot: SwitchHead Seed Robustness

Date: 2026-05-23

This memo extends the key induced SwitchHead conditions from seeds 1-5 to seeds
6-10.

## Question

```text
Do the core induced SwitchHead modularity results remain reliable beyond the
original five random seeds?
```

## One-Layer Output-Selector Condition

Setup:

```text
n_layers = 1
expert_supervision_selector = output
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
```

| Seed set | Local acc. | Induction acc. | Routed match | Output gate dist. | Value gate dist. | Causal dist. |
|---|---:|---:|---:|---:|---:|---:|
| 1-5 | 1.0000 | 1.0000 | 1.00 | 0.9645 | 0.0134 | 0.5663 |
| 6-10 | 1.0000 | 1.0000 | 1.00 | 0.9609 | 0.0194 | 0.5721 |

Across seeds 1-10:

```text
routed expert match: 10/10
local top:           L0E0 in 10/10
induction top:       L0E1 in 10/10
```

## One-Layer Spontaneous Baseline

Setup:

```text
n_layers = 1
no selector supervision
```

| Seed set | Local acc. | Induction acc. | Routed match | Output gate dist. | Causal dist. |
|---|---:|---:|---:|---:|---:|
| 1-5 | 1.0000 | 1.0000 | 0.20 | 0.0032 | 0.0087 |
| 6-10 | 1.0000 | 1.0000 | 0.20 | 0.0061 | 0.0086 |

Across seeds 1-10:

```text
routed expert match: 2/10
gate split:          absent / tiny
causal distance:     near zero
```

## Two-Layer All-Layer Output-Selector Condition

Setup:

```text
n_layers = 2
expert_supervision_selector = output
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
expert_supervision_layers = all
```

| Seed set | Local acc. | Induction acc. | Routed match | Output gate dist. | Causal dist. |
|---|---:|---:|---:|---:|---:|
| 1-5 | 1.0000 | 1.0000 | 1.00 | 0.7066 | 0.6148 |
| 6-10 | 1.0000 | 1.0000 | 1.00 | 0.7777 | 0.5551 |

Across seeds 1-10:

```text
routed expert match: 10/10
local top:           L1E0 in 10/10
induction top:       L1E1 in 10/10
```

## Two-Layer Spontaneous Baseline

Setup:

```text
n_layers = 2
no selector supervision
```

| Seed set | Local acc. | Induction acc. | Routed match | Output gate dist. | Causal dist. |
|---|---:|---:|---:|---:|---:|
| 1-5 | 1.0000 | 1.0000 | 0.20 | 0.0017 | 0.1617 |
| 6-10 | 1.0000 | 1.0000 | 0.00 | 0.0041 | 0.1209 |

Across seeds 1-10:

```text
routed expert match: 1/10
gate split:          absent / tiny
top causal layer:    late layer, usually shared across roles
```

## Interpretation

The core positive SwitchHead result is not a five-seed accident:

```text
output-selector pressure with weight 0.05 through step 800 reliably induces
causal role modularity across 10/10 one-layer and 10/10 two-layer seeds.
```

The two-layer localization result also survives the expanded seed set:

```text
the induced causal role modules localize to layer 1 across all 10 two-layer
seeds tested.
```

The negative spontaneous result also survives the expanded seed set:

```text
SwitchHead solves the task across spontaneous seeds, but does not reliably form
role-aligned expert modules without the role-informed output selector cue.
```

## Result Directories

- `results/phase3_toy_switchhead_selector_output_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_selector_output_w005_end800_seed6_10_steps2000/`
- `results/phase3_toy_switchhead_competition_seed5_steps2000/`
- `results/phase3_toy_switchhead_spontaneous_seed6_10_steps2000/`
- `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_v2/`
- `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed6_10_steps2000/`
- `results/phase3_toy_switchhead_2layer_spontaneous_seed5_steps2000_v2/`
- `results/phase3_toy_switchhead_2layer_spontaneous_seed6_10_steps2000/`
