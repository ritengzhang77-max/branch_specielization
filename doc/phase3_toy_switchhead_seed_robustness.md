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

## Result Directories

- `results/phase3_toy_switchhead_selector_output_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_selector_output_w005_end800_seed6_10_steps2000/`
- `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_v2/`
- `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed6_10_steps2000/`
