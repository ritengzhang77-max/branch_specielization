# Phase 3 Toy Pilot: Two-Layer SwitchHead Selector-Type Control

Date: 2026-05-23

This memo extends the selector-type control to the two-layer setting, focusing on
layer 1 because the two-layer causal modules localize there.

## Question

```text
In the causal layer of a two-layer SwitchHead model, is output selection still
the clean sufficient cue, or can value selection induce the role split?
```

## Setup

Shared setup:

```text
n_layers = 2
n_heads = 2
n_experts = 2
moe_k = 1
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
expert_supervision_layers = 1
seeds = 1, 2, 3, 4, 5
```

The sweep changes only `--expert-supervision-selector`.

## Results

| Selector supervised on layer 1 | Local acc. | Induction acc. | Routed match | Causal same top | Output gate dist. | Value gate dist. | Causal dist. |
|---|---:|---:|---:|---:|---:|---:|---:|
| output only | 1.0000 | 1.0000 | 1.00 | 0.00 | 0.4301 | 0.0063 | 0.6143 |
| value only | 1.0000 | 1.0000 | 0.80 | 0.20 | 0.0046 | 0.4197 | 0.5522 |
| both | 1.0000 | 1.0000 | 0.80 | 0.20 | 0.2764 | 0.3518 | 0.7956 |

## Interpretation

Output-only layer-1 supervision is cleanly sufficient:

```text
local top:     L1E0 in 5/5 seeds
induction top: L1E1 in 5/5 seeds
```

Value-only layer-1 supervision is not null in the two-layer setting. Unlike the
one-layer value-only run, it reaches `4/5` routed match while leaving output
gates shared. However, it still fails the reliability criterion.

Both-selector layer-1 supervision also remains `4/5`. Adding value-selector
pressure does not improve over output-only pressure here, and in this run it
reintroduces a failure seed.

## Project Takeaway

Across one-layer and two-layer SwitchHead controls, the cleanest sufficient cue
is still:

```text
output selector pressure at the causal layer.
```

Value selector pressure can affect causal specialization when it is applied at
the causal layer in a deeper model, but it is less reliable and does not replace
output selector pressure.

This sharpens the larger-model measurement recommendation:

```text
measure output-selection routing separately from value-selection routing, and do
not assume that all expert-routing signals have the same causal meaning.
```

## Result Directories

- `results/phase3_toy_switchhead_2layer_selector_output_l1_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_2layer_selector_value_l1_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_2layer_selector_both_l1_w005_end800_seed5_steps2000/`
