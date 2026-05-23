# Phase 3 Toy Pilot: Two-Layer SwitchHead

Date: 2026-05-23

This experiment asks whether the one-layer SwitchHead result survives when the
toy model has two routed-attention layers.

## Question

```text
With multiple SwitchHead layers, does induced functional modularity localize to
one layer, distribute across layers, or collapse into redundancy?
```

## Implementation Note

While analyzing the first two-layer run, I found that `expert_scores.csv` was
duplicating gate metrics averaged across layers. I fixed
`scripts/toy_switchhead_competition.py` so per-expert gate metrics are now
reported per layer. The v2 result directories below are the authoritative
two-layer results.

## Setup

Shared setup:

```text
task: bidirectional_lookup
model: two SwitchHeadRope blocks
n_heads: 2 per layer
n_experts: 2 per layer
moe_k: 1
steps: 2000
seeds: 1, 2, 3, 4, 5
```

Conditions:

```text
spontaneous: no selector loss
induced: expert_supervision_weight=0.05, expert_supervision_end_step=800
```

## Aggregate Results

| Condition | Local acc. | Induction acc. | Gate same top | Causal same top | Routed match | Gate distance | Causal distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| spontaneous | 1.0000 | 1.0000 | 1.00 | 0.80 | 0.20 | 0.0017 | 0.1617 |
| induced | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.7066 | 0.6148 |

## Layer Localization

In the spontaneous condition, the model solves the task but mostly uses the same
late expert for both roles:

```text
local top:     L1E0 in 5/5 seeds
induction top: L1E0 in 4/5 seeds, L1E1 in 1/5 seeds
```

The gates also remain shared:

```text
gate same top expert: 5/5
gate distribution distance: 0.0017
```

In the induced condition, the top causal components are cleanly role-aligned and
localized to the second layer:

```text
local top:     L1E0 in 5/5 seeds
induction top: L1E1 in 5/5 seeds
```

Layer 0 receives role-aligned gate pressure, but its causal deltas are much
smaller than layer 1's. The strongest causal role modules therefore concentrate
in layer 1, not evenly across both layers.

## Interpretation

The two-layer result preserves the one-layer conclusion:

```text
SwitchHead depth does not make spontaneous role modularity appear, but
role-informed selector pressure can still induce persistent causal expert
modularity.
```

It adds one refinement:

```text
in a deeper toy SwitchHead, induced causal modularity localizes to a late layer,
even though selector/gate pressure is applied to every layer.
```

That is useful for future real-model analysis: routed attention experts may show
role-aligned gates in several layers while the causal role module is concentrated
in a narrower layer range.

## Caveats

- This is still a tiny synthetic model.
- The selector loss is applied to every SwitchHead layer, so this does not test
  whether supervising only one layer is sufficient.
- The causal intervention zeros one expert's value and output rows in one layer;
  it does not patch intermediate activations between layers.

## Result Directories

- `results/phase3_toy_switchhead_2layer_spontaneous_seed5_steps2000_v2/`
- `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_v2/`
