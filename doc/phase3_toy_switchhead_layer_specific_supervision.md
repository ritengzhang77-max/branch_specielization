# Phase 3 Toy Pilot: Layer-Specific SwitchHead Supervision

Date: 2026-05-23

This follow-up asks whether the two-layer SwitchHead induced-modularity result
requires selector pressure in every layer.

## Question

```text
If causal role modules localize to layer 1, is supervising layer 1 alone
sufficient? Can supervising only layer 0 indirectly induce the same late-layer
causal split?
```

## Setup

I added:

```text
--expert-supervision-layers
```

to `scripts/toy_switchhead_competition.py`. If unspecified, selector supervision
applies to all layers. The layer-specific runs use:

```text
n_layers = 2
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
```

## Results

| Condition | Gate same top | Causal same top | Routed match | Gate distance | Causal distance |
|---|---:|---:|---:|---:|---:|
| spontaneous | 1.00 | 0.80 | 0.20 | 0.0017 | 0.1617 |
| supervise layer 0 only | 0.00 | 0.60 | 0.40 | 0.4862 | 0.2499 |
| supervise layer 1 only | 0.20 | 0.20 | 0.80 | 0.4155 | 0.5791 |
| supervise both layers | 0.00 | 0.00 | 1.00 | 0.7066 | 0.6148 |

## Layer Behavior

Layer-0-only supervision:

```text
layer 0 gates split strongly;
layer 1 gates remain near-neutral;
top causal components remain mostly shared in layer 1.
```

Layer-1-only supervision:

```text
layer 0 gates remain near-neutral;
layer 1 gates usually split;
causal modularity improves substantially but remains 4/5.
```

Both-layer supervision:

```text
layer 0 gates split;
layer 1 gates split or partially split;
top causal components are L1E0 for local and L1E1 for induction in 5/5 seeds.
```

## Interpretation

Layer-0-only routing pressure does not reliably induce the late-layer causal
expert split. It can create an upstream gate split, but the causal computation
still mostly forms in shared layer-1 experts.

Layer-1-only routing pressure is much closer to sufficient, which matches the
causal localization result. But it still fails in one seed, so all-layer
pressure is the most robust condition tested.

The working interpretation is:

```text
direct role-aligned pressure on the layer where the causal role module forms is
important, and upstream gate splitting alone is not enough.
```

## Project Takeaway

This refines the strength-duration story:

```text
the cue must also reach the relevant layer, not merely exist somewhere upstream.
```

For larger models, this argues for measuring gates and causal expert ablations
layer-by-layer. A model can show role-aligned routing in one layer while the
causal module is elsewhere.

## Result Directories

- `results/phase3_toy_switchhead_2layer_spontaneous_seed5_steps2000_v2/`
- `results/phase3_toy_switchhead_2layer_supervise_l0_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_2layer_supervise_l1_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_v2/`
