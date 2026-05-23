# Phase 3 Toy Pilot: SwitchHead Selector-Type Control

Date: 2026-05-23

SwitchHead has separate expert selectors for value projection and output
projection. Previous weak-selector experiments supervised the output selector.
This control separates the two selectors.

## Question

```text
Which SwitchHead selector controls the induced causal expert split: output
selection, value selection, or both?
```

## Setup

I added:

```text
--expert-supervision-selector output|value|both
```

to `scripts/toy_switchhead_competition.py`, and added value-selector metrics to
`model_summary.csv` and `expert_scores.csv`.

Shared setup:

```text
n_layers = 1
n_heads = 2
n_experts = 2
moe_k = 1
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
seeds = 1, 2, 3, 4, 5
```

## Results

| Selector supervised | Local acc. | Induction acc. | Routed match | Causal same top | Output gate dist. | Value gate dist. | Causal dist. |
|---|---:|---:|---:|---:|---:|---:|---:|
| output only | 1.0000 | 1.0000 | 1.00 | 0.00 | 0.9645 | 0.0134 | 0.5663 |
| value only | 0.9752 | 1.0000 | 0.00 | 1.00 | 0.0049 | 0.9299 | 0.0667 |
| both | 0.9506 | 1.0000 | 1.00 | 0.00 | 0.6978 | 0.7860 | 0.6327 |

## Interpretation

Output-selector supervision is cleanly sufficient:

```text
output-only pressure gives 5/5 causal role modularity, even though the value
selector remains mostly unsplit.
```

Value-selector supervision is not sufficient:

```text
value-only pressure strongly splits value routing, but the causal top expert
remains shared in 5/5 seeds.
```

Supervising both selectors recovers the causal split, but two seeds fail to fully
learn the local role by 2000 steps:

```text
both-selector pressure can force modularity, but at this weight it hurts
optimization relative to output-only pressure.
```

## Project Takeaway

For this toy SwitchHead setup, the relevant structural cue is specifically the
output expert selector, not arbitrary expert routing inside the attention module.
However, later swap-intervention results refine what "relevant" means here:
output-selector pressure is the clean sufficient training cue, but the final
inference-time bottleneck is not simply the output selector itself.

This refines the mechanism:

```text
output routing controls which expert writes the role-specific result back to the
residual stream; value routing alone can split internal reads without producing
separable causal role modules.
```

Follow-up expert swaps found that swapping the value projection `v` or value
selector `sel_v` alone destroys the trained computation, while swapping the
output projection `o` or output selector `sel_o` alone is tolerated. Coherently
swapping `v` with `sel_v` restores performance. So the best current
interpretation is:

```text
output-selector pressure induces the split, but the learned role computation
consolidates into a value-side expert codebook.
```

This matters for larger-model analysis: measuring only token-to-value expert
routing can miss the training cue, while measuring only output-selector routing
can miss where the causal computation consolidates.

A two-layer extension found the same overall ordering at the causal layer:
layer-1 output-only supervision was reliable in 5/5 seeds, while layer-1
value-only and both-selector supervision were both 4/5. See
`doc/phase3_toy_switchhead_two_layer_selector_type.md`.

For the swap-intervention refinement, see
`doc/phase3_toy_switchhead_swap_interventions.md`.

## Result Directories

- `results/phase3_toy_switchhead_selector_output_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_selector_value_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_selector_both_w005_end800_seed5_steps2000/`
