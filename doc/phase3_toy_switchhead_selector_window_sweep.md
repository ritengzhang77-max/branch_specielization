# Phase 3 Toy Pilot: SwitchHead Selector-Window Sweep

Date: 2026-05-23

This follow-up asks how long weak role-informed SwitchHead expert-selection
pressure must remain active before the induced expert split persists without the
auxiliary loss.

## Question

```text
Does early structural routing pressure consolidate into persistent functional
modularity, and where is the reliability threshold in this toy setup?
```

## Setup

Base setup:

```text
task: bidirectional_lookup
model: one SwitchHeadRope block
n_heads: 2
n_experts: 2
moe_k: 1
steps: 2000
seeds: 1, 2, 3, 4, 5
expert_supervision_weight: 0.05
```

Auxiliary selector target:

```text
local positions -> expert 0
induction positions -> expert 1
```

The sweep changes only `--expert-supervision-end-step`. The auxiliary loss is
active for steps `< end_step`, then removed for the rest of training.

## Results

| Selector end step | Local acc. | Induction acc. | Gate same top | Causal same top | Routed match | Gate distance | Causal distance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 / none | 1.0000 | 1.0000 | 1.00 | 0.80 | 0.20 | 0.0032 | 0.0087 |
| 400 | 1.0000 | 1.0000 | 0.20 | 0.20 | 0.80 | 0.3030 | 0.3290 |
| 425 | 1.0000 | 1.0000 | 0.00 | 0.20 | 0.80 | 0.3728 | 0.3780 |
| 450 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.4476 | 0.4240 |
| 500 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.5986 | 0.5188 |
| 600 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.8427 | 0.5646 |
| 800 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.9645 | 0.5664 |
| full run | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.9982 | 0.5675 |

Interpretation:

```text
gate separation appears before fully reliable causal expert separation.
```

At end step 425, all 5 seeds have different gate top experts for local and
induction positions, but seed 4 still has the same causal top expert for both
roles. At end step 450, all 5 seeds have both different gate top experts and
different causal top experts.

## Threshold

In this deterministic 5-seed toy sweep, the provisional reliability boundary is:

```text
400 steps: partial
425 steps: gate split reliable, causal split partial
450 steps: gate split and causal split reliable
```

The effect continues to strengthen with longer selector pressure. Gate distance
rises monotonically from `0.3030` at end step 400 to `0.9645` at end step 800.
Causal expert distance also rises, then saturates around `0.56`.

## Main Takeaway

This strengthens the structural-to-functional result:

```text
weak early structural routing pressure can seed a persistent functional module
split in SwitchHead attention experts.
```

It also sharpens the ordering:

```text
selection/gate specialization can precede causal functional modularity.
```

This matches the earlier hand-built router trajectory, where role-aligned gate
behavior appeared before full causal branch separation.

## Caveats

- The threshold is provisional and specific to this toy setup, optimizer, seed
  set, and selector target.
- The sweep uses fixed seeds 1-5. A larger seed set could shift the boundary.
- The selector target is hand-specified, so this remains an induced-modularity
  result, not a spontaneous-modularity result.
- The gate metric uses normalized sigmoid output-selection scores for
  interpretability; SwitchHead's actual selection uses the raw sigmoid scores.

## Result Directories

- `results/phase3_toy_switchhead_competition_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end400_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end425_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end500_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end600_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_seed5_steps2000/`
