# Phase 3 Toy Pilot: SwitchHead Strength-Duration Tradeoff

Date: 2026-05-23

This memo combines the selector-window and selector-weight sweeps to test whether
selector cue duration can compensate for selector cue strength.

## Question

```text
Is induced SwitchHead functional modularity controlled by a strength-duration
tradeoff, rather than by a fixed minimum selector weight?
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
```

The selector target is fixed:

```text
local positions -> expert 0
induction positions -> expert 1
```

## Results

### End Step 450

| Weight | Routed match | Gate same top | Causal same top | Gate distance | Causal distance |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.80 | 0.40 | 0.20 | 0.0443 | 0.1060 |
| 0.03 | 0.80 | 0.20 | 0.20 | 0.1068 | 0.1946 |
| 0.04 | 0.80 | 0.40 | 0.20 | 0.2571 | 0.2888 |
| 0.045 | 0.80 | 0.20 | 0.20 | 0.3483 | 0.3460 |
| 0.05 | 1.00 | 0.00 | 0.00 | 0.4476 | 0.4240 |

At 450 steps, the first tested reliable weight is `0.05`.

### End Step 800

| Weight | Routed match | Gate same top | Causal same top | Gate distance | Causal distance |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.80 | 0.00 | 0.20 | 0.3864 | 0.3405 |
| 0.025 | 1.00 | 0.00 | 0.00 | 0.5823 | 0.4834 |
| 0.03 | 1.00 | 0.00 | 0.00 | 0.7632 | 0.5528 |
| 0.05 | 1.00 | 0.00 | 0.00 | 0.9645 | 0.5664 |

At 800 steps, the first tested reliable weight is `0.025`.

## Interpretation

The reliable selector weight drops when the selector cue lasts longer:

```text
450-step window: reliable between 0.045 and 0.05
800-step window: reliable between 0.02 and 0.025
```

This supports a strength-duration tradeoff. The cue is not merely an active
constraint at evaluation time, because all end-step runs remove the auxiliary
loss before the final evaluation. It is also not enough for the model to solve
the task: every row in these sweeps reaches 1.0 local and induction accuracy,
but only some rows produce reliable causal expert modularity.

## Mechanistic Reading

The recurring pattern is:

```text
weaker or shorter cue -> gate split can appear -> one seed may remain causally
collapsed -> stronger or longer cue -> causal expert roles become reliable.
```

This is consistent with the checkpoint trajectory:

```text
gate specialization precedes causal functional modularity.
```

## Project Takeaway

The most defensible Stage 3 claim is now:

```text
structural expert routing can become persistent functional modularity when the
role cue crosses a strength-duration threshold.
```

This is stronger than a one-off positive intervention, but still narrower than
claiming spontaneous modularity or arbitrary weak-cue sufficiency.

## Result Directories

- `results/phase3_toy_switchhead_weight_w002_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w003_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w004_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w0045_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_tradeoff_w002_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_tradeoff_w0025_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_tradeoff_w003_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end800_seed5_steps2000/`
