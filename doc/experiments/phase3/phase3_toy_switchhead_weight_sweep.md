# Phase 3 Toy Pilot: SwitchHead Selector-Weight Sweep

Date: 2026-05-23

This follow-up fixes the selector-pressure window at 450 steps and sweeps the
auxiliary selector-loss weight.

## Question

```text
At the shortest reliable 450-step window, how strong must the role-informed
selector cue be before SwitchHead experts become reliably functional modules?
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
expert_supervision_end_step: 450
```

The sweep changes only `--expert-supervision-weight`.

## Results

| Weight | Local acc. | Induction acc. | Gate same top | Causal same top | Routed match | Gate distance | Causal distance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.0000 | 1.0000 | 1.00 | 0.80 | 0.20 | 0.0032 | 0.0087 |
| 0.02 | 1.0000 | 1.0000 | 0.40 | 0.20 | 0.80 | 0.0443 | 0.1060 |
| 0.03 | 1.0000 | 1.0000 | 0.20 | 0.20 | 0.80 | 0.1068 | 0.1946 |
| 0.04 | 1.0000 | 1.0000 | 0.40 | 0.20 | 0.80 | 0.2571 | 0.2888 |
| 0.045 | 1.0000 | 1.0000 | 0.20 | 0.20 | 0.80 | 0.3483 | 0.3460 |
| 0.05 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.4476 | 0.4240 |

## Interpretation

All tested weights solve the task, but task performance is not enough to
guarantee functional modularity.

At the fixed 450-step selector window, the first tested weight with reliable
5/5 role-aligned causal expert modularity is:

```text
expert_supervision_weight = 0.05
```

Lower weights still partially influence the model. Gate distance and causal
expert distance generally increase as the selector cue strengthens. However,
weights from `0.02` through `0.045` remain only `4/5` on routed expert match.

The recurring failure mode is that one seed keeps the local causal role on the
induction expert despite solving the task. This means the selector cue must pass
a strength-duration threshold before it reliably changes the learned causal
decomposition.

## Project Takeaway

This prevents an overclaim:

```text
arbitrarily tiny structural routing cues do not reliably induce functional
modularity in this SwitchHead toy setup.
```

The supported claim is narrower and more useful:

```text
once the role-informed selector cue is strong enough and present long enough,
SwitchHead expert routing can consolidate into persistent causal functional
modularity.
```

A follow-up strength-duration check found that the reliable weight threshold
drops when the selector cue lasts longer: at end step 450, the first tested
reliable weight was `0.05`; at end step 800, the first tested reliable weight was
`0.025`. See `doc/experiments/phase3/phase3_toy_switchhead_strength_duration_tradeoff.md`.

## Result Directories

- `results/phase3_toy_switchhead_competition_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w002_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w003_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w004_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_weight_w0045_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end450_seed5_steps2000/`
