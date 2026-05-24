# Phase 3 Toy Pilot: SwitchHead Expert-Label Control

Date: 2026-05-23

This control tests whether the induced SwitchHead roles are tied to fixed expert
identities or follow the role-informed selector cue.

## Question

```text
If the selector target is reversed, do the learned causal expert roles reverse?
```

## Setup

I added target-label arguments to `scripts/toy_switchhead_competition.py`:

```text
--local-target-expert
--induction-target-expert
```

Default experiments use:

```text
local -> expert 0
induction -> expert 1
```

This control reverses the target:

```text
local -> expert 1
induction -> expert 0
```

## Results

| Target | End step | Weight | Routed match | Gate same top | Causal same top | Gate distance | Causal distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| default 0/1 | 450 | 0.05 | 1.00 | 0.00 | 0.00 | 0.4476 | 0.4240 |
| reversed 1/0 | 450 | 0.05 | 0.80 | 0.20 | 0.20 | 0.3936 | 0.4014 |
| default 0/1 | 800 | 0.05 | 1.00 | 0.00 | 0.00 | 0.9645 | 0.5664 |
| reversed 1/0 | 800 | 0.05 | 1.00 | 0.00 | 0.00 | 0.9646 | 0.5609 |

For the successful reversed 800-step run, every seed has:

```text
local causal top:     L0E1
induction causal top: L0E0
gate local top:       E1
gate induction top:   E0
```

## Interpretation

The 800-step reversed-label result shows that expert identity is not inherently
tied to the role:

```text
the induced functional modules follow the role cue.
```

The 450-step reversed-label result is only `4/5`, while the default 450-step
assignment is `5/5`. This means the exact label assignment can affect the
short-window optimization threshold, even though a longer cue resolves the
asymmetry.

## Project Takeaway

This strengthens the induced-modularity interpretation:

```text
SwitchHead expert routing is not merely revealing a fixed expert-0/local,
expert-1/induction bias; under sufficient cue duration, the target labels can be
reversed and the causal roles reverse too.
```

It also cautions that threshold numbers are not universal:

```text
the strength-duration boundary depends on optimization details, including the
specific role-to-expert label assignment.
```

## Result Directories

- `results/phase3_toy_switchhead_competition_weak_w005_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_reversed_targets_w005_end450_seed5_steps2000/`
- `results/phase3_toy_switchhead_competition_weak_w005_end800_seed5_steps2000/`
- `results/phase3_toy_switchhead_reversed_targets_w005_end800_seed5_steps2000/`
