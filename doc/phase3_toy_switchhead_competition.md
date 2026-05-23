# Phase 3 Toy Pilot: SwitchHead Expert Competition

Date: 2026-05-23

This checkpoint is the first bridge from the hand-built two-branch toy router to
a less hand-designed routed-attention module. It uses the official plug-in
`SwitchHeadRope` implementation from `RobertCsordas/switchhead`, not the custom
two-branch tower used in the earlier Phase 3 toy runs.

## Question

The test asks:

```text
Does SwitchHead attention-expert routing spontaneously produce role-aligned
functional modularity on the conflict-heavy local-vs-induction task?
```

This is a stricter version of the previous learned-router question. The module
has attention experts rather than separate full branch towers, and the expert
selection mechanism is not designed around the local/induction roles.

## Setup

Task:

```text
bidirectional_lookup
local role:      y_i -> y_{i-1}
induction role:  y_i -> y_{i+1}
```

Model:

```text
embedding + one SwitchHeadRope block + MLP + unembedding
d_model = 128
n_layers = 1
n_heads = 2
n_experts = 2
d_head = 32
moe_k = 1
steps = 2000
seeds = 1, 2, 3, 4, 5
```

Command:

```bash
CUDA_VISIBLE_DEVICES=3 python3 -u scripts/toy_switchhead_competition.py \
  --seeds 1 2 3 4 5 \
  --steps 2000 \
  --batch-size 128 \
  --eval-examples 512 \
  --n-layers 1 \
  --n-heads 2 \
  --n-experts 2 \
  --d-head 32 \
  --moe-k 1 \
  --device cuda \
  --output-dir results/phase3_toy_switchhead_competition_seed5_steps2000
```

## Metrics

Task metrics:

- local and induction accuracy;
- local and induction loss.

Expert-selection metrics:

- mean normalized output-expert selection distribution at local positions;
- mean normalized output-expert selection distribution at induction positions;
- gate distribution distance between the two role distributions;
- whether local and induction have the same top selected expert.

Causal expert metrics:

- ablate each expert by temporarily zeroing that expert's value and output
  projection rows across all heads in the SwitchHead block;
- measure local and induction loss deltas;
- normalize positive deltas into role-specific expert-specialization
  distributions;
- compare local vs induction expert-specialization distributions.

Implementation note: the first ablation smoke test exposed an indexing bug.
Advanced indexing with `attn.v[row_idx].zero_()` zeroed a copy, not the original
parameter. The script now uses `index_fill_` and `index_copy_`, and a separate
logit-difference smoke test verified that expert ablation changes model outputs.

## Results

### Two Experts, One Active Expert

Aggregate over 5 seeds:

| Metric | Value |
|---|---:|
| Local accuracy | 1.0000 |
| Induction accuracy | 1.0000 |
| Gate same top expert | 1.00 |
| Causal same top expert | 0.80 |
| Routed expert match | 0.20 |
| Gate distribution distance | 0.0032 |
| Causal expert distribution distance | 0.0087 |
| Local top expert loss delta | 4.5838 |
| Induction top expert loss delta | 4.5197 |

Per-seed causal top expert:

| Seed | Local top | Induction top | Same top? | Causal distance | Gate distance |
|---:|---|---|---:|---:|---:|
| 1 | L0E0 | L0E1 | 0 | 0.0166 | 0.0026 |
| 2 | L0E1 | L0E1 | 1 | 0.0055 | 0.0090 |
| 3 | L0E0 | L0E0 | 1 | 0.0013 | 0.0003 |
| 4 | L0E1 | L0E1 | 1 | 0.0113 | 0.0005 |
| 5 | L0E0 | L0E0 | 1 | 0.0087 | 0.0039 |

The single separated-top case, seed 1, is not strong modularity. Its expert
specialization distributions are nearly tied:

```text
local:     E0 0.5125, E1 0.4875
induction: E0 0.4959, E1 0.5041
```

### Four Experts, Two Active Experts

I also ran the immediate capacity variant recommended below:

```text
n_experts = 4
moe_k = 2
steps = 2000
seeds = 1, 2, 3, 4, 5
```

Result:

| Metric | Value |
|---|---:|
| Local accuracy | 1.0000 |
| Induction accuracy | 1.0000 |
| Gate same top expert | 0.80 |
| Causal same top expert | 1.00 |
| Routed expert match | 0.00 |
| Gate distribution distance | 0.0083 |
| Causal expert distribution distance | 0.0486 |
| Local top expert loss delta | 0.0236 |
| Induction top expert loss delta | 0.0238 |

Per-seed causal top expert:

| Seed | Local top | Induction top | Same top? | Causal distance | Gate distance |
|---:|---|---|---:|---:|---:|
| 1 | L0E2 | L0E2 | 1 | 0.0768 | 0.0082 |
| 2 | L0E3 | L0E3 | 1 | 0.0608 | 0.0091 |
| 3 | L0E1 | L0E1 | 1 | 0.0587 | 0.0063 |
| 4 | L0E0 | L0E0 | 1 | 0.0252 | 0.0100 |
| 5 | L0E2 | L0E2 | 1 | 0.0215 | 0.0076 |

This variant is even less modular by top-expert identity. It also changes the
causal pattern: single-expert ablations are tiny because two experts are active,
so the computation is more redundant across active experts.

## Interpretation

This is a negative result for spontaneous SwitchHead expert-level modularity in
this toy setup.

Supported:

```text
SwitchHead can solve the conflict-heavy local-vs-induction task.
```

Not supported:

```text
SwitchHead attention experts spontaneously separate the local and induction
roles into different experts.
```

The strongest evidence is not just the top-expert count. It is the combination:

- gate same top expert is 5/5;
- causal same top expert is 4/5;
- gate distribution distance is only `0.0032`;
- causal expert distribution distance is only `0.0087`;
- both expert ablations strongly damage both roles.

The 4-expert `moe_k=2` variant reinforces the same conclusion in a different
failure mode:

- causal same top expert is 5/5;
- routed expert match is 0/5;
- gate distance remains tiny (`0.0083`);
- single-expert causal deltas become tiny, indicating redundancy across active
  experts rather than role separation.

### Two Experts With Weak Role-Informed Selection Pressure

I then added a weak auxiliary loss on SwitchHead's output-expert selector:

```text
local positions -> expert 0
induction positions -> expert 1
expert supervision weight = 0.05
```

The task loss remained the primary objective. The auxiliary loss was active for
the full 2000-step run.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/toy_switchhead_competition.py \
  --seeds 1 2 3 4 5 \
  --steps 2000 \
  --batch-size 128 \
  --eval-examples 512 \
  --n-layers 1 \
  --n-heads 2 \
  --n-experts 2 \
  --d-head 32 \
  --moe-k 1 \
  --expert-supervision-weight 0.05 \
  --device cuda \
  --output-dir results/phase3_toy_switchhead_competition_weak_w005_seed5_steps2000
```

Aggregate over 5 seeds:

| Metric | Value |
|---|---:|
| Local accuracy | 1.0000 |
| Induction accuracy | 1.0000 |
| Gate same top expert | 0.00 |
| Causal same top expert | 0.00 |
| Routed expert match | 1.00 |
| Gate distribution distance | 0.9982 |
| Causal expert distribution distance | 0.5675 |
| Local top expert loss delta | 7.0359 |
| Induction top expert loss delta | 7.0235 |

Per-seed causal top expert:

| Seed | Local top | Induction top | Routed match? | Causal distance | Gate distance |
|---:|---|---|---:|---:|---:|
| 1 | L0E0 | L0E1 | 1 | 0.5507 | 0.9983 |
| 2 | L0E0 | L0E1 | 1 | 0.6537 | 0.9983 |
| 3 | L0E0 | L0E1 | 1 | 0.5447 | 0.9981 |
| 4 | L0E0 | L0E1 | 1 | 0.5214 | 0.9983 |
| 5 | L0E0 | L0E1 | 1 | 0.5670 | 0.9980 |

This is the first SwitchHead result where structural expert selection and causal
functional modularity align cleanly across all tested seeds. Expert 0 is the
local-role causal component in 5/5 seeds, and expert 1 is the induction-role
causal component in 5/5 seeds.

The result should be stated carefully. It is not spontaneous modularity. It
shows that weak role-informative selector pressure can make SwitchHead's
attention experts become role-aligned functional modules on this toy task.

### Two Experts With Transient Selection Pressure

To test whether the result is merely compliance with an active auxiliary loss, I
reran the same setup with the selector loss turned off after step 800:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/toy_switchhead_competition.py \
  --seeds 1 2 3 4 5 \
  --steps 2000 \
  --batch-size 128 \
  --eval-examples 512 \
  --n-layers 1 \
  --n-heads 2 \
  --n-experts 2 \
  --d-head 32 \
  --moe-k 1 \
  --expert-supervision-weight 0.05 \
  --expert-supervision-end-step 800 \
  --device cuda \
  --output-dir results/phase3_toy_switchhead_competition_weak_w005_end800_seed5_steps2000
```

Aggregate over 5 seeds:

| Metric | Value |
|---|---:|
| Local accuracy | 1.0000 |
| Induction accuracy | 1.0000 |
| Gate same top expert | 0.00 |
| Causal same top expert | 0.00 |
| Routed expert match | 1.00 |
| Gate distribution distance | 0.9645 |
| Causal expert distribution distance | 0.5664 |
| Local top expert loss delta | 7.0446 |
| Induction top expert loss delta | 7.0217 |

Per-seed causal top expert:

| Seed | Local top | Induction top | Routed match? | Causal distance | Gate distance |
|---:|---|---|---:|---:|---:|
| 1 | L0E0 | L0E1 | 1 | 0.5514 | 0.9764 |
| 2 | L0E0 | L0E1 | 1 | 0.6486 | 0.9545 |
| 3 | L0E0 | L0E1 | 1 | 0.5452 | 0.9650 |
| 4 | L0E0 | L0E1 | 1 | 0.5209 | 0.9635 |
| 5 | L0E0 | L0E1 | 1 | 0.5656 | 0.9630 |

This strengthens the induced-modularity result. The selector pressure is gone
for the final 1200 training steps, but both the gate separation and causal
expert separation persist in 5/5 seeds.

A follow-up selector-window sweep narrowed the provisional threshold. With the
same weight, end step 400 was partial (`routed expert match=0.80`), end step 425
had reliable gate splitting but one causal failure, and end step 450 was reliable
in 5/5 seeds. See `doc/phase3_toy_switchhead_selector_window_sweep.md`.

A direct checkpointed trajectory with end step 450 confirmed the temporal
ordering: reliable gate separation appeared by checkpoint 425, while reliable
causal expert separation appeared by checkpoint 500. See
`doc/phase3_toy_switchhead_checkpoint_trajectory.md`.

At the same 450-step window, a selector-weight sweep found that weights `0.02`,
`0.03`, `0.04`, and `0.045` all solved the task but remained only `4/5` on
routed expert match. The first tested reliable 5/5 weight was `0.05`. See
`doc/phase3_toy_switchhead_weight_sweep.md`.

An additional strength-duration check showed that longer selector pressure lowers
the reliable weight threshold. At an 800-step window, `0.02` was still partial
but `0.025` and `0.03` were reliable in 5/5 seeds. See
`doc/phase3_toy_switchhead_strength_duration_tradeoff.md`.

So the best interpretation is:

```text
the model uses SwitchHead experts as shared computational resources, not as
role-specific functional modules.
```

## Project-Level Update

This strengthens the Phase 3 boundary condition:

```text
Moving from hand-built branch towers to SwitchHead attention experts does not by
itself create role-aligned functional modularity.
```

Together with the earlier router experiments, the current Phase 3 answer is:

```text
structural routing/expert design can support modularity, but spontaneous
role-aligned modularity is not automatic. Role-informative pressure remains the
only reliable mechanism observed so far.
```

The weak-supervision SwitchHead result sharpens that answer:

```text
when the structural router is given weak role information, SwitchHead can convert
expert selection into causal functional modularity.
```

The transient-pressure run adds:

```text
once induced, the role-aligned expert split can persist without the auxiliary
selection loss remaining active.
```

## Caveats

- These are very small SwitchHead models, not the paper-scale language model.
- Only two spontaneous configurations were tested: `n_experts=2, moe_k=1` and
  `n_experts=4, moe_k=2`.
- Expert ablation zeros both value and output expert rows. This is a coarse
  causal intervention but appropriate for a first pilot.
- The gate metric uses normalized sigmoid selection scores for interpretability;
  SwitchHead's actual top-k selection uses sigmoid scores directly.
- The task remains synthetic.
- The transient-pressure test turns off the auxiliary selector loss after step
  800, but it does not yet identify the minimum pressure duration or weight.

## Next Step

Do not claim a general negative result about SwitchHead. The next useful
SwitchHead-specific tests are:

1. Sweep shorter auxiliary selection windows, especially end steps 100, 200,
   and 400, to identify the minimum cue duration that survives later training.
2. Add checkpointed trajectories to ask whether gate/expert-selection separation
   ever precedes causal expert separation in this module.
3. Sweep the supervision weight and end step to test whether selection compliance
   precedes causal expert modularity as in the branch-router toy.

## Artifacts

- Experiment script:
  `scripts/toy_switchhead_competition.py`
- Result directory:
  `results/phase3_toy_switchhead_competition_seed5_steps2000/`
- Four-expert result directory:
  `results/phase3_toy_switchhead_competition_seed5_e4k2_steps2000/`
- Weak-selection result directory:
  `results/phase3_toy_switchhead_competition_weak_w005_seed5_steps2000/`
- Transient weak-selection result directory:
  `results/phase3_toy_switchhead_competition_weak_w005_end800_seed5_steps2000/`
- Selector-window sweep memo:
  `doc/phase3_toy_switchhead_selector_window_sweep.md`
- Checkpoint trajectory memo:
  `doc/phase3_toy_switchhead_checkpoint_trajectory.md`
- Selector-weight sweep memo:
  `doc/phase3_toy_switchhead_weight_sweep.md`
- Strength-duration tradeoff memo:
  `doc/phase3_toy_switchhead_strength_duration_tradeoff.md`
- Feasibility memo:
  `doc/switchhead_followup_feasibility.md`
