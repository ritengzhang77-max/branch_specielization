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

## Caveats

- These are very small SwitchHead models, not the paper-scale language model.
- Only two spontaneous configurations were tested: `n_experts=2, moe_k=1` and
  `n_experts=4, moe_k=2`.
- Expert ablation zeros both value and output expert rows. This is a coarse
  causal intervention but appropriate for a first pilot.
- The gate metric uses normalized sigmoid selection scores for interpretability;
  SwitchHead's actual top-k selection uses sigmoid scores directly.
- The task remains synthetic.

## Next Step

Do not claim a general negative result about SwitchHead. The next useful
SwitchHead-specific tests are:

1. Add a weak role-informative auxiliary selection loss, analogous to the
   weak-token-router toy.
2. Add checkpointed trajectories to ask whether gate/expert-selection separation
   ever precedes causal expert separation in this module.
3. If weak selection pressure works, sweep its weight to test whether
   selection compliance precedes causal expert modularity as in the branch-router
   toy.

## Artifacts

- Experiment script:
  `scripts/toy_switchhead_competition.py`
- Result directory:
  `results/phase3_toy_switchhead_competition_seed5_steps2000/`
- Four-expert result directory:
  `results/phase3_toy_switchhead_competition_seed5_e4k2_steps2000/`
- Feasibility memo:
  `doc/switchhead_followup_feasibility.md`
