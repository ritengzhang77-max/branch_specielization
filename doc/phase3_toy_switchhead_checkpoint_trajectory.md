# Phase 3 Toy Pilot: SwitchHead Checkpoint Trajectory

Date: 2026-05-23

This memo records a direct within-run trajectory for the weak role-informed
SwitchHead selector experiment. The selector loss is active through step 449 and
removed after step 450.

## Question

```text
Within a single training run, does gate specialization precede causal expert
modularity?
```

## Setup

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
  --expert-supervision-end-step 450 \
  --trajectory-eval-steps 0 100 200 300 400 425 450 500 600 800 1000 1500 2000 \
  --device cuda \
  --output-dir results/phase3_toy_switchhead_trajectory_w005_end450_seed5_steps2000
```

Implementation note: adding trajectory evaluation exposed a SwitchHead-specific
RoPE cache issue. `torch.inference_mode()` can leave cached inference tensors in
the rotary-position module, causing later backpropagation to fail. The script now
uses `torch.no_grad()` for evaluation so training can safely resume after
checkpoint analysis.

## Aggregate Trajectory

| Step | Local acc. | Induction acc. | Gate same top | Causal same top | Routed match | Gate distance | Causal distance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0060 | 0.0073 | 1.00 | 0.60 | 0.00 | 0.0000 | 0.1858 |
| 100 | 0.0766 | 0.0718 | 0.40 | 1.00 | 0.00 | 0.0390 | 0.0080 |
| 200 | 0.8817 | 0.9857 | 0.00 | 0.40 | 0.60 | 0.0226 | 0.1006 |
| 300 | 0.9497 | 1.0000 | 0.20 | 0.20 | 0.80 | 0.0911 | 0.1949 |
| 400 | 0.9520 | 1.0000 | 0.20 | 0.40 | 0.60 | 0.3520 | 0.2811 |
| 425 | 0.9676 | 0.9999 | 0.00 | 0.20 | 0.80 | 0.4387 | 0.3240 |
| 450 | 0.9741 | 1.0000 | 0.00 | 0.20 | 0.80 | 0.5266 | 0.3715 |
| 500 | 0.9752 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.5348 | 0.3905 |
| 600 | 0.9751 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.5125 | 0.3943 |
| 800 | 0.9751 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.4912 | 0.3977 |
| 1000 | 0.9752 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.4788 | 0.4008 |
| 1500 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.4592 | 0.4191 |
| 2000 | 1.0000 | 1.0000 | 0.00 | 0.00 | 1.00 | 0.4482 | 0.4257 |

## Milestones

| Milestone | First checkpoint |
|---|---:|
| Gate same-top expert = 0/5 with meaningful distance | 425 |
| Causal same-top expert = 0/5 | 500 |
| Routed expert match = 5/5 | 500 |
| Local and induction accuracy = 1.0 mean | 1500 |

The step-200 row is a warning against relying only on top expert identity:
`gate_same_top=0.00`, but gate distance is only `0.0226`, so the top split is
mostly a tiny-margin effect. By step 425 the gate split is much stronger
(`0.4387`) while causal modularity is still partial.

## Per-Seed Boundary Behavior

The boundary case is seed 4:

| Step | Gate local | Gate induction | Local causal top | Induction causal top | Routed match |
|---:|---:|---:|---|---|---:|
| 400 | E1 | E1 | L0E1 | L0E1 | 0 |
| 425 | E0 | E1 | L0E1 | L0E1 | 0 |
| 450 | E0 | E1 | L0E1 | L0E1 | 0 |
| 500 | E0 | E1 | L0E0 | L0E1 | 1 |

This seed makes the ordering especially clear:

```text
the gate split appears first; the causal local role moves to the routed local
expert later.
```

## Interpretation

This directly supports the temporal mechanism suggested by the selector-window
sweep:

```text
role-informed expert selection can seed a gate split, and that gate split can
later consolidate into causal functional modularity.
```

The result does not show spontaneous modularity. It shows a plausible mechanism
for converting structural routing pressure into persistent functional
modularity.

## Artifacts

- Script: `scripts/toy_switchhead_competition.py`
- Result directory:
  `results/phase3_toy_switchhead_trajectory_w005_end450_seed5_steps2000/`
- Main trajectory file:
  `results/phase3_toy_switchhead_trajectory_w005_end450_seed5_steps2000/trajectory_model_summary.csv`
- Per-expert trajectory file:
  `results/phase3_toy_switchhead_trajectory_w005_end450_seed5_steps2000/trajectory_expert_scores.csv`
