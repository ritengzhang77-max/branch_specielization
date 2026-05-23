# SwitchHead Follow-Up Feasibility Check

Date: 2026-05-23

This note checks the next proposed bridge after the toy-router and Pythia timing
results:

```text
test whether a less hand-designed routed attention model shows the same
probe/gate-before-causal-modularity lag.
```

## Correct Codebase

The earlier roadmap listed `github.com/RobertCsordas/moe` as the SwitchHead code
target. That is not the right repository for SwitchHead. It is a related MoE-MLP
repository for "Approximating Two-Layer Feedforward Networks for Efficient
Transformers."

The correct SwitchHead repositories are:

| Repository | Purpose | Local checkout | Commit inspected |
|---|---|---|---|
| `https://github.com/RobertCsordas/moe_attention` | official SwitchHead training and experiment code | `.tools/moe_attention/` | `7169ad3` |
| `https://github.com/RobertCsordas/switchhead` | user-friendly plug-in SwitchHead attention implementation | `.tools/switchhead/` | `0bb2f61` |
| `https://github.com/RobertCsordas/moe` | different MoE-MLP paper code, not the SwitchHead target | `.tools/csordas_moe/` | `6b175aa` |

The SwitchHead README describes the paper as a NeurIPS 2024 paper and points
training-code users to `moe_attention`; the plug-in implementation points users
back to `moe_attention` for full training code.

## Environment Check

Local environment:

```text
Python 3.10
PyTorch 2.4.0+cu121
Triton installed
GPU: Quadro RTX 6000 available
```

The plug-in `switchhead` package imports successfully from `.tools/switchhead`.

GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.tools/switchhead python3 - <<'PY'
import torch
import switchhead

x = torch.randn(2, 16, 32, device="cuda")
attn = switchhead.SwitchHeadRope(
    32,
    n_heads=2,
    n_experts=2,
    d_head=8,
    moe_k=1,
).cuda()
out, cache = attn(x, x, x, mask=None)
torch.cuda.synchronize()
print(tuple(out.shape), cache)
PY
```

Result:

```text
(2, 16, 32) None
```

CPU smoke test fails, as expected, because the implementation uses a Triton CVMM
kernel for expert projection and Triton cannot access CPU tensors in that path.
So any SwitchHead experiment should be GPU-only.

## Training-Code Practicality

`moe_attention` is a full training framework with W&B sweep integration. It is
useful for reproducing the paper, but not the fastest path for the next narrow
experiment because:

- it expects W&B-centered sweeps;
- it is a separate framework from the current toy/Pythia scripts;
- adapting checkpointed causal modularity metrics inside it would require
  nontrivial instrumentation.

The plug-in `switchhead` implementation is more practical for the next local
experiment because:

- it exposes `SwitchHeadRope` as an ordinary PyTorch module;
- it can be dropped into a small local transformer block;
- expert selection scores are computable from the module's `sel_v`, `sel_o`,
  and `get_sel` methods;
- GPU forward pass works in the current environment.

## Recommended Next Experiment

Use the plug-in implementation rather than the full W&B training repo for the
first bridge experiment.

Proposed minimal experiment:

```text
TinySwitchHeadCompetition:
  task_variant = bidirectional_lookup
  attention = SwitchHeadRope
  n_heads = 2 or 4
  n_experts = 2 or 4
  moe_k = 1 or 2
  checkpoint steps = 0, 400, 800, 1200, 1600
```

Measurements:

- task accuracy for local and induction roles;
- gate/expert-selection distance between local and induction positions;
- top expert used by local vs induction positions;
- causal expert ablation, by zeroing or masking selected value/output expert
  contributions;
- timing comparison between expert-selection separation and causal expert
  separation.

Key comparison to the existing toy router:

```text
Does SwitchHead expert selection show the same gate-before-causality lag, or
does attention-expert routing consolidate causal modularity differently?
```

## Implementation Notes

The first local implementation should not depend on the ignored `.tools/`
checkout at runtime unless the script checks for it and gives a clear error.
Better options:

1. Add a script that accepts `--switchhead-path .tools/switchhead` and appends it
   to `sys.path`.
2. Later vendor only the minimal `switchhead/` package or add an installation
   step if this becomes a central experiment.

Expert ablation is the main technical task. The plug-in module does not expose a
high-level "disable expert k" argument, so the experiment needs one of:

- temporary masking of selection indices after `get_sel`;
- temporary zeroing of expert rows in `v` and/or `o`;
- a small fork/subclass that records selections and applies an expert mask in
  forward.

The cleanest first pass is a small local wrapper/subclass, not modifying the
external checkout.

## Decision

SwitchHead is feasible in the current environment, but the next implementation
should be scoped as a local plug-in experiment, not a full reproduction of the
paper training framework.

The roadmap should use:

```text
training repo: https://github.com/RobertCsordas/moe_attention
plug-in repo:  https://github.com/RobertCsordas/switchhead
```

not `github.com/RobertCsordas/moe`.

## First Pilot Result

I implemented the local plug-in experiment in
`scripts/toy_switchhead_competition.py` and ran a 5-seed one-layer
`SwitchHeadRope` pilot on the conflict-heavy task.

Result:

```text
local accuracy: 1.0000
induction accuracy: 1.0000
gate same top expert: 1.00
causal same top expert: 0.80
gate distribution distance: 0.0032
causal expert distribution distance: 0.0087
```

Interpretation:

```text
SwitchHead solves the task, but its experts remain shared across roles in this
pilot. This is not spontaneous role-aligned functional modularity.
```

See `doc/phase3_toy_switchhead_competition.md`.
