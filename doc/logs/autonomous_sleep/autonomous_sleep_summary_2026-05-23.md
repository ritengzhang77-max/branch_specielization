# Autonomous Sleep Research Summary

Date: 2026-05-23

Window covered: 2026-05-23 00:15:59 PDT to late morning PDT.

This summary covers the later SwitchHead work completed after the seed-robustness
checkpoint. The detailed running log is
`doc/logs/autonomous_sleep/autonomous_sleep_log_2026-05-23.md`.

## Main Result

The strongest new conclusion is:

```text
Weak output-selector pressure induces a stable local/induction expert split, but
the frozen inference-time mechanism is mostly a value-side expert codebook.
```

The evidence is causal, not just probe-based:

- one-layer induced models solve the task and have routed match `1.00`;
- swapping `v` or `sel_v` alone destroys performance;
- coherently swapping `v + sel_v` restores performance;
- swapping `o` or `sel_o` alone is tolerated;
- full expert relabeling restores performance exactly.

This means output-selector pressure is still the clean sufficient training cue,
but after training the fragile code lives in the value projection / value
selector pairing.

## Key Experiments

### One-Layer Swap Grid

Result directory:

```text
results/phase3_toy_switchhead_swap_interventions_w005_end800_seed5_steps2000/
```

Mean over 5 seeds:

| Intervention | Local acc. | Induction acc. |
|---|---:|---:|
| baseline | 1.0000 | 1.0000 |
| swap_v | 0.0804 | 0.0706 |
| swap_value_selector | 0.0804 | 0.0706 |
| swap_v_and_value_selector | 1.0000 | 1.0000 |
| swap_o | 1.0000 | 1.0000 |
| swap_output_selector | 1.0000 | 1.0000 |
| swap_all | 1.0000 | 1.0000 |

Memo:

```text
doc/experiments/phase3/phase3_toy_switchhead_swap_interventions.md
```

### Two-Layer Layer-Grouped Swap Grid

Result directory:

```text
results/phase3_toy_switchhead_2layer_swap_groups_w005_end800_seed5_steps2000/
```

Mean over 5 seeds:

| Swap layers | Intervention | Local acc. | Induction acc. |
|---|---|---:|---:|
| 0 | swap_v | 0.8913 | 0.9995 |
| 0 | swap_o | 0.9254 | 1.0000 |
| 1 | swap_v | 0.6117 | 0.5892 |
| 1 | swap_o | 1.0000 | 1.0000 |
| all | swap_v | 0.5156 | 0.5743 |
| all | swap_o | 0.9237 | 1.0000 |
| all | swap_v_and_value_selector | 1.0000 | 1.0000 |
| all | swap_all | 1.0000 | 1.0000 |

Interpretation:

```text
the main two-role value-side codebook is in layer 1. Layer 0 is not irrelevant,
but its fragility is mostly local-role support, not the final induction module.
```

Memo:

```text
doc/experiments/phase3/phase3_toy_switchhead_two_layer_swap_interventions.md
```

### Attention-Weighted Value-Gate Diagnostic

Result directory:

```text
results/phase3_toy_switchhead_attended_value_gate_w005_end800_seed5_steps2000/
```

The diagnostic reconstructs attention weights from Q/K and weights source-token
value-selector distributions by the actual attention from local and induction
query positions.

Result:

```text
attended value-gate distance = 0.0091
```

Interpretation:

```text
the value-side swap fragility is not explained by simple marginal or
attention-weighted role-wise value-expert usage.
```

### Parameter Geometry

One-layer checkpoints:

```text
results/phase3_toy_switchhead_1layer_induced_w005_end800_seed5_steps2000_checkpoints/
results/phase3_toy_switchhead_1layer_parameter_diagnostics/
```

Two-layer checkpoints / diagnostics:

```text
results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_checkpoints/
results/phase3_toy_switchhead_2layer_induced_w005_end800_seed3_checkpoint_retry/
results/phase3_toy_switchhead_seed1245_parameter_diagnostics/
results/phase3_toy_switchhead_seed3_parameter_diagnostics/
```

Main geometry:

| Setting | Output `o` cosine | Value `v` cosine |
|---|---:|---:|
| one-layer | 0.7887 | 0.0451 |
| two-layer layer 1, seeds 1/2/4/5 | 0.8295 | 0.2884 |

Interpretation:

```text
output projections are much more similar than value projections, matching the
causal swap result where output-side swaps are tolerated and value-side swaps
are destructive.
```

## Infrastructure Added

Updated `scripts/toy_switchhead_competition.py` with:

- `--run-swap-interventions`;
- `--swap-intervention-layers`;
- `--swap-intervention-layer-groups`;
- source-position value-gate metrics;
- attention-weighted value-gate metrics;
- `--save-final-checkpoints`;
- `--load-final-checkpoints`.

Added:

```text
scripts/switchhead_checkpoint_parameter_diagnostics.py
doc/experiments/phase3/phase3_switchhead_activation_patching_plan.md
```

## Pushed Commits

Latest pushed commits:

```text
d515d98 Add SwitchHead expert swap interventions
f59ce65 Add two-layer SwitchHead swap interventions
b8a8a1f Add attended SwitchHead value gate diagnostic
920903f Add SwitchHead checkpoint saving
8c2ae92 Add SwitchHead checkpoint loading
869dd4d Document loaded SwitchHead swap workflow
f5b0b46 Add SwitchHead checkpoint parameter diagnostics
e33b11b Document one-layer SwitchHead parameter diagnostics
a715dcb Add autonomous sleep research summary
f39c0be Add SwitchHead activation patching plan
```

## Important Caveat

The two-layer checkpoint-saving pass saved seed 1-5 checkpoints, but that
particular run was only `4/5` on routed match because seed 3 landed in a shared
top-expert basin. A seed-3 retry recovered the intended `L1E0/L1E1` split.

This does not overturn the earlier 10/10 robustness result, but it means future
checkpoint-based patching should use explicitly validated checkpoints, not assume
that every nominally identical CUDA/Triton rerun lands in the same basin.

## Next Best Step

Add activation-patching utilities on top of checkpoint loading.

Planning doc:

```text
doc/experiments/phase3/phase3_switchhead_activation_patching_plan.md
```

The most direct next test is:

```text
load a validated induced checkpoint, run baseline and value-side-mismatched
models, then patch layer-1 value-side activations or attention outputs at the
local/induction positions to determine exactly where the codebook mismatch
enters the residual stream.
```

This should now be much cheaper because the loader avoids retraining.
