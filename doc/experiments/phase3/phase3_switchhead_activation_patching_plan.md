# Phase 3 SwitchHead Activation-Patching Plan

Date: 2026-05-23

## Goal

Use saved SwitchHead checkpoints to localize where the value-side expert-codebook
mismatch enters the residual stream.

The swap experiments established:

```text
value-side weight/selector mismatch is destructive;
coherent value-side relabeling restores performance;
output-side swaps are tolerated in the final layer.
```

The next question is:

```text
which activation produced by the value-side codebook has to be restored to
repair a value-side mismatch?
```

## Checkpoints To Use

Use explicitly validated checkpoints:

```text
results/phase3_toy_switchhead_1layer_induced_w005_end800_seed5_steps2000_checkpoints/checkpoints/
results/phase3_toy_switchhead_2layer_induced_w005_end800_seed3_checkpoint_retry/checkpoints/
```

Do not assume every nominally identical rerun is canonical; the two-layer
checkpoint-saving pass had a seed-3 shared-top failure.

## Minimal Patch Targets

### One-layer model

Run four forward variants:

1. baseline;
2. `swap_v` mismatch;
3. `swap_v` mismatch with patched attention output at local scored positions;
4. `swap_v` mismatch with patched attention output at induction scored
   positions.

The main recovery metric is local/induction accuracy and loss.

Expected result if the codebook mismatch enters through the layer attention
write:

```text
patching the role's scored positions should selectively recover that role.
```

### Two-layer model

Start with the validated seed-3 retry checkpoint, then expand to more validated
seeds.

Patch targets:

- layer 1 attention output at local scored positions;
- layer 1 attention output at induction scored positions;
- layer 0 attention output at local scored positions;
- layer 0 attention output at induction scored positions.

Expected result from swap evidence:

```text
layer-1 patches should recover both local and induction under layer-1 `swap_v`;
layer-0 patches should mostly affect local.
```

## Implementation Sketch

Add a context-managed hook to `SwitchHeadBlock.forward` or a wrapper method on
`TinySwitchHeadTransformer`:

```text
capture clean attention output per layer;
run corrupted model under `swap_v`;
replace selected positions in selected layer's attention output with clean
values;
finish the forward pass normally;
score local and induction losses.
```

Keep patching at the attention-output tensor before residual addition:

```text
x = x + attn_out
```

This is the cleanest first target because it avoids reimplementing CVMM expert
selection internals.

## Controls

- Patch unscored positions with equal count as a negative control.
- Patch the wrong role positions to test selectivity.
- Compare `swap_v` against `swap_v_and_value_selector`, which should already be
  fully restored.
- Report loss recovery, not only accuracy, because many corrupted conditions are
  partially solved.

## Decision Criterion

If role-position attention-output patching restores the corrupted role, the
project can say:

```text
the value-side codebook mismatch becomes causally visible at the attention write
into the residual stream.
```

If it does not restore, patch one level earlier:

```text
post-attention pre-output-projection result;
selected value projection output;
selected expert assignment.
```

Those earlier patches require more invasive SwitchHead internals and should wait
until the attention-output patch is tested.
