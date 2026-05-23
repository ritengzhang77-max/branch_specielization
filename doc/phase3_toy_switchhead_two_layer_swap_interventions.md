# Phase 3 Toy Pilot: Two-Layer SwitchHead Swap Interventions

Date: 2026-05-23

This memo repeats the expert-swap intervention in the two-layer induced
SwitchHead condition. The goal is to test whether the one-layer value-side
expert-codebook result localizes to the causal layer found by ablation.

## Question

```text
In the two-layer induced SwitchHead model, are the fragile expert identities in
layer 0, layer 1, or both?
```

## Setup

Training condition:

```text
n_layers = 2
n_heads = 2
n_experts = 2
moe_k = 1
expert_supervision_selector = output
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
steps = 2000
seeds = 1, 2, 3, 4, 5
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/toy_switchhead_competition.py \
  --seeds 1 2 3 4 5 \
  --steps 2000 \
  --batch-size 128 \
  --eval-examples 512 \
  --n-layers 2 \
  --n-heads 2 \
  --n-experts 2 \
  --d-head 32 \
  --moe-k 1 \
  --expert-supervision-weight 0.05 \
  --expert-supervision-selector output \
  --expert-supervision-end-step 800 \
  --run-swap-interventions \
  --swap-intervention-layer-groups 0 1 all \
  --device cuda \
  --output-dir results/phase3_toy_switchhead_2layer_swap_groups_w005_end800_seed5_steps2000
```

Using `--swap-intervention-layer-groups 0 1 all` evaluates layer-0-only,
layer-1-only, and all-layer swaps on the same trained model for each seed.

## Baseline Replication

The two-layer induced result replicated:

| Metric | Value |
|---|---:|
| Local accuracy | 1.0000 |
| Induction accuracy | 1.0000 |
| Routed expert match | 1.00 |
| Causal expert distance | 0.6360 |
| Output gate distance | 0.7275 |
| Query-position value gate distance | 0.0132 |
| Source-position value gate distance | 0.0019 |

Per-seed causal tops were local `L1E0` and induction `L1E1` in 5/5 seeds.

## Key Swap Results

Mean over 5 seeds:

| Swap layers | Intervention | Local acc. | Induction acc. | Eval loss |
|---|---|---:|---:|---:|
| 0 | baseline | 1.0000 | 1.0000 | 0.0003 |
| 0 | swap_v | 0.8913 | 0.9995 | 0.3597 |
| 0 | swap_o | 0.9254 | 1.0000 | 0.1826 |
| 0 | swap_v_o | 0.8761 | 0.9987 | 0.4890 |
| 0 | swap_v_and_value_selector | 1.0000 | 1.0000 | 0.0003 |
| 0 | swap_o_and_output_selector | 1.0000 | 1.0000 | 0.0003 |
| 0 | swap_all | 1.0000 | 1.0000 | 0.0003 |
| 1 | baseline | 1.0000 | 1.0000 | 0.0003 |
| 1 | swap_v | 0.6117 | 0.5892 | 1.8281 |
| 1 | swap_o | 1.0000 | 1.0000 | 0.0005 |
| 1 | swap_v_o | 0.5995 | 0.5953 | 1.8376 |
| 1 | swap_v_and_value_selector | 1.0000 | 1.0000 | 0.0003 |
| 1 | swap_o_and_output_selector | 1.0000 | 1.0000 | 0.0003 |
| 1 | swap_all | 1.0000 | 1.0000 | 0.0003 |
| all | baseline | 1.0000 | 1.0000 | 0.0003 |
| all | swap_v | 0.5156 | 0.5743 | 2.2171 |
| all | swap_o | 0.9237 | 1.0000 | 0.1865 |
| all | swap_v_o | 0.4945 | 0.5801 | 2.3228 |
| all | swap_v_and_value_selector | 1.0000 | 1.0000 | 0.0003 |
| all | swap_o_and_output_selector | 1.0000 | 1.0000 | 0.0003 |
| all | swap_all | 1.0000 | 1.0000 | 0.0003 |

## Interpretation

Layer 1 reproduces the one-layer result:

```text
swapping `v` or `sel_v` in layer 1 breaks both roles; coherently swapping
`v + sel_v` restores both roles; swapping `o` or `sel_o` is tolerated.
```

The failure is less total than in the one-layer model (`~0.60` accuracy instead
of `~0.08`), which is expected because the two-layer residual stream and layer 0
still carry useful information. But the qualitative codebook pattern is the
same.

Layer 0 is not irrelevant, but it is asymmetric:

```text
layer-0 swaps mainly hurt the local role and leave induction near-perfect.
```

This refines the earlier layer-localization result. Single-expert ablations
identified layer 1 as the top causal role layer for both local and induction.
The swap intervention shows that layer 0 still carries local-supporting
representations whose expert-label consistency matters. It just does not carry
the final induction module.

All-layer swaps combine the two effects:

- value-side swaps damage both roles, dominated by layer 1;
- output-side swaps mainly damage local, inherited from layer 0;
- coherent paired relabeling and full relabeling restore performance.

## Measurement Lesson

The two-layer result strengthens two warnings:

```text
top ablation localization is useful but incomplete;
paired expert relabeling is needed to distinguish codebook mismatch from
ordinary component importance.
```

Marginal gate statistics again under-explain the causal swap result. Source-
position value-gate distance was only `0.0019`, yet layer-1 value-side mismatch
substantially damaged both roles.

## Next Step

The next implementation improvement is to save trained checkpoints and add
activation-patching utilities. That would allow additional swaps and codebook
patches on exactly the same trained models without rerunning training.
