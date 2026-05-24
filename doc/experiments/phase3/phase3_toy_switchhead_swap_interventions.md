# Phase 3 Toy Pilot: SwitchHead Expert-Swap Interventions

Date: 2026-05-23

This memo follows the successful one-layer SwitchHead induced-modularity
condition with a stronger causal intervention: swap expert identities at
inference time and test which swaps break or preserve the learned computation.

## Question

```text
After weak output-selector pressure induces local -> expert 0 and induction ->
expert 1, is the role split carried by the selector, the value expert, the
output expert, or only by a coherent relabeling of expert identities?
```

## Setup

Shared training condition:

```text
n_layers = 1
n_heads = 2
n_experts = 2
moe_k = 1
expert_supervision_selector = output
expert_supervision_weight = 0.05
expert_supervision_end_step = 800
steps = 2000
seeds = 1, 2, 3, 4, 5
```

Result directory:

```text
results/phase3_toy_switchhead_swap_interventions_w005_end800_seed5_steps2000/
```

The script now supports:

```text
--run-swap-interventions
```

and writes:

```text
swap_intervention_summary.csv
```

Each intervention swaps expert 0 and expert 1 rows across all heads, then
evaluates the frozen model without further training. For example, `swap_v`
swaps value-projection expert rows only; `swap_v_and_value_selector` swaps both
the value projection and the value selector rows, which is a coherent value-side
expert relabeling.

I also added a source-aligned value-selector metric. The older value-gate metric
measured the value selector at the scored query positions. In SwitchHead,
however, value selection is computed on source/key tokens, so the new metric
also compares value selection at the local and induction source positions.

## Baseline Replication

The trained models reproduced the induced modularity result:

| Metric | Value |
|---|---:|
| Local accuracy | 1.0000 |
| Induction accuracy | 1.0000 |
| Routed expert match | 1.00 |
| Causal expert distance | 0.5664 |
| Output gate distance | 0.9645 |
| Query-position value gate distance | 0.0134 |
| Source-position value gate distance | 0.0029 |

Per-seed causal tops were local `L0E0` and induction `L0E1` in 5/5 seeds.

## Swap Results

Mean over 5 seeds:

| Intervention | Eval loss | Local acc. | Induction acc. | Interpretation |
|---|---:|---:|---:|---|
| baseline | 0.0004 | 1.0000 | 1.0000 | Original model. |
| swap_v | 6.5388 | 0.0804 | 0.0706 | Value expert identity is causal. |
| swap_o | 0.0005 | 1.0000 | 1.0000 | Output expert swap alone is tolerated. |
| swap_v_o | 6.5340 | 0.0676 | 0.0827 | Swapping both matrices without selectors breaks the value-side code. |
| swap_value_selector | 6.5388 | 0.0804 | 0.0706 | Value selector swap alone is equivalent to `swap_v`. |
| swap_output_selector | 0.0005 | 1.0000 | 1.0000 | Output selector swap alone is tolerated. |
| swap_both_selectors | 6.5340 | 0.0676 | 0.0827 | Selector-only relabeling breaks because value side is mismatched. |
| swap_v_and_value_selector | 0.0004 | 1.0000 | 1.0000 | Coherent value-side relabeling restores exactly. |
| swap_o_and_output_selector | 0.0004 | 1.0000 | 1.0000 | Coherent output-side relabeling restores exactly. |
| swap_v_and_output_selector | 6.5340 | 0.0676 | 0.0827 | Crossed value/output relabeling breaks. |
| swap_o_and_value_selector | 6.5340 | 0.0676 | 0.0827 | Crossed output/value relabeling breaks. |
| swap_v_o_and_value_selector | 0.0005 | 1.0000 | 1.0000 | Value side restored; output swap alone is tolerated. |
| swap_v_o_and_output_selector | 6.5388 | 0.0804 | 0.0706 | Output side restored; value swap remains destructive. |
| swap_v_and_both_selectors | 0.0005 | 1.0000 | 1.0000 | Value side restored; output selector swap tolerated. |
| swap_o_and_both_selectors | 6.5388 | 0.0804 | 0.0706 | Output side restored; value selector swap remains destructive. |
| swap_all | 0.0004 | 1.0000 | 1.0000 | Full expert relabeling is invariant. |

## Interpretation

The induced role split survives a full coherent expert relabeling:

```text
swap_all restores the baseline exactly.
```

So expert identity is still a label symmetry at the whole-module level.

But partial swaps reveal where the fragile causal machinery sits:

```text
the value projection and value selector form the causal expert codebook.
```

Swapping either `v` or `sel_v` alone destroys both roles. Swapping both together
restores performance. In contrast, swapping `o` or `sel_o` alone leaves accuracy
at 1.0, with only a tiny loss increase.

This refines the earlier selector-type conclusion. Output-selector supervision
is still the clean sufficient training cue, because it reliably induces the
causal expert split. But the final inference-time mechanism is not simply "the
output selector chooses the role expert." The role-relevant computation has
consolidated into the value-side expert codebook, while the output side is
largely tolerant to expert relabeling.

A checkpoint parameter diagnostic supports this interpretation:

```text
output projection expert cosine = 0.7887
value projection expert cosine = 0.0451
```

So the tolerated output-side swap is not mysterious: the output projections are
much more similar than the value projections, while the value projections form
distinct expert-specific bases.

## Measurement Lesson

The new source-aligned value-gate metric remained near zero:

```text
source-position value gate distance = 0.0029
```

I then added an attention-weighted value-gate diagnostic, reconstructing the
block's attention weights from Q/K and weighting source-token value-selector
distributions by the actual attention paid from local and induction query
positions. This also remained small:

```text
attended value-gate distance = 0.0091
```

So simple role-averaged value-gate statistics do not explain the destructive
value-side swaps, even when weighted by attention flow. The relevant structure
is not an obvious local-vs-induction split in marginal or attended value-gate
usage. It is an internal consistency relation between selected value experts and
their learned value projections.

This is another instance of the project-level warning:

```text
gate specialization, marginal routing statistics, and causal modularity can
come apart.
```

## Trustworthiness

This is a strong toy-level causal check:

- all claims are frozen-model interventions, not probes;
- the result is stable across 5/5 seeds;
- full relabeling and paired relabeling controls behave as expected;
- the exact equalities between several intervention pairs provide useful
  implementation sanity checks.

Main limits:

- one-layer, two-expert toy model only;
- swaps are global expert-row swaps, not position-specific activation patches;
- no trained checkpoints are saved yet, so adding new intervention types still
  requires rerunning training;
- the source-aligned value-gate metric is still coarse because it averages over
  positions and heads instead of weighting by actual attention flow.

## Next Step

This was completed in `doc/experiments/phase3/phase3_toy_switchhead_two_layer_swap_interventions.md`.
The two-layer result preserves the main value-side codebook pattern in layer 1,
while showing that layer 0 swaps mainly damage the local role.

The next implementation improvement is to save trained checkpoints and add
activation-patching utilities, so additional interventions can be run on exactly
the same models without retraining.
