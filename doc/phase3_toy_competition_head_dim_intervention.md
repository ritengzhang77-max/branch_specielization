# Phase 3 Toy Pilot: Local-vs-Induction Competition

Date: 2026-05-22

This checkpoint tests a stronger version of the heterogeneous-head-dimension
hypothesis. Prior toy runs showed that heterogeneous head dimensions can make a
64-dim head a stable causal slot on a single global retrieval/induction task.
This run asks whether, under task competition, different head dimensions
specialize into different roles:

```text
local/previous-token-style copying + global induction-style copying
```

The desired positive result would be:

```text
small heads -> local role
large head -> global induction role
```

That clean result did not appear.

## Task

Each training sequence contains two supervised regions.

Local region:

```text
[x, SEP, x]
```

At `SEP`, the model must predict the immediately previous token `x`.

Global induction region:

```text
[y_1, ..., y_16, y_1, ..., y_16]
```

At the second occurrence of `y_i`, the model must predict `y_{i+1}`.

The combined loss averages local and induction losses with equal weight.

## Configurations

All models are tiny decoder-only transformers:

- `d_model = 128`
- `n_layers = 2`
- `mlp_dim = 256`
- total attention head dimension = 128
- 5 seeds per config
- 1200 training steps per seed

| Config | Head dimensions | Purpose |
|---|---:|---|
| `uniform4` | `[32, 32, 32, 32]` | Same head count, uniform baseline |
| `hetero4` | `[16, 16, 32, 64]` | Same head count and total head dimension, heterogeneous intervention |
| `uniform2` | `[64, 64]` | Fewer/wider uniform control |
| `hetero4_64first` | `[64, 16, 16, 32]` | Position control |

## Command

```bash
python3 -u scripts/toy_competition_head_dim_intervention.py \
  --configs uniform4 hetero4 uniform2 hetero4_64first \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --random-controls 8 \
  --random-permutations 100 \
  --output-dir results/phase3_toy_competition_head_dim_intervention
```

## Metric

Each role gets its own causal specialization score:

```text
S_local(h) = max(local_loss_after_ablating_h - local_baseline_loss, 0)
S_induction(h) = max(induction_loss_after_ablating_h - induction_baseline_loss, 0)
```

Scores are normalized across heads within a layer before computing
specialization concentration and cross-seed consistency.

## Aggregate Results

All configs learned both objectives well enough for comparison.

| Config | Local acc. | Induction acc. | Local top spec. | Induction top spec. | Same top slot |
|---|---:|---:|---:|---:|---:|
| `uniform4` | 1.0000 | 0.9999 | 0.4924 | 0.5999 | 0.00 |
| `hetero4` | 1.0000 | 0.9992 | 0.8857 | 0.8188 | 0.60 |
| `uniform2` | 1.0000 | 0.9976 | 0.6461 | 0.7039 | 0.40 |
| `hetero4_64first` | 0.9999 | 0.9986 | 0.9187 | 0.7220 | 0.20 |

Heterogeneity still increases specialization concentration relative to
`uniform4`, especially for the local objective:

```text
uniform4 local top specialization: 0.4924
hetero4 local top specialization:  0.8857
64-first local top specialization: 0.9187
```

## Role Partition Result

The clean small-local / large-global partition did not appear.

Top-head dimension counts:

| Config | Local top dims | Induction top dims |
|---|---|---|
| `uniform4` | `{"32": 5}` | `{"32": 5}` |
| `hetero4` | `{"16": 1, "64": 4}` | `{"16": 1, "64": 4}` |
| `uniform2` | `{"64": 5}` | `{"64": 5}` |
| `hetero4_64first` | `{"64": 5}` | `{"16": 3, "64": 2}` |

The strongest stable pattern is:

```text
the 64-dim head is very attractive for the local role;
the induction role is also often attracted to the 64-dim head, but not always;
role assignment under competition depends on the heterogeneous layout.
```

In `hetero4`, the local and induction top roles use the same head dimension in
5/5 seeds and the same exact `(layer, head)` slot in 3/5 seeds.

In `hetero4_64first`, the local top role uses the 64-dim head in 5/5 seeds, but
the induction top role uses a 16-dim head in 3/5 seeds. This means moving the
64-dim head changes the role partition, not just the label of the same solution.

## Interpretation

This checkpoint refines the project hypothesis.

Still supported:

```text
heterogeneous head dimensions can increase causal specialization concentration.
```

Not yet supported:

```text
small heads reliably become local heads while large heads reliably become
global/induction heads.
```

The better current framing is:

```text
structural heterogeneity creates stable high-capacity slots, but task competition
and layout determine which function occupies which slot.
```

This is important for the eventual paper: the Stage 3 claim should not be
"heterogeneity automatically induces a semantic role taxonomy." It should be
"heterogeneity breaks permutation symmetry and can make function-to-slot mappings
more stable; the mapping itself depends on task pressure and layout."

## Caveats

- This is still a toy task, not LM pretraining.
- Some seeds have slightly worse induction eval loss, though all configs remain
  near-perfect in accuracy.
- Single-head ablation can overstate importance when redundant heads compensate
  nonlinearly.
- The local role is a simple previous-token copy from `SEP`, not a broad syntax
  or local-context task.
- The result is sensitive enough to layout that more heterogeneous permutations
  should be tested before making a strong dimension-function claim.

## Decision

Continue Stage 3, but narrow the claim:

```text
test heterogeneity as a permutation-symmetry-breaking stabilizer, not as a
guaranteed semantic role allocator.
```

Next decisive experiment:

1. Run more heterogeneous permutations, e.g. `[16, 64, 16, 32]` and
   `[16, 32, 64, 16]`, on the same competition task.
2. Add a slot-conditional analysis: compare function stability by structural
   slot `(head_dim, position)` rather than by free Hungarian matching alone.
3. If the function-to-slot mapping remains layout-sensitive, frame this as a
   real constraint on the architectural-intervention story.
