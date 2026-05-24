# Phase 3 Toy Pilot: Heterogeneous Head-Dimension Intervention

Date: 2026-05-21

This checkpoint is the first direct structural-intervention test in the project.
It does not claim language-model evidence. It asks whether the core Stage 3
hypothesis is even measurable in a controlled toy setting:

```text
Does structural head-dimension heterogeneity induce more stable functional
specialization across seeds?
```

## Task

Each example is a synthetic key-value recall sequence:

```text
[k_1, v_1, k_2, v_2, ..., k_8, v_8, k_q] -> v_q
```

The model sees eight key-value pairs and a final query key. It must predict the
value associated with that key.

All models are tiny decoder-only transformers:

- `d_model = 128`
- `n_layers = 2`
- `mlp_dim = 256`
- total attention head dimension = 128 for every config
- 5 seeds per config
- 1200 training steps per seed

## Configurations

| Config | Head dimensions | Purpose |
|---|---:|---|
| `uniform4` | `[32, 32, 32, 32]` | Same head count, uniform baseline |
| `hetero4` | `[16, 16, 32, 64]` | Same head count and total head dimension, heterogeneous intervention |
| `uniform2` | `[64, 64]` | Fewer/wider uniform control |
| `hetero4_64first` | `[64, 16, 16, 32]` | Position control: move the 64-dim head from index 3 to index 0 |

## Commands

Main pilot:

```bash
python3 -u scripts/toy_head_dim_intervention.py \
  --configs uniform4 hetero4 uniform2 \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --random-controls 8 \
  --random-permutations 100 \
  --role-target value \
  --role-metric ablation \
  --output-dir results/phase3_toy_head_dim_intervention
```

Position control:

```bash
python3 -u scripts/toy_head_dim_intervention.py \
  --configs hetero4_64first \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --random-controls 8 \
  --random-permutations 100 \
  --role-target value \
  --role-metric ablation \
  --output-dir results/phase3_toy_head_dim_position_control
```

## Metric

For this toy task, the primary specialization score is causal:

```text
S(h, t) = max(loss_after_ablating_head_h - baseline_loss, 0)
```

Scores are normalized across heads within a layer before computing
specialization concentration and cross-seed role-distribution similarity.

The script also records attention mass from the final query to the correct value
position, but the main metric is single-head ablation loss delta. This choice
matters because the trained toy models can solve the task with distributed
attention patterns even when a head has a large causal effect.

## Aggregate Results

All configurations learned the task, so the comparison is not confounded by
failure to train.

| Config | Eval acc. | Top specialization | Top-head loss delta | Random loss delta | Raw top-head match | Raw role similarity |
|---|---:|---:|---:|---:|---:|---:|
| `uniform4` | 0.9992 | 0.4414 | 0.1380 | 0.0586 | 0.30 | 0.6185 |
| `hetero4` | 0.9980 | 0.9741 | 1.6084 | 0.0098 | 1.00 | 0.9530 |
| `uniform2` | 0.9996 | 0.7704 | 0.9343 | 0.2069 | 0.40 | 0.6734 |
| `hetero4_64first` | 1.0000 | 0.9807 | 1.4581 | 0.0084 | 1.00 | 0.9611 |

## Key Finding

The heterogeneous model learns a much cleaner single-head causal solution than
the uniform 4-head baseline:

```text
uniform4 top-head loss delta: 0.1380
hetero4 top-head loss delta: 1.6084

uniform4 top specialization: 0.4414
hetero4 top specialization: 0.9741
```

The effect is stable across seeds and follows the high-capacity 64-dim slot:

```text
hetero4 [16,16,32,64]:     top causal head = head 3, dim 64, in 5/5 seeds
hetero4_64first [64,16,16,32]: top causal head = head 0, dim 64, in 5/5 seeds
```

This position control rules out the simplest "last head wins" artifact. In this
toy setting, the functional role follows the structurally larger head dimension.

## Important Interpretation

This is the first positive evidence for the architectural-intervention thesis:

```text
structural heterogeneity can induce stable functional specialization.
```

It also changes the metric lesson. In Pythia, raw-score Hungarian alignment was
needed because head index was unstable. In the heterogeneous toy model, the
structural slot itself becomes meaningful: same-index / same-dimension matching
is better aligned with the causal role than unconstrained raw-score matching.

So Stage 3 should include both:

- permutation-invariant matching, for same-architecture uniform heads;
- structure-aware slot matching, for heterogeneous heads where the architecture
  intentionally breaks head permutation symmetry.

## Caveats

- This is a toy key-value recall task, not a pretrained language model.
- The causal specialization metric is single-head ablation, not path patching.
- The task may naturally favor a high-capacity retrieval head; that is useful for
  a pilot, but the effect must be tested on richer tasks.
- `uniform2` shows that fewer/wider heads also create stronger causal heads than
  `uniform4`; however, `uniform2` does not produce the same cross-seed top-head
  consistency because both heads have the same structural type.
- The result supports continuing Stage 3, not claiming the final paper result.

## Decision

Continue the architectural-intervention branch. The next decisive experiment is
a larger toy or small language-model run with at least three arms:

```text
uniform same-head-count baseline
heterogeneous head-dim intervention
uniform fewer/wider control
```

The primary outcomes should be:

- matched validation loss;
- causal specialization concentration;
- same-slot vs Hungarian-aligned cross-seed stability;
- whether the large head consistently takes global retrieval / induction-style
  roles while smaller heads take local or low-impact roles.
