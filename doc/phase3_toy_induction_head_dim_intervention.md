# Phase 3 Toy Pilot: Induction-Style Head-Dimension Intervention

Date: 2026-05-22

This checkpoint tests whether the heterogeneous-head-dimension result from the
key-value toy task survives a richer induction-style task. This matters because
the project's strongest Pythia evidence so far is repeat-match / induction-style
behavior, not key-value lookup.

## Task

Each example is a repeated random-token sequence:

```text
[x_1, ..., x_16, x_1, ..., x_16]
```

Loss is scored only on second-half continuation positions:

```text
x_i(second occurrence) -> x_{i+1}(second occurrence)
```

This is a minimal induction-style next-token prediction task. The script also
records attention to the first occurrence of `x_i` and to the first occurrence
of `x_{i+1}`, but the primary specialization score is causal single-head
ablation loss delta.

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
| `hetero4_64first` | `[64, 16, 16, 32]` | Position control: move the 64-dim head from index 3 to index 0 |

## Command

```bash
python3 -u scripts/toy_induction_head_dim_intervention.py \
  --configs uniform4 hetero4 uniform2 hetero4_64first \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --repeat-length 16 \
  --random-controls 8 \
  --random-permutations 100 \
  --output-dir results/phase3_toy_induction_head_dim_intervention
```

## Metric

The primary specialization score is:

```text
S(h, t) = max(loss_after_ablating_head_h - baseline_loss, 0)
```

Scores are normalized across heads within a layer before computing
specialization concentration and cross-seed role-distribution similarity.

## Aggregate Results

All configurations solved the task.

| Config | Eval acc. | Top specialization | Top-head loss delta | Random loss delta | Raw top-head match | Raw role similarity |
|---|---:|---:|---:|---:|---:|---:|
| `uniform4` | 0.9993 | 0.5796 | 0.0587 | 0.0141 | 0.35 | 0.6591 |
| `hetero4` | 0.9991 | 0.9830 | 1.0195 | 0.0066 | 1.00 | 0.9685 |
| `uniform2` | 0.9993 | 0.6578 | 0.2989 | 0.1832 | 0.50 | 0.8747 |
| `hetero4_64first` | 0.9992 | 0.9882 | 1.5637 | 0.0040 | 1.00 | 0.9882 |

## Main Finding

The structural-intervention signal reproduces on the induction-style task:

```text
uniform4 top-head loss delta: 0.0587
hetero4 top-head loss delta: 1.0195
64-first top-head loss delta: 1.5637

uniform4 top specialization: 0.5796
hetero4 top specialization: 0.9830
64-first top specialization: 0.9882
```

The top causal role follows the 64-dim head, not its absolute index:

```text
hetero4 [16,16,32,64]:
  layer 0 top head = head 3, dim 64, in 5/5 seeds
  layer 1 top head = head 3, dim 64, in 5/5 seeds

hetero4_64first [64,16,16,32]:
  layer 0 top head = head 0, dim 64, in 5/5 seeds
  layer 1 top head = head 0, dim 64, in 5/5 seeds
```

This is stronger than the key-value toy result because it uses a repeated-token
next-token task that is closer to the Pythia repeat-match setup.

## Interpretation

The positive Stage 3 signal is now less likely to be a narrow key-value lookup
artifact. In two different synthetic tasks, heterogeneous head dimensions make
the high-capacity head a stable causal specialization slot across seeds.

The uniform fewer/wider control remains important:

- `uniform2` has wider 64-dim heads and stronger ablation effects than
  `uniform4`;
- but because both heads are structurally identical, top-head identity is not as
  stable as the heterogeneous 64-dim slot.

The current best framing is:

```text
heterogeneous head dimensions break head permutation symmetry in a way that can
turn a functional role into a stable structural slot.
```

## Caveats

- This is still a toy task, not a language-model pretraining run.
- The causal metric is single-head ablation, not full path patching.
- The task likely favors a high-capacity global retrieval head, so the next test
  should include competing local and global sub-tasks.
- Raw-score Hungarian alignment is not the right sole success metric for
  heterogeneous heads; structure-aware slot matching must be reported too.

## Decision

Continue the architectural-intervention branch. The next decisive experiment
should add task competition:

```text
local/previous-token prediction + global induction prediction
```

The key question is whether small heads reliably take local roles while the
large head reliably takes global induction roles. That would move from "large
head dominates one task" toward genuine structural-to-functional role
partitioning.
