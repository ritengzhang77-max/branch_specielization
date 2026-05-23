# Phase 3 Toy Pilot: Router Causal Consolidation Window

Date: 2026-05-23

This checkpoint narrows the timing mechanism from the router-trajectory run.
The earlier trajectory showed:

```text
step 400: gate match 1.00, causal routed role match 0.00
step 600: gate match 1.00, causal routed role match 1.00
```

This run adds denser evaluation between steps 400 and 800 to locate when the
causal branch split forms after gate alignment.

## Claim Tested

The claim is:

```text
role-aligned gates appear before branch computations become causally modular;
causal modularity consolidates later under continued weak role-routing pressure.
```

The key alternative is:

```text
once the gate is role-aligned, causal branch modularity appears immediately.
```

## Setup

Same conflict-heavy routed toy setup as the trajectory experiment:

```text
task_variant = bidirectional_lookup
config = weak_token_router
branch_head_dims = [64]
router_supervision_weight = 0.05
router_supervision_end_step = 800
local_weight = 1.0
induction_weight = 1.0
seeds = 1, 2, 3, 4, 5
training steps = 800
```

Dense trajectory evaluation steps:

```text
0, 400, 450, 500, 550, 600, 650, 700, 750, 800
```

Command:

```bash
CUDA_VISIBLE_DEVICES=3 python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs weak_token_router \
  --task-variant bidirectional_lookup \
  --seeds 1 2 3 4 5 \
  --steps 800 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 1.0 \
  --induction-weight 1.0 \
  --branch-head-dims 64 \
  --router-supervision-weight 0.05 \
  --router-supervision-end-step 800 \
  --trajectory-eval-steps 0 400 450 500 550 600 650 700 750 800 \
  --device cuda \
  --output-dir results/phase3_toy_trajectory_consolidation_end800
```

Analysis:

```bash
python3 -u scripts/analyze_router_consolidation_window.py \
  --input-dir results/phase3_toy_trajectory_consolidation_end800 \
  --output-dir results/phase3_toy_trajectory_consolidation_end800_analysis
```

## Results

| Step | Local acc. | Ind. acc. | Gate match | Gate dist. | Causal match | Branch dist. | Same top |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0063 | 0.0064 | 0.0000 | 0.0000 | 0.0000 | 0.3903 | 0.6000 |
| 400 | 0.9991 | 0.9990 | 1.0000 | 0.1127 | 0.0000 | 0.1525 | 1.0000 |
| 450 | 0.9979 | 0.9987 | 1.0000 | 0.1398 | 0.4000 | 0.1845 | 0.6000 |
| 500 | 0.9989 | 0.9982 | 1.0000 | 0.1651 | 0.6000 | 0.2193 | 0.4000 |
| 550 | 0.9990 | 0.9994 | 1.0000 | 0.1914 | 1.0000 | 0.2663 | 0.0000 |
| 600 | 0.9981 | 0.9986 | 1.0000 | 0.2183 | 1.0000 | 0.3056 | 0.0000 |
| 650 | 0.9982 | 0.9982 | 1.0000 | 0.2343 | 1.0000 | 0.3458 | 0.0000 |
| 700 | 0.9981 | 0.9978 | 1.0000 | 0.2519 | 0.8000 | 0.3613 | 0.2000 |
| 750 | 0.9979 | 0.9991 | 1.0000 | 0.2702 | 1.0000 | 0.4125 | 0.0000 |
| 800 | 0.9986 | 0.9992 | 1.0000 | 0.3058 | 1.0000 | 0.4996 | 0.0000 |

Milestones:

| Milestone | Step |
|---|---:|
| First solved checkpoint with gate match 5/5 | 400 |
| First solved checkpoint with causal routed-role match 5/5 | 550 |
| First solved checkpoint with branch distance >= 0.30 | 600 |
| First solved checkpoint with branch distance >= 0.40 | 750 |

At step 800, the final model summary reproduces the prior end-800 result:

```text
local top branch: branch 0 in 5/5 seeds
induction top branch: branch 1 in 5/5 seeds
routed role match: 1.00
branch distance: 0.4996
```

## Interpretation

This strengthens the gate-before-causality result.

Supported:

```text
gate alignment is not sufficient for immediate causal modularity.
```

At step 400, the task is already solved and gate routed-role match is `1.00`,
but causal routed-role match is still `0.00`: local and induction depend most on
the same branch in 5/5 seeds.

Also supported:

```text
causal modularity consolidates quickly but not instantly after gate alignment.
```

The transition happens between steps 400 and 550:

- step 450: causal routed-role match `0.40`;
- step 500: causal routed-role match `0.60`;
- step 550: causal routed-role match `1.00`.

Separation strength continues to grow after the top-branch split appears:

- branch distance `0.2663` at step 550;
- branch distance `0.3056` at step 600;
- branch distance `0.4996` at step 800.

The step-700 dip in top-branch match (`0.80`) is a useful caution: the discrete
top-branch assignment can wobble near the transition, while branch-distance
strength continues to increase on average.

## Project-Level Update

The Phase 3 mechanism is now sharper:

```text
Role-informative routing pressure first aligns the gate, then gradually
consolidates branch computations. Causal branch modularity appears after the
task is solved and after the gate is already aligned.
```

This supports the broader paper warning:

```text
router metrics and probe metrics must be paired with causal ablations, because
they can become positive before the corresponding causal functional structure
has formed.
```

## Next Step

Do not run another generic router-regularization sweep. The next useful move is
to bridge this timing mechanism to a less hand-designed setting:

```text
test whether a routed attention model or SwitchHead-style small model shows the
same probe/gate-before-causal-modularity lag under checkpointed training.
```

Within the current toy setup, the remaining optional check is to repeat the
dense 400-800 window with more seeds only if this timing threshold becomes a
central quantitative claim.

## Artifacts

- Training/evaluation script:
  `scripts/toy_branch_isolation_intervention.py`
- Consolidation-window analyzer:
  `scripts/analyze_router_consolidation_window.py`
- Dense trajectory output:
  `results/phase3_toy_trajectory_consolidation_end800/`
- Analysis output:
  `results/phase3_toy_trajectory_consolidation_end800_analysis/`
