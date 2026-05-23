# Phase 3 Toy Pilot: Router Trajectory Checkpoint

Date: 2026-05-22

This checkpoint measures when role-aligned gate behavior and causally separable
branch computation appear during training. It follows the annealed weak-router
result, where labels through step 400 failed, labels through step 800 partially
persisted, and labels through step 1200 fully preserved top-branch role
separation after removal.

## Claim Tested

The prior result left two possibilities:

```text
Early labels may create the right router gate, but causal branch computations
may need more time under that routing pressure before they become separable.
```

or:

```text
The decisive variable may simply be final gate quality; once the gate points the
right way, branch modularity should follow.
```

The trajectory experiment distinguishes these by evaluating gate metrics and
causal branch-ablation metrics at intermediate optimizer-update counts.

## Setup

All conditions use:

```text
task_variant = bidirectional_lookup
branch_head_dims = [64]
local_weight = 1.0
induction_weight = 1.0
5 seeds
1600 training steps
```

Trajectory steps:

```text
0, 50, 100, 200, 400, 401, 600, 800, 801, 1000, 1200, 1201, 1400, 1600
```

Conditions:

| Condition | Router | Role-label schedule |
|---|---|---:|
| unlabeled entropy+balance | `learned_token_router` | none |
| end 400 | `weak_token_router` | steps `< 400` |
| end 800 | `weak_token_router` | steps `< 800` |
| end 1200 | `weak_token_router` | steps `< 1200` |
| always | `weak_token_router` | full run |

Representative command:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs weak_token_router \
  --task-variant bidirectional_lookup \
  --seeds 1 2 3 4 5 \
  --steps 1600 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 1.0 \
  --induction-weight 1.0 \
  --branch-head-dims 64 \
  --router-supervision-weight 0.05 \
  --router-supervision-end-step 800 \
  --trajectory-eval-steps 0 50 100 200 400 401 600 800 801 1000 1200 1201 1400 1600 \
  --device cuda \
  --output-dir results/phase3_toy_trajectory_end800
```

Analysis:

```bash
python3 -u scripts/analyze_router_trajectory_experiment.py
```

## Main Result

The gate learns the role split before the branches become causally modular.

At step 400, the model has already solved both roles and the gate is
role-aligned, but causal branch ablation still says the two roles are
co-located:

| Schedule | Step | Local acc. | Ind. acc. | Gate match | Gate distance | Routed causal match | Branch distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| end 400 | 400 | 0.9991 | 0.9990 | 1.00 | 0.1127 | 0.00 | 0.1525 |
| end 400 | 600 | 0.9994 | 0.9994 | 1.00 | 0.0688 | 0.00 | 0.1141 |
| end 400 | 1600 | 0.9995 | 0.9992 | 0.60 | 0.0395 | 0.00 | 0.0945 |

Continuing labels past step 400 changes the causal branch computation:

| Schedule | Step | Gate match | Gate distance | Routed causal match | Branch distance |
|---|---:|---:|---:|---:|---:|
| end 800 | 400 | 1.00 | 0.1127 | 0.00 | 0.1525 |
| end 800 | 600 | 1.00 | 0.2183 | 1.00 | 0.3056 |
| end 800 | 800 | 1.00 | 0.3058 | 1.00 | 0.4996 |
| end 800 | 1600 | 1.00 | 0.1488 | 0.80 | 0.3337 |

Labels through step 1200 make the split stronger before removal:

| Schedule | Step | Gate match | Gate distance | Routed causal match | Branch distance |
|---|---:|---:|---:|---:|---:|
| end 1200 | 800 | 1.00 | 0.3058 | 1.00 | 0.4996 |
| end 1200 | 1200 | 1.00 | 0.5539 | 1.00 | 0.8701 |
| end 1200 | 1400 | 1.00 | 0.4324 | 1.00 | 0.7983 |
| end 1200 | 1600 | 1.00 | 0.3831 | 1.00 | 0.7652 |

Always-on labels keep strengthening the split:

| Schedule | Step | Gate match | Gate distance | Routed causal match | Branch distance |
|---|---:|---:|---:|---:|---:|
| always | 1200 | 1.00 | 0.5539 | 1.00 | 0.8701 |
| always | 1400 | 1.00 | 0.6779 | 1.00 | 0.9499 |
| always | 1600 | 1.00 | 0.7497 | 1.00 | 0.9773 |

The unlabeled entropy+balance reference never becomes reliably role-aligned:

| Schedule | Step | Gate match | Gate distance | Routed causal match | Branch distance |
|---|---:|---:|---:|---:|---:|
| unlabeled | 400 | 0.20 | 0.0250 | 0.00 | 0.0433 |
| unlabeled | 800 | 0.20 | 0.0598 | 0.20 | 0.1188 |
| unlabeled | 1200 | 0.20 | 0.0980 | 0.20 | 0.2156 |
| unlabeled | 1600 | 0.00 | 0.1275 | 0.00 | 0.2069 |

## Interpretation

The important mechanism is a two-stage lag:

1. The router gate becomes role-aligned first.
2. Causal branch computations become role-separated later, only if the
   role-aligned routing pressure remains active long enough.

This explains why end 400 failed. It removed labels after task accuracy and gate
alignment were already high, but before the branches had become causally
separable. End 800 removed labels after the causal split had appeared, so the
top-branch role assignment mostly persisted but weakened. End 1200 removed
labels after the split had become much stronger, so top-branch modularity
persisted in all seeds, with some decay in branch distance.

Important caveat: early routed-match values before the task is solved should not
be overinterpreted. The decisive comparison is around step 400 and later, after
local and induction accuracies are near one.

## Project-Level Update

The strongest toy conclusion is now:

```text
Structural routing plus a role-informative training signal can produce
functional branch modularity, but gate alignment is only an intermediate state.
The branches need sustained role-aligned routing pressure after the task is
solved before causal functional modularity consolidates.
```

This refines the research framing from:

```text
Does structural modularity cause functional modularity?
```

to:

```text
Under what training signals and time windows does structural modularity become
causal functional modularity?
```

## Decision

The current toy branch has produced a coherent mechanism:

1. Unlabeled routing pressure does not reliably find role-aligned modularity.
2. Weak role labels make the gate role-aligned early.
3. Causal branch modularity appears later than gate alignment.
4. Removing labels after causal modularity appears preserves the split, but the
   split decays without continued pressure.
5. Removing labels before causal modularity appears fails, even if task accuracy
   and gate alignment are already high.

Recommended next step:

```text
Move from this toy mechanism to a paper-facing experiment: test whether the same
gate-before-causality lag appears in a less hand-designed routed attention setup
or in a small real transformer task with role probes and causal patching.
```

Within the toy setup, the most useful next narrow check would be a checkpointed
loss-trajectory analysis around steps 400 to 700 to locate the causal
consolidation window more precisely.

## Artifacts

- Training/evaluation script:
  `scripts/toy_branch_isolation_intervention.py`
- Trajectory analyzer:
  `scripts/analyze_router_trajectory_experiment.py`
- Analysis output:
  `results/phase3_toy_router_trajectory_analysis/`
- Plot:
  `results/phase3_toy_router_trajectory_analysis/router_trajectory_metrics.png`
- Checkpoint deck:
  `presentations/2026-05-22-1745-router-trajectory/router_trajectory_checkpoint.pdf`
- Per-condition trajectory outputs:
  `results/phase3_toy_trajectory_end400/`
  `results/phase3_toy_trajectory_end800/`
  `results/phase3_toy_trajectory_end1200/`
  `results/phase3_toy_trajectory_always_label/`
  `results/phase3_toy_trajectory_unlabeled_entropy_balance/`
