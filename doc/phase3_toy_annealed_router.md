# Phase 3 Toy Pilot: Annealed Weak Router Supervision

Date: 2026-05-22

This checkpoint tests whether role-informative routing pressure is needed only
as an early symmetry breaker, or whether it must remain active through most of
training for branch-level functional modularity to persist.

The prior conflict-heavy checkpoint showed:

```text
direct predecessor-vs-successor role conflict still did not make unlabeled
entropy/balance routing discover causal modularity.
```

But always-on weak routing labels did work. This run anneals those weak labels:
the router sees the local/induction branch target only for the first `N` training
steps, then the auxiliary routing loss is removed.

## Claim Tested

Positive early-symmetry-breaking outcome:

```text
brief early role labels are enough; once a branch split forms, task loss keeps it
intact after the labels are removed.
```

Continuous-pressure outcome:

```text
brief early role labels are not enough; causal modularity appears only when role
labels remain active through a large fraction of training.
```

## Setup

All annealed conditions use:

```text
task_variant = bidirectional_lookup
config = weak_token_router
branch_head_dims = [64]
router_supervision_weight = 0.05
local_weight = 1.0
induction_weight = 1.0
5 seeds
1600 training steps
```

The new schedule flag is:

```text
--router-supervision-end-step N
```

This means weak routing supervision is active only for optimization steps `< N`.
The default `-1` preserves the old always-on behavior.

Conditions:

| Condition | Label steps | Fraction of training |
|---|---:|---:|
| `anneal_label_end50` | 50 | 0.03125 |
| `anneal_label_end100` | 100 | 0.0625 |
| `anneal_label_end200` | 200 | 0.125 |
| `anneal_label_end400` | 400 | 0.25 |
| `anneal_label_end800` | 800 | 0.50 |
| `anneal_label_end1200` | 1200 | 0.75 |
| `always_label_0.05` | 1600 | 1.00 |

Representative command:

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs weak_token_router \
  --task-variant bidirectional_lookup \
  --seeds 1 \
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
  --output-dir results/phase3_toy_conflict_wide64_anneal_label_end800_seed1
```

Analysis:

```bash
python3 -u scripts/analyze_annealed_router_experiment.py
```

## Results

All annealed conditions solved the task. The amount of role supervision strongly
controlled whether causal modularity persisted.

| Condition | Local acc. | Ind. acc. | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|---:|---:|
| unlabeled entropy+balance | 0.9997 | 0.9998 | 1.00 | 0.00 | 0.2069 |
| `anneal_label_end50` | 0.9992 | 0.9998 | 1.00 | 0.00 | 0.0670 |
| `anneal_label_end100` | 0.9996 | 1.0000 | 1.00 | 0.00 | 0.0395 |
| `anneal_label_end200` | 1.0000 | 1.0000 | 1.00 | 0.00 | 0.0560 |
| `anneal_label_end400` | 0.9995 | 0.9992 | 1.00 | 0.00 | 0.0945 |
| `anneal_label_end800` | 0.9982 | 0.9989 | 0.20 | 0.80 | 0.3337 |
| `anneal_label_end1200` | 0.9999 | 1.0000 | 0.00 | 1.00 | 0.7652 |
| always label 0.05 | 0.9986 | 0.9997 | 0.00 | 1.00 | 0.9773 |
| oracle route | 0.9994 | 0.9995 | 0.00 | 1.00 | 1.0000 |

Router metrics:

| Condition | Gate routed match | Gate role distance | Global gate entropy |
|---|---:|---:|---:|
| `anneal_label_end50` | 0.00 | 0.0051 | 0.6894 |
| `anneal_label_end100` | 0.40 | 0.0135 | 0.6899 |
| `anneal_label_end200` | 0.40 | 0.0149 | 0.6901 |
| `anneal_label_end400` | 0.60 | 0.0395 | 0.6887 |
| `anneal_label_end800` | 1.00 | 0.1488 | 0.6780 |
| `anneal_label_end1200` | 1.00 | 0.3831 | 0.6408 |
| always label 0.05 | 1.00 | 0.7497 | 0.5093 |

## Interpretation

This is a mixed but informative result.

Supported:

```text
role-informative routing pressure does not need to remain active until the final
step for a top-branch split to persist.
```

`anneal_label_end1200` removed labels for the final 400 of 1600 steps and still
got routed role match 1.00 across 5/5 seeds. `anneal_label_end800` removed
labels halfway through training and still got routed role match 0.80.

Not supported:

```text
brief early labels are enough as a small symmetry-breaking nudge.
```

Removing labels at 50, 100, 200, or 400 steps left same-top-branch rate 1.00 and
routed role match 0.00 across 5/5 seeds. These models solved the task, but the
functions causally co-located.

Important nuance:

```text
top-branch modularity can persist after late label removal, but its strength
decays without continuous role pressure.
```

Branch distance:

- `anneal_label_end800`: 0.3337;
- `anneal_label_end1200`: 0.7652;
- always-on weak label: 0.9773.

So late removal preserved the discrete top-branch assignment, but did not fully
preserve near-oracle causal separation strength.

## Project-Level Update

The strongest toy conclusion is now:

```text
Functional branch modularity is not produced by structural branches, generic
unlabeled routing pressure, branch bottlenecks, or conflict alone. It is
reliably produced by role-informative routing pressure. That pressure can be
removed late in training with partial persistence, but brief early supervision is
not enough.
```

This reframes the project away from a naive "structure causes modularity" claim.
The more defensible research question is:

```text
What kind of training signal makes structural modularity become functional
modularity, and when during training must that signal be present?
```

## Decision

Do not continue broad sweeps of generic unlabeled regularizers. The toy evidence
has converged enough for this branch:

1. Explicit routing architecture is a substrate.
2. Role-informative pressure is the causal ingredient observed so far.
3. Continuous pressure is strongest, but late removal can preserve a weakened
   split.

Recommended next step:

```text
inspect training-time trajectories for the annealed runs, especially around
steps 400, 800, and 1200, to determine when gate separation and causal branch
separation first appear and when they decay after label removal.
```

That requires checkpointing intermediate models during training; the current
script only evaluates the final model.

## Artifacts

- Experiment script:
  `scripts/toy_branch_isolation_intervention.py`
- Analysis script:
  `scripts/analyze_annealed_router_experiment.py`
- Analysis output:
  `results/phase3_toy_annealed_router_analysis/`
- Checkpoint deck:
  `presentations/2026-05-22-1709-annealed-router/annealed_router_checkpoint.pdf`
- Per-seed outputs:
  `results/phase3_toy_conflict_wide64_anneal_label_end50_seed*/`
  `results/phase3_toy_conflict_wide64_anneal_label_end100_seed*/`
  `results/phase3_toy_conflict_wide64_anneal_label_end200_seed*/`
  `results/phase3_toy_conflict_wide64_anneal_label_end400_seed*/`
  `results/phase3_toy_conflict_wide64_anneal_label_end800_seed*/`
  `results/phase3_toy_conflict_wide64_anneal_label_end1200_seed*/`
