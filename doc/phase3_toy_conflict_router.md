# Phase 3 Toy Pilot: Conflict-Heavy Branch Routing

Date: 2026-05-22

This checkpoint tests whether direct task conflict makes unlabeled routing
pressure align with functional roles.

The prior bottleneck checkpoint showed that shrinking each branch attention head
from 64 dims to 16 dims did not rescue unlabeled modularity. The remaining
hypothesis was that the task itself was still too compatible: local and induction
could cohabit one branch because their target rules did not directly disagree.

This run adds a new `bidirectional_lookup` task variant where the same query
token requires different retrieval directions in the two roles.

## Task Variant

For each sequence, sample a prefix:

```text
[y_0, y_1, ..., y_15]
```

For local scored positions, the query token is `y_i` and the target is its
prefix predecessor:

```text
local role:      y_i -> y_{i-1}
```

For induction scored positions, the same query token `y_i` appears in the second
copy of the prefix and the target is its prefix successor:

```text
induction role:  y_i -> y_{i+1}
```

Thus the same token identity requires incompatible predecessor-vs-successor
lookups depending on role. This is meant to make functional cohabitation more
costly than in the original local-copy plus induction task.

## Claim Tested

Positive outcome:

```text
when local and induction require incompatible lookup directions, unlabeled
entropy/balance routing pressure becomes sufficient to split the roles across
branches.
```

Negative outcome:

```text
even under direct role conflict, unlabeled routers solve the task while keeping
both roles causally co-located in the same branch.
```

## Setup

All conflict-task conditions use:

```text
task_variant = bidirectional_lookup
branch_head_dims = [64]
two branch towers
shared token embedding and unembedding
local_weight = 1.0
induction_weight = 1.0
5 seeds
1600 training steps
```

Conditions:

| Condition | Config | Entropy | Balance | Weak role labels |
|---|---|---:|---:|---:|
| `conflict_wide64_unconstrained` | `learned_token_router` | 0.00 | 0.00 | 0.00 |
| `conflict_wide64_balance_only_1.0` | `learned_token_router` | 0.00 | 1.00 | 0.00 |
| `conflict_wide64_entropy_0.05_balance_1.0` | `learned_token_router` | 0.05 | 1.00 | 0.00 |
| `conflict_wide64_weak_label_0.05` | `weak_token_router` | 0.00 | 0.00 | 0.05 |
| `conflict_wide64_oracle_route` | `oracle_route` | n/a | n/a | n/a |

Representative command:

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs learned_token_router \
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
  --router-supervision-weight 0.0 \
  --router-entropy-weight 0.05 \
  --router-balance-weight 1.0 \
  --output-dir results/phase3_toy_conflict_wide64_unlabeled_entropy005_balance1_seed1
```

Analysis:

```bash
python3 -u scripts/analyze_conflict_router_experiment.py
```

## Results

All conditions solved the conflict task. The unlabeled conditions still did not
produce role-aligned causal branch modularity.

| Condition | Local acc. | Ind. acc. | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|---:|---:|
| `conflict_wide64_unconstrained` | 0.9992 | 1.0000 | 1.00 | 0.00 | 0.0237 |
| `conflict_wide64_balance_only_1.0` | 0.9991 | 0.9994 | 1.00 | 0.00 | 0.0283 |
| `conflict_wide64_entropy_0.05_balance_1.0` | 0.9997 | 0.9998 | 1.00 | 0.00 | 0.2069 |
| `conflict_wide64_weak_label_0.05` | 0.9986 | 0.9997 | 0.00 | 1.00 | 0.9773 |
| `conflict_wide64_oracle_route` | 0.9994 | 0.9995 | 0.00 | 1.00 | 1.0000 |

Router metrics:

| Condition | Gate routed match | Gate role distance | Global balance error | Global gate entropy |
|---|---:|---:|---:|---:|
| `conflict_wide64_unconstrained` | 0.00 | 0.0068 | 0.0461 | 0.6893 |
| `conflict_wide64_balance_only_1.0` | 0.00 | 0.0032 | 0.0020 | 0.6903 |
| `conflict_wide64_entropy_0.05_balance_1.0` | 0.00 | 0.1275 | 0.0124 | 0.3160 |
| `conflict_wide64_weak_label_0.05` | 1.00 | 0.7497 | 0.2125 | 0.5093 |
| `conflict_wide64_oracle_route` | 1.00 | 1.0000 | 0.0000 | 0.4757 |

Comparison to the original standard task:

| Condition | Routed role match | Branch distance |
|---|---:|---:|
| standard wide64 entropy 0.05 + balance 1.0 | 0.40 | 0.3082 |
| conflict wide64 entropy 0.05 + balance 1.0 | 0.00 | 0.2069 |
| standard wide64 weak label 0.05 | 1.00 | 0.8957 |
| conflict wide64 weak label 0.05 | 1.00 | 0.9773 |
| conflict wide64 oracle route | 1.00 | 1.0000 |

## Interpretation

This is a negative result for the tested spontaneous-modularity hypothesis.

Supported:

```text
The conflict task is learnable and can support clean branch-level modularity
when routing is role-informative.
```

Weak labels and oracle routing both achieved routed role match 1.00 across 5/5
seeds, with branch distances 0.9773 and 1.0000 respectively.

Not supported:

```text
Direct predecessor-vs-successor conflict is enough for generic unlabeled
entropy/balance routing pressure to discover causal branch modularity.
```

The most important observation is that entropy+balance changed gate statistics
and increased branch-distribution distance somewhat, but the causal top branches
still co-located in 5/5 seeds. Balance-only made global usage nearly 50/50 while
leaving gate role distance and causal branch distance close to zero.

## Project-Level Update

The toy evidence now strongly separates three mechanisms:

1. **Correct routing is sufficient.** Oracle routing works on the standard,
   bottlenecked, and conflict-heavy variants.
2. **Role-informative pressure is sufficient.** Weak scored-position labels make
   branches modular even on the conflict task.
3. **Generic unlabeled pressure is not sufficient in these pilots.** Entropy,
   load balancing, branch bottlenecks, and direct role conflict have not produced
   spontaneous causal role splits.

Current strongest formulation:

```text
Structural branch design creates a substrate for modularity, but these toy
experiments do not support spontaneous role-aligned functional modularity from
generic unlabeled routing pressure. A role-informative signal, even weak, is the
reliable mechanism observed so far.
```

## Decision

The project should not claim that structural modularity by itself causes
functional modularity. The stronger and cleaner claim is conditional:

```text
branch structure plus role-informative routing pressure can create functional
modularity; branch structure plus generic unlabeled routing pressure often does
not, even when the task contains direct role conflict.
```

Recommended next experiment:

```text
annealed weak-label routing: provide role labels only early in training, then
remove them, to test whether modularity is an early symmetry-breaking effect or
requires continuous role supervision.
```

This is the next best test because the current evidence says the missing factor
is not capacity or conflict alone, but role-informative pressure.

## Artifacts

- Experiment script:
  `scripts/toy_branch_isolation_intervention.py`
- Analysis script:
  `scripts/analyze_conflict_router_experiment.py`
- Analysis output:
  `results/phase3_toy_conflict_router_analysis/`
- Per-seed outputs:
  `results/phase3_toy_conflict_wide64_unlabeled_unconstrained_seed*/`
  `results/phase3_toy_conflict_wide64_unlabeled_balance1_seed*/`
  `results/phase3_toy_conflict_wide64_unlabeled_entropy005_balance1_seed*/`
  `results/phase3_toy_conflict_wide64_weak_label005_seed*/`
  `results/phase3_toy_conflict_wide64_oracle_route_seed*/`
