# Phase 3 Toy Pilot: Bottlenecked Branch Routing

Date: 2026-05-22

This checkpoint tests whether a branch-capacity bottleneck makes unlabeled
routing pressure align with functional roles.

The prior unlabeled-router checkpoint showed that entropy minimization and load
balancing changed gate statistics but did not reliably create role-aligned
causal branch modularity. A plausible explanation was that each 64-dim branch
could cheaply carry both local-copy and induction computations. This run shrinks
each branch attention head from 64 dimensions to 16 dimensions.

## Claim Tested

Positive outcome:

```text
when branches are bottlenecked, sharp and balanced unlabeled routing becomes
sufficient to push local and induction computations into different branches.
```

Negative outcome:

```text
even with narrower branches, unlabeled routing regularizers can solve the task
while local and induction remain causally co-located.
```

## Setup

All bottlenecked conditions use:

```text
branch_head_dims = [16]
two branch towers
shared token embedding and unembedding
local_weight = 0.25
induction_weight = 1.0
5 seeds
1200 training steps
```

Conditions:

| Condition | Config | Entropy | Balance | Weak role labels |
|---|---|---:|---:|---:|
| `bottleneck16_unconstrained` | `learned_token_router` | 0.00 | 0.00 | 0.00 |
| `bottleneck16_balance_only_1.0` | `learned_token_router` | 0.00 | 1.00 | 0.00 |
| `bottleneck16_entropy_0.05_balance_1.0` | `learned_token_router` | 0.05 | 1.00 | 0.00 |
| `bottleneck16_weak_label_0.05` | `weak_token_router` | 0.00 | 0.00 | 0.05 |
| `bottleneck16_oracle_route` | `oracle_route` | n/a | n/a | n/a |

Representative command:

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs learned_token_router \
  --seeds 1 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --branch-head-dims 16 \
  --router-supervision-weight 0.0 \
  --router-entropy-weight 0.05 \
  --router-balance-weight 1.0 \
  --output-dir results/phase3_toy_bottleneck16_unlabeled_entropy005_balance1_seed1
```

Analysis:

```bash
python3 -u scripts/analyze_bottleneck_router_experiment.py
```

## Results

All conditions solved the task. The unlabeled bottlenecked conditions did not
produce role-aligned causal branch modularity.

| Condition | Local acc. | Ind. acc. | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|---:|---:|
| `bottleneck16_unconstrained` | 1.0000 | 0.9987 | 1.00 | 0.00 | 0.0220 |
| `bottleneck16_balance_only_1.0` | 1.0000 | 0.9997 | 1.00 | 0.00 | 0.0551 |
| `bottleneck16_entropy_0.05_balance_1.0` | 1.0000 | 0.9982 | 1.00 | 0.00 | 0.0004 |
| `bottleneck16_weak_label_0.05` | 0.9999 | 0.9991 | 0.20 | 0.80 | 0.6768 |
| `bottleneck16_oracle_route` | 0.9999 | 1.0000 | 0.00 | 1.00 | 1.0000 |

Router metrics:

| Condition | Gate routed match | Gate role distance | Global balance error | Global gate entropy |
|---|---:|---:|---:|---:|
| `bottleneck16_unconstrained` | 0.00 | 0.0594 | 0.5491 | 0.4759 |
| `bottleneck16_balance_only_1.0` | 0.60 | 0.0801 | 0.0017 | 0.6753 |
| `bottleneck16_entropy_0.05_balance_1.0` | 0.00 | 0.1041 | 0.0040 | 0.1589 |
| `bottleneck16_weak_label_0.05` | 1.00 | 0.8314 | 0.3096 | 0.3668 |
| `bottleneck16_oracle_route` | 1.00 | 1.0000 | 0.1228 | 0.4135 |

Comparison against the prior 64-dim branch result:

| Condition | Routed role match | Branch distance |
|---|---:|---:|
| wide64 entropy 0.05 + balance 1.0 | 0.40 | 0.3082 |
| bottleneck16 entropy 0.05 + balance 1.0 | 0.00 | 0.0004 |
| wide64 weak label 0.05 | 1.00 | 0.8957 |
| bottleneck16 weak label 0.05 | 0.80 | 0.6768 |
| bottleneck16 oracle route | 1.00 | 1.0000 |

## Interpretation

This is a negative result for the tested bottleneck hypothesis.

Supported:

```text
16-dim branches can support clean functional modularity when routing is correct.
```

The oracle-route condition proves this directly: same top branch was 0.00,
routed role match was 1.00, and branch distance was 1.00 across 5/5 seeds.

Not supported:

```text
a simple branch-capacity bottleneck makes generic unlabeled router regularizers
discover role-aligned causal branch modularity.
```

The key failure mode is sharper now than in the wide-branch experiment:

- Balance-only made global usage almost exactly 50/50, but causal local and
  induction dependence still landed in the same branch in 5/5 seeds.
- Entropy plus balance made gates sharp and balanced, but again kept both
  functions causally co-located in 5/5 seeds.
- Weak labels remained mostly effective, but less robust than with 64-dim
  branches: 4/5 causal routed role match instead of 5/5.

## Project-Level Update

The current toy evidence now separates four claims:

1. **Branch capacity is sufficient when routing is fixed.** Oracle routing works
   even with 16-dim branches.
2. **Weak role-informative routing pressure can induce functional modularity.**
   It works strongly with 64-dim branches and mostly with 16-dim branches.
3. **Generic unlabeled routing pressure is not enough in this setup.** Entropy
   and balance can produce sharp or balanced routing without causal modularity.
4. **Capacity bottleneck alone is not the missing ingredient.** Shrinking branch
   head dim from 64 to 16 did not rescue unlabeled modularity.

Current strongest formulation:

```text
Structural branch design and routing can create a substrate for functional
modularity, but in this toy setup role-informative pressure is needed. Generic
unlabeled gate pressure and simple branch bottlenecks do not reliably make local
and induction computations separate.
```

## Decision

Do not spend the next iteration on more entropy/balance weights for the same
task. The next decisive experiment should change the task so cohabitation is
directly costly, not only make branches narrower.

Recommended next experiment:

```text
conflict-heavy local-vs-induction task, where the same token/position patterns
create competing targets unless the model routes roles differently.
```

Useful variants:

1. Anti-correlated role labels: the local region and induction region require
   incompatible mappings for the same token identities.
2. Ambiguous shared-token layouts: local and induction positions share repeated
   tokens so one branch cannot solve both by a single copy heuristic.
3. Annealed weak-label control: compare pure unlabeled routing to routing labels
   that are present early and then removed, to test whether modularity can
   persist without ongoing labels.

## Artifacts

- Experiment script:
  `scripts/toy_branch_isolation_intervention.py`
- Analysis script:
  `scripts/analyze_bottleneck_router_experiment.py`
- Analysis output:
  `results/phase3_toy_bottleneck16_router_analysis/`
- Per-seed outputs:
  `results/phase3_toy_bottleneck16_unlabeled_unconstrained_seed*/`
  `results/phase3_toy_bottleneck16_unlabeled_balance1_seed*/`
  `results/phase3_toy_bottleneck16_unlabeled_entropy005_balance1_seed*/`
  `results/phase3_toy_bottleneck16_weak_label005_seed*/`
  `results/phase3_toy_bottleneck16_oracle_route_seed*/`
