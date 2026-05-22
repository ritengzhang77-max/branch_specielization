# Phase 3 Toy Pilot: Unlabeled Router Regularization

Date: 2026-05-22

This checkpoint tests whether unlabeled router pressure can produce the same
causal branch modularity that weak role labels produced.

The prior checkpoint found:

```text
gate routed match can become correct before causal branch modularity appears;
weak role supervision at weight 0.05 produced 5/5 causal modularity.
```

This run removes the role labels and instead uses generic router regularizers:

- entropy minimization: make per-position gates sharper;
- load balancing: keep global branch usage near 50/50;
- entropy plus load balancing.

## Claim Tested

The positive outcome would be:

```text
unlabeled routing pressure is sufficient for learned branches to become
functionally modular.
```

The negative outcome would be:

```text
regularizers change router statistics, but branch ablations still show local and
induction cohabiting the same branch in most seeds.
```

## Setup

All unlabeled conditions use:

```text
config = learned_token_router
two branch towers
branch_head_dims = [64]
shared token embedding and unembedding
local_weight = 0.25
induction_weight = 1.0
5 seeds
1200 training steps
no role-label routing loss
```

Conditions:

| Condition | Entropy weight | Balance weight |
|---|---:|---:|
| `unconstrained_learned_token` | 0.00 | 0.00 |
| `entropy_only_0.05` | 0.05 | 0.00 |
| `balance_only_1.0` | 0.00 | 1.00 |
| `entropy_0.05_balance_1.0` | 0.05 | 1.00 |
| `entropy_0.10_balance_1.0` | 0.10 | 1.00 |

The weak-label and oracle rows are comparison references, not unlabeled
conditions.

## Commands

Representative unlabeled command:

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
  --router-supervision-weight 0.0 \
  --router-entropy-weight 0.05 \
  --router-balance-weight 1.0 \
  --output-dir results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed1
```

Analysis:

```bash
python3 -u scripts/analyze_unlabeled_router_regularization.py
```

## Results

All unlabeled conditions solved the task. None produced reliable role-aligned
causal branch modularity.

| Condition | Local acc. | Ind. acc. | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|---:|---:|
| `unconstrained_learned_token` | 1.0000 | 0.9989 | 1.00 | 0.00 | 0.0006 |
| `entropy_only_0.05` | 1.0000 | 0.9994 | 1.00 | 0.00 | 0.0002 |
| `balance_only_1.0` | 1.0000 | 0.9993 | 1.00 | 0.00 | 0.0033 |
| `entropy_0.05_balance_1.0` | 1.0000 | 0.9993 | 0.60 | 0.40 | 0.3082 |
| `entropy_0.10_balance_1.0` | 1.0000 | 0.9984 | 0.80 | 0.20 | 0.1484 |
| `weak_label_0.05` | 0.9998 | 0.9983 | 0.00 | 1.00 | 0.8957 |
| `oracle_route` | 1.0000 | 0.9988 | 0.00 | 1.00 | 1.0000 |

Router metrics:

| Condition | Global balance error | Global gate entropy | Local-vs-induction gate distance | Gate routed match |
|---|---:|---:|---:|---:|
| `entropy_only_0.05` | 0.9597 | 0.0351 | 0.0095 | 0.00 |
| `balance_only_1.0` | 0.0017 | 0.5385 | 0.1718 | 0.00 |
| `entropy_0.05_balance_1.0` | 0.0038 | 0.2830 | 0.2667 | 0.40 |
| `entropy_0.10_balance_1.0` | 0.0118 | 0.1247 | 0.1512 | 0.20 |

Per-seed causal top-branch counts:

| Condition | Local top branches | Induction top branches |
|---|---|---|
| `entropy_only_0.05` | `{0:3, 1:2}` | `{0:3, 1:2}` |
| `balance_only_1.0` | `{0:2, 1:3}` | `{0:2, 1:3}` |
| `entropy_0.05_balance_1.0` | `{0:4, 1:1}` | `{0:2, 1:3}` |
| `entropy_0.10_balance_1.0` | `{0:3, 1:2}` | `{0:2, 1:3}` |
| `weak_label_0.05` | `{0:5}` | `{1:5}` |

## Interpretation

This is a negative result for the tested unlabeled router regularizers.

Supported:

```text
unlabeled regularizers can strongly change gate statistics.
```

Not supported:

```text
entropy or load-balancing pressure alone reliably induces role-aligned causal
branch modularity.
```

The failure modes are different:

- Entropy-only makes gates sharp, but each seed collapses both roles onto the
  same branch.
- Balance-only makes global branch usage almost exactly 50/50, but still keeps
  local and induction causally co-located.
- Entropy plus balance sometimes finds a role split, but not reliably. The best
  unlabeled condition reached routed role match 0.40, far below weak labels at
  1.00.

## Project-Level Update

This checkpoint separates three notions that can be confused:

1. **Sharp routing**: low gate entropy.
2. **Balanced routing**: both branches used equally.
3. **Functional modularity**: local and induction depend on different branches
   under ablation.

The toy evidence says that (1) and (2), even together, are not sufficient for
(3) in this setup.

Current strongest formulation:

```text
Structural branches and learned gates provide a substrate for modularity, but
functional modularity requires role-informative routing pressure or an equivalent
task pressure. Generic unlabeled gate regularizers are not enough in this toy
setting.
```

## Caveats

- Only a small set of entropy and balance weights was tested.
- The positive weak-label result used explicit role information, so it is a
  sufficiency result, not a spontaneous-emergence result.
- The task may still be too easy; local and induction can cohabit one branch
  while solving the objective.
- The branch labels are exchangeable, so top-branch counts should be read as
  co-location vs separation, not literal branch identity unless role labels are
  used.

## Decision

The next experiment should change the task pressure rather than add more generic
router regularizer weights:

```text
make local and induction computations conflict more directly, then test whether
unlabeled routing pressure begins to align with functional roles.
```

Candidate directions:

1. Anti-correlated role targets, where sharing a branch creates interference.
2. Branch bottlenecks, where a single branch cannot cheaply carry both roles.
3. Routing regularizers plus a conflict-heavy task, compared to the same
   regularizers on the current easy task.

## Artifacts

- Updated model script:
  `scripts/toy_branch_isolation_intervention.py`
- Analysis script:
  `scripts/analyze_unlabeled_router_regularization.py`
- Analysis output:
  `results/phase3_toy_unlabeled_router_regularization_analysis/`
- Per-seed outputs:
  `results/phase3_toy_unlabeled_token_router_entropy005_lw025_seed*/`
  `results/phase3_toy_unlabeled_token_router_balance1_lw025_seed*/`
  `results/phase3_toy_unlabeled_token_router_entropy005_balance1_lw025_seed*/`
  `results/phase3_toy_unlabeled_token_router_entropy010_balance1_lw025_seed*/`
