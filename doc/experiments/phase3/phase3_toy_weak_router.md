# Phase 3 Toy Pilot: Weakly Supervised Branch Routing

Date: 2026-05-22

This checkpoint tests the next question after the unconstrained learned-router
result:

```text
What routing signal is sufficient for learned functional modularity to emerge?
```

The prior result showed that unconstrained position and token routers can solve
the task, but do not discover the local-vs-induction branch split on their own.

This run adds a small auxiliary routing loss on scored positions only:

```text
local scored positions -> branch 0
induction scored positions -> branch 1
```

The main task, architecture, seeds, and branch-ablation metrics are otherwise the
same as the learned-router checkpoint.

## Claim Tested

The positive outcome would be:

```text
a weak routing signal causes learned routers to become functionally modular,
as measured by branch ablations rather than only by gate values.
```

The negative outcome would be:

```text
the gates follow the auxiliary target, but both branches remain causally
entangled or both roles still depend most on the same branch.
```

## Configurations

| Config | Meaning |
|---|---|
| `weak_position_router` | Position router plus weak scored-position routing loss |
| `weak_token_router` | Token/state router plus weak scored-position routing loss |

Shared setup:

```text
two branch towers
branch_head_dims = [64]
shared token embedding and unembedding
local_weight = 0.25
induction_weight = 1.0
router_supervision_weight = 0.05
5 seeds
1200 training steps
```

## Metrics

Task metrics:

- local and induction accuracy;
- local and induction loss.

Causal branch metrics:

```text
S_local(branch)     = max(local_loss_after_ablating_branch - local_baseline_loss, 0)
S_induction(branch) = max(induction_loss_after_ablating_branch - induction_baseline_loss, 0)
```

Modularity metrics:

- `same_top_branch_rate`: local and induction depend most on the same branch;
- `routed_role_match_rate`: local top branch is 0 and induction top branch is 1;
- `branch_distribution_distance`: distance between local and induction causal
  branch-specialization distributions.

Gate metrics:

- mean local and induction gate weights by branch;
- gate entropy;
- gate distribution distance;
- `gate_target_nll_mean`, the negative log-probability of the weak routing
  targets.

## Commands

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs weak_position_router weak_token_router \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --router-supervision-weight 0.05 \
  --output-dir results/phase3_toy_weak_router_w005_lw025
```

Analysis:

```bash
python3 -u scripts/analyze_branch_isolation.py \
  --config-summary results/phase3_toy_weak_router_w005_lw025/config_summary.csv \
  --output-dir results/phase3_toy_weak_router_w005_analysis
```

## Results

Both weakly supervised routers learned the task.

| Config | Local acc. | Induction acc. | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|---:|---:|
| `weak_position_router` | 0.9999 | 0.9988 | 0.20 | 0.80 | 0.6446 |
| `weak_token_router` | 0.9998 | 0.9983 | 0.00 | 1.00 | 0.8957 |

Comparison to prior baselines:

| Config | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|
| `branch_sum` | 0.80 | 0.20 | 0.0951 |
| `learned_position_router` | 1.00 | 0.00 | 0.0665 |
| `learned_token_router` | 1.00 | 0.00 | 0.0006 |
| `weak_position_router` | 0.20 | 0.80 | 0.6446 |
| `weak_token_router` | 0.00 | 1.00 | 0.8957 |
| `oracle_route` | 0.00 | 1.00 | 1.0000 |

Gate behavior:

| Config | Gate routed match | Gate distance | Local entropy | Induction entropy | Gate target NLL |
|---|---:|---:|---:|---:|---:|
| `weak_position_router` | 1.00 | 0.4700 | 0.4811 | 0.6422 | 0.3139 |
| `weak_token_router` | 1.00 | 0.9049 | 0.0125 | 0.1944 | 0.0634 |

Branch ablation effects:

| Config | Local B0 | Local B1 | Induction B0 | Induction B1 |
|---|---:|---:|---:|---:|
| `weak_position_router` | 0.9813 | 0.0406 | 0.5780 | 1.3542 |
| `weak_token_router` | 5.6793 | 0.0000 | 0.5184 | 4.8444 |

Per-seed inspection:

- `weak_position_router` put local on branch 0 in 5/5 seeds, and induction on
  branch 1 in 4/5 seeds.
- `weak_token_router` put local on branch 0 and induction on branch 1 in 5/5
  seeds.

## Interpretation

This is a positive result for weakly supervised learned routing in this toy
setup.

Supported:

```text
a small scored-position routing objective can make learned branch routing
functionally modular.
```

Not supported:

```text
unconstrained routing alone is sufficient.
```

The token router is much cleaner than the position router:

- it makes gates nearly deterministic at local positions;
- it routes induction positions strongly but not perfectly to branch 1;
- branch ablations become close to the oracle pattern;
- residual induction dependence on branch 0 remains nonzero, so the solution is
  not perfectly oracle-like.

## Project-Level Update

The current toy evidence now suggests a more precise mechanism:

1. Heterogeneous capacity can stabilize functional specialization.
2. Extra capacity or separate branch towers do not automatically create
   functional modularity.
3. Unconstrained learned routing can solve the task without modularity.
4. Explicit oracle routing can produce modularity.
5. Weak scored-position routing supervision is sufficient to make a learned
   token router produce near-oracle branch-level functional modularity.

This is useful because it separates:

```text
structural branch capacity
learned gates
training signal on the gates
causal modularity under ablation
```

## Caveats

- The auxiliary loss directly names the desired role split, so this is not
  evidence for spontaneous modularity.
- Only one supervision weight, `0.05`, was tested here.
- The token router still leaves nonzero induction dependence on branch 0.
- This remains a synthetic toy task with known scored positions.

## Decision

Continue with a routing-pressure sweep and a harder unsupervised alternative:

1. Sweep lower router supervision weights, e.g. `0.005`, `0.01`, `0.02`, `0.05`,
   to estimate the minimum signal needed.
2. Test entropy and load-balancing regularization without explicit role labels.
3. Make the local and induction objectives more conflict-heavy to see whether
   modularity can emerge from task pressure rather than labels.

## Artifacts

- Updated model script:
  `scripts/toy_branch_isolation_intervention.py`
- Weak-router output:
  `results/phase3_toy_weak_router_w005_lw025/`
- Weak-router analysis:
  `results/phase3_toy_weak_router_w005_analysis/`
- Combined baseline comparison:
  `results/phase3_toy_weak_router_w005_combined/`
  `results/phase3_toy_weak_router_w005_combined_analysis/`
