# Phase 3 Toy Pilot: Learned Branch Routing

Date: 2026-05-22

This checkpoint tests whether a model can learn the branch routing that produced
functional modularity in the previous oracle-routing experiment.

The previous checkpoint showed:

```text
oracle routing can produce clean functional modularity,
but separate branch towers without routing stay entangled.
```

This run asks:

```text
Can learned routing discover role-specific branches without hard-coded local ->
branch 0 and induction -> branch 1 assignment?
```

## Claim Tested

The positive outcome would be:

```text
learned branch gates route local and induction positions to different branches,
and branch ablations become role-specific.
```

The competing outcome:

```text
learned gates solve the task but route both roles through the same branch or
keep both branches functionally entangled.
```

## Configurations

| Config | Meaning |
|---|---|
| `branch_sum` | Two branch towers, both active everywhere |
| `learned_position_router` | Softmax branch gate learned per sequence position |
| `learned_token_router` | Softmax branch gate learned from token/position representation |
| `oracle_route` | Local scored positions use branch 0; induction scored positions use branch 1 |

All configs use:

```text
two branch towers
branch_head_dims = [64]
shared token embedding and unembedding
local_weight = 0.25
induction_weight = 1.0
5 seeds
1200 training steps
```

## Metrics

Task metrics:

- local and induction accuracy;
- local and induction loss.

Branch-level causal metrics:

```text
S_local(branch)     = max(local_loss_after_ablating_branch - local_baseline_loss, 0)
S_induction(branch) = max(induction_loss_after_ablating_branch - induction_baseline_loss, 0)
```

Modularity metrics:

- `same_top_branch_rate`: local and induction depend most on the same branch;
- `routed_role_match_rate`: local top branch is 0 and induction top branch is 1;
- `branch_distribution_distance`: distance between local and induction
  branch-specialization distributions.

Gate metrics:

- mean local and induction gate weights for each branch;
- gate entropy;
- gate distribution distance between local and induction positions.

## Commands

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs branch_sum learned_position_router learned_token_router oracle_route \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --output-dir results/phase3_toy_learned_router_lw025
```

Analysis:

```bash
python3 -u scripts/analyze_branch_isolation.py \
  --config-summary results/phase3_toy_learned_router_lw025/config_summary.csv \
  --output-dir results/phase3_toy_learned_router_analysis
```

## Results

All configs learned the task.

| Config | Local acc. | Induction acc. | Same top branch | Routed role match | Branch distance |
|---|---:|---:|---:|---:|---:|
| `branch_sum` | 1.0000 | 0.9997 | 0.80 | 0.20 | 0.0951 |
| `learned_position_router` | 0.9999 | 0.9987 | 1.00 | 0.00 | 0.0665 |
| `learned_token_router` | 1.0000 | 0.9989 | 1.00 | 0.00 | 0.0006 |
| `oracle_route` | 1.0000 | 0.9988 | 0.00 | 1.00 | 1.0000 |

Gate behavior:

| Config | Gate routed match | Gate distance | Local entropy | Induction entropy |
|---|---:|---:|---:|---:|
| `branch_sum` | 0.00 | 0.0000 | 0.6931 | 0.6931 |
| `learned_position_router` | 0.40 | 0.0200 | 0.6924 | 0.6914 |
| `learned_token_router` | 0.00 | 0.1045 | 0.1593 | 0.3340 |
| `oracle_route` | 1.00 | 1.0000 | 0.0000 | 0.0000 |

Branch ablation effects:

| Config | Local B0 | Local B1 | Induction B0 | Induction B1 |
|---|---:|---:|---:|---:|
| `branch_sum` | 0.2017 | 0.1316 | 0.8995 | 0.7907 |
| `learned_position_router` | 0.2437 | 0.2964 | 0.7832 | 1.0852 |
| `learned_token_router` | 2.2221 | 2.7767 | 2.3307 | 3.2506 |
| `oracle_route` | 5.8198 | 0.0000 | 0.0000 | 6.1639 |

## Interpretation

This is a negative result for unconstrained learned routing.

Supported:

```text
learned routers can solve the task.
```

Not supported:

```text
unconstrained learned routers discover role-specific branch modularity.
```

The two learned routers failed in different ways:

- `learned_position_router` kept gates almost uniform and high-entropy. It showed
  tiny local-vs-induction gate differences, but causal branch ablations stayed
  entangled and both roles had the same top branch in 5/5 seeds.
- `learned_token_router` learned low-entropy gates, but it routed local and
  induction through the same top branch in every seed. Its causal branch
  distributions were almost identical across roles.

The oracle upper bound still works:

```text
hard-coded role routing gives same_top_branch_rate = 0.00 and
routed_role_match_rate = 1.00.
```

## Project-Level Update

The current toy evidence now separates four levels:

1. **Capacity heterogeneity** can stabilize functional specialization.
2. **Extra capacity / extra large heads** does not imply functional modularity.
3. **Separate branch towers without routing** do not imply functional modularity.
4. **Explicit routing** can produce functional modularity.

This checkpoint adds:

```text
unconstrained learned routing is not enough, at least in this toy setup.
```

So the next claim to test is not simply "learned routers exist," but:

```text
what training signal or routing constraint is sufficient for learned functional
modularity?
```

## Caveats

- The learned routers were unconstrained: no entropy penalty, load-balancing
  penalty, or weak routing supervision was used.
- The task may be too easy for modular routing to be necessary; both roles can
  be solved by shared or cohabiting computation.
- The token router can use position information through the hidden state, but it
  was not forced to represent the local/induction distinction.
- This remains a toy task.

## Decision

Continue, but sharpen the learned-routing question:

```text
Does a weak routing objective, load-balancing pressure, or a more conflict-heavy
task make learned routing become functionally modular?
```

Recommended next tests:

1. Add weak router supervision on scored positions while keeping the main task
   unchanged.
2. Add entropy or load-balancing regularization to avoid trivial same-branch
   routing.
3. Make the two roles more conflicting so a shared branch solution is less
   attractive.

## Artifacts

- Updated model script:
  `scripts/toy_branch_isolation_intervention.py`
- Analysis script:
  `scripts/analyze_branch_isolation.py`
- Run output:
  `results/phase3_toy_learned_router_lw025/`
- Analysis output:
  `results/phase3_toy_learned_router_analysis/`
