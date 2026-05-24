# Phase 3 Toy Pilot: Explicit Branch Isolation

Date: 2026-05-22

This checkpoint tests whether explicit structural separation/routing is needed
to turn functional specialization into functional modularity.

The previous two-attractor experiment showed:

```text
adding a second high-dimensional head does not automatically make local and
induction split into separate modules.
```

So this run moves from capacity heterogeneity to explicit branch structure.

## Claim Tested

The claim under test:

```text
structural separation alone may be insufficient; explicit routing may be needed
for robust functional modularity.
```

The key contrast:

| Config | Meaning |
|---|---|
| `branch_sum` | Two separate branch towers, both active at every scored position |
| `oracle_route` | Local scored positions use branch 0; induction scored positions use branch 1 |

Both configs use:

```text
branch_head_dims = [64]
two branches
shared token embedding and unembedding
separate attention/MLP towers
```

## Task and Metric

Same local-vs-induction task:

- local copy: `[x, SEP, x]`, scored at `SEP`;
- induction: `[y_1, ..., y_16, y_1, ..., y_16]`, scored on second-half
  continuation.

Training weights:

```text
local_weight = 0.25
induction_weight = 1.0
```

Branch-level causal metrics:

```text
S_local(branch)     = max(local_loss_after_ablating_branch - local_baseline_loss, 0)
S_induction(branch) = max(induction_loss_after_ablating_branch - induction_baseline_loss, 0)
```

Modularity readouts:

- `same_top_branch_rate`: whether local and induction depend most on the same
  branch;
- `routed_role_match_rate`: whether local top branch is 0 and induction top
  branch is 1;
- `branch_distribution_distance`: distance between local and induction branch
  specialization distributions.

## Commands

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs branch_sum oracle_route \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --output-dir results/phase3_toy_branch_isolation_lw025
```

Analysis:

```bash
python3 -u scripts/analyze_branch_isolation.py
```

## Results

Both configs learned the task.

| Config | Local acc. | Induction acc. | Same top branch | Routed role match | Branch role distance |
|---|---:|---:|---:|---:|---:|
| `branch_sum` | 1.0000 | 0.9998 | 0.60 | 0.20 | 0.0411 |
| `oracle_route` | 0.9999 | 0.9987 | 0.00 | 1.00 | 1.0000 |

Branch ablation effects:

| Config | Local B0 | Local B1 | Induction B0 | Induction B1 |
|---|---:|---:|---:|---:|
| `branch_sum` | 0.1860 | 0.1672 | 0.9538 | 0.8308 |
| `oracle_route` | 6.0104 | 0.0000 | 0.0000 | 6.1801 |

In `branch_sum`, both branches support both roles. The local and induction
branch distributions are almost identical:

```text
branch_distribution_distance_mean = 0.0411
```

In `oracle_route`, ablation is perfectly role-specific:

```text
local top branch:     branch 0 in 5/5 seeds
induction top branch: branch 1 in 5/5 seeds
```

## Interpretation

This is the cleanest modularity result so far.

Supported:

```text
explicit routing can produce functional modularity in this toy setting.
```

Also supported:

```text
branch separation alone is not enough. Without routing, the two branches remain
functionally entangled and both support both roles.
```

Not supported:

```text
separate branch towers automatically discover separate local and induction
modules.
```

The current best project-level conclusion:

```text
structural heterogeneity can stabilize specialization;
explicit routing/separation is needed to get robust modularity.
```

## Why This Matters

This directly answers the user's reframing:

```text
Does structural modularity or specialization lead to functional modularity or
specialization?
```

Current toy answer:

```text
structural heterogeneity -> functional specialization stability: yes, often.
structural heterogeneity -> functional modularity: no, not by itself.
explicit structural routing -> functional modularity: yes, in the oracle toy.
```

The distinction is now experimentally grounded rather than just conceptual.

## Caveats

- `oracle_route` is an upper-bound intervention, not a naturally learned router.
- It uses task-region knowledge at the scored positions.
- The two branch towers have more parameters than a single shared transformer
  block, so this is not a parameter-matched final result.
- The task is still synthetic.

## Decision

Continue Phase 3, but split the architectural story into two claims:

1. Capacity heterogeneity supports stable specialization.
2. Routing or stronger structural separation is required for modularity.

Next decisive experiment:

```text
replace oracle routing with learned or weakly supervised routing, then test
whether modularity emerges without hard-coded task-region assignment.
```

Candidate variants:

- learned position router with entropy/load-balancing penalty;
- soft branch gates conditioned on token/position;
- region-blind learned router tested for whether it discovers local vs
  induction branches.

## Artifacts

- New script:
  `scripts/toy_branch_isolation_intervention.py`
- Analysis script:
  `scripts/analyze_branch_isolation.py`
- Run output:
  `results/phase3_toy_branch_isolation_lw025/`
- Analysis output:
  `results/phase3_toy_branch_isolation_analysis/`
