# Phase 3 Toy Pilot: Weak Token-Router Supervision Sweep

Date: 2026-05-22

This checkpoint follows the weak-router result by asking how much scored-position
routing supervision is needed before learned branch routing becomes causally
modular.

The tested router is `weak_token_router`, because the previous checkpoint showed
that token/state-dependent routing was the clean positive case at supervision
weight `0.05`.

## Claim Tested

The target claim is:

```text
there is a low but nonzero routing-pressure threshold above which learned gates
produce branch-level functional modularity.
```

The key failure mode to test is:

```text
the gates follow the target direction, but branch ablations still show that the
induction role depends more on the wrong branch.
```

## Setup

Shared setup:

```text
config = weak_token_router
two branch towers
branch_head_dims = [64]
shared token embedding and unembedding
local_weight = 0.25
induction_weight = 1.0
5 seeds
1200 training steps
```

Swept router supervision weights:

```text
0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05
```

The `0.05` run is the previous weak-router checkpoint. The `0.0` row below is
the earlier unconstrained `learned_token_router` baseline.

## Commands

Representative command:

```bash
python3 -u scripts/toy_branch_isolation_intervention.py \
  --configs weak_token_router \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --router-supervision-weight 0.02 \
  --output-dir results/phase3_toy_weak_token_router_w002_lw025
```

Analysis:

```bash
python3 -u scripts/analyze_weak_router_sweep.py
```

## Results

All weights solved the task. The threshold is not about task learning; it is
about whether the learned branch split becomes causal under ablation.

| Router weight | Local acc. | Induction acc. | Gate routed match | Routed role match | Branch distance | Induction top branches |
|---:|---:|---:|---:|---:|---:|---|
| 0.000 | 1.0000 | 0.9989 | 0.00 | 0.00 | 0.0006 | `{0:2, 1:3}` |
| 0.005 | 1.0000 | 0.9978 | 0.40 | 0.40 | 0.2789 | `{0:3, 1:2}` |
| 0.010 | 0.9998 | 0.9987 | 0.40 | 0.20 | 0.2050 | `{0:4, 1:1}` |
| 0.020 | 0.9994 | 0.9985 | 1.00 | 0.20 | 0.2844 | `{0:4, 1:1}` |
| 0.030 | 0.9999 | 0.9994 | 1.00 | 0.40 | 0.3910 | `{0:3, 1:2}` |
| 0.040 | 1.0000 | 0.9983 | 1.00 | 0.40 | 0.4504 | `{0:3, 1:2}` |
| 0.045 | 0.9999 | 0.9985 | 1.00 | 0.80 | 0.7459 | `{0:1, 1:4}` |
| 0.050 | 0.9998 | 0.9983 | 1.00 | 1.00 | 0.8957 | `{1:5}` |

Key per-seed pattern:

- local routing becomes branch-0-specific at every nonzero weight;
- induction routing is the limiting role;
- gate routed match reaches 1.00 by weight `0.02`;
- causal routed role match does not reach 1.00 until weight `0.05`;
- weight `0.045` is close but still fails in one seed.

## Gate-vs-Causality Dissociation

The most important finding is that gate compliance is not sufficient.

At weight `0.02`:

```text
gate_routed_role_match_rate = 1.00
routed_role_match_rate      = 0.20
```

So the router is already assigning local and induction positions to the intended
branches on average, but the branch towers have not reorganized causally. Ablating
branch 0 still hurts induction more than ablating branch 1 in 4/5 seeds.

At weight `0.05`:

```text
gate_routed_role_match_rate = 1.00
routed_role_match_rate      = 1.00
```

Only then do the gates and branch ablations agree across all seeds.

## Interpretation

This sweep strengthens the routing-pressure story:

```text
learned functional modularity appears only after the routing objective is strong
enough to reshape branch computations, not merely strong enough to bias gate
probabilities.
```

Current toy threshold:

```text
reliable 5/5 causal modularity: 0.05
near-threshold 4/5 modularity: 0.045
mixed or failed causal modularity: <= 0.04
```

This makes the project framing sharper:

- structural branches provide the substrate;
- weak labels can select a modular branch assignment;
- gate metrics alone can overstate modularity;
- causal branch ablation remains necessary.

## Caveats

- The threshold is specific to this toy architecture, task, optimizer, and seed
  set.
- The sweep used 5 seeds per weight; near-threshold behavior should be repeated
  with more seeds if it becomes a central claim.
- The supervision target explicitly names the role split; this is still not
  spontaneous modularity.
- The routing loss weight is relative to the averaged task loss, so different
  task weights could shift the threshold.

## Decision

Continue, but do not run more labeled-router sweeps until testing an unlabeled
alternative. The next decisive experiment should ask:

```text
Can routing entropy or load-balancing pressure, without role labels, produce any
of the same causal modularity shift?
```

If unlabeled pressure fails, the project can make a clean distinction:

```text
structural branches + learned gates are insufficient;
structural branches + explicit routing labels are sufficient;
unlabeled routing pressure is the open middle case.
```

## Artifacts

- Sweep analysis script:
  `scripts/analyze_weak_router_sweep.py`
- Sweep output:
  `results/phase3_toy_weak_token_router_sweep_analysis/`
- Per-weight run outputs:
  `results/phase3_toy_weak_token_router_w0005_lw025/`
  `results/phase3_toy_weak_token_router_w001_lw025/`
  `results/phase3_toy_weak_token_router_w002_lw025/`
  `results/phase3_toy_weak_token_router_w003_lw025/`
  `results/phase3_toy_weak_token_router_w004_lw025/`
  `results/phase3_toy_weak_token_router_w0045_lw025/`
  `results/phase3_toy_weak_router_w005_lw025/`
