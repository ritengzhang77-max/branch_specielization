# Phase 3 Toy Pilot: Local-vs-Induction Weight Sweep

Date: 2026-05-22

This checkpoint tests whether the local and induction roles are competing for
the heterogeneous high-capacity slot. The previous layout-permutation checkpoint
showed:

- local/previous-token behavior almost always follows the 64-dim head;
- induction behavior is layout-sensitive and often moves to smaller layer-0
  heads.

The question here is:

```text
If local-task pressure is reduced, does induction reclaim the 64-dim slot?
```

## Claim Tested

Capacity-slot competition predicts:

```text
induction should use the 64-dim head when local pressure is weak or absent, but
may be displaced to smaller heads when local pressure is strong enough.
```

This run focuses on the most diagnostic layout from the prior checkpoint:

```text
hetero4_64second = [16, 64, 16, 32]
```

Under equal weights, induction used the 64-dim top slot in only 1/5 seeds.

## Task and Metric

Same mixed objective as before:

- local copy: `[x, SEP, x]`, scored at `SEP`;
- induction: `[y_1, ..., y_16, y_1, ..., y_16]`, scored on second-half
  continuation.

Each role is measured by single-head causal ablation:

```text
S_local(h)     = max(local_loss_after_ablating_h - local_baseline_loss, 0)
S_induction(h) = max(induction_loss_after_ablating_h - induction_baseline_loss, 0)
```

## Commands

New runs:

```bash
python3 -u scripts/toy_competition_head_dim_intervention.py \
  --configs hetero4_64second \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight <0.0|0.01|0.10|0.25> \
  --induction-weight 1.0 \
  --random-controls 8 \
  --random-permutations 100 \
  --output-dir results/phase3_toy_competition_weight_lw*
```

The `local_weight = 1.0` baseline is reused from
`results/phase3_toy_competition_layout_permutations`.

Combined analysis:

```bash
python3 -u scripts/analyze_competition_weight_sweep.py
```

## Results

| Local weight | Local acc. | Induction acc. | Local top 64 rate | Induction top 64 rate | Same top slot |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.2264 | 0.9981 | 0.20 | 1.00 | 0.20 |
| 0.01 | 0.9963 | 0.9976 | 1.00 | 1.00 | 0.60 |
| 0.10 | 0.9998 | 0.9976 | 0.80 | 0.40 | 0.60 |
| 0.25 | 0.9999 | 0.9980 | 0.80 | 0.40 | 0.20 |
| 1.00 | 1.0000 | 1.0000 | 1.00 | 0.20 | 0.20 |

Top induction dimension counts:

| Local weight | Induction top dimensions |
|---:|---|
| 0.00 | `{"64": 5}` |
| 0.01 | `{"64": 5}` |
| 0.10 | `{"16": 3, "64": 2}` |
| 0.25 | `{"16": 1, "32": 2, "64": 2}` |
| 1.00 | `{"16": 4, "64": 1}` |

## Interpretation

This supports the capacity-slot competition story, but the mechanism is not a
simple monotonic transfer.

Supported:

```text
When local pressure is absent, induction uses the 64-dim head in 5/5 seeds.
Therefore the [16, 64, 16, 32] layout does not intrinsically prevent induction
from occupying the high-capacity slot.
```

Also supported:

```text
As local pressure becomes substantial, induction is often displaced from the
64-dim top slot to 16-dim or 32-dim layer-0 slots.
```

The interesting nonlinear part:

```text
At local_weight = 0.01, both local and induction use 64-dim top slots in 5/5
seeds. The local objective is already learned, but the two roles can still share
or split across the 64-dim head in different layers.
```

So the current best mechanistic statement is:

```text
heterogeneous head dimensions create high-capacity attractor slots; weak task
pressure can cohabit those slots, while stronger competing pressure can displace
one role into secondary slots.
```

## Why This Matters

This is the strongest evidence so far that the architectural intervention is not
just producing arbitrary seed noise. The same structural slot changes role usage
systematically as the objective changes.

However, it also narrows the paper claim:

```text
The result is about symmetry breaking, capacity attraction, and competition
between functions. It is not evidence that a head dimension has a fixed semantic
meaning.
```

## Caveats

- This is one layout, chosen because it was maximally diagnostic.
- The sweep is still a toy task, not language-model pretraining.
- `local_weight = 0.0` leaves the local objective unsupervised, so local-role
  ablation scores at that point should not be interpreted as meaningful
  specialization.
- Single-head ablation can miss redundant circuits or overstate top-slot
  exclusivity.

## Decision

Continue Phase 3 with the capacity-slot competition framing.

Next decisive experiment:

1. Repeat the weight sweep across all four 64-dim placements to test whether
   the threshold pattern generalizes.
2. Add a matched-total-dimension two-capacity-slot condition, such as a
   heterogeneous layout with two medium-large heads, to test whether role
   displacement weakens when there is more than one attractive slot.
3. If both hold, Stage 3 has a coherent causal story: structural heterogeneity
   breaks permutation symmetry, creates capacity attractors, and task pressure
   determines role allocation among those attractors.

## Artifacts

- New run outputs:
  - `results/phase3_toy_competition_weight_lw000/`
  - `results/phase3_toy_competition_weight_lw001/`
  - `results/phase3_toy_competition_weight_lw010/`
  - `results/phase3_toy_competition_weight_lw025/`
- Combined analysis:
  `results/phase3_toy_competition_weight_sweep/`
- Analysis script:
  `scripts/analyze_competition_weight_sweep.py`
- Training/evaluation script:
  `scripts/toy_competition_head_dim_intervention.py`
