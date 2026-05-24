# Phase 3 Toy Pilot: Local-vs-Induction Layout Permutations

Date: 2026-05-22

This checkpoint follows up the local-vs-induction competition result. The prior
run showed that heterogeneous head dimensions increase causal specialization,
but the clean claim "small heads become local and large heads become induction"
did not hold. This run asks a narrower question:

```text
When the 64-dim head is moved to each possible head position, do functions follow
the head dimension, the head index, or the broader layout?
```

## Claim Tested

The positive version of the structural-heterogeneity claim would be:

```text
the same function repeatedly occupies the same structural slot type, even when
the 64-dim head is moved.
```

The competing explanation is:

```text
heterogeneity creates attractive high-capacity slots, but task competition and
layout decide which function occupies each slot.
```

## Task and Metric

The task is the same mixed objective from
`phase3_toy_competition_head_dim_intervention`:

- local copy region: `[x, SEP, x]`, scored at `SEP`;
- induction region: `[y_1, ..., y_16, y_1, ..., y_16]`, scored on second-half
  continuation positions.

Each role is measured by single-head causal ablation:

```text
S_local(h)     = max(local_loss_after_ablating_h - local_baseline_loss, 0)
S_induction(h) = max(induction_loss_after_ablating_h - induction_baseline_loss, 0)
```

The layout analysis reconstructs the top causal slot for each role from
`model_summary.csv`.

## Commands

New permutation run:

```bash
python3 -u scripts/toy_competition_head_dim_intervention.py \
  --configs hetero4_64second hetero4_64third \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --random-controls 8 \
  --random-permutations 100 \
  --output-dir results/phase3_toy_competition_layout_permutations
```

Combined layout analysis:

```bash
python3 -u scripts/analyze_competition_layout_permutations.py
```

## Layout Results

All new models learned both objectives, so the result is not explained by
undertraining.

| Config | Head dimensions | Local acc. | Induction acc. | Local top 64 rate | Induction top 64 rate |
|---|---:|---:|---:|---:|---:|
| `hetero4_64first` | `[64, 16, 16, 32]` | 0.9999 | 0.9986 | 1.00 | 0.40 |
| `hetero4_64second` | `[16, 64, 16, 32]` | 1.0000 | 1.0000 | 1.00 | 0.20 |
| `hetero4_64third` | `[16, 32, 64, 16]` | 1.0000 | 0.9993 | 1.00 | 0.60 |
| `hetero4` | `[16, 16, 32, 64]` | 1.0000 | 0.9992 | 0.80 | 0.80 |

Across the four heterogeneous layouts:

| Role | Models | Top head is 64-dim | Mean top specialization | Mean top loss delta |
|---|---:|---:|---:|---:|
| Local | 20 | 19/20 | 0.9180 | 0.1457 |
| Induction | 20 | 10/20 | 0.7388 | 2.5272 |

## Slot-Conditional Pattern

The local role follows the 64-dim head position almost perfectly:

| Config | Local top slot counts |
|---|---|
| `hetero4_64first` | `L1H0:d64` in 5/5 seeds |
| `hetero4_64second` | `L1H1:d64` in 5/5 seeds |
| `hetero4_64third` | `L1H2:d64` in 5/5 seeds |
| `hetero4` | `L1H3:d64` in 4/5 seeds, `L0H0:d16` in 1/5 |

The induction role does not follow dimension as cleanly:

| Config | Induction top dimension counts |
|---|---|
| `hetero4_64first` | `d16`: 3/5, `d64`: 2/5 |
| `hetero4_64second` | `d16`: 4/5, `d64`: 1/5 |
| `hetero4_64third` | `d16`: 2/5, `d64`: 3/5 |
| `hetero4` | `d16`: 1/5, `d64`: 4/5 |

## Interpretation

This strengthens the "heterogeneity as symmetry breaker" story, but weakens the
stronger "head dimension directly determines semantic role" story.

Supported:

```text
The 64-dim head forms a stable high-capacity slot, and the local/previous-token
role very often occupies that slot across seeds and across head positions.
```

Not supported:

```text
Large heads reliably become induction/global heads while small heads reliably
become local heads.
```

The current best statement is:

```text
structural heterogeneity can stabilize function-to-slot mappings, but the
function assigned to each slot is mediated by task pressure and layout.
```

This also explains why the previous competition checkpoint looked mixed: the
local role appears to compete strongly for the high-capacity 64-dim slot, leaving
the induction role to use either the 64-dim slot or a 16-dim layer-0 slot
depending on layout and seed.

## Decision

Continue Phase 3, but keep the claim narrow:

```text
heterogeneous head dimensions are evidence for structural symmetry breaking and
stable slot formation, not evidence for an automatic semantic taxonomy of head
dimensions.
```

Next decisive experiment:

1. Sweep local-vs-induction objective weights on the same task.
2. Test whether reducing local pressure lets induction occupy the 64-dim slot
   more often.
3. If the top role flips with objective weight, frame the mechanism as
   capacity-slot competition rather than dimension-to-function mapping.

## Artifacts

- New run output:
  `results/phase3_toy_competition_layout_permutations/`
- Combined analysis output:
  `results/phase3_toy_competition_layout_analysis/`
- Analysis script:
  `scripts/analyze_competition_layout_permutations.py`
- Training/evaluation script:
  `scripts/toy_competition_head_dim_intervention.py`
