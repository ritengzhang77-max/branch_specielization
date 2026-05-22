# Phase 3 Toy Pilot: All-Layout Local-Weight Sweep

Date: 2026-05-22

This checkpoint generalizes the previous local-vs-induction weight sweep across
all four placements of the 64-dim head. The prior result used only:

```text
hetero4_64second = [16, 64, 16, 32]
```

That result supported a capacity-slot competition story, but it could have been
layout-specific. This run tests whether the same pressure effect appears when
the 64-dim head is first, second, third, or last.

## Claim Tested

The claim under test is:

```text
heterogeneous head dimensions create high-capacity attractor slots; local task
pressure changes whether induction occupies that 64-dim slot or is displaced to
secondary heads.
```

The strongest falsifier would be:

```text
induction top-slot dimension is unaffected by local pressure once all layouts
are tested.
```

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

The main readout is whether the top causal head for each role has dimension 64.

## Configurations

All models use total attention head dimension 128.

| Config | Head dimensions |
|---|---:|
| `hetero4_64first` | `[64, 16, 16, 32]` |
| `hetero4_64second` | `[16, 64, 16, 32]` |
| `hetero4_64third` | `[16, 32, 64, 16]` |
| `hetero4` | `[16, 16, 32, 64]` |

For each layout:

- seeds: 1 through 5;
- local weights: 0.00, 0.01, 0.10, 0.25, 1.00;
- induction weight: 1.00;
- 1200 training steps per seed.

This gives 100 trained models in the full grid.

## Commands

New all-layout runs:

```bash
for LW in 0.0 0.01 0.10 0.25; do
  TAG=${LW/./}
  OUT="results/phase3_toy_competition_all_layout_weights_lw${TAG}"
  python3 -u scripts/toy_competition_head_dim_intervention.py \
    --configs hetero4_64first hetero4_64third hetero4 \
    --seeds 1 2 3 4 5 \
    --steps 1200 \
    --batch-size 128 \
    --eval-examples 512 \
    --local-pairs 8 \
    --repeat-length 16 \
    --local-weight "$LW" \
    --induction-weight 1.0 \
    --random-controls 8 \
    --random-permutations 100 \
    --output-dir "$OUT"
done
```

Combined analysis:

```bash
python3 -u scripts/analyze_competition_all_layout_weight_sweep.py
```

The analysis combines:

- new runs for `hetero4_64first`, `hetero4_64third`, and `hetero4`;
- prior runs for `hetero4_64second`;
- prior equal-weight baselines for all four layouts.

## Aggregate Results

| Local weight | Models | Local acc. | Induction acc. | Local top 64 | Induction top 64 | Same top slot |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 20 | 0.2006 | 0.9979 | 0.25 | 0.95 | 0.15 |
| 0.01 | 20 | 0.9955 | 0.9979 | 0.90 | 0.95 | 0.50 |
| 0.10 | 20 | 0.9998 | 0.9983 | 0.85 | 0.60 | 0.60 |
| 0.25 | 20 | 1.0000 | 0.9986 | 0.60 | 0.25 | 0.45 |
| 1.00 | 20 | 1.0000 | 0.9993 | 0.95 | 0.50 | 0.40 |

The strongest pattern:

```text
induction uses the 64-dim head in 19/20 models when local pressure is absent or
tiny, but only 5/20 models at local_weight = 0.25.
```

## Per-Layout Induction Top-64 Rates

| Config | 0.00 | 0.01 | 0.10 | 0.25 | 1.00 |
|---|---:|---:|---:|---:|---:|
| `hetero4_64first` | 1.00 | 1.00 | 0.60 | 0.20 | 0.40 |
| `hetero4_64second` | 1.00 | 1.00 | 0.40 | 0.40 | 0.20 |
| `hetero4_64third` | 1.00 | 0.80 | 0.80 | 0.20 | 0.60 |
| `hetero4` | 0.80 | 1.00 | 0.60 | 0.20 | 0.80 |

The effect generalizes, but not as a strict monotonic curve. The all-layout
pattern is:

```text
low local pressure strongly favors induction-on-64;
moderate local pressure often displaces induction;
equal weighting reintroduces layout-specific mixed solutions.
```

## Interpretation

This strengthens the capacity-slot competition claim and narrows it.

Supported:

```text
The 64-dim head is a real capacity attractor for induction when local pressure is
absent or tiny. This holds in 19/20 models across all four 64-dim placements.
```

Also supported:

```text
Increasing local pressure changes role allocation. At local_weight = 0.25,
induction moves away from the 64-dim top slot in 15/20 models.
```

Not supported:

```text
The relationship between local pressure and induction-on-64 is strictly
monotonic.
```

The current best mechanism is:

```text
structural heterogeneity creates capacity-attractor slots; role allocation among
those slots is pressure-sensitive, layout-sensitive, and optimization-basin
sensitive.
```

This is still a useful causal story. It is just not a one-variable law.

## How This Changes The Paper Framing

The project should avoid claiming:

```text
larger heads always implement global/induction functions.
```

The stronger and more defensible claim is:

```text
structural heterogeneity breaks permutation symmetry by creating differentiated
capacity slots, and task pressure controls which functions occupy those slots.
```

This directly matches the user's high-level framing:

```text
Does structural modularity/heterogeneity lead to functional specialization or
modularity?
```

Current answer in the toy setting:

```text
yes for functional specialization stability; not yet established for functional
modularity.
```

## Caveats

- This is still a toy task.
- Equal-weight nonmonotonicity needs explanation; it may reflect loss-scale
  interactions, optimization basins, or redundant circuits.
- Single-head ablation is causal but still a coarse probe.
- The result does not yet prove modularity. It shows specialization and stable
  slot allocation.

## Decision

This is a good checkpoint result and supports continuing Phase 3. The next
experiment should test whether adding more than one attractive slot reduces
role displacement.

Recommended next test:

```text
two-attractor heterogeneous controls such as [16, 48, 48, 16],
[48, 16, 48, 16], and [48, 48, 16, 16]
```

Question:

```text
If there are two medium-large heads, do local and induction split more cleanly
instead of competing for one 64-dim head?
```

That is now the cleanest direct test of the capacity-attractor mechanism.

## Artifacts

- New run outputs:
  - `results/phase3_toy_competition_all_layout_weights_lw00/`
  - `results/phase3_toy_competition_all_layout_weights_lw001/`
  - `results/phase3_toy_competition_all_layout_weights_lw010/`
  - `results/phase3_toy_competition_all_layout_weights_lw025/`
- Combined analysis:
  `results/phase3_toy_competition_all_layout_weight_sweep/`
- Analysis script:
  `scripts/analyze_competition_all_layout_weight_sweep.py`
- Training/evaluation script:
  `scripts/toy_competition_head_dim_intervention.py`
