# Phase 3 Toy Pilot: Two-Attractor Layouts

Date: 2026-05-22

This checkpoint tests the next implication of the capacity-attractor story. The
previous all-layout weight sweep showed that a single 64-dim head is an
attractor for induction when local pressure is absent or tiny, but induction is
often displaced under moderate local pressure.

The natural next hypothesis was:

```text
if there are two attractive heads, local and induction should split between them
instead of competing for one head.
```

That clean split did not happen.

## Claim Tested

The positive version of the claim:

```text
two medium-large heads should reduce role collision and increase modular role
separation.
```

The falsifier:

```text
even with two attractive heads, local and induction still collapse onto the same
top slot or move to smaller secondary heads.
```

## Task and Metric

Same local-vs-induction competition task:

- local copy: `[x, SEP, x]`, scored at `SEP`;
- induction: `[y_1, ..., y_16, y_1, ..., y_16]`, scored on second-half
  continuation.

Each role is measured by single-head causal ablation:

```text
S_local(h)     = max(local_loss_after_ablating_h - local_baseline_loss, 0)
S_induction(h) = max(induction_loss_after_ablating_h - induction_baseline_loss, 0)
```

The test uses the strongest competition setting from the previous checkpoint:

```text
local_weight = 0.25
induction_weight = 1.0
```

## Configurations

Two-48 heterogeneous layouts:

| Config | Head dimensions |
|---|---:|
| `hetero4_two48_center` | `[16, 48, 48, 16]` |
| `hetero4_two48_skip` | `[48, 16, 48, 16]` |
| `hetero4_two48_front` | `[48, 48, 16, 16]` |

Control:

| Config | Head dimensions | Purpose |
|---|---:|---|
| `uniform2` | `[64, 64]` | Two true 64-dim heads, fewer-head control |

The analysis compares these against the prior single-64 heterogeneous layouts at
the same local weight.

## Commands

Two-48 run:

```bash
python3 -u scripts/toy_competition_head_dim_intervention.py \
  --configs hetero4_two48_center hetero4_two48_skip hetero4_two48_front \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --random-controls 8 \
  --random-permutations 100 \
  --output-dir results/phase3_toy_competition_two48_lw025
```

Uniform two-64 control:

```bash
python3 -u scripts/toy_competition_head_dim_intervention.py \
  --configs uniform2 \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --local-pairs 8 \
  --repeat-length 16 \
  --local-weight 0.25 \
  --induction-weight 1.0 \
  --random-controls 8 \
  --random-permutations 100 \
  --output-dir results/phase3_toy_competition_uniform2_lw025
```

Combined analysis:

```bash
python3 -u scripts/analyze_competition_two_attractor.py
```

## Results

All models learned the task, so the result is not explained by failed training.

| Family | Models | Local acc. | Induction acc. | Local top max-dim | Induction top max-dim | Same top slot | Distinct max-dim slots |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single 64 hetero | 20 | 1.0000 | 0.9986 | 0.60 | 0.25 | 0.45 | 0.20 |
| Two 48 hetero | 15 | 0.9999 | 0.9986 | 0.33 | 0.27 | 0.80 | 0.07 |
| Two 64 uniform | 5 | 0.9999 | 0.9990 | 1.00 | 1.00 | 0.80 | 0.20 |

For the two-48 heterogeneous layouts:

```text
local top dimension counts:     {"16": 10, "48": 5}
induction top dimension counts: {"16": 11, "48": 4}
```

So the two 48-dim heads did not behave like two attractive role slots. The top
causal heads were usually 16-dim, and local/induction shared the same top slot
in 12/15 models.

For `uniform2 = [64, 64]`:

```text
local and induction were necessarily 64-dim in 5/5 models,
but they shared the same exact top slot in 4/5 models.
```

This also fails the clean modular split prediction.

## Interpretation

This is a negative result for the simple two-attractor hypothesis.

Supported:

```text
Adding more high-dimensional capacity does not automatically produce modular
role separation.
```

Also supported:

```text
The local and induction roles can cohabit or collapse onto the same top causal
slot even when there are multiple high-capacity heads available.
```

Not supported:

```text
two medium-large heads are enough to make local and induction split into
separate functional modules.
```

The current best mechanism becomes:

```text
structural heterogeneity can create stable functional specialization slots, but
functional modularity requires more than differentiated head capacity.
```

This directly sharpens the project's central conceptual distinction:

```text
specialization does not imply modularity.
```

## Why This Matters

This is important for the paper framing. Earlier results supported the idea that
structural heterogeneity can stabilize role allocation. This checkpoint shows
that stabilized role allocation is not the same as modular decomposition.

The toy evidence now says:

```text
structural heterogeneity can influence which heads become important, but it does
not force different functions into different heads.
```

This is a stronger and more careful claim than "heterogeneous branches produce
modular circuits."

## Caveats

- The two-48 heads may simply be below the threshold required to act like the
  64-dim head in this task.
- `uniform2` changes head count as well as head size, so it is a control, not a
  pure heterogeneous intervention.
- Local and induction may share enough computational structure that a single
  head can support both roles.
- Single-head ablation measures causal importance but not full circuit
  separability.

## Decision

This checkpoint narrows the Stage 3 claim:

```text
heterogeneous capacity can stabilize specialization, but it is insufficient
evidence for functional modularity.
```

The next decisive experiment should test whether explicit structural separation,
not just capacity heterogeneity, is needed for modularity.

Recommended next tests:

1. Add an explicit branch-isolation or routing intervention and compare against
   heterogeneous dimensions alone.
2. Use a more conflict-heavy two-role task where one shared head is less likely
   to solve both roles.
3. Add path-patching or multi-head ablation to distinguish role cohabitation
   from redundant separate circuits.

## Artifacts

- New run outputs:
  - `results/phase3_toy_competition_two48_lw025/`
  - `results/phase3_toy_competition_uniform2_lw025/`
- Combined analysis:
  `results/phase3_toy_competition_two_attractor/`
- Analysis script:
  `scripts/analyze_competition_two_attractor.py`
- Training/evaluation script:
  `scripts/toy_competition_head_dim_intervention.py`
