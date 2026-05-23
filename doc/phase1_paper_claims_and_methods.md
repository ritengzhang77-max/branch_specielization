# Phase 1 Paper-Facing Claims and Methods

Date: 2026-05-23

## Current Main Claim

For repeat/copy behavior in Pythia attention heads:

```text
Functional roles are stable across seeds after role-level relabeling, but the
right relabeling representation depends on the role strength.
```

This is a functional specialization and cross-seed role-stability claim. It is
not yet a functional modularity claim.

## Measurement Stack

The paper should report a two-stage alignment metric.

### 1. Generic Alignment Baseline

Use Hungarian matching over generic Phase 0 attention-pattern vectors.

Purpose:

- asks whether an unsupervised, task-agnostic representation recovers the role;
- gives a strong baseline against "we hand-picked the matching basis";
- works well for high-signal synthetic local-copy.

Limitation:

- underestimates weak natural-repeat roles;
- should not be treated as the only valid stability metric.

### 2. Role-Specific Alignment Measurement

Use Hungarian matching over attention vectors at the role-relevant positions on
the probe split, then evaluate causal transfer on held-out eval examples.

Purpose:

- asks whether heads implementing the same role can be relabeled across seeds;
- gives the role-level stability measurement;
- remains honest because matching and causal evaluation use disjoint examples.

Limitation:

- more role-informed than generic alignment;
- supports "role-level stability" rather than "generic attention patterns always
  recover the role."

## Required Reported Quantities

For each task/model:

- own-top excess over random heads;
- same-index source transfer;
- aligned source transfer;
- aligned-minus-same;
- pair-level CI and sign count;
- target-level CI and sign count;
- matched `step0` control where role-specific matching is used.

Do not report only aligned-minus-same. Same-index transfer is not a neutral null:
some raw slots occasionally transfer unusually well and can make
aligned-minus-same understate role transfer.

## Key Result Table

All values are aligned-minus-same loss-delta means.

| Task | Model | Generic Phase 0 | Role-specific | Step0 role-specific | Takeaway |
|---|---:|---:|---:|---:|---|
| synthetic local-copy | 160M | 1.7838 | 1.9593 | 0.0000 | high-signal role transfers under either basis |
| synthetic local-copy | 410M | 1.6554 | 3.7737 | -0.0004 | task-local matching recovers stronger 410M transfer |
| inserted WikiText repeated span | 160M | 0.0835 | 0.5645 | 0.0003 | natural repeated spans transfer strongly after role matching |
| inserted WikiText repeated span | 410M | 0.0455 | 0.1544 | -0.0003 | positive but weaker and more heterogeneous than 160M |
| exact 4-gram natural repeat | 160M | -0.0016 | 0.1897 | -0.0033 | unmodified natural repeats need role matching |
| exact 4-gram natural repeat | 410M | not run | 0.0215 | -0.0022 | weak/suggestive only |
| exact 8-gram natural repeat | 160M | 0.0063 | 0.2820 | -0.0018 | longer natural repeats strengthen 160M result |
| exact 8-gram natural repeat | 410M | -0.0022 | 0.0378 | -0.0002 | longer repeats improve 410M but remain weak |
| ordinary exact 8-gram repeat | 160M | 0.0137 | 0.2252 | 0.0012 | not just numbers/titles/names/tokenizer artifacts |
| ordinary exact 8-gram repeat | 410M | 0.0022 | 0.0327 | 0.0009 | filtered natural repeats give clean but small 410M transfer |

## Interpretation For The Research Question

The user-facing research question is:

```text
Does structural branch design induce stable functional specialization or
functional modularity across random seeds?
```

Phase 1 answers the prerequisite vanilla-attention question:

- yes, repeat/copy heads can be functionally specialized;
- yes, the function is cross-seed stable after role-level relabeling;
- no, the raw `(layer, head)` slot is not reliable enough as the unit;
- no, these experiments alone do not establish modularity.

The next paper step is to compare these vanilla-MHA role-stability measurements
against explicit branch or heterogeneous-head architectures.

## Caveats To State Explicitly

- Task-specific alignment is role-informed, so it is a measurement of role-level
  stability, not unsupervised discovery.
- Natural exact repeats are semantically mixed; ordinary-phrase filtering helps,
  but category and baseline-predictability analyses remain exploratory.
- 410M natural-repeat effects are smaller than 160M effects, so avoid simple
  monotonic model-size claims.
- Functional specialization is not functional modularity; modularity still needs
  separability/path/interaction tests.

## Suggested Paper Skeleton

1. Motivation: branch specialization claims need cross-seed functional evidence.
2. Definitions: structural heterogeneity, functional specialization, functional
   modularity, role-level relabeling.
3. Metrics: generic alignment, role-specific alignment, causal transfer,
   same-index baseline, `step0` controls.
4. Synthetic local-copy upper bound.
5. Inserted natural repeated-span validation.
6. Unmodified natural repeat validation and category filtering.
7. Heterogeneity analysis: same-index outliers and model-size caution.
8. Next stage: explicit branch/heterogeneous architectures and modularity tests.
