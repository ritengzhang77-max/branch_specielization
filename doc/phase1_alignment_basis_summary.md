# Phase 1: Alignment Basis Summary

Date: 2026-05-23

## Core Finding

The naturalistic experiments changed the methodological conclusion:

```text
For weak natural repeated-span roles, cross-seed transfer depends strongly on
the alignment representation.
```

Generic Phase 0 raw-score matching can detect the large synthetic local-copy
role, but it underestimates weaker natural roles. Matching on role-specific
attention vectors from the task probe split recovers much stronger held-out
causal transfer.

## Result Table

All values are aligned-minus-same loss-delta means. Positive means cross-seed
aligned source heads transfer the target function better than same-index source
heads.

| Task | Model | Generic Phase 0 alignment | Task-specific alignment | Step0 task-specific control | Interpretation |
|---|---:|---:|---:|---:|---|
| synthetic `[x, SEP, x]` local-copy | 160M | 1.7838 | 1.9593 | 0.0000 | high-signal role transfers under both generic and task-local alignment |
| synthetic `[x, SEP, x]` local-copy | 410M | 1.6554 | 3.7737 | -0.0004 | task-local alignment recovers much stronger final-checkpoint transfer |
| inserted WikiText repeated span, 64/64 | 160M | 0.0835 | 0.5645 | 0.0003 | task-specific alignment reveals much stronger natural transfer |
| inserted WikiText repeated span, 128/128 | 160M | 0.0816 | 0.4773 | not rerun | larger-sample replication stays strong |
| inserted WikiText repeated span, 64/64 | 410M | 0.0455 | 0.1544 | -0.0003 | task-specific alignment rescues 410M, but with seed-6 outlier |
| inserted WikiText repeated span, 128/128 | 410M | 0.0293 | 0.1158 | not rerun | larger-sample replication stays positive but weaker than 160M |
| naturally occurring exact 4-gram repeat | 160M | -0.0016 | 0.1897 | -0.0033 | unmodified natural repeats need role-specific alignment |
| naturally occurring exact 4-gram repeat | 410M | not run | 0.0215 | -0.0022 | weak own-head signal, no clean target-level transfer |
| naturally occurring exact 8-gram repeat, WikiText-103 | 160M | 0.0063 | 0.2820 | -0.0018 | longer natural repeats strengthen 160M task-specific transfer |
| naturally occurring exact 8-gram repeat, WikiText-103 | 410M | -0.0022 | 0.0378 | -0.0002 | longer repeats improve 410M but remain weak/heterogeneous |
| ordinary-phrase exact 8-gram repeat, WikiText-103 | 160M | 0.0137 | 0.2252 | 0.0012 | category filtering preserves 160M task-specific transfer |

## Interpretation

The cleanest current statement is:

```text
Functional repeat/copy roles are stable across seeds after role-level
relabeling, but weak natural roles require role-specific alignment features.
```

This refines the earlier framing. The project should not treat "Hungarian
alignment over generic attention score matrices" as one universal alignment
metric. It is a good generic baseline, but it is not sufficient for weak,
task-local roles.

The current hierarchy is:

1. Synthetic local-copy: high signal; generic alignment works, and task-local
   alignment acts as an upper-bound sanity check.
2. Inserted natural repeated spans: medium signal; generic alignment is positive
   but much too conservative; task-specific alignment is strong.
3. Naturally occurring exact repeats: low/noisy under 4-token WikiText-2, but
   cleaner with 8-token WikiText-103 repeats. The 160M result is clearly
   positive under task-specific alignment; 410M improves but remains weak and
   heterogeneous.

## What This Means For The Research Question

The project question was:

```text
Does structural/role specialization lead to functional specialization or
functional modularity?
```

For these Pythia repeat/copy roles, the answer is currently:

- trained heads are functionally specialized for repeat/copy behavior;
- this specialization is cross-seed transferable after the right role-level
  relabeling;
- the raw `(layer, head)` slot is not the right unit;
- the alignment representation is part of the measured phenomenon, not just a
  technical detail;
- none of these results by themselves prove modularity, only functional
  specialization and cross-seed role stability.

## Caution

Task-specific alignment is more role-informed than generic alignment. That is
not a flaw, but it changes the claim. The claim is no longer:

```text
Any generic attention-pattern alignment recovers the role.
```

The claim is:

```text
If we align heads using features that actually describe the role, source heads
transfer the role across seeds on held-out examples.
```

That distinction should be explicit in any paper draft.

## Next Best Checks

1. Inspect the remaining heterogeneous cases, especially 410M natural repeats
   and the 410M repeated-span seed-6 outlier, before making any model-size
   scaling claim.
2. Consider a two-stage metric in the paper: generic alignment as an unsupervised
   baseline, task-specific alignment as the role-level measurement.
3. Filter repeated spans by baseline predictability and semantic class to test
   whether copy-reliant repeats give cleaner cross-seed transfer than
   easy-to-predict boilerplate or entity repeats.

## Files

- Synthetic local-copy memo:
  `doc/phase1_pythia160m_local_copy_candidate_pool.md`.
- Inserted WikiText repeated-span memo:
  `doc/phase1_naturalistic_span_candidate_pool.md`.
- Naturally occurring repeat-ngram memo:
  `doc/phase1_natural_repeat_ngram_candidate_pool.md`.
- Natural-repeat heterogeneity memo:
  `doc/phase1_natural_repeat_heterogeneity.md`.
