# Phase 1: Natural-Repeat Heterogeneity Notes

Date: 2026-05-23

## Question

The WikiText-103 exact 8-gram runs made the natural-repeat result cleaner for
160M but still weak for 410M. This memo asks what drives the remaining
heterogeneity.

## Target-Level Pattern

Pythia-160M exact 8-grams:

- task-repeat alignment is positive on 8/9 targets;
- the negative target is seed 2, where same-index transfer is unusually high
  (`0.1662`) and task-aligned transfer is lower (`0.0696`);
- the strongest positive target is seed 7, where task-aligned transfer nearly
  matches the own-head effect (`1.0804` aligned transfer vs `1.0620` own excess).

Pythia-410M exact 8-grams:

- own-head repeat causality is positive on 8/9 targets;
- task-repeat aligned-minus-same is positive on 8/9 targets;
- the weak/negative cases are target seed 4 and target seed 6:
  - seed 4 has negative own-head excess (`-0.0958`), so there is little target
    role to transfer;
  - seed 6 has positive own-head excess (`0.0232`) but very high same-index
    transfer (`0.1177`), making aligned-minus-same negative (`-0.1058`).

## Pair-Level Outliers

The biggest 410M exact 8-gram failure is not a generally bad aligned mapping.
It is one very strong same-index source-target pair:

| Run | Target | Source | Same-index | Aligned | Aligned - same |
|---|---:|---:|---:|---:|---:|
| 410M exact 8-gram, task-repeat | 6 | 3 | 0.8055 | 0.0599 | -0.7456 |
| 410M exact 8-gram, Phase 0 | 6 | 3 | 0.8055 | 0.0502 | -0.7553 |
| 410M exact 4-gram, task-repeat | 6 | 3 | 0.9588 | 0.0577 | -0.9011 |

This same-index outlier persists across the 4-gram and 8-gram tasks, so it is
not an accident of the longer WikiText-103 sample. It means target seed 6 is a
bad target for the aligned-minus-same metric because one source seed already
transfers unusually well without relabeling.

There are also strong positive 410M pair-level examples:

| Run | Target | Source | Same-index | Aligned | Aligned - same |
|---|---:|---:|---:|---:|---:|
| 410M exact 8-gram, task-repeat | 3 | 5 | -0.0013 | 0.1914 | 0.1927 |
| 410M exact 8-gram, task-repeat | 3 | 7 | -0.0013 | 0.1914 | 0.1927 |
| 410M exact 8-gram, task-repeat | 5 | 3 | -0.0360 | 0.1239 | 0.1599 |

## Semantic Mixture

I added `scripts/analyze_natural_repeat_categories.py` to classify the repeated
span text with simple heuristics. The categories are not interpretability
labels; they are only a practical way to decide what to stratify next.

Full evaluation-set counts:

| Run | n | Ordinary | Numeric/date | Quoted/title | Proper-name-like | Tokenizer markup |
|---|---:|---:|---:|---:|---:|---:|
| 160M WikiText-2 exact 4-gram | 64 | 27 | 8 | 3 | 8 | 17 |
| 160M WikiText-103 exact 8-gram | 128 | 35 | 20 | 16 | 25 | 32 |
| 410M WikiText-103 exact 8-gram | 128 | 29 | 25 | 21 | 18 | 35 |

The exact-repeat task is therefore not one semantic behavior. It mixes ordinary
phrases, numbers/dates, titles/quotes, names, and tokenization artifacts such as
`@-@`. This supports stratifying future runs before making model-size claims.

## Interpretation

The 410M natural-repeat weakness should not be summarized as "alignment does not
work." A more accurate statement is:

```text
410M exact-repeat transfer is small because own-head causality is small in most
targets, and aligned-minus-same is sensitive to a few unusually strong same-index
source-target pairs.
```

This matters for the paper framing. Same-index transfer is a useful baseline,
but it is not a neutral null: occasionally the raw `(layer, head)` slot already
lands on a good source role. In those cases, aligned-minus-same can understate
role transfer even when aligned heads are directionally positive.

## Next Checks

1. Report aligned transfer, same-index transfer, and aligned-minus-same together;
   do not rely only on the difference metric.
2. For 410M natural repeats, inspect semantic classes and baseline
   predictability before claiming a model-size trend.
3. Consider a robustness summary that trims the largest same-index outlier, but
   treat that as exploratory unless pre-registered in the analysis plan.
