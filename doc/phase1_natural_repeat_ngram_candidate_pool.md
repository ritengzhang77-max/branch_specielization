# Phase 1: Naturally Occurring Repeat-Ngram Candidate-Pool Alignment

Date: 2026-05-23

## Question

The WikiText repeated-span probe inserts a second copy of a natural span:

```text
prefix + span + distractor + span
```

That is more natural than synthetic `[x, SEP, x]`, but it still constructs the
repeat. This follow-up asks a stricter question:

```text
Do trained heads causally support naturally occurring repeated phrases already
present in WikiText, and does cross-seed alignment transfer that function better
than same-index transfer?
```

## Method

Dataset:

- Hugging Face `wikitext`, config `wikitext-2-raw-v1`, train split;
- first 4000 non-heading rows, tokenized with the target Pythia tokenizer;
- token stream length: 50152.

Example construction:

- scan 96-token windows with stride 8;
- skip windows containing EOS;
- find an exact repeated 4-token n-gram whose first token starts on a
  whitespace/newline token boundary;
- require at least 8 intervening tokens beyond the repeated span;
- use the original 96-token window as the model input, without inserting or
  modifying tokens.

The 160M all-seed run found 722 candidate windows and sampled 64 probe plus 64
evaluation windows without replacement.

Scoring:

- attention probe: mean attention from the second occurrence of the repeated
  n-gram to the matching positions in the first occurrence;
- causal readout: next-token loss over positions in the second occurrence;
- selected heads: top 2 heads by probe score from all layers;
- alignment: Hungarian matching over raw attention-score vectors on the shared
  Phase 0 probe corpus;
- transfer comparison: own top heads, random candidate heads, same-index source
  heads, and cross-layer aligned source heads.

## Results

Pythia-160M all-layer exact-repeat results:

| Condition | Alignment source | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|---:|
| `step0` | Phase 0 generic | 9 | -0.0001 | 0.0015 | 0.0011 | -0.0004 | [-0.0030, 0.0019] |
| `step0` | task repeat | 9 | -0.0001 | 0.0015 | -0.0018 | -0.0033 | [-0.0060, -0.0006] |
| `step143000` 3-seed pilot | Phase 0 generic | 3 | 0.0956 | 0.0079 | 0.0400 | 0.0321 | [-0.0074, 0.0633] |
| `step143000` all seeds | Phase 0 generic | 9 | 0.1588 | 0.0464 | 0.0448 | -0.0016 | [-0.0548, 0.0360] |
| `step143000` all seeds | task repeat | 9 | 0.1588 | 0.0464 | 0.2361 | 0.1897 | [0.0737, 0.3140] |

All-seed final checkpoint details under generic Phase 0 alignment:

- own top excess was positive for 9/9 target seeds;
- target-level own top excess CI: `[0.0806, 0.2718]`;
- aligned-minus-same was positive for 6/9 target seeds;
- pair-level aligned-better count was 48/72, with sign test `p=0.0063`;
- despite that sign count, the mean aligned-minus-same was `-0.0016`, because a
  few same-index-transfer outliers offset many small positive pairwise gains.

The matched `step0` control was null:

- own top excess: `-0.0001`;
- aligned-minus-same: `-0.0004`;
- target-level CIs crossed zero for both own excess and aligned-minus-same.

Task-repeat alignment changes the transfer result:

- it aligns heads using their attention vectors on the natural-repeat probe
  positions rather than generic Phase 0 texts;
- evaluation remains held out: the alignment uses the 64 probe windows, while
  causal loss is measured on 64 separate evaluation windows;
- aligned transfer rises from `0.0448` to `0.2361`;
- aligned-minus-same rises from `-0.0016` to `0.1897`;
- aligned-minus-same is positive for 8/9 targets;
- pair-level aligned-better count is 66/72;
- target-level aligned-minus-same CI is `[0.0737, 0.3140]`.

The matched `step0` task-repeat alignment control is not positive:

- own top excess: `-0.0001`;
- aligned-minus-same: `-0.0033`;
- target-level aligned-minus-same CI: `[-0.0060, -0.0006]`.

## Interpretation

This probe separates two claims:

```text
Trained Pythia-160M heads causally support naturally occurring repeated phrases.
```

Supported. The own-head causal effect is positive in all 9 seeds and absent at
initialization.

```text
Cross-seed role alignment improves transfer over same-index heads on naturally
occurring exact repeats.
```

Supported only when the alignment basis is role-specific. Generic Phase 0
attention-score matching does not recover the transfer advantage, but
task-repeat attention matching does.

This is a useful methodological result. It says the stricter natural-repeat task
is causally real, and cross-seed transfer exists, but the matching representation
must be close to the role being transferred.

Likely reasons:

- exact naturally occurring 4-token repeats are often article-specific names,
  headings, boilerplate, or local phrase reuse rather than a clean induction
  role;
- the 4-token span is short, so there are only 3 scored next-token positions per
  example;
- the generic Phase 0 raw-score alignment is not specific enough for this weaker
  role;
- same-index heads may already capture enough of this broad natural-repeat
  behavior to obscure the advantage unless the matching basis is role-specific.

## Trustworthiness

Stronger parts:

- the script uses unmodified corpus windows;
- examples are inspectable in `example_rows.csv`;
- own-head causality is positive for 9/9 final-checkpoint seeds;
- both `step0` controls are null with respect to positive transfer;
- the task-specific alignment result uses held-out evaluation windows.

Weaker parts:

- exact 4-token repeats are short and semantically mixed;
- WikiText-2 yields only a small exact-repeat candidate pool;
- generic-alignment aligned-minus-same has high variance and target-level CI
  crosses zero;
- the task-specific alignment result is stronger but more role-informed, so it
  should be reported as a method-dependent positive result.

## Next Experiments

1. Use a larger corpus such as WikiText-103 or OpenWebText to support exact
   repeated spans of length 5-8 without replacement.
2. Filter repeated spans by lower baseline predictability, so the model must rely
   more on copying rather than ordinary language-model context.
3. Compare same-index vs aligned transfer separately for name/title repeats,
   numeric repeats, and ordinary phrase repeats.
4. Test whether task-specific alignment also improves the inserted-span WikiText
   result and the synthetic local-copy result.

## Files

- Script: `scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`.
- 14M smoke:
  `results/debug_pythia14m_natural_repeat_ngram_candidate_pool/`.
- Pythia-160M 3-seed pilot:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed3/`.
- Pythia-160M all-seed result:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed9/`.
- Pythia-160M all-seed `step0` control:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed9_step0/`.
- Pythia-160M all-seed task-repeat alignment result:
  `results/phase1_pythia160m_natural_repeat_ngram_task_alignment_seed9/`.
- Pythia-160M all-seed task-repeat alignment `step0` control:
  `results/phase1_pythia160m_natural_repeat_ngram_task_alignment_seed9_step0/`.
