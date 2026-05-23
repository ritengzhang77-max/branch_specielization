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

| Condition | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---:|---:|---:|---:|---:|---:|
| `step0` | 9 | -0.0001 | 0.0015 | 0.0011 | -0.0004 | [-0.0030, 0.0019] |
| `step143000` 3-seed pilot | 3 | 0.0956 | 0.0079 | 0.0400 | 0.0321 | [-0.0074, 0.0633] |
| `step143000` all seeds | 9 | 0.1588 | 0.0464 | 0.0448 | -0.0016 | [-0.0548, 0.0360] |

All-seed final checkpoint details:

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

Not supported under this exact-repeat setup. Aligned transfer and same-index
transfer are effectively tied in mean.

This is a useful neutral/negative result, not a failure of the project. It says
the stricter natural-repeat task is causally real but does not currently expose
the same cross-layer role-relabeling advantage seen in synthetic local-copy or
inserted WikiText repeated spans.

Likely reasons:

- exact naturally occurring 4-token repeats are often article-specific names,
  headings, boilerplate, or local phrase reuse rather than a clean induction
  role;
- the 4-token span is short, so there are only 3 scored next-token positions per
  example;
- the generic Phase 0 raw-score alignment may not be specific enough for this
  weaker role;
- same-index heads may already capture enough of this broad natural-repeat
  behavior, eliminating the aligned-transfer advantage.

## Trustworthiness

Stronger parts:

- the script uses unmodified corpus windows;
- examples are inspectable in `example_rows.csv`;
- own-head causality is positive for 9/9 final-checkpoint seeds;
- the matched `step0` control is null.

Weaker parts:

- exact 4-token repeats are short and semantically mixed;
- WikiText-2 yields only a small exact-repeat candidate pool;
- aligned-minus-same has high variance and target-level CI crosses zero;
- this should not be used as a positive cross-seed alignment claim.

## Next Experiments

1. Use a larger corpus such as WikiText-103 or OpenWebText to support exact
   repeated spans of length 5-8 without replacement.
2. Filter repeated spans by lower baseline predictability, so the model must rely
   more on copying rather than ordinary language-model context.
3. Try alignment vectors built from the natural-repeat probe itself instead of
   the generic Phase 0 probe corpus.
4. Compare same-index vs aligned transfer separately for name/title repeats,
   numeric repeats, and ordinary phrase repeats.

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
