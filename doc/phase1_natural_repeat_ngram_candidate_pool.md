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

Pythia-410M task-repeat alignment result:

| Condition | Alignment source | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|---:|
| `step0` | task repeat | 9 | -0.0013 | 0.0005 | -0.0017 | -0.0022 | [-0.0042, -0.0002] |
| `step143000` all seeds | task repeat | 9 | 0.0503 | 0.0173 | 0.0388 | 0.0215 | [-0.0166, 0.0564] |

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

The 410M exact-repeat result is weaker:

- own top excess is positive with target CI `[0.0068, 0.0873]`;
- own top excess is positive for 8/9 targets;
- aligned-minus-same is only `0.0215`;
- target-level aligned-minus-same CI crosses zero: `[-0.0166, 0.0564]`;
- target positives are 5/9.

So exact naturally occurring repeats extend to 410M as a weak trained causal
signal, but not as a clean 410M alignment-transfer result under the current
4-token WikiText setup.

## WikiText-103 Exact 8-Gram Extension

To reduce the concern that 4-token repeats are too short or too accidental, I
reran the natural-repeat probe on a larger standard corpus:

- dataset: Hugging Face `wikitext`, config `wikitext-103-raw-v1`, train split;
- first 20000 rows, token stream length `500024`;
- context length `128`;
- exact repeat span length `8`;
- minimum gap `8`;
- candidate windows found: `524`;
- sampled windows: 128 probe plus 128 evaluation, without replacement.

Candidate-count scan on the same 500k-token stream:

| Exact span length | Candidate windows |
|---:|---:|
| 5 | 3839 |
| 6 | 2049 |
| 7 | 980 |
| 8 | 524 |

Pythia-160M WikiText-103 exact 8-gram results:

| Condition | Alignment source | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|---:|
| `step0` | task repeat | 9 | -0.0010 | 0.0006 | -0.0012 | -0.0018 | [-0.0035, -0.0001] |
| `step143000` | Phase 0 generic | 9 | 0.3718 | 0.0334 | 0.0397 | 0.0063 | [-0.0253, 0.0344] |
| `step143000` | task repeat | 9 | 0.3718 | 0.0334 | 0.3155 | 0.2820 | [0.0995, 0.5164] |

Additional task-repeat details:

- own top excess is positive for 9/9 target seeds;
- task-repeat aligned-minus-same is positive for 8/9 target seeds;
- task-repeat pair-level CI is `[0.1822, 0.3918]`;
- task-repeat aligned-better count is 63/72;
- generic Phase 0 aligned-minus-same has target CI crossing zero and only 5/9
  positive targets.

Pythia-410M WikiText-103 exact 8-gram results:

| Condition | Alignment source | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|---:|
| `step0` | task repeat | 9 | 0.0004 | -0.0000 | -0.0003 | -0.0002 | [-0.0019, 0.0013] |
| `step143000` | Phase 0 generic | 9 | 0.0580 | 0.0193 | 0.0171 | -0.0022 | [-0.0333, 0.0196] |
| `step143000` | task repeat | 9 | 0.0580 | 0.0193 | 0.0571 | 0.0378 | [-0.0042, 0.0708] |

The 410M run found 491 candidate windows and sampled 128 probe plus 128
evaluation windows without replacement.

Additional 410M details:

- own top excess is positive for 8/9 target seeds;
- task-repeat aligned-minus-same is positive for 8/9 target seeds;
- task-repeat pair-level CI is `[0.0065, 0.0612]`;
- task-repeat aligned-better count is 57/72;
- target-level bootstrap CI still slightly crosses zero;
- generic Phase 0 aligned-minus-same is neutral/slightly negative.

Interpretation: the larger-corpus exact 8-gram check strengthens the natural
repeat result, especially for 160M. The task is more natural than inserted
spans, uses longer exact repeated phrases, avoids sampling replacement, and
still shows trained own-head causality plus task-specific cross-seed transfer.
It also preserves the alignment-basis lesson: generic Phase 0 matching is
essentially neutral on this weak natural role, while role-specific matching is
positive. However, the 410M effect remains much smaller and more heterogeneous
than 160M.

## Ordinary-Phrase Filtered Check

I added `--span-primary-category` to the natural-repeat runner and used the
category helper's same heuristics to filter exact 8-gram candidates. The first
filtered check uses only `ordinary_phrase` spans.

Pythia-160M WikiText-103 ordinary-phrase exact 8-gram setup:

- token stream length: `1000066`;
- ordinary-phrase candidate windows: `147`;
- sampled windows: 64 probe plus 64 evaluation, without replacement.

Results:

| Condition | Alignment source | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|---:|
| `step0` | task repeat | 9 | 0.0009 | -0.0000 | 0.0012 | 0.0012 | [-0.0005, 0.0028] |
| `step143000` | Phase 0 generic | 9 | 0.3133 | 0.0248 | 0.0384 | 0.0137 | [-0.0044, 0.0305] |
| `step143000` | task repeat | 9 | 0.3133 | 0.0248 | 0.2500 | 0.2252 | [0.1096, 0.3776] |

Additional task-repeat details:

- own top excess is positive for 9/9 targets;
- task-repeat aligned-minus-same is positive for 8/9 targets;
- task-repeat pair-level CI is `[0.1510, 0.3070]`;
- task-repeat aligned-better count is 68/72.

Interpretation: filtering to ordinary phrases does not eliminate the 160M
natural-repeat result. The result remains role-specific: generic Phase 0
alignment is neutral, while task-repeat alignment gives clear held-out transfer.

Pythia-410M WikiText-103 ordinary-phrase exact 8-gram setup:

- token stream length: `1000066`;
- ordinary-phrase candidate windows: `140`;
- sampled windows: 64 probe plus 64 evaluation, without replacement.

Results:

| Condition | Alignment source | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|---:|
| `step0` | task repeat | 9 | 0.0010 | -0.0008 | 0.0002 | 0.0009 | [-0.0003, 0.0022] |
| `step143000` | task repeat | 9 | 0.0559 | 0.0102 | 0.0429 | 0.0327 | [0.0027, 0.0599] |

Additional 410M task-repeat details:

- own top excess target CI is `[0.0189, 0.0949]`;
- task-repeat aligned-minus-same is positive for 8/9 targets;
- task-repeat pair-level CI is `[0.0112, 0.0506]`;
- task-repeat aligned-better count is 56/72.

Interpretation: filtering to ordinary phrases makes the 410M exact-repeat
result cleaner than the mixed 8-gram run. The effect is still much smaller than
160M, but the target-level CI is now positive. I did not rerun the expensive
410M ordinary-phrase generic Phase 0 comparison; the full mixed 410M exact
8-gram generic comparison was neutral, and the 160M ordinary-phrase generic
comparison was also neutral.

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

- exact naturally occurring repeats are often article-specific names, headings,
  boilerplate, or local phrase reuse rather than a clean induction role;
- the original 4-token span was short, so there were only 3 scored next-token
  positions per example; WikiText-103 exact 8-grams reduce this issue but do
  not remove the semantic mixture;
- the generic Phase 0 raw-score alignment is not specific enough for this weaker
  role;
- same-index heads may already capture enough of this broad natural-repeat
  behavior to obscure the advantage unless the matching basis is role-specific.

## Trustworthiness

Stronger parts:

- the script uses unmodified corpus windows;
- examples are inspectable in `example_rows.csv`;
- own-head causality is positive for 9/9 final-checkpoint seeds;
- matched `step0` controls are null with respect to positive transfer;
- the task-specific alignment result uses held-out evaluation windows.
- the WikiText-103 exact 8-gram extension uses longer repeated spans and enough
  candidates to sample probe/evaluation windows without replacement.

Weaker parts:

- exact repeats are semantically mixed;
- the 410M exact-repeat result is still weak/heterogeneous even after moving to
  WikiText-103 exact 8-grams;
- generic-alignment aligned-minus-same has high variance and target-level CI
  crosses zero;
- the task-specific alignment result is stronger but more role-informed, so it
  should be reported as a method-dependent positive result.

## Next Experiments

1. Filter repeated spans by lower baseline predictability, so the model must rely
   more on copying rather than ordinary language-model context.
2. Compare same-index vs aligned transfer separately for name/title repeats,
   numeric repeats, and ordinary phrase repeats.
3. Inspect the remaining 410M weak cases by repeat type and target seed, rather
   than making a simple monotonic model-size claim.

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
- Pythia-410M all-seed task-repeat alignment result:
  `results/phase1_pythia410m_natural_repeat_ngram_task_alignment_seed9/`.
- Pythia-410M all-seed task-repeat alignment `step0` control:
  `results/phase1_pythia410m_natural_repeat_ngram_task_alignment_seed9_step0/`.
- Pythia-160M WikiText-103 exact 8-gram Phase 0 alignment:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_phase0_alignment_seed9_n128/`.
- Pythia-160M WikiText-103 exact 8-gram task-repeat alignment:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128/`.
- Pythia-160M WikiText-103 exact 8-gram task-repeat `step0` control:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128_step0/`.
- Pythia-410M WikiText-103 exact 8-gram Phase 0 alignment:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_phase0_alignment_seed9_n128/`.
- Pythia-410M WikiText-103 exact 8-gram task-repeat alignment:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128/`.
- Pythia-410M WikiText-103 exact 8-gram task-repeat `step0` control:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128_step0/`.
- Pythia-160M WikiText-103 ordinary-phrase exact 8-gram Phase 0 alignment:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_phase0_alignment_seed9_n64/`.
- Pythia-160M WikiText-103 ordinary-phrase exact 8-gram task-repeat alignment:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`.
- Pythia-160M WikiText-103 ordinary-phrase exact 8-gram task-repeat `step0`
  control:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0/`.
- Pythia-410M WikiText-103 ordinary-phrase exact 8-gram task-repeat alignment:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`.
- Pythia-410M WikiText-103 ordinary-phrase exact 8-gram task-repeat `step0`
  control:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0/`.
- Natural-repeat heterogeneity memo:
  `doc/phase1_natural_repeat_heterogeneity.md`.
- Category helper:
  `scripts/analyze_natural_repeat_categories.py`.
