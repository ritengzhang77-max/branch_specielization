# Autonomous Sleep Research Log

Start: 2026-05-23 00:15:59 PDT

Planned stop: 2026-05-23 12:15:59 PDT

## Initial Plan

The user chose the standard-dataset route for the naturalistic local-copy /
induction probe. I did not find a local skill file named "midnight"; the matching
installed skill is `autonomous-sleep-research`, which instructs a durable
12-hour autonomous research loop with commits and pushes at coherent
checkpoints.

Immediate experiment:

```text
Replace the synthetic [x, SEP, x] local-copy task with a standard-dataset
repeated-span task, then test whether cross-layer candidate-pool alignment still
transfers function across seeds.
```

Planned data source:

- Hugging Face `wikitext`, config `wikitext-2-raw-v1`, train split.

Task definition:

- Construct sequences of the form `prefix + span + distractor + span`.
- Probe attention from each token in the second span occurrence back to the
  matching token in the first span occurrence.
- Causal readout: next-token loss over the second span occurrence.
- Compare own top heads, random candidate heads, same-index source heads, and
  cross-layer aligned source heads.

## Progress: Naturalistic Probe Implemented

Added `scripts/pythia_naturalistic_span_candidate_pool_alignment.py`.

First 14M smoke test completed but exposed a data-construction issue: some
sampled repeated spans began mid-word after tokenization. I added boundary
filtering so valid windows exclude EOS and the first span token must decode as
starting with whitespace/newline. The second 14M smoke test validated the full
probe/alignment/ablation path with more interpretable examples.

## Progress: Pythia-160M Naturalistic Runs

Three-seed narrow-window pilot:

- result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed3/`;
- candidate layers: 2-4;
- own top excess: `0.3150`;
- aligned-minus-same: `0.0047`;
- target CI: `[-0.0858, 0.1435]`.

Three-seed all-layer control:

- result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed3_all_layers/`;
- candidate layers: 0-11;
- own top excess: `0.3609`;
- aligned-minus-same: `0.1474`;
- target CI: `[-0.1666, 0.3088]`.

All-seed all-layer run:

- result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers/`;
- own top excess: `0.6458`;
- same-index transfer: `-0.0170`;
- aligned transfer: `0.0665`;
- aligned-minus-same: `0.0835`;
- pair CI: `[0.0216, 0.1430]`;
- target CI: `[0.0334, 0.1343]`;
- target positives: 8/9.

Interpretation: this is a positive but small naturalistic transfer result.

## Progress: Pythia-410M Naturalistic Runs

Three-seed narrow-window pilot:

- result directory:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed3_layers2_6/`;
- candidate layers: 2-6;
- own top excess: `0.1664`;
- aligned-minus-same: `-0.0164`;
- target CI: `[-0.0356, -0.0025]`.

Three-seed all-layer control:

- result directory:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed3_all_layers/`;
- candidate layers: 0-23;
- own top excess: `0.2714`;
- aligned-minus-same: `0.1610`;
- target CI: `[0.0952, 0.2332]`.

All-seed all-layer run:

- result directory:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed9_all_layers/`;
- own top excess: `0.2416`;
- same-index transfer: `0.0007`;
- aligned transfer: `0.0462`;
- aligned-minus-same: `0.0455`;
- pair CI: `[-0.0014, 0.0873]`;
- target CI: `[-0.0190, 0.0894]`;
- target positives: 8/9.

Interpretation: this is suggestive but not decisive. The sign pattern is
positive, but bootstrap intervals cross zero.

## Checkpoint Interpretation

The naturalistic repeated-span result supports the external-validity direction:
cross-layer role alignment is not purely a synthetic-token artifact. However,
effect sizes are much smaller than in synthetic local-copy. The current claim
should therefore be:

```text
Synthetic local-copy gives the clean high-signal result; WikiText repeated spans
show a weaker naturalistic validation, especially in Pythia-160M.
```

Next decisive checks:

1. rerun the 160M all-layer naturalistic experiment with more probe/evaluation
   sequences;
2. add a 160M `step0` all-layer naturalistic control;
3. test naturally occurring repeated n-grams without inserted second spans.

## Progress: Larger-Sample 160M Replication

Reran the Pythia-160M all-layer naturalistic experiment with 128 probe sequences
and 128 evaluation sequences.

- result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers_n128/`;
- own top excess: `0.6060`;
- same-index transfer: `-0.0281`;
- aligned transfer: `0.0534`;
- aligned-minus-same: `0.0816`;
- pair CI: `[0.0237, 0.1379]`;
- target CI: `[0.0333, 0.1300]`;
- target positives: 8/9.

Interpretation: the larger-sample replication preserves the 64-example effect
almost exactly (`0.0816` vs `0.0835` aligned-minus-same).

## Progress: Matched 160M Step0 Control

Ran a matched Pythia-160M all-layer `step0` control with 128 probe and 128
evaluation sequences.

- result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers_step0_n128/`;
- own top excess: `-0.0005`;
- same-index transfer: `-0.0006`;
- aligned transfer: `0.0000`;
- aligned-minus-same: `0.0007`;
- pair CI: `[-0.0006, 0.0020]`;
- target CI: `[-0.0004, 0.0016]`.

Interpretation: the initialization control is null. The naturalistic 160M
aligned-transfer effect is therefore small but training-created, not a raw
matching artifact.

## Progress: Larger-Sample 410M Replication

Reran the Pythia-410M all-layer naturalistic experiment with 128 probe sequences
and 128 evaluation sequences.

- result directory:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed9_all_layers_n128/`;
- own top excess: `0.1809`;
- same-index transfer: `-0.0046`;
- aligned transfer: `0.0247`;
- aligned-minus-same: `0.0293`;
- pair CI: `[-0.0102, 0.0636]`;
- target CI: `[-0.0237, 0.0630]`;
- target positives: 8/9.

Target seed 6 remained a stable negative outlier:

- 64-example aligned-minus-same: `-0.1853`;
- 128-example aligned-minus-same: `-0.1659`.

Interpretation: the 410M naturalistic result is weaker than the 160M result and
weaker after doubling examples. The sign pattern is positive, but CIs cross zero
and the seed-6 failure is stable. This should be presented as weak /
heterogeneous, not as a clean 410M naturalistic confirmation.

## Progress: Naturally Occurring Repeat-Ngram Probe

Implemented `scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`.
Unlike the inserted-span probe, this script uses unmodified WikiText windows and
finds exact repeated n-grams already present in the corpus.

Default task:

- 96-token windows;
- exact repeated 4-token n-gram;
- window stride 8;
- minimum gap 8;
- no EOS in the window;
- repeated n-gram starts on a whitespace/newline token boundary.

Smoke:

- result directory:
  `results/debug_pythia14m_natural_repeat_ngram_candidate_pool/`;
- completed after one loader bug fix.

Pythia-160M 3-seed pilot:

- result directory:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed3/`;
- own top excess: `0.0956`;
- aligned-minus-same: `0.0321`;
- target CI for aligned-minus-same: `[-0.0074, 0.0633]`.

Pythia-160M all-seed final checkpoint:

- result directory:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed9/`;
- own top excess: `0.1588`;
- own top excess target CI: `[0.0806, 0.2718]`;
- own top excess positives: 9/9;
- same-index transfer: `0.0464`;
- aligned transfer: `0.0448`;
- aligned-minus-same: `-0.0016`;
- target CI for aligned-minus-same: `[-0.0548, 0.0360]`;
- target positives for aligned-minus-same: 6/9.

Pythia-160M all-seed `step0` control:

- result directory:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed9_step0/`;
- own top excess: `-0.0001`;
- aligned-minus-same: `-0.0004`;
- target CI for aligned-minus-same: `[-0.0030, 0.0019]`.

Interpretation: trained 160M heads causally support naturally occurring exact
repeats, and this is absent at initialization. Generic Phase 0 alignment does
not beat same-index transfer in mean.

## Progress: Natural-Repeat Task-Specific Alignment

Added `--alignment-source task_repeat` to
`scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`. This aligns
heads using attention vectors on the repeated n-gram probe positions instead of
using generic Phase 0 texts. The alignment still uses probe windows only; causal
loss is evaluated on held-out evaluation windows.

Pythia-160M all-seed final checkpoint:

- result directory:
  `results/phase1_pythia160m_natural_repeat_ngram_task_alignment_seed9/`;
- own top excess: `0.1588`;
- same-index transfer: `0.0464`;
- task-repeat aligned transfer: `0.2361`;
- aligned-minus-same: `0.1897`;
- pair CI: `[0.0908, 0.2843]`;
- target CI: `[0.0737, 0.3140]`;
- target positives: 8/9;
- aligned better count: 66/72.

Matched `step0` task-alignment control:

- result directory:
  `results/phase1_pythia160m_natural_repeat_ngram_task_alignment_seed9_step0/`;
- own top excess: `-0.0001`;
- same-index transfer: `0.0015`;
- task-repeat aligned transfer: `-0.0018`;
- aligned-minus-same: `-0.0033`;
- target CI: `[-0.0060, -0.0006]`.

Updated interpretation: the stricter natural-repeat task is not truly
alignment-neutral. It is neutral under generic Phase 0 matching, but positive
under role-specific task-repeat matching. This makes alignment basis a key
methodological variable.

## Progress: Inserted-Span Task-Specific Alignment

Added `--alignment-source task_span` to
`scripts/pythia_naturalistic_span_candidate_pool_alignment.py`. This applies the
same alignment-basis test to the inserted WikiText repeated-span task.

Pythia-160M all-seed final checkpoint:

- result directory:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers/`;
- own top excess: `0.6458`;
- same-index transfer: `-0.0170`;
- task-span aligned transfer: `0.5475`;
- aligned-minus-same: `0.5645`;
- pair CI: `[0.4501, 0.6855]`;
- target CI: `[0.3653, 0.8068]`;
- target positives: 9/9;
- aligned better count: 68/72.

Matched `step0` task-span alignment control:

- result directory:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers_step0/`;
- own top excess: `0.0002`;
- same-index transfer: `-0.0003`;
- task-span aligned transfer: `0.0001`;
- aligned-minus-same: `0.0003`;
- target CI: `[-0.0008, 0.0015]`.

Updated interpretation: the inserted-span naturalistic effect was not merely
weak. It was weak under generic Phase 0 matching. Role-specific task-span
matching gives a strong held-out transfer effect with a null initialization
control.

## Progress: 410M Inserted-Span Task-Specific Alignment

Applied the same task-span alignment to Pythia-410M.

Pythia-410M all-seed final checkpoint:

- result directory:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers/`;
- own top excess: `0.2416`;
- same-index transfer: `0.0007`;
- task-span aligned transfer: `0.1551`;
- aligned-minus-same: `0.1544`;
- pair CI: `[0.0939, 0.2075]`;
- target CI: `[0.0430, 0.2460]`;
- target positives: 8/9;
- aligned better count: 54/72.

Matched `step0` task-span alignment control:

- result directory:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers_step0/`;
- own top excess: `-0.0008`;
- same-index transfer: `0.0004`;
- task-span aligned transfer: `0.0002`;
- aligned-minus-same: `-0.0003`;
- target CI: `[-0.0005, 0.0000]`.

Updated interpretation: task-span alignment also rescues 410M naturalistic
transfer, though less cleanly than 160M because target seed 6 remains a negative
outlier. The earlier weak 410M result should be framed as a generic-alignment
failure, not absence of a transferable role.

## Progress: Task-Span Alignment 128/128 Replications

Pythia-160M task-span alignment with 128 probe and 128 evaluation spans:

- result directory:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers_n128/`;
- own top excess: `0.6060`;
- same-index transfer: `-0.0281`;
- task-span aligned transfer: `0.4492`;
- aligned-minus-same: `0.4773`;
- pair CI: `[0.3682, 0.5899]`;
- target CI: `[0.2829, 0.6852]`;
- target positives: 8/9.

Pythia-410M task-span alignment with 128 probe and 128 evaluation spans:

- result directory:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers_n128/`;
- own top excess: `0.1809`;
- same-index transfer: `-0.0046`;
- task-span aligned transfer: `0.1112`;
- aligned-minus-same: `0.1158`;
- pair CI: `[0.0650, 0.1604]`;
- target CI: `[0.0222, 0.1884]`;
- target positives: 8/9.

Interpretation: task-span alignment remains positive under doubled sample count
for both 160M and 410M. The 410M effect is smaller and retains the seed-6
negative outlier; the 160M effect remains large.

## Progress: Alignment-Basis Summary Memo

Wrote `doc/phase1_alignment_basis_summary.md`.

Current consolidated conclusion:

```text
Functional repeat/copy roles are stable across seeds after role-level
relabeling, but weak natural roles require role-specific alignment features.
```

The hierarchy is now:

1. synthetic local-copy: high signal, generic alignment works;
2. inserted WikiText repeated spans: generic alignment positive but too
   conservative, task-specific alignment strong;
3. naturally occurring exact repeats: generic alignment neutral, task-specific
   alignment positive.

## Progress: 410M Naturally Occurring Repeat-Ngram Check

Ran Pythia-410M all-seed naturally occurring exact-repeat task-specific
alignment.

- result directory:
  `results/phase1_pythia410m_natural_repeat_ngram_task_alignment_seed9/`;
- own top excess: `0.0503`;
- own top excess target CI: `[0.0068, 0.0873]`;
- own top excess positives: 8/9;
- same-index transfer: `0.0173`;
- task-repeat aligned transfer: `0.0388`;
- aligned-minus-same: `0.0215`;
- pair CI: `[-0.0120, 0.0461]`;
- target CI: `[-0.0166, 0.0564]`;
- target positives: 5/9.

Matched `step0` control:

- result directory:
  `results/phase1_pythia410m_natural_repeat_ngram_task_alignment_seed9_step0/`;
- own top excess: `-0.0013`;
- aligned-minus-same: `-0.0022`;
- target CI: `[-0.0042, -0.0002]`.

Interpretation: exact natural repeats extend to 410M as a weak trained
own-head causal signal, but not as a clean 410M alignment-transfer result under
the current 4-token WikiText setup.

## Progress: Synthetic Local-Copy Task-Specific Alignment

Added `--alignment-source task_local_copy` to
`scripts/pythia_local_copy_candidate_pool_alignment.py`. This aligns heads using
attention to the repeated-value positions in the synthetic local-copy probe
split, while keeping causal loss on held-out evaluation sequences.

Pythia-160M all-seed final checkpoint:

- result directory:
  `results/phase1_pythia160m_local_copy_task_alignment_layers2_4_top2/`;
- own top excess: `2.2896`;
- same-index transfer: `0.4876`;
- task-local aligned transfer: `2.4469`;
- aligned-minus-same: `1.9593`;
- pair CI: `[1.6902, 2.2171]`;
- target CI: `[1.5948, 2.5006]`;
- target positives: 9/9;
- aligned better count: 66/72.

Matched Pythia-160M `step0` control:

- result directory:
  `results/phase1_pythia160m_local_copy_task_alignment_layers2_4_top2_step0/`;
- own top excess: `-0.0004`;
- aligned-minus-same: `0.0000`;
- target CI: `[-0.0004, 0.0003]`.

Pythia-410M all-seed final checkpoint:

- result directory:
  `results/phase1_pythia410m_local_copy_task_alignment_layers2_6_top2/`;
- own top excess: `4.1723`;
- same-index transfer: `0.2562`;
- task-local aligned transfer: `4.0299`;
- aligned-minus-same: `3.7737`;
- pair CI: `[3.2483, 4.2896]`;
- target CI: `[2.4658, 4.8657]`;
- target positives: 8/9;
- aligned better count: 66/72.

Matched Pythia-410M `step0` control:

- result directory:
  `results/phase1_pythia410m_local_copy_task_alignment_layers2_6_top2_step0/`;
- own top excess: `-0.0009`;
- aligned-minus-same: `-0.0004`;
- target CI: `[-0.0007, -0.0001]` at negligible absolute scale.

Interpretation: task-local alignment is an upper-bound sanity check for the
synthetic task. It agrees with the strong 160M generic result and substantially
strengthens the 410M final-checkpoint result, showing that even for a large
synthetic causal role, generic Phase 0 matching can miss part of the
cross-seed relabeling.

## Progress: WikiText-103 Exact 8-Gram Natural Repeats

Checked whether the naturally occurring exact-repeat result survives longer
spans on a larger standard corpus.

Candidate-count scan on 500024 WikiText-103 train tokens:

- exact 5-gram candidates: 3839;
- exact 6-gram candidates: 2049;
- exact 7-gram candidates: 980;
- exact 8-gram candidates: 524.

Ran Pythia-160M exact 8-gram task-repeat alignment with 128 probe and 128
evaluation windows, sampled without replacement.

- result directory:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128/`;
- own top excess: `0.3718`;
- own top target CI: `[0.1494, 0.6444]`;
- own top positives: 9/9;
- same-index transfer: `0.0334`;
- task-repeat aligned transfer: `0.3155`;
- aligned-minus-same: `0.2820`;
- pair CI: `[0.1822, 0.3918]`;
- target CI: `[0.0995, 0.5164]`;
- target positives: 8/9;
- aligned better count: 63/72.

Matched `step0` task-repeat control:

- result directory:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128_step0/`;
- own top excess: `-0.0010`;
- aligned-minus-same: `-0.0018`;
- target CI: `[-0.0035, -0.0001]` at negligible absolute scale.

Generic Phase 0 comparison on the same deterministic 8-gram sample:

- result directory:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_phase0_alignment_seed9_n128/`;
- own top excess: `0.3718`;
- same-index transfer: `0.0334`;
- generic aligned transfer: `0.0397`;
- aligned-minus-same: `0.0063`;
- pair CI: `[-0.0424, 0.0426]`;
- target CI: `[-0.0253, 0.0344]`;
- target positives: 5/9.

Interpretation: longer exact natural repeats strengthen the 160M external
validity result. The role is causally stronger than in the 4-gram WikiText-2
task, but the alignment-basis conclusion remains: generic Phase 0 matching is
neutral, while task-repeat matching gives positive held-out transfer.

## Progress: 410M WikiText-103 Exact 8-Gram Check

Ran the same exact 8-gram setup on Pythia-410M with all 24 layers in the
candidate pool.

Task-repeat alignment:

- result directory:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128/`;
- repeat candidate count: 491;
- own top excess: `0.0580`;
- own top target CI: `[0.0059, 0.1088]`;
- own top positives: 8/9;
- same-index transfer: `0.0193`;
- task-repeat aligned transfer: `0.0571`;
- aligned-minus-same: `0.0378`;
- pair CI: `[0.0065, 0.0612]`;
- target CI: `[-0.0042, 0.0708]`;
- target positives: 8/9;
- aligned better count: 57/72.

Matched `step0` task-repeat control:

- result directory:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128_step0/`;
- own top excess: `0.0004`;
- aligned-minus-same: `-0.0002`;
- target CI: `[-0.0019, 0.0013]`.

Generic Phase 0 comparison:

- result directory:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_phase0_alignment_seed9_n128/`;
- own top excess: `0.0580`;
- same-index transfer: `0.0193`;
- generic aligned transfer: `0.0171`;
- aligned-minus-same: `-0.0022`;
- pair CI: `[-0.0303, 0.0183]`;
- target CI: `[-0.0333, 0.0196]`;
- target positives: 6/9.

Interpretation: 410M improves when the exact-repeat task uses WikiText-103
8-grams rather than WikiText-2 4-grams, but it is still much weaker than 160M.
The right wording is not "larger model improves natural-repeat transfer"; it is
"longer natural repeats reveal a small trained 410M signal, still
heterogeneous and alignment-basis dependent."

## Progress: Natural-Repeat Heterogeneity Inspection

Wrote `doc/phase1_natural_repeat_heterogeneity.md`.

Main diagnostic points:

- 410M target seed 4 has negative own-head excess on exact 8-grams (`-0.0958`),
  so there is little target role to transfer.
- 410M target seed 6 has positive own-head excess (`0.0232`) but unusually
  strong same-index transfer (`0.1177`), making aligned-minus-same negative
  even though aligned transfer is positive (`0.0119`).
- The biggest 410M exact 8-gram failure is target seed 6 from source seed 3:
  same-index transfer `0.8055`, task-repeat aligned transfer `0.0599`,
  aligned-minus-same `-0.7456`.
- The same source-target outlier appears in the 410M exact 4-gram run:
  same-index transfer `0.9588`, aligned transfer `0.0577`,
  aligned-minus-same `-0.9011`.

Interpretation: aligned-minus-same is a useful metric, but it is not a neutral
truth oracle. A few raw same-index source heads can transfer unusually well and
make aligned-minus-same understate role transfer. The paper should report
aligned transfer, same-index transfer, and their difference together.

## Progress: Natural-Repeat Category Counts

Added `scripts/analyze_natural_repeat_categories.py` and used it to reconstruct
the full deterministic evaluation sets from existing result directories.

Primary category counts:

| Run | n | Ordinary | Numeric/date | Quoted/title | Proper-name-like | Tokenizer markup |
|---|---:|---:|---:|---:|---:|---:|
| 160M WikiText-2 exact 4-gram | 64 | 27 | 8 | 3 | 8 | 17 |
| 160M WikiText-103 exact 8-gram | 128 | 35 | 20 | 16 | 25 | 32 |
| 410M WikiText-103 exact 8-gram | 128 | 29 | 25 | 21 | 18 | 35 |

Interpretation: exact natural repeats are not one clean semantic behavior. They
mix ordinary phrases, numeric/date spans, quoted titles, proper-name-like spans,
and tokenizer-artifact spans. The next filtering experiment should stratify or
filter examples before interpreting model-size differences.

## Progress: Ordinary-Phrase Filtered Exact 8-Gram Check

Added `--span-primary-category` to
`scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py` and ran a
Pythia-160M ordinary-phrase-only check.

Setup:

- dataset: WikiText-103 train;
- token stream length: `1000066`;
- repeated span: exact 8-gram;
- primary category: `ordinary_phrase`;
- candidates: `147`;
- probe/eval: 64/64, no replacement.

Task-repeat alignment:

- result directory:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`;
- own top excess: `0.3133`;
- same-index transfer: `0.0248`;
- task-repeat aligned transfer: `0.2500`;
- aligned-minus-same: `0.2252`;
- pair CI: `[0.1510, 0.3070]`;
- target CI: `[0.1096, 0.3776]`;
- target positives: 8/9;
- aligned better count: 68/72.

Matched `step0` task-repeat control:

- result directory:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0/`;
- own top excess: `0.0009`;
- aligned-minus-same: `0.0012`;
- target CI: `[-0.0005, 0.0028]`.

Generic Phase 0 comparison:

- result directory:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_phase0_alignment_seed9_n64/`;
- own top excess: `0.3133`;
- generic aligned transfer: `0.0384`;
- aligned-minus-same: `0.0137`;
- target CI: `[-0.0044, 0.0305]`.

Interpretation: filtering to ordinary phrases does not remove the 160M
natural-repeat effect. The result remains strongly alignment-basis dependent:
task-repeat matching transfers, generic Phase 0 matching is neutral.

## Progress: 410M Ordinary-Phrase Filtered Exact 8-Gram Check

Ran the matched Pythia-410M ordinary-phrase-only task-repeat check.

Setup:

- dataset: WikiText-103 train;
- token stream length: `1000066`;
- repeated span: exact 8-gram;
- primary category: `ordinary_phrase`;
- candidates: `140`;
- probe/eval: 64/64, no replacement.

Task-repeat alignment:

- result directory:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`;
- own top excess: `0.0559`;
- own top target CI: `[0.0189, 0.0949]`;
- same-index transfer: `0.0102`;
- task-repeat aligned transfer: `0.0429`;
- aligned-minus-same: `0.0327`;
- pair CI: `[0.0112, 0.0506]`;
- target CI: `[0.0027, 0.0599]`;
- target positives: 8/9;
- aligned better count: 56/72.

Matched `step0` task-repeat control:

- result directory:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0/`;
- own top excess: `0.0010`;
- aligned-minus-same: `0.0009`;
- target CI: `[-0.0003, 0.0022]`.

Generic Phase 0 comparison:

- result directory:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_phase0_alignment_seed9_n64/`;
- generic aligned transfer: `0.0124`;
- aligned-minus-same: `0.0022`;
- pair CI: `[-0.0164, 0.0169]`;
- target CI: `[-0.0191, 0.0182]`.

Interpretation: filtering to ordinary phrases makes the 410M exact-repeat result
cleaner than the mixed 8-gram run: the target-level CI is now positive. The
effect is still small relative to 160M, so this supports "filtered natural
repeat transfer exists in 410M" but not a strong monotonic scaling claim.
Generic Phase 0 matching remains neutral.

## Progress: Phase 1 Paper-Facing Claims Memo

Wrote `doc/phase1_paper_claims_and_methods.md`.

Current consolidated claim:

```text
Functional repeat/copy roles are stable across seeds after role-level
relabeling, but weak natural roles require role-specific alignment features.
```

The memo separates:

- generic Phase 0 alignment as the task-agnostic baseline;
- role-specific alignment as the held-out role-stability measurement;
- functional specialization from functional modularity.

It also sketches a paper structure: definitions, metrics, synthetic upper bound,
naturalistic inserted spans, unmodified natural repeats, heterogeneity analysis,
and then explicit branch/heterogeneous architectures as the next stage.

## Progress: Phase 3 Structural-to-Functional Synthesis

Wrote `doc/phase3_structural_to_functional_synthesis.md`.

Current answer to the main framing:

```text
Structural heterogeneity can create stable functional specialization slots.
Structural modularity/routing can support functional modularity, but in the toy
evidence so far it does not reliably create it by itself.
```

The strongest positive mechanism is role-informative routing pressure:

- oracle routing produces clean functional modularity;
- weak scored-position router supervision produces near-oracle modularity;
- annealed and trajectory runs show that gates become role-aligned before branch
  computations become causally modular.

The strongest negative boundary:

- separate branches, unconstrained learned routers, entropy/balance
  regularization, bottlenecked branches, and conflict-heavy lookup all failed to
  reliably produce spontaneous role-aligned causal modularity.

Next experiment:

```text
Run a denser weak-token-router trajectory between steps 400 and 800 to locate
the causal consolidation window after gate alignment.
```

## Progress: Dense Router Consolidation-Window Run

Ran the planned dense weak-token-router trajectory:

- output:
  `results/phase3_toy_trajectory_consolidation_end800/`;
- analysis:
  `results/phase3_toy_trajectory_consolidation_end800_analysis/`;
- memo:
  `doc/phase3_toy_router_consolidation_window.md`.

Key milestones in the solved regime:

| Milestone | Step |
|---|---:|
| Gate routed-role match 5/5 | 400 |
| Causal routed-role match 5/5 | 550 |
| Branch distance >= 0.30 | 600 |
| Branch distance >= 0.40 | 750 |

Interpretation:

```text
role-aligned routing gates precede causal branch modularity, and causal
separation strength keeps growing after the top-branch split appears.
```

## Progress: Pythia-160M Ordinary Natural-Repeat Trajectory

Ran the WikiText-103 ordinary-phrase exact 8-gram task-repeat alignment setup at
three additional Pythia-160M checkpoints: `step4000`, `step16000`, and
`step64000`. Combined them with the existing `step0` and final checkpoint runs.

Trajectory:

| Checkpoint | Probe spec. | Own top - random | Aligned - same | Target CI |
|---|---:|---:|---:|---:|
| step0 | 0.0077 | 0.0009 | 0.0012 | [-0.0005, 0.0028] |
| step4000 | 0.1115 | 0.0205 | 0.0154 | [-0.0152, 0.0448] |
| step16000 | 0.1481 | 0.1756 | 0.0460 | [-0.0446, 0.1336] |
| step64000 | 0.1576 | 0.1693 | 0.1174 | [0.0387, 0.2099] |
| step143000 | 0.1623 | 0.3133 | 0.2252 | [0.1096, 0.3776] |

Interpretation:

```text
natural ordinary-repeat probes appear before robust causal cross-seed transfer;
own causal importance is visible by step16000, but aligned transfer becomes
target-level robust only by step64000.
```
