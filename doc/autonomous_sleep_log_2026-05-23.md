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
