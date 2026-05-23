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
