# Naturalistic Local-Copy / Induction Probe Design

Date: 2026-05-22

## Why This Is Needed

The strongest current local-copy result uses synthetic `[x, SEP, x]` triples.
That is useful because it isolates a clean causal behavior, but it leaves one
paper-risk:

```text
The cross-layer candidate-pool result might be specific to arbitrary synthetic
tokens rather than a general local-copy / induction-like attention role.
```

The next experiment should test the same alignment-transfer method on a more
naturalistic repeated-span task.

## Proposed Task

Use repeated natural token spans:

```text
prefix + span + distractor + span
```

At each token in the second occurrence of `span`, the model should predict the
next token in that natural span. This is closer to induction than the current
`[x, SEP, x]` copy task:

- the copied content is a multi-token natural phrase;
- the target is the next token in the repeated span;
- the attention probe is from the second occurrence back to the matching token
  in the first occurrence;
- the causal readout is next-token loss over the second occurrence.

This preserves the key experimental logic while removing the arbitrary
single-token separator structure.

## Data Options

### Option A: Local fixed corpus

Create a small checked-in corpus of hand-written, license-clean sentences and
sample natural spans from it.

Pros:

- fully reproducible;
- no external dataset dependency;
- easy to inspect examples.

Cons:

- small and artificial;
- weaker paper evidence.

### Option B: WikiText / OpenWebText-style dataset

Use a standard open text dataset through Hugging Face datasets, cache it under
`HF_HOME`, and sample spans deterministically.

Pros:

- better external validity;
- easier to scale examples;
- more paper-like.

Cons:

- extra dependency and download;
- requires dataset provenance/citation;
- more moving parts for reproducibility.

## Metrics

Reuse the candidate-pool machinery:

1. Probe candidate heads by attention from the second span occurrence to the
   first matching occurrence.
2. Select top `k=2` candidate heads across a layer window.
3. Compute cross-seed raw-score Hungarian alignment over the full candidate
   pool.
4. Compare:
   - own top heads vs random candidate heads;
   - source same-index heads;
   - source cross-layer aligned heads.
5. Report:
   - own top excess over random;
   - aligned-minus-same transfer;
   - pair-level and target-level bootstrap CIs;
   - pair-level and target-level sign tests;
   - per-target seed heterogeneity.

## Initial Recommended Settings

For a fast but meaningful first run:

| Setting | Value |
|---|---|
| model | Pythia-160M |
| seeds | 1-9 |
| checkpoint | `step143000` |
| candidate layers | 2-4 |
| selected heads | top 2 |
| examples | 128 repeated-span sequences |
| span length | 8-16 tokens after tokenization |
| distractor length | 16-32 tokens |
| eval batch size | 8 |

Then replicate on Pythia-410M layers 2-6 if the 160M run is positive.

## Decision Rule

The naturalistic probe is useful if:

- own top excess over random is clearly positive;
- aligned-minus-same is positive at both pair and target levels;
- target-level bootstrap CI excludes zero;
- examples are interpretable on inspection.

If own top excess is weak, the result should be interpreted as task/corpus
failure, not as evidence against alignment.

## Implementation Plan

1. Add a script parallel to
   `scripts/pythia_local_copy_candidate_pool_alignment.py`.
2. Replace `[x, SEP, x]` sequence generation with deterministic repeated-span
   generation.
3. Add an `example_rows.csv` output with decoded spans and scored positions for
   manual inspection.
4. Run a Pythia-160M smoke test with seeds 1-3.
5. If examples and own-head causality look sane, run all 9 seeds and analyze
   with `scripts/analyze_transfer_significance.py`.

## Expected Contribution

If positive, this would make the current claim much more paper-ready:

```text
Cross-layer role alignment transfers an induction-like local-copy function
across seeds, and the result is not limited to arbitrary synthetic token triples.
```

## Completed First Pass

The first standard-dataset version was implemented on 2026-05-23 using
WikiText-2 repeated natural spans. The boundary-filtered version excludes EOS
inside the sampled window and requires the first repeated-span token to start on
a whitespace/newline boundary.

Summary:

- Pythia-160M, all 9 seeds, all-layer candidate pool: own top excess `0.6458`;
  aligned-minus-same `0.0835`; target-level bootstrap CI `[0.0334, 0.1343]`;
  aligned-minus-same positive for 8/9 target seeds.
- A 128-probe / 128-eval Pythia-160M replication kept the same effect size:
  own top excess `0.6060`; aligned-minus-same `0.0816`; target-level bootstrap
  CI `[0.0333, 0.1300]`; aligned-minus-same positive for 8/9 target seeds.
- A matched Pythia-160M `step0` control was null: own top excess `-0.0005`;
  aligned-minus-same `0.0007`; target-level bootstrap CI
  `[-0.0004, 0.0016]`.
- Pythia-410M, all 9 seeds, all-layer candidate pool: own top excess `0.2416`;
  aligned-minus-same `0.0455`; target-level bootstrap CI
  `[-0.0190, 0.0894]`; aligned-minus-same positive for 8/9 target seeds.
- A 128-probe / 128-eval Pythia-410M replication weakened the estimate:
  own top excess `0.1809`; aligned-minus-same `0.0293`; target-level bootstrap
  CI `[-0.0237, 0.0630]`; target seed 6 remained a large negative outlier.

Interpretation: the naturalistic result is positive for 160M and weak /
heterogeneous for 410M, and both are much smaller than the synthetic
`[x, SEP, x]` result. This supports the external-validity direction while
keeping the current paper claim modest: natural text preserves a weak
aligned-transfer signal in 160M, not the large synthetic effect size.

Full memo: `doc/phase1_naturalistic_span_candidate_pool.md`.
