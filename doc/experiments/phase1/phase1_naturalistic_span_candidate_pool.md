# Phase 1: Naturalistic Repeated-Span Candidate-Pool Alignment

Date: 2026-05-23

## Question

The strongest local-copy result so far used synthetic `[x, SEP, x]` triples.
That design isolates a clean copy/induction-like function, but it leaves one
external-validity concern:

```text
Does cross-layer role alignment still transfer a local-copy function when the
repeated content is natural text rather than arbitrary synthetic tokens?
```

This experiment tests the same candidate-pool machinery on WikiText repeated
natural spans.

## Method

Dataset:

- Hugging Face `wikitext`, config `wikitext-2-raw-v1`, train split;
- first 4000 non-heading rows, tokenized with the target Pythia tokenizer;
- collected token stream length: 50152;
- deterministic construction of 64 probe and 64 evaluation sequences.

Task:

```text
prefix + span + distractor + span
```

Default sequence geometry:

| Part | Tokens |
|---|---:|
| prefix | 16 |
| repeated span | 12 |
| distractor | 24 |

Scoring:

- attention probe: mean attention from the second span occurrence to the
  matching token positions in the first span occurrence;
- causal readout: next-token loss over positions in the second span occurrence;
- selected heads: top 2 heads by probe score from a candidate layer pool;
- alignment: Hungarian matching over raw attention-score vectors on the shared
  Phase 0 probe corpus;
- transfer comparison: own top heads, random candidate heads, same-index source
  heads, and cross-layer aligned source heads.

Boundary filtering was added after an initial smoke test: examples with EOS in
the sampled window are excluded, and the first repeated-span token must decode
as a whitespace/newline-starting token. This avoids accidental mid-word
concatenations such as a repeated span beginning halfway through a word.

## Main Results

All-layer candidate pools were needed for the naturalistic task. Narrow
synthetic-task layer windows were too brittle, especially for Pythia-410M.

| Model / candidate pool | Seeds | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Aligned better |
|---|---:|---:|---:|---:|---:|---:|
| 160M layers 2-4 | 3 | 0.3150 | -0.1560 | -0.1513 | 0.0047 | 4/6 |
| 160M all layers | 3 | 0.3609 | -0.1501 | -0.0027 | 0.1474 | 3/6 |
| 160M all layers | 9 | 0.6458 | -0.0170 | 0.0665 | 0.0835 | 47/72 |
| 410M layers 2-6 | 3 | 0.1664 | -0.0268 | -0.0432 | -0.0164 | 3/6 |
| 410M all layers | 3 | 0.2714 | -0.0361 | 0.1249 | 0.1610 | 5/6 |
| 410M all layers | 9 | 0.2416 | 0.0007 | 0.0462 | 0.0455 | 47/72 |

All-seed significance summaries:

| Model | Own top excess target CI | Aligned - same pair CI | Pair sign p | Aligned - same target CI | Target sign p |
|---|---:|---:|---:|---:|---:|
| 160M all layers | [0.4146, 0.9090] | [0.0216, 0.1430] | 0.0128 | [0.0334, 0.1343] | 0.0391 |
| 410M all layers | [0.1163, 0.3562] | [-0.0014, 0.0873] | 0.0128 | [-0.0190, 0.0894] | 0.0391 |

Per-target target-level behavior:

- Pythia-160M: own top excess positive for 9/9 targets; aligned-minus-same
  positive for 8/9 targets.
- Pythia-410M: own top excess positive for 8/9 targets; aligned-minus-same
  positive for 8/9 targets. Seed 6 is the main negative outlier with
  aligned-minus-same `-0.1853`.

## Larger-Sample Replication And Initialization Control

I reran the Pythia-160M all-layer experiment with 128 probe and 128 evaluation
sequences, then matched it with a `step0` initialization control using the same
settings.

| 160M condition | Probe/eval sequences | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---:|---:|---:|---:|---:|---:|
| `step0`, all layers | 128 / 128 | -0.0005 | -0.0006 | 0.0000 | 0.0007 | [-0.0004, 0.0016] |
| `step143000`, all layers | 64 / 64 | 0.6458 | -0.0170 | 0.0665 | 0.0835 | [0.0334, 0.1343] |
| `step143000`, all layers | 128 / 128 | 0.6060 | -0.0281 | 0.0534 | 0.0816 | [0.0333, 0.1300] |

The 128-example replication leaves the main estimate almost unchanged:
aligned-minus-same is `0.0816`, compared with `0.0835` in the 64-example run.
The `step0` control is a clean null: own top excess is `-0.0005`, and
aligned-minus-same is only `0.0007` with confidence intervals crossing zero.

This makes the naturalistic 160M claim more trustworthy:

```text
The WikiText repeated-span effect is small, but it is stable to doubling the
sample count and absent at initialization.
```

I also reran the Pythia-410M all-layer final-checkpoint experiment with 128
probe and 128 evaluation sequences. This did not strengthen the 410M result.

| 410M condition | Probe/eval sequences | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---:|---:|---:|---:|---:|---:|
| `step143000`, all layers | 64 / 64 | 0.2416 | 0.0007 | 0.0462 | 0.0455 | [-0.0190, 0.0894] |
| `step143000`, all layers | 128 / 128 | 0.1809 | -0.0046 | 0.0247 | 0.0293 | [-0.0237, 0.0630] |

The sign pattern remains mostly positive, but the effect is smaller. Target seed
6 remains a stable negative outlier (`aligned-minus-same=-0.1659` in the
128-example run, compared with `-0.1853` in the 64-example run). The right
interpretation is therefore:

```text
Pythia-410M naturalistic repeated-span transfer is weak and heterogeneous under
this probe, despite strong synthetic local-copy transfer.
```

## Alignment-Basis Ablation

The initial naturalistic runs used the same generic Phase 0 raw-score alignment
corpus as the synthetic experiments. After the naturally occurring repeat-ngram
follow-up showed that weak natural roles can be missed by generic alignment, I
added a task-specific alignment option:

```text
--alignment-source task_span
```

This aligns heads using attention vectors on the repeated-span probe positions.
It still evaluates causality on held-out evaluation spans.

Pythia-160M all-layer result:

| 160M condition | Alignment source | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|
| `step0` | task span | 0.0002 | -0.0003 | 0.0001 | 0.0003 | [-0.0008, 0.0015] |
| `step143000` | Phase 0 generic | 0.6458 | -0.0170 | 0.0665 | 0.0835 | [0.0334, 0.1343] |
| `step143000` | task span | 0.6458 | -0.0170 | 0.5475 | 0.5645 | [0.3653, 0.8068] |
| `step143000` | task span, 128/128 | 0.6060 | -0.0281 | 0.4492 | 0.4773 | [0.2829, 0.6852] |

Task-span alignment is much stronger than generic alignment:

- aligned-minus-same rises from `0.0835` to `0.5645`;
- the larger 128/128 run remains strongly positive at `0.4773`;
- target positives rise from 8/9 to 9/9;
- aligned-better count rises to 68/72;
- the matched `step0` task-span control is null.

Interpretation:

```text
For natural repeated spans, the alignment representation matters. Generic
attention-score matching detects a weak transfer effect; role-specific matching
recovers a much larger cross-seed functional transfer effect.
```

Pythia-410M shows the same pattern, though less cleanly:

| 410M condition | Alignment source | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---|---:|---:|---:|---:|---:|
| `step0` | task span | -0.0008 | 0.0004 | 0.0002 | -0.0003 | [-0.0005, 0.0000] |
| `step143000` | Phase 0 generic | 0.2416 | 0.0007 | 0.0462 | 0.0455 | [-0.0190, 0.0894] |
| `step143000` | task span | 0.2416 | 0.0007 | 0.1551 | 0.1544 | [0.0430, 0.2460] |
| `step143000` | task span, 128/128 | 0.1809 | -0.0046 | 0.1112 | 0.1158 | [0.0222, 0.1884] |

The 410M task-span result is positive at pair and target levels in both 64/64
and 128/128 runs, but seed 6 remains a stable negative outlier
(`aligned-minus-same=-0.2138` in the 64/64 run and `-0.1987` in the 128/128
run). So the right 410M interpretation is not "no naturalistic transfer"; it is:

```text
410M naturalistic transfer is present under role-specific alignment, but more
heterogeneous than 160M.
```

## Comparison To Synthetic Local-Copy

The naturalistic result is positive but much smaller than the synthetic
`[x, SEP, x]` result.

| Model / task | Own top excess | Same-index transfer | Aligned transfer | Aligned - same | Target CI for aligned - same |
|---|---:|---:|---:|---:|---:|
| 160M synthetic local-copy | 2.2896 | 0.4876 | 2.2714 | 1.7838 | [1.3341, 2.3715] |
| 160M WikiText repeated span | 0.6458 | -0.0170 | 0.0665 | 0.0835 | [0.0334, 0.1343] |
| 410M synthetic local-copy | 4.1723 | 0.2562 | 1.9116 | 1.6554 | [1.0261, 2.2362] |
| 410M WikiText repeated span | 0.2416 | 0.0007 | 0.0462 | 0.0455 | [-0.0190, 0.0894] |

Interpretation:

- The WikiText task validates that the synthetic effect is not purely an
  artifact of arbitrary token triples: Pythia-160M still shows positive
  own-head causality and positive cross-seed transfer after alignment.
- The effect is far weaker on natural text. A likely reason is that the model can
  often predict repeated natural spans from ordinary language-model context, so
  ablating a local-copy head changes the loss less than in the synthetic task.
- Pythia-410M is only suggestive under this setup. The sign tests are positive,
  but bootstrap intervals for aligned-minus-same cross zero at both the pair and
  target levels.
- Same-index transfer is near zero in the all-seed naturalistic runs. The
  observed transfer is therefore a role-relabeling effect, not a fixed raw
  `(layer, head)` slot effect.

## Trustworthiness

The 160M all-seed result is currently the strongest naturalistic evidence:

- positive own-head causal excess with target-level CI excluding zero;
- positive aligned-minus-same at both pair and target levels;
- 8/9 target seeds have positive aligned-minus-same;
- a 128-example replication gives essentially the same aligned-minus-same
  estimate as the 64-example run;
- a matched `step0` control gives near-zero own-head and aligned-transfer
  effects;
- task-span alignment gives a much larger held-out transfer effect
  (`aligned-minus-same=0.5645`) with a null `step0` task-alignment control;
- examples were manually inspected through `example_rows.csv` after boundary
  filtering.

The 410M result should be treated as weaker:

- own-head causal excess is positive, but smaller than 160M;
- aligned-minus-same is positive by sign count but has bootstrap CIs crossing
  zero;
- one target seed has a large negative aligned-transfer gap;
- the 128-example replication reduces the aligned-minus-same mean from `0.0455`
  to `0.0293`, so the 410M naturalistic result should not be used as a strong
  positive claim.

Main limitations:

- only 64 probe and 64 evaluation sequences in the all-seed runs;
- the task still inserts an artificial second occurrence of the span, so it is
  naturalistic rather than fully naturally occurring repetition;
- all-layer pools increase robustness, but they also make the selected role less
  anatomically localized;
- WikiText-2 is small and stylistically narrow.

## Next Experiments

1. Rerun the 160M all-layer WikiText experiment with more probe/evaluation
   sequences again at 256 if compute allows, but the 128-example replication has
   already passed the main stability check.
2. Build a naturally occurring repeated-ngram variant that does not insert the
   second span manually. This is lower-control but higher-validity.
3. Diagnose why 410M synthetic local-copy is strong while 410M WikiText
   repeated-span transfer is weak. Candidate explanations: natural-token
   predictability reduces causal reliance on copy heads; all-layer matching
   selects a less coherent role; or the local-copy role is distributed
   differently in larger models.

## Files

- Script: `scripts/pythia_naturalistic_span_candidate_pool_alignment.py`.
- Pythia-160M all-seed result:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers/`.
- Pythia-160M 128-example replication:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers_n128/`.
- Pythia-160M 128-example `step0` control:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers_step0_n128/`.
- Pythia-160M task-span alignment result:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers/`.
- Pythia-160M task-span alignment 128/128 replication:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers_n128/`.
- Pythia-160M task-span alignment `step0` control:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers_step0/`.
- Pythia-410M all-seed result:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed9_all_layers/`.
- Pythia-410M 128-example replication:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed9_all_layers_n128/`.
- Pythia-410M task-span alignment result:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers/`.
- Pythia-410M task-span alignment 128/128 replication:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers_n128/`.
- Pythia-410M task-span alignment `step0` control:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers_step0/`.
- Initial design memo: `doc/experiments/phase1/naturalistic_local_copy_probe_design.md`.
