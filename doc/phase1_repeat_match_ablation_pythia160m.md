# Phase 1: Pythia-160M Repeat-Match Head Ablation

Date: 2026-05-21

This checkpoint tests whether the repeat-match heads identified by the role
proxy are causally useful for a repeated-token next-token prediction task. The
previous Phase 1 result was correlational: repeat-match attention roles were
concentrated in early layers and became stable across seeds after raw-score
Hungarian alignment. This run asks whether ablating those heads actually hurts
behavior.

## Command

```bash
python3 -u scripts/repeat_match_ablation.py \
  --model-size 160m \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --revision step143000 \
  --role-scores results/phase1_pythia160m_attention_role_specialization/head_role_scores.csv \
  --alignment-summary results/phase0_pythia160m_seed1_to_9_step143000_raw_scores/summary.json \
  --layers 0,1 \
  --top-k-per-layer 1 \
  --random-controls 8 \
  --eval-sequences 64 \
  --repeat-length 32 \
  --batch-size 8 \
  --output-dir results/phase1_repeat_match_ablation_pythia160m
```

## Method

The script selects the top repeat-match head in layers 0 and 1 for each seed,
using the role scores from `attention_role_specialization.py`.

The evaluation task uses synthetic repeated-token sequences:

```text
[x_1, ..., x_32, x_1, ..., x_32]
```

Loss is measured only on second-half continuation positions, where the model
must predict `x_{i+1}` after seeing the second occurrence of `x_i`. The same 64
synthetic evaluation sequences are reused across all seeds.

The ablation zeros selected per-head attention outputs before the GPT-NeoX
attention output projection. Four conditions are compared:

- **own_top**: ablate the target seed's own top repeat-match heads in layers
  0 and 1.
- **own_random**: ablate random same-layer heads in the target model, excluding
  the top repeat-match heads.
- **source_same_index**: take another seed's top repeat-match head indices and
  ablate those same layer/head indices in the target model.
- **source_aligned**: take another seed's top repeat-match heads and map them
  into the target seed with the Phase 0 raw-score Hungarian alignment, then
  ablate the mapped heads.

## Aggregate Results

Positive loss delta means the ablation made the repeated-token task worse.
Negative target-logit delta means the ablation reduced the logit of the correct
next token.

| Condition | n | Mean loss delta | Std | Mean target-logit delta | Std |
|---|---:|---:|---:|---:|---:|
| own_random | 72 | 0.2403 | 0.4842 | -0.7184 | 1.2687 |
| own_top | 9 | 1.5244 | 0.4411 | -2.0689 | 0.6306 |
| source_aligned | 72 | 1.0538 | 0.7024 | -1.5750 | 0.9408 |
| source_same_index | 72 | 0.2571 | 0.3631 | -0.7014 | 1.1375 |

## Paired Comparisons

The most important comparison is paired by target seed and source seed:
`source_aligned - source_same_index`.

| Comparison | Mean difference | Std | Directional count |
|---|---:|---:|---:|
| Aligned minus same-index loss delta | 0.7967 | 0.7750 | 59 / 72 positive |
| Aligned minus same-index target-logit delta | -0.8736 | 1.4090 | 58 / 72 negative |

The own-top condition also beats the target seed's own random controls:

```text
mean(own_top - target_random_mean) = 1.2841
positive in 9 / 9 seeds
```

## Main Finding

The repeat-match heads are not just attention-pattern artifacts. Ablating each
seed's own top repeat-match heads causes a large repeated-token loss increase.
More importantly for the cross-seed question, heads transferred across seeds by
raw-score alignment are much more damaging than heads transferred by same index:

```text
source_aligned loss delta:    1.0538
source_same_index loss delta: 0.2571
own_random loss delta:        0.2403
```

This supports the current project thesis:

```text
functional specialization exists;
same-index stability is weak;
alignment-based stability is substantially stronger;
the aligned role is causally relevant for task behavior.
```

## Caveats

- This is still a simple synthetic repeated-token task, not a broad language
  modeling benchmark.
- The ablation zeros head outputs; it does not isolate downstream paths or prove
  that the head is the only route for the behavior.
- Random controls are same-layer controls, but some random heads may have other
  causal functions on the synthetic task.
- The alignment is cross-seed within the same architecture. It does not yet test
  the project's final heterogeneous-branch architecture question.
- Only Pythia-160M at `step143000` was tested here.

## Decision

Continue with repeat-match / induction-style behavior as the first causal test
bed. The next research step should move from same-architecture cross-seed
stability to the structural question:

```text
Does structural branch design or structural heterogeneity induce stable
functional specialization?
```

The immediate implementation target is a small branched-attention training or
fine-tuning setup where branch structure can be varied, then measured with the
same raw-score alignment, role specialization, and causal ablation pipeline.
