# Phase 3 Toy Ontology v2: Attention-Head Dimension Intervention

Date: 2026-05-23

## Executive Summary

This is the first larger ordinary-attention-head experiment after the user
clarified the project direction. The unit is still an ordinary attention head,
not a MoE expert, not a branch tower, and not a router module.

The tested question is:

```text
If attention heads have different structural shapes, especially different head
dimensions, are some functional roles more likely to be learned by particular
head types, and does that change specialization or modularity?
```

The answer from this round:

1. Structural role affinity is strong. Larger head types are selected as the
   top causal structural type far more often than chance, and this gets stronger
   as the dimension imbalance increases.
2. Functional specialization is strong. Heterogeneous head dimensions usually
   make each role's causal effect more concentrated into fewer heads.
3. Functional modularity is mixed. Some heterogeneous layouts improve family
   clustering over uniform baselines, but heterogeneity alone does not
   automatically make related role families cluster.

This means the project is still worth continuing. The strongest current claim
is not "heterogeneous heads always create modularity." The stronger, cleaner
claim is:

```text
Heterogeneous ordinary attention-head dimensions induce structural role affinity
and increase role-level specialization; family-level modularity is a separate,
harder property that depends on the role ontology and exact layout.
```

## What Was Tested

The experiment trained small decoder-only transformers on a single synthetic
multi-role next-token dataset. Each role contributes target positions in the
sequence. After training, each ordinary attention head is ablated one at a time.
For each role, the accuracy drop caused by each head is converted into one row
of a role x head causal matrix.

The three project questions are computed from the same role x head matrix.

| Question | Plain Meaning | Metric Used Here |
|---|---|---|
| Structural role affinity | Which head type does a role prefer? | Top structural dimension per role row; largest-dim top rate; top-dim mass. |
| Functional specialization | Is a role concentrated in one/few heads? | Top role mass and effective heads from the ablation-effect distribution. |
| Functional modularity | Do related roles use similar head distributions? | Within-family minus between-family similarity, plus adjusted Rand index (ARI). |

The important distinction is that specialization is role-level concentration,
while modularity is family-level clustering. A role can be highly specialized in
one head without the related roles in its family clustering together.

## Ontology And Dataset Setup

The ontology hierarchy is:

```text
ontology
  family
    role / subrole
      synthetic scene
        target positions
          role x head causal row
```

Toy Ontology v2 has 5 families, each with 4 roles, for 20 role rows.

| Family | Roles | Main Literature Anchor |
|---|---|---|
| `copy_transport` | `local_copy`, `previous_token`, `kv_lookup`, `duplicate_token` | Transformer Circuits QK/OV framing; BERT/head pattern papers. |
| `induction` | `induction_short`, `induction_long`, `induction_ngram`, `false_induction_control` | Olsson et al. 2022 induction heads; Elhage et al. 2021 previous-token composition. |
| `position_boundary` | `bos_sink`, `sep_sink`, `fixed_offset_prev`, `punctuation_boundary` | Clark et al. 2019 and Voita et al. 2019 positional/delimiter heads. |
| `suppression_conflict` | `distractor_suppression`, `anti_copy`, `wrong_key_suppression`, `recency_conflict` | Copy suppression and IOI-style negative/suppression heads. |
| `entity_coreference` | `repeated_name_detection`, `pronoun_antecedent`, `simple_ioi_name_mover`, `negative_name_control` | Clark et al. 2019 coreference heads; Wang et al. 2022/2023 IOI circuit. |

### Literature Anchors

| Source | Venue / Year | Roles It Motivates Here |
|---|---:|---|
| Clark et al., "What Does BERT Look at? An Analysis of BERT's Attention" | BlackboxNLP at ACL 2019 | delimiter/sink, fixed-position, syntax, and coreference-style head patterns. |
| Voita et al., "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" | ACL 2019 | specialized heads, positional heads, syntactic heads, rare-token attention. |
| Elhage et al., "A Mathematical Framework for Transformer Circuits" | Transformer Circuits Thread 2021 | QK/OV decomposition, previous-token heads, attention-head circuits. |
| Olsson et al., "In-context Learning and Induction Heads" | Transformer Circuits Thread / arXiv 2022 | induction heads and previous-token-head composition. |
| Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" | ICLR 2023 | name movers, negative name movers, duplicate-token/name roles. |
| McDougall et al., "Copy Suppression" | arXiv/OpenReview 2023; BlackboxNLP 2024 version | copy suppression, anti-copy, negative copy effects, self-repair. |
| Htut et al., "Do Attention Heads in BERT Track Syntactic Dependencies?" | arXiv 2019 | syntax-family follow-up roles, not yet in this toy sweep. |
| Pande et al., "The Heads Hypothesis" | AAAI 2021 | head-role taxonomy and possible co-location of roles in one head. |

The current run uses synthetic scenes, not natural-language datasets. The
papers motivate the role vocabulary, but the evidence in this document comes
from controlled toy causal training and ablation.

### Concrete Role Scenes

These are the exact style of examples generated for each row. Tokens are random
IDs in the actual dataset, so the model cannot solve the task by memorizing one
fixed target token.

| Family | Role | Synthetic Scene | Target Meaning |
|---|---|---|---|
| `copy_transport` | `local_copy` | `[x, LOCAL_SEP, x]` | At `LOCAL_SEP`, predict prior token `x`. |
| `copy_transport` | `previous_token` | `[x, y, PREV_SEP, y]` | At `PREV_SEP`, copy immediately previous token `y`. |
| `copy_transport` | `kv_lookup` | `[KV_SEP, k1,v1,...,kq,vq]` | At query key `kq`, predict paired value `vq`. |
| `copy_transport` | `duplicate_token` | `[DUP_SEP, a,b,c,a,DUP_QUERY,a]` | At query, predict repeated token `a`. |
| `induction` | `induction_short` | `[IND_SHORT_SEP, x1..x6, x1..x6]` | On the second copy, predict the following token from the first copy. |
| `induction` | `induction_long` | `[IND_LONG_SEP, x1..x12, x1..x12]` | Same induction rule over longer context. |
| `induction` | `induction_ngram` | `[IND_NGRAM_SEP, x1..x9, x1..x9]` | Repeated n-gram continuation. |
| `induction` | `false_induction_control` | `[FALSE_IND_SEP, a,b,c,a,c]` | At second `a`, predict `c`, not the earlier continuation `b`. |
| `position_boundary` | `bos_sink` | `[anchor,n1,n2,n3,BOS_QUERY,anchor]` | Retrieve the first token in the segment. |
| `position_boundary` | `sep_sink` | `[SEP_ANCHOR,value,n1,n2,SEP_QUERY,value]` | Retrieve value after the earlier separator. |
| `position_boundary` | `fixed_offset_prev` | `[a,b,c,d,OFFSET_QUERY,b]` | Retrieve the token three positions back. |
| `position_boundary` | `punctuation_boundary` | `[n1,PUNCT,value,n2,PUNCT_QUERY,value]` | Retrieve the token after punctuation. |
| `suppression_conflict` | `distractor_suppression` | `[SUPPRESS_SEP,target,distractor,distractor,SUPPRESS_QUERY,target]` | Ignore repeated distractor and predict target. |
| `suppression_conflict` | `anti_copy` | `[a,b,c,a,ANTI_COPY_QUERY,c]` | Ignore misleading repeated-token continuation. |
| `suppression_conflict` | `wrong_key_suppression` | `[WK_SEP,key,right,wrong,wrong_value,wrong,trap,key,right]` | Use final key and ignore wrong-key decoys. |
| `suppression_conflict` | `recency_conflict` | `[RECENCY_SEP,key,old,key,recent,key,old]` | Prefer older value, not the closer recent value. |
| `entity_coreference` | `repeated_name_detection` | `[REP_NAME_SEP,A,B,A,REP_NAME_QUERY,A]` | Identify repeated name `A`. |
| `entity_coreference` | `pronoun_antecedent` | `[name,PRONOUN,n1,n2,PRONOUN_QUERY,name]` | Retrieve name before pronoun marker. |
| `entity_coreference` | `simple_ioi_name_mover` | `[A,B,A,IOI_QUERY,B]` | Predict the non-repeated name `B`. |
| `entity_coreference` | `negative_name_control` | `[A,B,B,NEG_NAME_QUERY,A]` | Predict non-repeated name `A`; repeated `B` is distractor. |

The exact sampled token examples are saved in:

```text
results/phase3_toy_role_ontology_v2_full_1600_20260523/role_dataset_examples.json
```

## Configurations

All comparisons use the same total attention dimension, 128. Non-uniform
settings use all-distinct dimensions, as requested.

| Config | Head Dims | Why It Is Included |
|---|---|---|
| `uniform4` | `[32,32,32,32]` | Four-head uniform baseline. |
| `uniform2` | `[64,64]` | Strong fewer/wider uniform baseline. |
| `hetero4_unique_mild` | `[16,24,40,48]` | Four all-distinct heads, mild imbalance. |
| `hetero4_unique_64` | `[8,16,40,64]` | Four all-distinct heads with one 64-dim head. |
| `hetero4_unique_extreme` | `[8,16,24,80]` | Four all-distinct heads with extreme imbalance. |
| `hetero2_unique_mild` | `[48,80]` | Two all-distinct heads, mild imbalance. |
| `hetero2_unique_mid` | `[32,96]` | Two all-distinct heads, medium imbalance. |
| `hetero2_unique_extreme` | `[16,112]` | Two all-distinct heads, extreme imbalance. |

Training setting:

```text
role_set: v2_full
roles: 20
families: 5
seeds: 1,2,3,4,5
steps: 1600
eval examples: 512
sequence length: 162
```

## Metrics

Let `score(role, head)` be the positive causal importance of an ordinary
attention head for a role, measured as the accuracy drop caused by ablating that
head. Negative scores are clipped to zero and each role row is normalized over
all layer-head slots.

### Structural Role Affinity

This asks: "Which structural head type does this role use?"

For each role and seed:

1. Compute the normalized ablation-effect distribution over heads.
2. Sum mass by head dimension.
3. Record the dimension with the largest mass as `top_dim`.
4. Report how often the largest dimension is `top_dim`.

For heterogeneous four-head configs, chance largest-dim top rate is 0.25 if the
top type is random across the four distinct head dimensions. For heterogeneous
two-head configs, chance is 0.50. Uniform baselines have no meaningful type
preference because all dimensions are identical.

### Functional Specialization

This asks: "How concentrated is the role into one or a few heads?"

Metrics:

- `specialization`: largest mass in the role's normalized head distribution.
  Higher means one head dominates more.
- `effective_heads`: `exp(entropy(distribution))`. Lower means fewer effective
  heads carry the role.

### Functional Modularity

This asks: "Do related roles in the ontology use similar head distributions?"

Metrics:

- `family_gap`: mean same-family similarity minus mean different-family
  similarity. Higher means roles inside the same family are more similar to
  each other than to roles in other families.
- `ARI`: adjusted Rand index after clustering role distributions into the known
  number of families. Higher means the learned clusters align with the ontology
  family labels. ARI near 0 means little better than chance; 1 means perfect
  family recovery.

## Main Results

### Learnability Check

All configs learned the 20-role dataset. Minimum role accuracy averaged across
seeds stayed at or above 0.991. This matters because the role-affinity and
specialization results are not artifacts of one architecture failing the task.

| Config | Min Role Accuracy | Interpretation |
|---|---:|---|
| `uniform4` | 0.998 | Learned. |
| `uniform2` | 0.991 | Learned, but weakest minimum-role margin. |
| `hetero4_unique_mild` | 0.999 | Learned. |
| `hetero4_unique_64` | 0.999 | Learned. |
| `hetero4_unique_extreme` | 0.999 | Learned. |
| `hetero2_unique_mild` | 0.999 | Learned. |
| `hetero2_unique_mid` | 0.996 | Learned. |
| `hetero2_unique_extreme` | 0.997 | Learned. |

### Question 1: Structural Role Affinity

Baseline:

- In `uniform4` and `uniform2`, all heads have the same dimension, so there is
  no real structural type choice.
- The correct baseline for heterogeneous configs is chance over distinct head
  types: 0.25 for four distinct heads and 0.50 for two distinct heads.

Observed result:

| Config | Head Dims | Chance Largest-Top Rate | Observed Largest-Top Rate | Mean Top-Dim Mass |
|---|---|---:|---:|---:|
| `hetero4_unique_mild` | `[16,24,40,48]` | 0.25 | 0.50 | 0.759 |
| `hetero4_unique_64` | `[8,16,40,64]` | 0.25 | 0.65 | 0.821 |
| `hetero4_unique_extreme` | `[8,16,24,80]` | 0.25 | 0.82 | 0.912 |
| `hetero2_unique_mild` | `[48,80]` | 0.50 | 0.66 | 0.839 |
| `hetero2_unique_mid` | `[32,96]` | 0.50 | 0.82 | 0.871 |
| `hetero2_unique_extreme` | `[16,112]` | 0.50 | 0.94 | 0.937 |

Interpretation:

This is the strongest result. The larger head type attracts many role rows, and
the effect increases monotonically with imbalance. This directly supports the
user's intended claim that, in heterogeneous ordinary-head models, some roles
are more likely to be learned by particular structural head types.

### Question 2: Functional Specialization

Baseline:

| Uniform Baseline | Specialization | Effective Heads |
|---|---:|---:|
| `uniform4 [32,32,32,32]` | 0.733 | 2.400 |
| `uniform2 [64,64]` | 0.684 | 2.166 |

Heterogeneous results:

| Config | Specialization | Effective Heads | Comparison |
|---|---:|---:|---|
| `hetero4_unique_mild` | 0.726 | 2.206 | Similar to `uniform4`, better than `uniform2` on specialization. |
| `hetero4_unique_64` | 0.776 | 1.906 | Better than both uniform baselines. |
| `hetero4_unique_extreme` | 0.849 | 1.522 | Strongest role-level specialization. |
| `hetero2_unique_mild` | 0.782 | 1.802 | Better than both uniform baselines. |
| `hetero2_unique_mid` | 0.776 | 1.692 | Better than both uniform baselines. |
| `hetero2_unique_extreme` | 0.780 | 1.636 | Better specialization, but see modularity caveat below. |

Interpretation:

Heterogeneous head dimensions generally concentrate a role's causal effect into
fewer heads. The most extreme four-head setting, `[8,16,24,80]`, has the
highest specialization and lowest effective-head count.

### Question 3: Functional Modularity

Baseline:

| Uniform Baseline | Family Gap | ARI |
|---|---:|---:|
| `uniform4 [32,32,32,32]` | 0.153 | 0.149 |
| `uniform2 [64,64]` | 0.117 | 0.108 |

Heterogeneous results:

| Config | Family Gap | ARI | Comparison |
|---|---:|---:|---|
| `hetero4_unique_mild` | 0.137 | 0.082 | Gap near uniform, ARI lower. |
| `hetero4_unique_64` | 0.127 | 0.113 | Similar to `uniform2`, below `uniform4`. |
| `hetero4_unique_extreme` | 0.138 | 0.111 | Strong specialization, only moderate family clustering. |
| `hetero2_unique_mild` | 0.128 | 0.112 | Similar to `uniform2`. |
| `hetero2_unique_mid` | 0.151 | 0.159 | Best or tied-best modularity in the main grid. |
| `hetero2_unique_extreme` | 0.050 | 0.039 | Extreme collapse hurts family modularity. |

Interpretation:

This is mixed. The `[32,96]` two-head hetero config is promising and slightly
beats the uniform baselines on ARI, but the strongest specialization config does
not automatically have the strongest modularity. The `[16,112]` setting is the
clearest warning: it has high structural affinity and high specialization, but
poor family modularity. Too much imbalance can collapse many roles into one
large head type, which is specialization without useful family-level modularity.

## Layout Permutation Control

The user asked whether the role is really following the structural type or just
memorizing a head index like `H3`. To test that, the `[8,16,40,64]` vector was
permuted across head indices:

```text
[8,16,40,64]
[64,8,16,40]
[8,64,16,40]
[8,16,64,40]
```

Each layout used 5 seeds and the same 20-role ontology.

### Aggregate Layout Result

| Layout | 64-Dim Head Position | Largest-Top Rate | Specialization | Family Gap | ARI |
|---|---:|---:|---:|---:|---:|
| `[8,16,40,64]` | H3 | 0.65 | 0.776 | 0.127 | 0.113 |
| `[64,8,16,40]` | H0 | 0.68 | 0.803 | 0.100 | 0.089 |
| `[8,64,16,40]` | H1 | 0.67 | 0.747 | 0.086 | 0.096 |
| `[8,16,64,40]` | H2 | 0.60 | 0.779 | 0.186 | 0.189 |

Interpretation:

The largest-dim preference persists when the 64-dim head moves. This supports a
structural-head-type interpretation rather than a fixed-head-index story.
However, family modularity is layout-sensitive: the H2 placement has the best
family gap and ARI, while other placements do not.

### Per-Role 64-Dim Preference Across Layouts

There are 20 trials per role here: 4 layouts x 5 seeds.

| Role | Family | 64-Dim Top Cases | Rate |
|---|---|---:|---:|
| `local_copy` | `copy_transport` | 20/20 | 1.00 |
| `negative_name_control` | `entity_coreference` | 19/20 | 0.95 |
| `recency_conflict` | `suppression_conflict` | 19/20 | 0.95 |
| `wrong_key_suppression` | `suppression_conflict` | 19/20 | 0.95 |
| `pronoun_antecedent` | `entity_coreference` | 18/20 | 0.90 |
| `kv_lookup` | `copy_transport` | 17/20 | 0.85 |
| `simple_ioi_name_mover` | `entity_coreference` | 17/20 | 0.85 |
| `duplicate_token` | `copy_transport` | 13/20 | 0.65 |
| `repeated_name_detection` | `entity_coreference` | 13/20 | 0.65 |
| `bos_sink` | `position_boundary` | 13/20 | 0.65 |
| `false_induction_control` | `induction` | 12/20 | 0.60 |
| `anti_copy` | `suppression_conflict` | 12/20 | 0.60 |
| `induction_short` | `induction` | 10/20 | 0.50 |
| `fixed_offset_prev` | `position_boundary` | 10/20 | 0.50 |
| `distractor_suppression` | `suppression_conflict` | 10/20 | 0.50 |
| `punctuation_boundary` | `position_boundary` | 9/20 | 0.45 |
| `previous_token` | `copy_transport` | 8/20 | 0.40 |
| `induction_long` | `induction` | 8/20 | 0.40 |
| `induction_ngram` | `induction` | 7/20 | 0.35 |
| `sep_sink` | `position_boundary` | 6/20 | 0.30 |

Interpretation:

Not every role wants the biggest head. This is scientifically useful because it
means the result is not merely "all functions collapse to the biggest head."
Local copy, KV lookup, wrong-key suppression, recency conflict, and several
entity/coreference roles strongly prefer the 64-dim type. Several induction and
position-boundary roles are much less locked to 64. That gives the next paper a
more nuanced claim: different role families have different structural
affinities.

## How Strong Are The Results?

| Claim | Current Evidence Strength | Why |
|---|---|---|
| Heterogeneous dimensions create structural role affinity. | Strong in this toy setting. | Largest-dim top rates exceed chance in every hetero config and increase with imbalance. Layout controls show roles follow the moved 64-dim type. |
| Heterogeneous dimensions increase role specialization. | Strong in this toy setting. | Most hetero configs beat both uniform baselines on top-role mass and effective-head count; all hetero configs beat `uniform2` on specialization. |
| Heterogeneous dimensions create functional modularity. | Mixed / not yet a broad claim. | `hetero2_unique_mid` and one layout beat uniform baselines, but other hetero configs do not, and extreme imbalance can hurt modularity. |
| The ontology maps to real language-model roles. | Plausible but not proven by this run. | The role names are literature-grounded, but the dataset is synthetic. Real-model validation remains needed. |

## Why This Does Not Yet Prove Full Language-Model Modularity

This run is important but still toy evidence.

Limitations:

1. The dataset is synthetic. It is designed to isolate role families, not to
   represent natural language.
2. The model is small. The result may change with depth, scale, or pretrained
   corpora.
3. The modularity ontology is researcher-defined. Family labels are meaningful
   for this synthetic world, but they are not yet a complete ontology of all LM
   functions.
4. Role measurement uses single-head ablation. That is causal and useful, but
   it can miss distributed backup paths and nonlinear compensation.
5. Family-level ARI is modest. The best values in this round are informative,
   not decisive.

## Decision

Continue the project.

The results support the central direction strongly enough to keep investing:

```text
different-shaped ordinary attention heads can bias which head type learns which
role, and this can make roles more specialized.
```

The next stage should not claim too much about modularity yet. It should use
this round to sharpen the claim and then test whether modularity appears under
specific layouts, richer role ontologies, and real-model probes.

## Next Steps

1. Extend layout controls for the best modularity layout `[8,16,64,40]` and the
   best specialization layout `[8,16,24,80]`.
2. Add more roles inside families that currently show different behavior:
   induction and position-boundary roles are less 64-locked than copy/conflict
   roles.
3. Run a real-model validation pass using ordinary pretrained attention heads:
   local-copy, induction/repeated n-gram, delimiter/BOS, copy suppression, and
   IOI/name-mover probes.
4. Add confidence intervals or bootstrap tests for the main tables before paper
   writing.
5. Keep every future result in the same format:

```text
baseline result -> heterogeneous result -> interpretation
```

## Artifacts

Scripts:

```text
scripts/toy_role_ontology_v2_head_dim_intervention.py
scripts/analyze_role_ontology_v2.py
```

Main result root:

```text
results/phase3_toy_role_ontology_v2_full_1600_20260523
```

Layout control root:

```text
results/phase3_toy_role_ontology_v2_layout_1600_20260523
```

Useful analysis files:

```text
results/phase3_toy_role_ontology_v2_full_1600_20260523/analysis/model_metric_table.csv
results/phase3_toy_role_ontology_v2_full_1600_20260523/analysis/affinity_table.csv
results/phase3_toy_role_ontology_v2_full_1600_20260523/analysis/family_metric_table.csv
results/phase3_toy_role_ontology_v2_full_1600_20260523/analysis/role_top_dim_counts.csv
results/phase3_toy_role_ontology_v2_layout_1600_20260523/analysis/model_metric_table.csv
results/phase3_toy_role_ontology_v2_layout_1600_20260523/analysis/affinity_table.csv
results/phase3_toy_role_ontology_v2_layout_1600_20260523/analysis/role_top_dim_counts.csv
```

Reproduction commands:

```bash
python scripts/toy_role_ontology_v2_head_dim_intervention.py \
  --role-set v2_full \
  --configs uniform4 uniform2 hetero4_unique_mild hetero4_unique_64 hetero4_unique_extreme hetero2_unique_mild hetero2_unique_mid hetero2_unique_extreme \
  --seeds 1 2 3 4 5 \
  --steps 1600 \
  --batch-size 128 \
  --eval-examples 512 \
  --output-dir results/phase3_toy_role_ontology_v2_full_1600_20260523

python scripts/analyze_role_ontology_v2.py \
  results/phase3_toy_role_ontology_v2_full_1600_20260523 \
  --title "Toy Ontology v2 Full 20-Role Sweep"

python scripts/toy_role_ontology_v2_head_dim_intervention.py \
  --role-set v2_full \
  --configs 8,16,40,64 64,8,16,40 8,64,16,40 8,16,64,40 \
  --seeds 1 2 3 4 5 \
  --steps 1600 \
  --batch-size 128 \
  --eval-examples 512 \
  --output-dir results/phase3_toy_role_ontology_v2_layout_1600_20260523

python scripts/analyze_role_ontology_v2.py \
  results/phase3_toy_role_ontology_v2_layout_1600_20260523 \
  --title "Toy Ontology v2 64-Dim Layout Control"
```
