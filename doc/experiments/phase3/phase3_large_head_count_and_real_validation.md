# Phase 3: Larger Head-Count Control And Real-Model Role Validation

Date: 2026-05-24

## Why This Round Was Run

The user pointed out a serious possible confound in Toy Ontology v2:

```text
Maybe family-level modularity looked weak because there were too few heads for a
sparse modular distribution to appear.
```

That concern is valid. The previous Toy Ontology v2 models had:

| Previous Setting | Heads Per Layer | Layers | Total Ordinary Head Slots |
|---|---:|---:|---:|
| `uniform2` / hetero2 | 2 | 2 | 4 |
| `uniform4` / hetero4 | 4 | 2 | 8 |

Those settings were enough to test structural role affinity, but maybe too
small to test ontology-level modularity fairly. This round therefore reran the
same 20-role ontology with more ordinary attention heads.

## Scope Lock

- Unit: ordinary attention heads.
- Intervention: different attention-head dimensions.
- No MoE experts, no branch towers, no SwitchHead units.
- Same Toy Ontology v2: 20 roles, 5 families.

## Larger-Head Configurations

The new configs use 8 heads per layer. Uniform and heterogeneous configs are
matched on total attention dimension.

| Config | Head Dims | Total Attention Dim | Notes |
|---|---|---:|---|
| `uniform8` | `[48,48,48,48,48,48,48,48]` | 384 | 8-head uniform baseline. |
| `hetero8_unique_spread` | `[16,24,32,40,48,56,72,96]` | 384 | All-distinct, spread-out dimensions. |
| `hetero8_unique_extreme` | `[8,16,24,32,40,48,64,152]` | 384 | All-distinct, one very large head. |

Two model-depth settings were tested:

| Setting | Heads Per Layer | Layers | Total Head Slots | Steps |
|---|---:|---:|---:|---:|
| 32-slot larger model | 8 | 4 | 32 | 1000 |
| 16-slot control | 8 | 2 | 16 | 1000, then 2000 after undertraining |

## Metrics

The same metrics as Toy Ontology v2 were used.

| Question | Metric |
|---|---|
| Structural role affinity | largest-dimension top rate; chance is `1/8 = 0.125` for hetero8. |
| Functional specialization | top role mass and effective heads. |
| Functional modularity | family gap and ARI over exact layer-head slots. |
| Structural-type modularity control | family gap and ARI after collapsing each role distribution by head dimension. |

## Main Result Table

### 32-Slot Larger Model

This model uses 8 heads per layer and 4 layers.

| Config | Min Acc | Largest-Top Rate | Specialization | Effective Heads | Family Gap | ARI | Dim-Family Gap | Dim-ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform8` | 0.996 | n/a | 0.517 | 9.117 | 0.048 | 0.060 | 0.000 | 0.000 |
| `hetero8_unique_spread` | 0.991 | 0.50 | 0.488 | 8.020 | 0.050 | 0.051 | 0.019 | 0.013 |
| `hetero8_unique_extreme` | 0.990 | 0.72 | 0.619 | 4.589 | 0.038 | 0.025 | 0.023 | 0.018 |

Interpretation:

- More head slots did not rescue family-level modularity.
- Structural role affinity remained strong: largest-dim top rates were far
  above the 8-way chance baseline of `0.125`.
- Extreme heterogeneity increased specialization, but family modularity stayed
  weak.

### 16-Slot Control, Clean 2000-Step Run

This model uses 8 heads per layer and 2 layers. The first 1000-step run showed
undertraining in `uniform8`, so this cleaner 2000-step run is the one to use.

| Config | Min Acc | Largest-Top Rate | Specialization | Effective Heads | Family Gap | ARI | Dim-Family Gap | Dim-ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform8` | 0.978 | n/a | 0.514 | 5.314 | 0.132 | 0.130 | 0.000 | 0.000 |
| `hetero8_unique_spread` | 0.988 | 0.62 | 0.609 | 3.923 | 0.079 | 0.046 | 0.042 | 0.044 |
| `hetero8_unique_extreme` | 0.997 | 0.76 | 0.784 | 2.247 | 0.081 | 0.058 | 0.046 | 0.063 |

Interpretation:

- With 16 slots, heterogeneity clearly increased specialization.
- It still did not beat the uniform8 baseline on family modularity.
- The 16-slot setting had better family gap than the 32-slot setting, which
  suggests extra depth/slots can fragment exact-slot clustering.
- But even after fixing the undertraining issue, hetero8 modularity remained
  below the matched uniform8 baseline.

## Answer To The Head-Count Concern

The larger-head experiment does not support the hypothesis that the previous
mixed modularity result was merely caused by too few heads.

The cleaner statement is:

```text
More ordinary head slots preserve or strengthen structural role affinity and
specialization, but they do not by themselves make the heterogeneous model more
family-modular than a matched uniform-head baseline.
```

This is not bad for the project. It makes the result sharper:

- structural role affinity: strong;
- role-level specialization: strong;
- family-level modularity: not automatic.

## Failure Analysis

The negative/mixed modularity result has several plausible causes.

| Possible Factor | Evidence From This Round | Status |
|---|---|---|
| Too few heads | Larger 16-slot and 32-slot runs still did not make hetero beat uniform on modularity. | Less likely. |
| Too much depth fragments exact slots | 32-slot family gap was weaker than 16-slot family gap. | Plausible. |
| Heterogeneity attracts by role difficulty, not family | Largest head attracts many unrelated roles, especially in extreme configs. | Plausible. |
| Family ontology too coarse | Some roles inside a family behave differently, especially induction and boundary roles. | Plausible. |
| Single-head ablation undercounts distributed paths | Larger models can distribute redundant behavior over more heads. | Still possible. |
| Extreme imbalance causes collapse | `hetero8_unique_extreme` has strong specialization but weak modularity. | Supported. |

Targeted tweaks already run:

1. More heads and more slots: 32-slot model.
2. Less depth with more heads: 16-slot model.
3. Longer training for the 16-slot baseline after undertraining appeared.
4. Dimension-level modularity analysis, to check whether families cluster by
   structural type rather than exact slot.

Result of these tweaks:

```text
The modularity conclusion did not flip. Heterogeneity robustly biases role
assignment and can concentrate roles, but family-level clustering remains
weaker than the matched uniform baseline in the larger-head settings.
```

## Real-Model Role Validation

This part does not test heterogeneous dimensions, because standard Pythia
models have uniform head dimensions. It tests a different prerequisite:

```text
Are the role families we use in the toy ontology measurable in real ordinary
attention heads?
```

### New Pythia-160M-Deduped Attention Probe

Run:

```text
EleutherAI/pythia-160m-deduped, revision step143000, float32
```

The first float16 run produced NaNs in later-layer attention summaries, so it
was rerun in float32. The float32 run had no NaN rows.

| Role Probe | Best Layer | Best Head | Top Specialization | Effective Heads | Interpretation |
|---|---:|---:|---:|---:|---|
| `repeat_match` | 0 | 7 | 0.834 | 2.088 | Very concentrated repeat/induction-like attention role. |
| `repeat_match` | 6 | 10 | 0.725 | 3.445 | Second strong repeat-match layer. |
| `previous_token` | 10 | 9 | 0.332 | 9.169 | Measurable but more distributed. |
| `bos` | 8 | 6 | 0.140 | 9.809 | Diffuse BOS/sink attention. |

Interpretation:

- Repeat-match / induction-like behavior is strongly measurable in real
  ordinary heads.
- Previous-token behavior is present but less concentrated.
- BOS/sink behavior is present but diffuse in this probe.

### Existing Causal Pythia Validation Results

These are previously generated results in the repo, reused here as real-model
evidence because they are causal ablation tests over ordinary attention heads.

| Real-Model Probe | Model / Dataset | Own Top Loss Delta | Random Loss Delta | Own Top Excess | Aligned Transfer | Same-Index Transfer |
|---|---|---:|---:|---:|---:|---:|
| Local copy | Pythia-160M seeds, synthetic local-copy | 2.519 | 0.229 | 2.290 | 2.271 | 0.488 |
| Natural repeated 8-gram | Pythia-160M seeds, WikiText-103 | 0.387 | 0.016 | 0.372 | 0.315 | 0.033 |

Interpretation:

- The local-copy and natural-repeat roles are not just attention-pattern labels;
  ablating the selected ordinary heads causes meaningful loss increases.
- Cross-seed alignment transfers the role much better than same head index.
- This supports the broader research program's role-measurement machinery, but
  it still does not prove structural heterogeneity in pretrained Pythia because
  Pythia does not have heterogeneous head dimensions.

## Updated Project Interpretation

The current best claim is now:

```text
Heterogeneous ordinary attention-head dimensions create structural role affinity
and usually increase role-level specialization. Larger-head controls do not
show that heterogeneity automatically improves family-level modularity.
```

The paper should separate the claims:

1. Structural role affinity: which head type a role chooses.
2. Functional specialization: how concentrated a role is.
3. Functional modularity: whether related role families cluster.

The first two are strong in the toy experiments. The third is still a research
question, not an established positive result.

## What To Do Next

Recommended next step:

1. Keep the larger-head result as a negative/moderating result for modularity.
2. Expand the ontology where family labels are weak:
   - induction variants;
   - boundary/sink variants;
   - suppression/conflict variants.
3. Add a role-difficulty analysis:
   - sequence distance;
   - conflict level;
   - target entropy;
   - whether the role prefers the biggest head independent of family.
4. For real-model validation, continue with ordinary-head causal probes:
   - repeat/induction;
   - local copy;
   - copy suppression;
   - IOI/name mover.
5. Do not claim that heterogeneous dimensions cause full functional modularity
   unless a future family-level experiment beats matched uniform baselines.

## Artifacts

Larger-head scripts:

```text
scripts/toy_role_ontology_v2_head_dim_intervention.py
scripts/analyze_role_ontology_v2.py
```

Larger-head result roots:

```text
results/phase3_toy_role_ontology_v2_large_heads_1000_20260523
results/phase3_toy_role_ontology_v2_large_heads_2layer_1000_20260523
results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523
```

Real-model probe root:

```text
results/phase3_real_model_role_probe_pythia160m_deduped_float32_20260524
```

Existing causal real-model result roots used:

```text
results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2
results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128
```

Main commands:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/toy_role_ontology_v2_head_dim_intervention.py \
  --role-set v2_full \
  --configs uniform8 hetero8_unique_spread hetero8_unique_extreme \
  --seeds 1 2 3 4 5 \
  --steps 1000 \
  --batch-size 64 \
  --eval-examples 512 \
  --d-model 256 \
  --n-layers 4 \
  --mlp-dim 512 \
  --output-dir results/phase3_toy_role_ontology_v2_large_heads_1000_20260523

CUDA_VISIBLE_DEVICES=2 python scripts/toy_role_ontology_v2_head_dim_intervention.py \
  --role-set v2_full \
  --configs uniform8 hetero8_unique_spread hetero8_unique_extreme \
  --seeds 1 2 3 4 5 \
  --steps 2000 \
  --batch-size 64 \
  --eval-examples 512 \
  --d-model 256 \
  --n-layers 2 \
  --mlp-dim 512 \
  --output-dir results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523

CUDA_VISIBLE_DEVICES=3 python scripts/attention_role_specialization.py \
  --model-template 'EleutherAI/pythia-{model_size}-deduped' \
  --model-size 160m \
  --seeds base \
  --revision step143000 \
  --num-texts 16 \
  --max-length 64 \
  --batch-size 4 \
  --synthetic-repeat-sequences 64 \
  --synthetic-repeat-length 32 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase3_real_model_role_probe_pythia160m_deduped_float32_20260524
```
