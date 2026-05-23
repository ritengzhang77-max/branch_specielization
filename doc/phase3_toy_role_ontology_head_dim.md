# Phase 3 Toy Role Ontology: Ordinary Attention Heads

Date: 2026-05-23

## Scope

Unit of analysis:

```text
ordinary attention head
```

This is not a SwitchHead, MoE, or branch-tower experiment.

## Why This Experiment Exists

The previous local-vs-induction experiment only measured pairwise separability.
That is a valid diagnostic, but it is not full functional modularity. This
experiment adds a small role ontology:

```text
local_copy:
  local_a
  local_b

kv_lookup:
  kv_a
  kv_b

induction:
  induction_short
  induction_long
```

The goal is to test three separate quantities:

1. Structural role affinity: do roles reliably prefer a structural head type?
2. Functional specialization: do roles concentrate into fewer heads?
3. Ontology-level modularity: do related subroles have similar head
   distributions and unrelated families separate?

## Experiment

Each model is a tiny decoder-only transformer with ordinary attention heads.
Each sequence contains all six supervised roles. For every role `r` and head
slot `h`, the script computes:

```text
S_r(h) = max(loss_r(model with h ablated) - loss_r(base model), 0)
p_r(h) = S_r(h) / sum_h S_r(h)
```

The role distribution `p_r` is over flattened `(layer, head)` slots.

Command:

```bash
python3 -u scripts/toy_role_ontology_head_dim_intervention.py \
  --configs uniform4 hetero4 hetero4_64first hetero4_64second \
            hetero4_64third uniform2 \
  --seeds 1 2 3 4 5 \
  --steps 1200 \
  --batch-size 128 \
  --eval-examples 512 \
  --device cuda \
  --output-dir results/phase3_toy_role_ontology_head_dim_20260523
```

Artifacts:

```text
results/phase3_toy_role_ontology_head_dim_20260523/config_summary.csv
results/phase3_toy_role_ontology_head_dim_20260523/model_summary.csv
results/phase3_toy_role_ontology_head_dim_20260523/role_summary.csv
results/phase3_toy_role_ontology_head_dim_20260523/head_role_scores.csv
results/phase3_toy_role_ontology_head_dim_20260523/role_pair_summary.csv
results/phase3_toy_role_ontology_head_dim_20260523/summary.json
```

## Metrics

Structural role affinity:

```text
top_dim count per role across seeds and layout permutations
```

Functional specialization:

```text
global_top_specialization = max_h p_r(h)
effective_heads = exp(entropy(p_r))
```

Ontology-level modularity:

```text
within_family_similarity = mean similarity(p_r, p_s) for same-family role pairs
between_family_similarity = mean similarity(p_r, p_s) for different-family pairs
family_gap = within_family_similarity - between_family_similarity
family_cluster_ari = clustering ARI against role-family labels
```

Here, `similarity = 1 - TV_distance`.

## Aggregate Results

All configs solved the task well enough to interpret head-role organization.

| Config | Acc. mean | Acc. min | Spec. | Eff. heads | Within | Between | Family gap | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform4` | 0.9994 | 0.9961 | 0.449 | 4.15 | 0.821 | 0.310 | 0.511 | 0.586 |
| `hetero4` | 0.9997 | 0.9961 | 0.663 | 2.30 | 0.856 | 0.351 | 0.505 | 0.889 |
| `hetero4_64first` | 0.9992 | 0.9961 | 0.695 | 2.28 | 0.904 | 0.347 | 0.557 | 1.000 |
| `hetero4_64second` | 0.9997 | 0.9980 | 0.723 | 2.08 | 0.918 | 0.310 | 0.607 | 0.889 |
| `hetero4_64third` | 0.9992 | 0.9961 | 0.723 | 2.09 | 0.895 | 0.307 | 0.587 | 0.889 |
| `uniform2` | 0.9993 | 0.9934 | 0.636 | 2.25 | 0.923 | 0.271 | 0.653 | 1.000 |

## Structural Role Affinity Result

Across the four one-64 heterogeneous layouts:

```text
local_a -> 64-dim top structural type in 20/20
local_b -> 64-dim top structural type in 20/20
kv_a    -> 64-dim top structural type in 20/20
kv_b    -> 64-dim top structural type in 20/20
```

Combined:

```text
local-copy and KV-lookup subroles choose the 64-dim structural type in 80/80
cases.
```

The induction family is different:

```text
induction_short -> 64 in 9/20, 32 in 8/20, 16 in 3/20
induction_long  -> 64 in 2/20, 32 in 12/20, 16 in 6/20
```

Interpretation:

```text
heterogeneous head dimensions create strong structural role affinity, but the
preferred head type is role-family dependent.
```

The strongest pattern is:

```text
local-copy and KV lookup prefer the large 64-dim head;
induction roles often prefer smaller or intermediate heads.
```

This is closer to the user's intended claim than the word "modularity" alone.

## Functional Specialization Result

Heterogeneous one-64 layouts increase role specialization relative to `uniform4`:

```text
uniform4 specialization:       0.449
hetero4 specialization:        0.663
hetero4_64first specialization: 0.695
hetero4_64second specialization: 0.723
hetero4_64third specialization: 0.723
```

The effective number of heads also drops:

```text
uniform4 effective heads:       4.15
one-64 hetero effective heads:  2.08 to 2.30
```

Interpretation:

```text
heterogeneous head dimensions make roles more causally concentrated.
```

The `uniform2` control is important:

```text
uniform2 specialization: 0.636
uniform2 effective heads: 2.25
```

So some concentration comes from fewer/wider heads, not only heterogeneity.
However, the one-64 heterogeneous layouts generally match or exceed `uniform2`
on specialization while also revealing role-dependent structural affinity.

## Ontology-Level Modularity Result

The small role ontology produces family clustering in every condition, but
heterogeneity is not the only way to get it.

Compared with `uniform4`, several one-64 heterogeneous layouts improve the
family gap and ARI:

```text
uniform4 family gap:        0.511, ARI 0.586
hetero4_64first family gap: 0.557, ARI 1.000
hetero4_64second family gap: 0.607, ARI 0.889
hetero4_64third family gap: 0.587, ARI 0.889
```

But `uniform2` is the strongest ontology-modularity control in this sweep:

```text
uniform2 family gap: 0.653, ARI 1.000
```

Interpretation:

```text
heterogeneous head dimensions can improve ontology-level role-family clustering
over uniform4, but they do not uniquely explain clustering. A fewer/wider
uniform-head baseline can also produce strong family modularity.
```

## Current Conclusion

This experiment supports two claims strongly:

```text
Claim A: Heterogeneous head dimensions create structural role affinity.
Claim B: Heterogeneous head dimensions increase functional specialization.
```

This experiment supports a weaker, controlled claim about modularity:

```text
Claim C: Heterogeneous head dimensions can improve ontology-level family
clustering over uniform4 in some layouts, but uniform2 is a strong competing
baseline.
```

The cleanest current framing is:

```text
Different-shaped attention heads bias which roles attach to which structural
head types. This reliably increases role concentration. Whether it improves
full functional modularity depends on the baseline and may be partly explained
by head capacity rather than heterogeneity alone.
```

## Next Step

The next decisive test should separate heterogeneity from head capacity:

1. Add matched controls with the same number of heads but less extreme
   heterogeneity, e.g. `[24, 24, 40, 40]`, `[16, 48, 16, 48]`, and
   `[8, 40, 40, 40]`.
2. Add a same-head-count wide-control if possible by increasing total attention
   dimension for a separate non-matched ablation, clearly labeled as
   non-matched.
3. Expand the ontology with more subroles per family, especially induction
   variants, to test whether the induction split is robust.
4. Validate structural role affinity on real pretrained ordinary heads using
   Pythia/MultiBERT local-copy, KV-like, and induction probes.

Do not change the unit of analysis without explicit approval.
