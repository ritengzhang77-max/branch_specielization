# Metric Literature Review and Provenance

This note lists the metrics proposed for the branch-specialization-in-attention project, separates which ones are project-specific designs from which ones come directly from prior work, and gives a trustworthiness assessment for each metric.

Citation counts are OpenAlex `cited_by_count` snapshots retrieved on 2026-05-21. They are not stable facts and will drift over time. Google Scholar counts are usually higher.

## Summary Table

| Metric | Designed here? | Main source(s) | Venue / year | OpenAlex citations | What it measures | Trustworthiness for this project |
|---|---:|---|---|---:|---|---|
| Attention-score / attention-pattern similarity | No, adopted | Bali et al., "Quantifying LLM Attention-Head Stability"; Kobayashi-style attention-matrix comparison as cited in plan | arXiv 2026 | 0 | Whether two heads attend to the same token-token relationships on the same inputs | High for cross-seed head behavior. It compares heads in a shared token-token basis and avoids many weight/activation alignment problems. Main weakness: Bali et al. is very recent and not yet peer-reviewed. |
| Specialization score `S(h, t)` | Yes, project-specific adaptation | Voss et al. branch specialization; Voita et al. head importance/pruning; causal ablation literature | Distill 2021; ACL 2019 | 12; 1050 | How much head/branch `h` contributes to function/task `t` relative to other heads | High if the task taxonomy is clean and causal effect is measured by ablation or patching. Weak if tasks are vague or measured only with probes. |
| Cross-seed consistency `C(h, t)` | Yes, project-specific adaptation | Bali et al. cross-refit stability; Chughtai et al. universality framing | arXiv 2026; ICML 2023 | 0; 11 | Whether the same function lands in the same or corresponding heads across random seeds | High. This is one of the central target variables. Needs both raw head-index and matched versions. |
| Hungarian-matched head similarity gap | Yes, project-specific adaptation | Permutation/alignment idea related to Git Re-Basin; also standard assignment matching | ICLR 2023 for Git Re-Basin | 32 | Whether cross-seed similarity remains after optimal head relabeling, compared to random matching | High as a permutation-symmetry control. Risk: optimal matching can overstate structure, so always compare against random or null matching. |
| Weight-rebasined stability | Added from user suggestion; adopted from model merging | Ainsworth et al., "Git Re-Basin: Merging Models modulo Permutation Symmetries" | ICLR 2023, notable top 5 percent | 32 | Whether model weights can be permuted/aligned before comparing head roles or specialization scores | Medium. Useful robustness check, but not the primary stability metric because the scientific claim is functional stability, not just weight-space alignability. |
| CKA / Procrustes representation similarity | No, adopted as secondary metric | Kornblith et al.; Davari et al. reliability critique | ICML 2019; arXiv 2022 | 431; 3 | Whether activation representations are geometrically similar across layers/models/seeds | Low-medium. Useful sanity check, but can be misleading for head function, permutation symmetry, and high-dimensional representation comparisons. |
| Functional head probes | No, adopted/adapted | Voita et al.; Clark et al.; Pande et al. | ACL 2019; BlackboxNLP 2019; AAAI 2021 | 1050; 79; 1 | Whether head activations predict head-role labels such as syntactic, positional, BOS, rare-word, induction, etc. | Medium. Good for naming roles and building a taxonomy. Weakness: probes are correlational and can extract information not causally used by the model. |
| Causal attribution / path patching | No, adopted | Wang et al., "Interpretability in the Wild"; broader activation/path patching literature | ICLR 2023 | 50 | Whether a head causally supports a task by patching or ablating its activation and measuring task recovery/loss | High for specific tasks. It is the strongest evidence for functional role, but expensive and task-specific. |
| Lesion / pruning recovery curves | No, adopted | Michel et al.; Voita et al. | NeurIPS 2019; ACL 2019 | 45; 1050 | How performance changes as heads are removed, especially top-k important heads for a task | Medium-high. Strong for importance and redundancy. Less precise for exact function identity because heads interact and compensatory effects can occur. |
| Importance entropy | Yes, project-specific summary statistic | Built from normalized ablation/pruning importance scores; inspired by Filan-style importance framing | Project metric | n/a | Whether task importance is concentrated in one/few heads or spread across many heads | Medium. Simple and interpretable, but only as trustworthy as the underlying importance estimates. |
| Graph modularity / clusterability | No, adopted/adapted | Filan et al.; Hod et al. | arXiv 2021; arXiv 2021/2022 | 5; 1 | Whether heads/neurons form strongly connected within-cluster and weakly connected cross-cluster groups | Medium. Good structural modularity signal, but graph clusters are not automatically functional modules. Needs causal/task validation. |
| Conditional mutual information between heads | Yes, project-specific formalization | Inspired by information-theoretic modularity ideas; not a standard transformer-head metric here | Project metric | n/a | Whether head outputs carry independent information after conditioning on the residual stream input | Low-medium. Conceptually clean but hard to estimate reliably in high-dimensional transformer states. Better as an exploratory metric. |
| Csordas-style mask IoU / IoMin | No, adopted/adapted | Csordas et al., "Are Neural Nets Modular?" | ICLR 2021 | 13 | Whether learned functional weight masks for tasks overlap across tasks/seeds | Medium-high for modularity. It directly links weights/subnets to functions. Weakness: mask optimization can be unstable, expensive, and sensitive to regularization. |
| MoE router / expert specialization metrics | No, adopted/adapted | OLMoE; "The Expert Strikes Back"; SwitchHead | arXiv 2024; arXiv 2026; NeurIPS 2024 | 7; 0; 6 | Routing entropy, expert saturation, co-activation, domain specialization, vocabulary specialization, expert-level functional importance | High for MoE/expert architectures. Less relevant to vanilla MHA unless routed heads or SwitchHead-style attention experts are used. |

## Metric Details

### 1. Attention-Score / Attention-Pattern Similarity

**Status:** Adopted from prior work.

**Goal:** Compare heads across seeds by asking whether they attend to the same token-token relationships.

**How to compute:**

1. Run a fixed probe dataset through each model seed.
2. For each layer/head, save attention score or attention probability matrices `A^(l,h)(x)` of shape `T x T`.
3. Compare two heads by Frobenius distance, cosine similarity, or correlation over matrices and examples.
4. Average over examples.

**Why it is useful:** Attention matrices live in a shared token-token coordinate system. That makes them easier to compare across seeds than raw activations or weights.

**Trustworthiness:** High for behavioral head similarity. It does not prove causal importance by itself, so pair it with ablation or patching.

### 2. Specialization Score `S(h, t)`

**Status:** Designed here as a project-specific adaptation.

**Goal:** Measure whether a head/branch is disproportionately responsible for a function/task.

**Definition:**

```text
S(h, t) = |effect of ablating head h on task t| / sum_h' |effect of ablating head h' on task t|
```

**How to compute:**

1. Pick a task/function `t`: induction copying, previous-token attention, IOI name mover, BOS sink, syntactic relation, rare-word behavior, etc.
2. Measure baseline performance or logit difference.
3. Ablate or patch each head one at a time.
4. Normalize the absolute effect over heads.

**Interpretation:**

- High `S(h, t)`: function `t` is concentrated in head `h`.
- Low, diffuse `S`: function is distributed or redundant.

**Trustworthiness:** High if the effect is causal. Lower if `S` is estimated from probes alone.

### 3. Cross-Seed Consistency `C(h, t)`

**Status:** Designed here as a project-specific adaptation.

**Goal:** Measure whether specialization is stable across seeds.

**How to compute:**

1. Compute `S(h, t)` for each seed.
2. Compare the `S` vectors across seed pairs.
3. Report both raw and matched versions:
   - raw: compare the same `(layer, head)` index across seeds;
   - matched: first align heads by Hungarian matching, then compare.

**Possible summaries:**

```text
mean pairwise correlation of S vectors
residual variance after optimal head matching
fraction of tasks assigned to the same head slot
```

**Trustworthiness:** High. This is the direct operationalization of "strong universality" or cross-seed branch-specialization stability.

### 4. Hungarian-Matched Head Similarity Gap

**Status:** Designed here using a standard assignment method and inspired by model-alignment work.

**Goal:** Separate "no consistency" from "same roles, different labels."

**How to compute:**

1. For seed A and seed B, build a head-head similarity matrix.
2. Use Hungarian assignment to find the best one-to-one head matching.
3. Compare the matched score against random permutations.

**Core number:**

```text
matched_similarity - random_permutation_similarity
```

**Trustworthiness:** High as a permutation-symmetry control. The key is to report the null baseline, because optimal matching always finds some match.

### 5. Weight-Rebasined Stability

**Status:** Added from the user's suggestion, adopted from model-merging literature.

**Goal:** Ask whether weights can be aligned first, then specialization compared in the aligned coordinate system.

**How to compute:**

1. Choose a reference seed.
2. Use Git-Re-Basin-style permutation matching to align another seed's hidden units/heads/MLP channels to the reference.
3. After alignment, recompute head-level or branch-level metric agreement.
4. Compare against behavior-only Hungarian matching.

**Why this matters:** It answers a slightly different question from functional stability:

```text
Can models be made weight-space comparable by permutation?
```

rather than:

```text
Did the same function naturally land in the same branch/head?
```

**Trustworthiness:** Medium. It is a strong robustness check, especially if the project wants to talk to the model-merging literature. But it should not replace attention-pattern and causal metrics, because successful weight matching can hide functionally meaningful differences.

### 6. CKA / Procrustes Similarity

**Status:** Adopted as a secondary metric.

**Goal:** Compare representation geometry across heads/layers/seeds.

**How to compute:**

1. Collect activations for the same dataset.
2. Compute CKA, Procrustes distance, CCA, or SVCCA between representations.
3. Compare across seeds and layers.

**Trustworthiness:** Low-medium. CKA is widely used and useful for sanity checks, but not enough for this project because representation similarity does not necessarily imply same circuit role or causal function.

### 7. Functional Head Probes

**Status:** Adopted/adapted from head-analysis literature.

**Goal:** Assign interpretable labels to heads.

**How to compute:**

1. Define labels or targets for candidate functions.
2. Train a linear probe on each head's output activation or attention pattern.
3. Report probe accuracy or selectivity by `(layer, head)`.
4. Compare role labels across seeds.

**Trustworthiness:** Medium. Good for exploratory taxonomy, but probes can detect information that the model does not actually use. Treat as descriptive until validated by ablation/patching.

### 8. Causal Attribution / Path Patching

**Status:** Adopted from mechanistic interpretability.

**Goal:** Test whether a head is causally involved in a task.

**How to compute:**

1. Construct clean and corrupted input pairs.
2. Run activation patching or path patching for each head.
3. Measure recovery in task performance, logit difference, or loss.
4. Compare patching effects across seeds.

**Trustworthiness:** High. This is one of the best metrics for "what the head actually does." Main limitations are compute cost and task specificity.

### 9. Lesion / Pruning Recovery Curves

**Status:** Adopted from pruning and head-importance literature.

**Goal:** Measure importance concentration and redundancy.

**How to compute:**

1. Rank heads by importance for a task.
2. Remove top-k heads or bottom-k heads.
3. Plot task performance as k increases.

**Interpretation:**

- Sharp drop after removing one/few heads: high concentration.
- Slow drop: distributed or redundant behavior.

**Trustworthiness:** Medium-high. Strong for importance and redundancy, weaker for mechanistic identity because ablation can cause distribution shift or compensatory behavior.

### 10. Importance Entropy

**Status:** Designed here as a compact summary metric.

**Goal:** Quantify concentration of a task over heads.

**How to compute:**

1. Compute normalized head importance scores for task `t`.
2. Treat them as a distribution over heads.
3. Compute entropy.

```text
H(t) = - sum_h S(h, t) log S(h, t)
```

**Interpretation:**

- Low entropy: the task is concentrated in a few heads.
- High entropy: the task is spread across many heads.

**Trustworthiness:** Medium. It is clear and useful, but depends entirely on the quality of `S(h,t)`.

### 11. Graph Modularity / Clusterability

**Status:** Adopted/adapted from modularity literature.

**Goal:** Measure whether heads form modular groups.

**How to compute:**

1. Build a graph with heads as nodes.
2. Define edge weights using activation correlation, attention-flow overlap, causal interaction, or weight connectivity.
3. Run spectral clustering, Louvain, or Leiden.
4. Report Newman's `Q`, normalized cut, or cluster stability across seeds.
5. Compare cluster assignments using adjusted Rand index.

**Trustworthiness:** Medium. Good for structural organization. Not sufficient by itself because clusters can be mathematically clean without being functionally meaningful.

### 12. Conditional Mutual Information Between Heads

**Status:** Designed here as a formal modularity target.

**Goal:** Measure whether heads are informationally coupled after conditioning on shared input.

**Formal target:**

```text
I(f_h ; f_h' | residual stream input)
```

**Interpretation:**

- Lower conditional MI: more separable/modular heads.
- Higher conditional MI: more entangled heads.

**Trustworthiness:** Low-medium. The idea is clean, but estimating conditional mutual information in high-dimensional neural activations is hard. This should be exploratory unless carefully validated.

### 13. Csordas-Style Mask IoU / IoMin

**Status:** Adopted/adapted from Csordas et al.

**Goal:** Identify which weights/subnets support a function and compare them across tasks or seeds.

**How to compute:**

1. Freeze a trained model.
2. Learn differentiable binary masks over weights for task/function `t`.
3. Threshold or otherwise discretize masks.
4. Compare masks using IoU or IoMin.

```text
IoU(mask_a, mask_b) = |mask_a intersect mask_b| / |mask_a union mask_b|
```

**Trustworthiness:** Medium-high for modularity. It directly links function to subnet structure. Weaknesses are mask-training instability, regularization sensitivity, and cost.

### 14. MoE Router / Expert Specialization Metrics

**Status:** Adopted/adapted from MoE literature.

**Goal:** Measure branch specialization in expert/routed architectures.

**Metrics:**

- routing entropy: how concentrated routing is;
- router saturation: whether a few experts dominate;
- expert co-activation: which experts fire together;
- domain specialization: whether experts specialize by data domain;
- vocabulary specialization: whether experts specialize by token classes;
- functional specialization: whether expert ablation affects specific tasks.

**Trustworthiness:** High for MoE and SwitchHead-style architectures. Less useful for vanilla MHA unless attention heads are routed or expert-like.

## Recommended Measurement Stack

For the main paper-level claim, use the following stack:

1. **Attention-score similarity** as the primary cross-seed behavioral similarity metric.
2. **Causal specialization score `S(h,t)`** for task-specific branch/head roles.
3. **Raw and Hungarian-matched cross-seed consistency `C(h,t)`** to separate head-index stability from relabeled stability.
4. **Weight-rebasined stability** as a robustness check and bridge to model-merging literature.
5. **Mask IoU and graph modularity** to test whether specialization is accompanied by real modularity.

Avoid relying on any single metric. The central risk is confusing specialization with modularity:

```text
specialization = which component does which function
modularity = how separable the components are from each other
```

A head can be specialized but still highly entangled through the residual stream. Conversely, a model can have clean graph clusters that do not map to meaningful functions.

## Source Index

- Voss et al., "Branch Specialization", Distill 2021, DOI `10.23915/distill.00024.008`.
- Bali et al., "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality", arXiv 2026, `2602.16740`.
- Chughtai, Chan, Nanda, "A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations", ICML 2023, arXiv `2302.03025`.
- Ainsworth, Hayase, Srinivasa, "Git Re-Basin: Merging Models modulo Permutation Symmetries", ICLR 2023, arXiv `2209.04836`.
- Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019, arXiv `1905.00414`.
- Davari et al., "Reliability of CKA as a Similarity Measure in Deep Learning", arXiv 2022, `2210.16156`.
- Voita et al., "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned", ACL 2019, DOI `10.18653/v1/P19-1580`.
- Clark et al., "What Does BERT Look at? An Analysis of BERT's Attention", BlackboxNLP 2019, DOI `10.18653/v1/W19-4828`.
- Pande et al., "The Heads Hypothesis: A Unifying Statistical Approach Towards Understanding Multi-Headed Attention in BERT", AAAI 2021, DOI `10.1609/aaai.v35i15.17605`.
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small", ICLR 2023, arXiv `2211.00593`.
- Michel, Levy, Neubig, "Are Sixteen Heads Really Better than One?", NeurIPS 2019, arXiv `1905.10650`.
- Filan et al., "Clusterability in Neural Networks", arXiv 2021, `2103.03386`.
- Hod et al., "Quantifying Local Specialization in Deep Neural Networks", arXiv 2021/2022, `2110.08058`.
- Csordas, van Steenkiste, Schmidhuber, "Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks", ICLR 2021, arXiv `2010.02066`.
- Muennighoff et al., "OLMoE: Open Mixture-of-Experts Language Models", arXiv 2024, `2409.02060`.
- "The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level", arXiv 2026, `2604.02178`.
- Csordas et al., "SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention", NeurIPS 2024, arXiv `2312.07987`.
