# Branch Specialization in Attention-Based Architectures: A Research Roadmap

## What This Project Is

This document scopes a research project that extends the **Branch Specialization** framework — originally developed by Olah, Voss, and collaborators on convolutional architectures like Inception (Distill 2020/2021) — to **transformer attention**. The core conceptual move is to treat each attention head (or each MoE expert, or each parallel branch in hybrid architectures like Hymba) as a "branch" in the Inception sense, and to ask the same questions the Branch Specialization literature has asked about CNNs: do branches functionally specialize, is that specialization consistent across random seeds, and how do architectural design choices (for example heterogeneous per-head dimensions, explicit branches, or routing) affect specialization and modularity?

The project is deliberately neutral about whether functions *should* separate
into different branches. The research question is whether they do, when they do,
and under which structural conditions. "Functions separate cleanly" and
"functions cohabit the same branch despite specialization" are both useful
findings.

The project builds directly on prior work by the author on cross-seed branch attribution variability in Inception variants (Zhang & Alvarez, advised by Prof. Sergio Alvarez at Boston College — "Analyzing Variations in Branch Attribution in Non-monolithic Models"). It is organized into two stages:

- **Stage 1 — Post-hoc analysis.** Using publicly available multi-seed checkpoint suites (Pythia, MultiBERTs) and existing branched attention architectures (OLMoE, Hymba, SwitchHead), quantify how consistent attention-head functional specialization is across random initializations. Establish baseline measurements that the Branch Specialization community has produced for CNNs but that essentially do not yet exist for transformers.
- **Stage 2 — Architectural intervention.** Test whether interventions such as *heterogeneous per-head dimensions*, explicit branch structure, or routing change the stability of functional specialization and/or the degree of functional modularity. This stage should measure modularity as an empirical outcome, not assume that separated functions are the desired or expected result.

## What This Document Is

This is a **research roadmap and literature review**, not a paper. Its purpose is to:

1. Map the design space and identify the closest prior work (the single most relevant paper turns out to be Bali et al. 2026, arXiv:2602.16740, which addresses cross-seed head stability but leaves the architectural-intervention question open).
2. Bridge two communities that have largely talked past each other — **Branch Specialization / Modularity** (Olah, Voss, Filan, Csordás, Hod, Lange) and **Mechanistic Interpretability** (Anthropic Transformer Circuits, Wang/Nanda IOI, Olsson induction heads) — by providing a translation table and a formal distinction between *specialization* and *modularity*.
3. Rank existing branched attention architectures by practicality as Stage-1 analysis targets, with concrete HuggingFace/GitHub paths.
4. Specify methodology for cross-seed specialization measurement that avoids known pitfalls (permutation symmetry, CKA misleading results) by using attention-score-matrix comparison plus Hungarian-matched alignment.
5. Lay out a 20-week staged plan with explicit milestones, pre-registered predictions, and decision thresholds.
6. List caveats — including a Differential Transformer follow-up result (arXiv:2505.16333) that constitutes a serious null prior for the architectural intervention and must be engaged with.

## How To Read This Document

Read the **TL;DR** for the three highest-level takeaways (the project is novel and well-positioned; Stage 1 is immediately tractable; Stage 2 appears genuinely new). Read **Key Findings §1–§5** for the conceptual framing (two-community bridge, specialization-vs-modularity distinction). Read **Key Findings §6** if you want the ranked table of architectures to analyze first. Read **Details §B** for the week-by-week plan. Read **Caveats** before committing to any specific experiment.

---

## TL;DR
- **The core idea is novel and well-positioned**: extending Olah/Voss et al.'s Branch Specialization framework from Inception branches to transformer attention heads is timely — only one paper to date (Bali, Stanley, Suresh, Bzdok, "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality," arXiv:2602.16740, Feb 2026) directly addresses cross-seed consistency of attention-head function, and it leaves the *architectural-intervention* question (does heterogeneity improve consistency?) entirely open.
- **Stage 1 (post-hoc) is immediately tractable**: Pythia (9 official seeds at 14M–410M, plus decoupled `weight-seed{1-3}` / `data-seed{1-3}` variants for 160M) + MultiBERTs (25 BERT-Base Uncased seeds with 28 intermediate checkpoints for the first 5 models, 140 total) give you a cheap, reproducible substrate; Hymba-1.5B-Base (parallel attention+SSM heads) and OLMoE-1B-7B (16 decoder layers × 64 routed experts, top-k=8 per token, ~1.3B active of 7B total, pretrained on 5T tokens) are the two most natural "branched" architectures to layer on top.
- **Stage 2 (architectural intervention) appears genuinely new**: no paper found trains transformers with mixed per-head dimensions ({32,64,128,256}) within a single attention layer and tests whether this changes cross-seed specialization or modularity. DeepSeek MLA, Differential Transformer, Bhojanapalli's low-rank-bottleneck fix, and "Allocation of Parameters in Transformers" (arXiv:2510.03784) all motivate the direction but stop short of explicit intra-layer head-dim heterogeneity, and none treats cross-seed function-to-branch mapping as the outcome variable.

## Key Findings

### 1. Two communities, mostly disjoint, that this project would bridge
- **Branch Specialization / Modularity community** (Olah et al. Distill 2020; Voss et al. 2021; Filan/Casper/Hod 2021–2022; Csordás/van Steenkiste/Schmidhuber ICLR 2021; Lange et al. 2022; Hamblin & Alvarez 2021; Dobs et al. 2021–2022; Golechha & Dao 2024). Canonical venues: Distill, ICLR, NeurIPS workshops on interpretability. Key metrics: graph clusterability (Newman's Q, n-cuts), differentiable weight masks (IoU/IoMin), lesioning effects, branch-conditional MSE. The community emphasizes *structural* modularity (weight/connectivity-based) and treats specialization as *which branch does what*.
- **Mechanistic Interpretability community** (Anthropic Transformer Circuits Thread; Wang et al. 2022 IOI; Conmy et al. 2023 ACDC; Chughtai/Chan/Nanda ICML 2023 on universality; Olsson et al. 2022 induction heads; Voita et al. 2019; Michel et al. 2019; Clark et al. 2019; Heimersheim 2024; Nanda attribution patching). Canonical venues: arXiv, transformer-circuits.pub, ICLR/NeurIPS. Key metrics: logit difference under activation/path patching, faithfulness/completeness/minimality of circuits, attention-pattern statistics, SAE features. The community emphasizes *causal* attribution and *circuits* spanning multiple components.
- **The gap**: Branch Specialization measures variance-across-seeds of *which branch holds which feature class*; Mech-Interp measures *what computation a specific head does in a specific model*. Almost no paper formally links them. The Bali et al. 2026 paper is the first systematic attempt.

### 2. The Olah/Voss "Branch Specialization" paper — what it claims and what's brittle
- **Site**: distill.pub/2020/circuits/branch-specialization (Voss, Goh, Cammarata, Petrov, Schubert, Olah; published April 5, 2021; doi 10.23915/distill.00024.008).
- **Core claim**: In branched CNNs (AlexNet, InceptionV1) and even implicitly in non-branched CNNs (via weight PCA), neurons with similar circuit-level function (curve detection, low/high-frequency detection, color contrast) cluster into the same branch. Canonical datapoint: *"all 30 of the curve-related features in mixed3b are in mixed3b_5x5, despite it being only 96 out of the 480 neurons in the layer. The probability of that happening by chance is less than 1/10²⁰."*
- **Mechanism conjectured**: a *positive feedback loop* during training — early layers in a branch are incentivized to form low-level features that later layers in the same branch use as primitives.
- **Limitations**: (a) qualitative, no formal modularity metric; (b) AlexNet-style 2-GPU branching is a historical artifact, not a designed inductive bias; (c) no cross-seed reproducibility study in the original paper; (d) no transformer / attention generalization.
- **Direct follow-ups**: Hamblin & Alvarez 2021 ("What Matters in Branch Specialization?"), which uses a Gabor toy task and confirms branches specialize early in training and don't change afterward, and that specialization is sensitive to task alternation frequency.

### 3. Modularity & clusterability literature (the Filan/Casper/Hod line)
- **Filan, Casper, Hod, Wild, Critch, Russell 2021, arXiv:2103.03386** — defines clusterability as "how well a network can be divided into groups of neurons with strong internal connectivity but weak external connectivity"; uses spectral graph clustering on weight matrices. Finds trained networks are more clusterable than random networks with the same weight distribution.
- **Hod, Casper, Filan, Wild, Critch, Russell 2022** — "Detecting Modularity in Deep Neural Networks" — extends to *functional* modularity via activation correlations.
- **Csordás, van Steenkiste, Schmidhuber, ICLR 2021** (arXiv:2010.02066) — **the key paper for distinguishing specialization from modularity**. Uses differentiable binary weight masks to identify subnets responsible for specific functions and defines two distinct properties: **P_specialize** (use different modules for separate functions) and **P_reuse** (use the same module for identical functions). Empirical finding: standard NNs partially achieve P_specialize but largely fail P_reuse — i.e., they are weakly modular in a way that does *not* support compositional generalization.
- **Lange et al. 2022** — finds clusters from upstream (input) information differ from downstream (output) information; modularity is multi-faceted.
- **Patil et al. 2023; Golechha & Dao 2024 (ICML 2024 Workshop on Mech Interp); arXiv:2409.15747 "Training Neural Networks for Modularity Aids Interpretability"; arXiv:2502.02470 "Modular Training of Neural Networks aids Interpretability"** — newer papers that train models to be more modular and assess whether this aids mech-interp goals. These bridge the two communities.

### 4. Cross-seed consistency of attention-head function — the SOTA gap
- **Bali, Stanley, Suresh, Bzdok 2026, arXiv:2602.16740**, "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality." Findings the user must build on:
  1. "Middle-layer heads are the least stable yet the most representationally distinct"
  2. "Deeper models exhibit stronger mid-depth divergence"
  3. "Unstable heads in deeper layers become more functionally important than their peers from the same layer"
  4. "Applying weight decay optimization substantially improves attention-head stability across random model initializations"
  5. "The residual stream is comparatively stable"
  
  Methodology: compare attention *score matrices* across refits (avoids permutation-symmetry / activation-space issues that hurt CKA/CCA/SVCCA). This is the most important methodological prior for Stage 1.
- **Chughtai, Chan, Nanda ICML 2023 (arXiv:2302.03025)** "A Toy Model of Universality" — finds *mixed* evidence: "using our algorithm, we can completely characterize the family of circuits and features that networks learn on this task, but for a given network the precise circuits learned — as well as the order they develop — are arbitrary." This is direct evidence that *weak* universality holds (same family of solutions) but *strong* universality (same head-index does the same thing) does not.
- **Wang/Ge/Shu et al. 2024 (arXiv:2410.06672)** "Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures" — compares Transformer vs Mamba via SAEs.
- **MultiBERTs (Sellam et al. ICLR 2022, arXiv:2106.16163)** — 25 BERT-Base Uncased checkpoints with different seeds (init + data order) plus 28 intermediate checkpoints for the first 5 models (140 total). Crucially: **the MultiBERTs paper itself does NOT contain a head-index functional-role consistency analysis** — this is a gap the user can directly fill.
- **Voita, Talbot, Moiseev, Sennrich, Titov 2019 (ACL)**, "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" (Proceedings of the 57th ACL, Florence, July 2019, pp. 5797–5808, doi:10.18653/v1/P19-1580, arXiv:1905.09418) — "the most important and confident heads play consistent and often linguistically-interpretable roles ... specialized heads are last to be pruned"; pruning 38 of 48 encoder heads drops only 0.15 BLEU on En-Ru WMT. Crucially analyzed *one* model — no cross-seed analysis.
- **Pande et al. 2021 (arXiv:2101.09115)** "The Heads Hypothesis" — formalizes head-role classification via hypothesis testing on a "sieve bias score" but analyzes a single BERT — not cross-seed.

### 5. Specialization vs Modularity — formal distinction
After the literature review the cleanest formal framing (the user's own synthesis is needed here — no single paper provides this) is:
- **Specialization**: a property of the function-to-component mapping. Component *c* is specialized for task/feature *f* iff *f* is disproportionately encoded by *c* (high mutual information, high attribution, or high lesion effect on *f* relative to other components).
- **Modularity**: a property of the *interface* between components. Components *c₁* and *c₂* are modular iff their internal computations are weakly coupled — formally, low conditional mutual information of c₁'s internal state given the rest of the network's state conditional on its inputs, or equivalently, low cross-cluster edge weight in a weighted-graph clusterability sense.
- **They can dissociate**: (a) specialization *without* modularity = polysemantic heads that each encode a unique feature mix but share weights/information heavily (superposition territory — Elhage et al. 2022; Henighan et al. 2023). (b) modularity *without* specialization = redundant branches doing the same computation (the "duplicate head" phenomenon — Michel et al. 2019). The project should measure these possibilities rather than treating any one of them as the intended outcome.
- **Mapping to mech-interp**: monosemanticity ≈ specialization at the neuron level + the absence of superposition; the mech-interp community's "universality" hypothesis ≈ cross-seed reproducibility of specialization. The Branch Specialization community's *modularity* concept = the mech-interp community's *circuit separability*.
- **Recommended formal metrics**:
  - For specialization: Filan-style *importance entropy* (how concentrated is feature *f* in a single head), Csordás-style weight-mask IoU between functional masks, lesion-recovery curves.
  - For modularity: cross-cluster edge weight in attention-graph spectral clustering, conditional mutual information between head outputs given residual stream, path-patching-derived effective separability.

### 6. Existing branched attention architectures, ranked by practicality for Stage 1
Practicality = (multi-seed checkpoints available) × (analogy to Inception branch is clean) × (compute cost).

| Architecture | Multi-seed availability | Branch analogy clarity | Stage-1 practicality |
|---|---|---|---|
| **Pythia (standard MHA, parallel attn+FFN)** | **9 seeds** at 14M/70M/160M/410M; plus `pythia-160m-weight-seed{1-3}` and `pythia-160m-data-seed{1-3}` decoupling init vs data ordering; all on HuggingFace `EleutherAI/pythia-*-seed*` | Each head = a branch; parallel attn+FFN gives a second "two-branch" partition | **Highest — start here** |
| **MultiBERTs** | 25 BERT-Base Uncased seeds + 28 intermediate checkpoints for first 5 (140 total); Google Cloud + HuggingFace | Each head = a branch; encoder-only is cleaner | **Highest — start here in parallel** |
| **OLMoE-1B-7B** | Single seed (one official checkpoint), but all training artifacts open (Muennighoff et al., arXiv:2409.02060: 16 decoder layers, 64 routed experts per layer, top-k=8 active per token, ~1.3B active of 7B total, 5T tokens) | Each expert = a branch; shared-vs-routed distinction is exactly the Inception-multi-branch analog | High for within-model analysis; multi-seed would require retraining |
| **Hymba-1.5B-Base** | Single seed; barebones repo on GitHub; the attention-vs-SSM head distinction is the clearest "heterogeneous branches" example in any production model | Each layer has parallel attention heads + Mamba2 heads; this IS the Inception analog | High for functional analysis; would need to re-pretrain for multi-seed; ICLR 2025 paper does NOT report multi-seed |
| **Mixtral / DeepSeek-MoE / Qwen-MoE** | Single seed each | MoE FFN experts as branches | Medium — analyze within-model expert specialization; existing work (Muennighoff et al. OLMoE; arXiv:2604.02178; arXiv:2505.24593) gives baseline |
| **SwitchHead (Csordás, Piękos, Irie, Schmidhuber, NeurIPS 2024, arXiv:2312.07987)** | Training code public; "For our 262M parameter model trained on C4, SwitchHead matches the perplexity of standard models with only 44% compute and 27% memory usage" — small enough to re-train multi-seed cheaply | Mixture of attention experts | Medium — best for explicit "MoE attention" branch analog; cheap |
| **MoH / MoA** | Code on GitHub | Each routed head = a branch | Medium |
| **Mixture of Depths (Raposo et al. 2024)** | No official checkpoints; community implementations | Each layer's "process vs skip" router is a 2-branch decision | Low for cross-seed; requires from-scratch |
| **Differential Transformer (Ye et al., arXiv:2410.05258, ICLR 2025)** | No multi-seed | Pairs of heads (subtraction) — heads are structurally matched, not heterogeneous | Low (one seed only) |
| **DeepSeek-V2/V3 MLA** | Single production checkpoint | Heterogeneity is intra-head (NoPE+RoPE 128+64=192; v_head_dim=128) NOT inter-head — heads are still uniform across the layer | Low for cross-seed |
| **ViT branched variants (Talking-Heads, etc.)** | Mixed | Each head = a branch | Medium |

### 7. The architectural intervention (Stage 2) — what's known and what isn't
The user's Stage 2 hypothesis (heterogeneous per-head head_dim like {32, 64, 128, 256} as an inductive bias for consistency) is genuinely under-explored:
- **"Low-Rank Bottleneck in Multi-head Attention Models" (Bhojanapalli et al., arXiv:2002.07028, Google Research)** decouples head_size from d_model/n_heads ("This fixed head size is also independent of both the number of heads and the embedding size... unlike the standard setting which requires the number of heads to be a factor of the embedding size, we are free to set an arbitrary number of heads as required for the task.") but uses *uniform* head_size.
- **"Allocation of Parameters in Transformers" (arXiv:2510.03784, Oct 2025)** gives a theoretical bound: "one component of the error decreases with larger head dimensions ('dim'), while another decreases with more heads ('head'). Given a fixed budget of total parameters (say, dim × head fixed), this naturally implies a trade-off between the number of heads and head dimensions... this saturation pattern suggests that one can operate more efficiently with reduced parameters (head dimensions) without significantly degrading performance, particularly for later layers." This argues for *layer-wise* heterogeneity but not within-layer.
- **Adaptive Attention Span (Sukhbaatar et al. ACL 2019, arXiv:1905.07799)**: per-head span z^(i) with L1 penalty; "lower layers do not require very long attention spans, while a few attention heads in higher layers may use exceptionally long spans." Closest analog to a learned per-head capacity inductive bias, but for *receptive field*, not *head_dim*, and no cross-seed analysis.
- **Differential Transformer (Ye et al. ICLR 2025, arXiv:2410.05258)** and its follow-up **arXiv:2505.16333 "Understanding Differential Transformer Unchains Pretrained Self-Attentions"** — the follow-up explicitly tests a "LLAMA-half" baseline with halved head count and doubled head dim and shows that **wider heads alone do NOT replicate Diff Transformer's specialization improvement**: "merely using fewer, wider heads does not replicate this effect, as demonstrated by our LLAMA-half baseline (green), configured with halved head count and doubled head dimension." This is highly relevant counter-evidence the user must engage with — it suggests that uniform width increases do not on their own produce specialization gains; the structural pairing (subtraction) matters.
- **DeepSeek MLA (V3 config: qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, 128 heads, kv_lora_rank=512, q_lora_rank=1536)** — heterogeneity is *intra*-head (RoPE vs no-RoPE sub-vectors and Q/K=192 ≠ V=128) but all heads share the same composite dim. Not the per-head heterogeneity the user proposes.

**Bottom line for Stage 2**: no paper found trains with explicit intra-layer per-head dim mixtures, and no paper found tests cross-seed consistency as the outcome of any head-dim manipulation. The intervention is novel; the LLaMA-half result is a *warning* (uniform width changes don't help) and motivates exactly the *heterogeneous* design.

### 8. Methodology for cross-seed specialization measurement in attention
Synthesizing Bali et al. 2026, Csordás et al. 2021, Ainsworth et al. 2023 (Git Re-Basin), Filan et al. 2021, Kobayashi et al. 2023:

- **Direct comparison without permutation alignment** — Use **attention score matrices** A^(l,h) ∈ ℝ^(T×T) as the comparison object across seeds (Kobayashi 2023; Bali 2026). Attention scores are token-token relationships on a common basis, so head-to-head similarity is directly meaningful. Concrete metric: Frobenius distance between attention matrices on a held-out probe set, averaged over many inputs.
- **Permutation-invariant head matching** (Hungarian assignment on similarity matrix) — when comparing seed A's heads to seed B's heads, solve an assignment problem to find the *best* matching, then report the *gap between matched-similarity and random-permutation-similarity*. This separates "no consistency" from "consistency under relabeling."
- **CKA / Procrustes** — useful but with the caveats Bali et al. flag ("not flawless and can be misleading," citing Kornblith 2019 and Davari 2022). Use as a secondary metric.
- **Functional probing of heads** — train linear probes on attention-output activations for a fixed taxonomy of head functions (induction-head copying, positional, syntactic, BOS-attention, rare-word, suppression) and report probe accuracy *by head index* across seeds.
- **Causal attribution** — path patching / activation patching of head outputs against a fixed task (IOI, induction, greater-than). Bali et al.'s metric: layer-wise correlation of patching effects across seeds.
- **Graph-based modularity of the head-head attention graph** — build a graph where node = head and edge weight = activation correlation or attention-flow overlap, then run Louvain/Leiden and compare community assignments across seeds (Newman's Q + adjusted Rand index across seeds).
- **For MoE / routed-head architectures**: Muennighoff et al. (OLMoE, arXiv:2409.02060) provides router-saturation, expert-co-activation, domain-specialization, vocabulary-specialization metrics that are directly importable. arXiv:2604.02178 "The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level" plots Routing Specialization vs Functional Specialization across layers and is a methodological template.

### 9. Cleanly open research questions in (Branch Specialization × attention × cross-seed × heterogeneous architecture)
1. **Do Pythia's 9 seeds at 410M produce the *same* induction-head circuit (Olsson et al. 2022) at the same head indices?** Not in the literature.
2. **In Hymba, across seeds, is the attention-vs-Mamba head assignment for "recall" tasks stable?** NVIDIA's paper analyzes input-adaptive importance but not cross-seed.
3. **Does heterogeneous head_dim {32,64,128,256} change Filan-clusterability of the attention graph relative to uniform head_dim?**
4. **Does heterogeneous head_dim *reduce* mid-layer instability** (Bali et al.'s "stability dip")?
5. **Is the consistency-improvement (if any) driven by the *small* heads or the *large* heads?** I.e., does forcing a 32-dim head create an information bottleneck that channels a specific feature class into it?
6. **For MultiBERTs, do the 25 seeds' BERT-base heads partition into the same clusters under spectral clustering?** Direct test of cross-seed modularity (not yet done in literature despite the data being public since 2021).
7. **Does Csordás-style weight-mask IoU between paired tasks vary across seeds with vs without architectural heterogeneity or routing?** This is a clean formal test of whether modularity changes, not a claim that modularity must increase.

## Details

### A. The "Branch Specialization" framework — precise formalization in the new setting
For an attention layer with H heads, let f_h(x) be the output contribution of head h to the residual stream, and let T = {t₁, …, t_K} be a taxonomy of *candidate functions* (induction copying, previous-token, positional, rare-word, syntactic, BOS-sink, name-mover, etc.). Define:

- **Specialization score** for head h on task t: S(h, t) = |effect of ablating h on t| / Σ_{h'} |effect of ablating h' on t|. High S means concentration.
- **Cross-seed consistency** for the (h, t) pair across seeds {1, …, S}: C(h, t) = mean correlation of S(h, t) across seed pairs, or equivalently — after Hungarian alignment of head indices — the residual variance in S after optimal permutation matching.
- **Branch-level modularity**: cross-head conditional mutual information I(f_h; f_h' | residual stream input) — if heads are modular, this should be small for h ≠ h'.

These three numbers (S, C, M) are what the project should report for each architecture × seed-set pair in Stage 1, and as a function of head_dim heterogeneity, explicit branch structure, or routing in Stage 2. The goal is to learn whether S, C, and M move together or dissociate.

### B. A staged roadmap with milestones

**Stage 0 (week 0–2): Infrastructure & calibration**
- Pull Pythia-160m-seed{1..9} and Pythia-160m-weight-seed{1..3}/data-seed{1..3} via `from_pretrained("EleutherAI/pythia-160m-seed3")` at `revision="step143000"`.
- Pull MultiBERTs seeds 0..4 with all 28 checkpoints from `google/multiberts-seed_X-step_Yk`.
- Re-implement Bali et al. 2026 (arXiv:2602.16740) attention-score-matrix similarity baseline as a sanity calibration; reproduce their "mid-layer stability dip" on Pythia-160m. Milestone: published-figure reproduction.

**Stage 1a (week 2–6): Post-hoc on standard MHA**
- Compute, for Pythia and MultiBERTs across all seeds: (i) attention-pattern similarity by (layer, head) index, (ii) Hungarian-matched cross-seed similarity, (iii) head-level circuit probes (induction-head, previous-token, positional). Use Voita 2019 head-role taxonomy and Olsson 2022 induction-head test.
- Compute Filan-style spectral clusterability of the head-graph weighted by attention-output correlation; compare clusterability scores across seeds.
- Compute Csordás-style differentiable weight masks for a small set of synthetic probing tasks (induction, copy, sort), then compute mask-IoU across seeds. Milestone: a heatmap showing "stability of head h on task t" across the 9 Pythia seeds.

**Stage 1b (week 6–10): Post-hoc on branched attention architectures**
- OLMoE-1B-7B: replicate router-saturation / expert co-activation / domain-specialization analysis from arXiv:2409.02060 §5; extend with the "expert level" interpretation pipeline from arXiv:2604.02178. Since only one seed exists, treat across-layer or across-checkpoint variation as a within-model proxy.
- Hymba-1.5B-Base: compute attention-vs-SSM head importance per task; identify whether the assignment is task-driven (NVIDIA's claim) or input-driven (also NVIDIA's claim) or stable across re-pretraining of small versions you train yourself (the open question).
- SwitchHead: easiest re-training target (Csordás code public; 262M model trains in 44% compute / 27% memory vs the parameter-matched baseline on C4 per arXiv:2312.07987); re-train ≥5 seeds at this scale. Compute cross-seed routing entropy and head-expert specialization. Milestone: a 4-panel figure (MHA Pythia / MultiBERTs / OLMoE / SwitchHead) of cross-seed specialization variance.

**Stage 2a (week 10–16): Architectural interventions**
- Modify the SwitchHead training code (smallest open codebase with MoE attention) — and a vanilla GPT-NeoX trainer at 70M–160M scale — to support per-head dimensions specified as a config tuple, e.g., `head_dims=[32, 32, 64, 64, 128, 128, 256, 256]` summing to the same total Q/K/V dim as the uniform baseline.
- Train each configuration (uniform vs heterogeneous; and, where tractable, explicit branch/routing variants; ≥5 seeds each) to matched validation loss on a Pile/C4 subset.
- Apply Stage 1a metrics (cross-seed similarity, Hungarian-matched gap, mask-IoU) to both configurations.
- Pre-register competing outcomes rather than a single desired direction:
  - heterogeneity changes specialization stability but not modularity;
  - heterogeneity changes modularity as well as specialization;
  - explicit routing changes modularity while capacity heterogeneity does not;
  - none of the interventions produce a reliable effect at matched loss.

**Stage 2b (week 16–20): Mechanistic interpretation**
- Apply path patching (Wang et al. 2022, arXiv:2211.00593) to identify circuits in each configuration; ask whether the *same* structural slot consistently hosts a given circuit role across seeds. Do not assume ahead of time that small heads should become local/positional heads or that large heads should become global/induction heads.
- Compare with the "LLaMA-half" warning from arXiv:2505.16333: confirm that uniform-width changes do not produce the same effect.

**Current Phase 3 toy-pilot evidence (2026-05-22):**
- Heterogeneous head dimensions can stabilize functional specialization in toy
  key-value and induction-style tasks.
- In local-vs-induction competition, capacity heterogeneity behaves like a
  capacity-slot and task-pressure mechanism, not a fixed semantic role taxonomy.
- Adding another high-capacity head, or adding separate branch towers without
  routing, does not automatically produce role-specific functional modularity.
- Oracle routing can produce branch-level functional modularity.
- Unconstrained learned position and token routers solve the task, but in the
  tested setup they did not discover the oracle role split. The next intervention
  should test weak routing supervision, routing regularization, or a more
  conflict-heavy task.
- Weak scored-position routing supervision with weight 0.05 made the token router
  recover near-oracle branch-level functional modularity in 5/5 seeds. The next
  question is how little supervision is needed, and whether unlabeled routing
  regularizers or harder task pressure can produce the same change.
- A weak-token-router supervision sweep found that gate behavior can become
  correctly routed before causal branch modularity follows: gate routed match was
  1.00 by weight 0.02, but 5/5 causal routed role match appeared only at weight
  0.05. This makes causal ablation necessary for the modularity claim and
  motivates unlabeled routing-pressure tests next.
- Unlabeled entropy and load-balancing regularizers changed router statistics
  but did not reliably create role-aligned causal modularity. Entropy-only
  produced sharp single-branch collapse, balance-only produced globally balanced
  but functionally co-located routing, and entropy+balance was mixed. This
  motivated testing stronger role conflict or branch bottlenecks.
- A follow-up bottleneck test shrank each branch attention head from 64 dims to
  16 dims. This did not rescue unlabeled modularity: unconstrained,
  balance-only, and entropy+balance bottlenecked routers all had same-top-branch
  rate 1.00 and routed role match 0.00 across 5 seeds. Oracle routing still
  produced routed role match 1.00, so the architecture can support modularity
  when routing is correct. The next intervention should change task conflict
  rather than only shrink branch capacity.
- A conflict-heavy `bidirectional_lookup` task then made the same query token
  require predecessor lookup in the local role and successor lookup in the
  induction role. This also did not rescue unlabeled modularity: the
  unconstrained, balance-only, and entropy+balance conflict-task routers all had
  same-top-branch rate 1.00 and routed role match 0.00 across 5 seeds. Weak
  labels and oracle routing both reached routed role match 1.00. The current toy
  evidence therefore points to role-informative routing pressure, not capacity
  bottlenecks or conflict alone, as the reliable mechanism for branch-level
  functional modularity.
- Annealed weak-label routing refined that mechanism. Role labels active only
  through steps 50, 100, 200, or 400 did not persist as causal branch modularity:
  all four conditions had same-top-branch rate 1.00 and routed role match 0.00.
  Labels through step 800 produced partial persistence with routed role match
  0.80 and branch distance 0.3337. Labels through step 1200 produced full
  top-branch role separation after 400 unlabeled final steps, with routed role
  match 1.00 and branch distance 0.7652, still weaker than always-on weak labels
  at branch distance 0.9773. The next decisive toy experiment is to checkpoint
  training trajectories and measure when gate separation and causal separation
  first form, and whether they decay after label removal.
- Checkpointed trajectory measurement found a gate-before-causality lag. By
  step 400, task accuracy was near one and gate routed match was 1.00, but
  causal routed role match was 0.00 and branch distance was only 0.1525. With
  continued weak labels, causal routed role match reached 1.00 by step 600 and
  branch distance reached 0.4996 by step 800. After label removal at step 800,
  the top-branch split mostly persisted but weakened to routed role match 0.80
  and branch distance 0.3337 by step 1600. After label removal at step 1200,
  routed role match stayed 1.00 but branch distance decayed from 0.8701 to
  0.7652. Always-on labels kept strengthening the split to branch distance
  0.9773. This supports a causal-consolidation framing: role-aligned gates form
  before causally separable branch computations.
- A small real-transformer follow-up on Pythia-14M checkpoints found an
  analogous probe-before-causality pattern for repeat-match heads. Across seeds
  1-3, selected layer-0/1 repeat-match specialization rose by step 4000
  (`0.3675` vs `0.2590` at step0), while top-head ablation was only `0.0556`
  loss-delta above random same-layer controls. The causal excess became
  substantial later: `1.3224` at step16000, `4.1506` at step64000, and `7.2982`
  at step143000. This does not establish branch modularity in Pythia, but it
  strengthens the methodological framing that specialization probes and causal
  importance can have different developmental timelines.
- A Pythia-160M follow-up over seeds 1-3 added checkpoint-specific raw-score
  alignment and cross-seed causal transfer. Repeat-match probe specialization
  rose by step4000 (`0.4794`), but causal own-top excess over random controls
  was still negative or near zero (`-0.0180`). By step16000, own-top causal
  excess became positive (`0.2324`) and aligned source-head transfer beat
  same-index source transfer by `0.1939`. At step143000, aligned transfer was
  much stronger than same-index transfer (`1.1774` vs `0.3046`), with aligned
  transfer better in all 6 ordered seed pairs. This strengthens the Phase 1
  framing: head-role identity is weakly universal after relabeling/alignment,
  not reliably tied to the same raw head index.
- Scaling the same analysis to all 9 official Pythia-160M seeds at selected
  checkpoints strengthened the alignment-transfer claim and revised the timing
  story. With 72 ordered source-target pairs per checkpoint, aligned transfer
  beat same-index transfer at step4000 (`0.1682` vs `-0.0049`, 62/72 pairs),
  step16000 (`0.3958` vs `0.0785`, 64/72 pairs), and step143000 (`1.0619` vs
  `0.2541`, 59/72 pairs). The 3-seed "probe before causal transfer" story was
  too sharp for Pythia-160M: aligned causal transfer is already detectable by
  step4000, but it strengthens substantially through training.
- A second-role test examined local-copy / previous-token behavior with
  `[x, SEP, x]` triples. The full Pythia-160M all-source/all-target final
  checkpoint result across all 9 seeds is positive in mean transfer but much
  more heterogeneous than repeat-match: selected local-copy specialization was
  `0.3262`, own-top excess over random was `1.6072`, aligned source transfer was
  `1.0137` versus `0.3142` for same-index transfer, and aligned transfer was
  better in `40/72` ordered pairs. The key qualifier is target heterogeneity.
  Target seeds 1-3 had aligned-minus-same `1.3993`; target seeds 4-6 had
  `-0.0116`; target seeds 7-9 had `0.7107`. Target own-head causal excess
  correlated strongly with aligned-minus-same transfer (`r ~= 0.97`), suggesting
  that probe-defined specialization only transfers functionally when the target
  seed actually uses the selected local-copy head causally.
- A layer-selection follow-up resolved the weak local-copy targets. All 9 seeds
  have a causal local-copy head, but the best causal layer is seed-dependent:
  layer 3 for seeds 1, 2, 3, 7, and 9; layer 2 for seed 4; and layer 4 for seeds
  5, 6, and 8. A fixed layers 2+4 transfer run rescued own-head causality in the
  previously weak targets (`own_top_excess=1.8528`) but was worse globally than
  layer 3 (`aligned-minus-same=0.2441` vs `0.6995`). This sharpens the framing:
  structural specialization can lead to functional specialization, but not at a
  fixed raw layer/head slot across seeds; candidate selection must distinguish
  probe behavior from causal role usage.
- A cross-layer candidate-pool run gave the strongest local-copy result so far.
  Selecting the top 2 local-copy heads across layers 2-4 and matching over the
  full 36-head candidate pool produced own-top excess `2.2896`, aligned transfer
  `2.2714` versus same-index transfer `0.4876`, aligned-minus-same `1.7838`,
  and aligned transfer better in `66/72` ordered source-target pairs. This is
  the clearest Phase 1 evidence that functional specialization is stable across
  seeds after role-level relabeling, while the raw structural slot can shift
  across nearby layers.
- The candidate-pool trajectory mirrors the repeat-match timing story but with a
  clean null at initialization. At `step0`, own-head excess and aligned transfer
  were essentially zero (`aligned-minus-same=-0.0004`). By `step4000`, aligned
  transfer was already positive (`0.4822` vs same-index `0.0631`,
  aligned-minus-same `0.4191`, 66/72 pairs). It strengthened at `step16000`
  (`aligned-minus-same=1.2037`) and final checkpoint
  (`aligned-minus-same=1.7838`). This supports a training-development claim:
  cross-seed functional role alignment appears early and then grows in causal
  magnitude.
- A Pythia-70M model-size check did not reproduce the 160M local-copy result.
  With layers 1-3, own-top excess was only `0.0508` and aligned-minus-same was
  `-0.0348`; with all layers 0-5, own-top excess rose only to `0.2692` and
  aligned-minus-same to `0.0810`. This is best interpreted as a capacity/task
  caveat: the 70M heads are only weakly causal for this synthetic local-copy
  behavior, so there is little functional role for alignment to transfer.
- A Pythia-410M check restored a strong positive result. Using candidate layers
  2-6 and top 2 local-copy heads, own-top excess was `4.1723`, aligned transfer
  was `1.9116` versus same-index transfer `0.2562`, and aligned-minus-same was
  `1.6554`. Aligned transfer was better in `49/72` ordered pairs and
  target-level aligned-minus-same was positive for all 9 seeds. This suggests
  the local-copy candidate-pool result is not unique to 160M, but the synthetic
  behavior may require enough model capacity to become a robust causal role.
- Bootstrap/sign-test summaries support the same model-size interpretation:
  aligned-minus-same target-level bootstrap CI crosses zero for 70M
  (`[-0.1332, 0.2989]`) but is clearly positive for 160M (`[1.3341, 2.3715]`)
  and 410M (`[1.0261, 2.2362]`).
- The 410M selected-checkpoint trajectory adds an important nuance. Candidate
  pool transfer is absent at `step0` (`aligned-minus-same=-0.0007`), already
  very strong at `step4000` (`aligned-minus-same=1.2062`, 72/72 pairs), peaks
  at `step16000` (`aligned-minus-same=3.4057`, 71/72 pairs), then remains
  positive but less clean at final (`aligned-minus-same=1.6554`, 49/72 pairs).
  Unlike 160M, the 410M trajectory is not monotonic under the fixed layers 2-6
  candidate window.
- A naturalistic WikiText repeated-span follow-up tested whether the synthetic
  `[x, SEP, x]` local-copy result survives a more natural induction-like setup:
  `prefix + span + distractor + span`, scored by next-token loss on the second
  span occurrence. With all-layer candidate pools, Pythia-160M across 9 seeds
  showed a positive but small aligned-transfer effect: own-head excess `0.6458`,
  aligned transfer `0.0665` versus same-index transfer `-0.0170`, and
  aligned-minus-same `0.0835` with target-level CI `[0.0334, 0.1343]`. A
  128-example replication kept the effect almost identical
  (`aligned-minus-same=0.0816`, target CI `[0.0333, 0.1300]`), while a matched
  `step0` control was null (`aligned-minus-same=0.0007`, target CI
  `[-0.0004, 0.0016]`). Pythia 410M was weaker: own-head excess `0.2416`,
  aligned-minus-same `0.0455`, and target-level CI `[-0.0190, 0.0894]`. A
  128-example 410M replication weakened this to aligned-minus-same `0.0293`
  with target CI `[-0.0237, 0.0630]`; target seed 6 remained a stable negative
  outlier. This reduces the risk that the synthetic result is purely artificial,
  but the naturalistic effect is much smaller and should be framed as supporting
  evidence rather than the main evidence. However, task-span alignment on the
  repeated-span probe split raised the 160M aligned-minus-same effect from
  `0.0835` to `0.5645` with target CI `[0.3653, 0.8068]`, 9/9 target positives,
  and a null `step0` control. The same task-span alignment rescued 410M as well:
  aligned-minus-same rose to `0.1544`, target CI `[0.0430, 0.2460]`, with 8/9
  target positives and a null `step0` control, though seed 6 remains a negative
  outlier. Larger 128/128 task-span replications remained positive for both
  160M (`aligned-minus-same=0.4773`, target CI `[0.2829, 0.6852]`) and 410M
  (`aligned-minus-same=0.1158`, target CI `[0.0222, 0.1884]`). This makes the
  methodological lesson sharper: weak natural roles require role-specific
  alignment representations.
- The alignment-basis summary as of 2026-05-23 is now a core Phase 1 result:
  generic Phase 0 alignment works for high-signal synthetic local-copy, but
  underestimates weak natural roles and can also miss some relabeling in 410M
  synthetic local-copy. Task-specific repeated-position alignment raises
  synthetic 160M aligned-minus-same from `1.7838` to `1.9593`, synthetic 410M
  from `1.6554` to `3.7737`, inserted WikiText 160M from `0.0835` to `0.5645`,
  inserted WikiText 410M from `0.0455` to `0.1544`, and naturally occurring
  exact-repeat 160M from `-0.0016` to `0.1897`. A 410M exact-repeat check was
  weaker: own-head causality was positive (`0.0503`), but aligned-minus-same was
  only `0.0215` with target CI crossing zero. The paper should report generic
  alignment as an unsupervised baseline and task-specific alignment as the
  role-level measurement.
- A stricter natural-repeat follow-up scanned unmodified WikiText windows for
  exact repeated 4-token n-grams. Pythia-160M all-layer candidate pools across 9
  seeds showed trained causal repeat heads (`own_top_excess=0.1588`, target CI
  `[0.0806, 0.2718]`, 9/9 positive), with a null `step0` control. However,
  generic Phase 0 aligned transfer did not beat same-index transfer
  (`aligned=0.0448`, same-index `0.0464`, aligned-minus-same `-0.0016`, target
  CI `[-0.0548, 0.0360]`). A task-repeat alignment basis reversed this result:
  aligned transfer rose to `0.2361`, aligned-minus-same to `0.1897`, target CI
  `[0.0737, 0.3140]`, with 8/9 target positives and 66/72 aligned-better pairs;
  the matched `step0` task-alignment control was null. This means unmodified
  natural repeats are causally real and can support role-relabeling, but only
  when the matching representation is specific to the repeated-span role.

### C. Resources concretely available (with HF / GitHub paths)
- Pythia seeds: `EleutherAI/pythia-{14m,70m,160m,410m}-seed{1..9}`, plus `pythia-160m-weight-seed{1-3}` and `pythia-160m-data-seed{1-3}`. 154 checkpoints per model (steps 0, 1, 2, 4, 8, …, 143000). GitHub: `EleutherAI/pythia`.
- MultiBERTs: `google/multiberts-seed_{0..24}-step_2000k`; 28 intermediate checkpoints for seeds 0–4 (`step_{20k..2000k}`). Cloud bucket `storage.googleapis.com/multiberts/public/intermediates/`.
- OLMoE-1B-7B: `allenai/OLMoE-1B-7B-0924`, all training artifacts open (data, code, logs); 16 decoder layers, 64 routed experts per layer, top-k=8.
- Hymba: `nvidia/Hymba-1.5B-Base`; barebones repo on GitHub (`barebones-hymba`).
- SwitchHead: code at `github.com/RobertCsordas/moe` (SwitchHead branch). 262M model on C4 reproduces at 44% compute / 27% memory of baseline.
- MultiBERTs analysis library: `language/multiberts` (Multi-Bootstrap statistical library for multi-seed inference).
- Csordás "Are Neural Nets Modular?" code: `github.com/RobertCsordas/modules` — directly importable for weight-mask analysis.
- Voita head-pruning code: `github.com/lena-voita/the-story-of-heads`.
- TransformerLens / activation patching: `TransformerLensOrg/TransformerLens` (Neel Nanda), plus `learnmechinterp.com` examples for IOI.

### D. Why "specialization without modularity" is the central conceptual distinction
Voss et al. 2021's Distill article documents specialization in InceptionV1 *without* a formal modularity test; the existence of "specialization" was inferred from feature visualizations. In transformers, Voita 2019 documented head specialization but Michel 2019 simultaneously showed massive head redundancy — both could be true because *specialized heads can still leak information through the residual stream*. This is exactly Csordás's P_specialize-without-P_reuse phenomenon at the head level. The user's project should explicitly compute both: a head can score high on specialization (S) and low on modularity (M), and reporting only S misses the second axis. The Branch Specialization community has historically reported only the analog of S; the user's contribution is to add M and to test whether architectural heterogeneity moves S, M, or both.

### E. The mech-interp / branch-specialization translation table
| Branch Specialization term | Mech-Interp term | Operationalization |
|---|---|---|
| Branch | Component / head | Index (layer, head) |
| Specialization | Universality (cross-model) / monosemanticity (within-model) | S(h, t) score |
| Modularity | Circuit separability | I(f_h; f_h' \| residual stream) |
| Functional region | Circuit | Path-patching-derived subgraph |
| Cross-seed consistency | Strong universality | Hungarian-matched S across seeds |
| Inductive bias of branching | Architectural prior | Head_dim distribution, gating function |

## Recommendations

**Immediately do (week 0–2)**:
1. Download Pythia-160m seeds 1–9, the weight-seed and data-seed variants, and MultiBERTs seeds 0–4. Total disk: ~10 GB. No retraining.
2. Reproduce Bali et al. 2026 "mid-layer stability dip" on Pythia-160m as your calibration baseline. If you can't reproduce it within ±10%, debug before any other step.
3. Implement the Hungarian-matched head-similarity metric on attention score matrices (Bali / Kobayashi approach — works without permutation alignment).

**Then do (week 2–10)**:
4. Apply the metric (and Filan clusterability + Csordás mask-IoU) to all four "branched attention" architectures, ranked by practicality: Pythia (MHA + parallel attn+FFN) → MultiBERTs (encoder MHA) → OLMoE (MoE FFN) → SwitchHead (MoE attention, the cheapest to multi-seed).
5. Skip Hymba and Mixture-of-Depths for Stage 1 — they have single seeds and re-pretraining is expensive; defer to Stage 2 if budget allows.

**Then do (week 10–20)**:
6. Architectural intervention: heterogeneous head_dim {32, 64, 128, 256}, uniform 64 baseline, and explicit branch/routing variants where tractable, trained ≥5 seeds each at 70M–160M scale on a Pile/C4 subset. Compare Stage 1 metrics.
7. Pre-register the competing hypotheses and decision criteria on OpenReview / OSF.
8. Engage with the LLaMA-half counter-result from arXiv:2505.16333: include a "uniform-but-wider" control arm to confirm the effect (if observed) is from *heterogeneity*, not from average width.

**Benchmarks/thresholds that would change the recommendation**:
- If Pythia 9-seed reproduction shows cross-seed *head* similarity already > 0.9 after Hungarian matching for most heads, the project's premise is weakened — pivot to studying the unstable mid-layer heads specifically.
- If the heterogeneous-head_dim intervention does not show a practically meaningful change in cross-seed variance of S(h, t), modularity metrics, or their dissociation at matched validation loss in pilot 70M experiments, reframe the intervention claim rather than forcing a positive modularity story.
- If Stage 1 finds that MoE-FFN expert specialization (OLMoE) is much more cross-seed consistent than MHA head specialization, that itself is a publishable finding and reorients the paper toward expert > head as the natural "branch."

**Venue targets**: ICLR / NeurIPS main track if Stage 2 produces a positive intervention result; ICML BlackboxNLP / ICLR Re-Align workshop / NeurIPS XAI workshop for Stage-1-only outcomes; transformer-circuits.pub / Distill-style report for a qualitative bridge between the two communities (which is itself valuable given how few papers exist).

## Caveats

- **The Bali et al. paper (arXiv:2602.16740) is very recent (Feb 2026) and the user should verify the latest version before reproducing**; its findings are pre-print and not peer-reviewed at time of writing.
- **Permutation symmetry is the technical hazard**: any cross-seed weight comparison without Hungarian alignment (or Ainsworth-style Git Re-Basin permutation matching) will report artifactually low similarity. Even with alignment, head index is not the only symmetry — Q/K columns are jointly permutable. Bali et al.'s use of attention *score* matrices (not weights) is the cleanest workaround and is the recommended primary metric.
- **CKA/CCA/SVCCA have known failure modes** (Kornblith 2019; Davari 2022) — use as secondary, not primary.
- **MoE routing instability** is well-documented (Mixtral router noise; arXiv:2505.24593 "Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis" reports that "deep Qwen 1.5-MoE mitigates expert failures (e.g., 43% MRR drop in geographic tasks when blocking top-10 experts) through shared expert redundancy, whereas shallow OLMoE suffers severe degradation (76% drop)"); cross-seed comparison of MoE expert specialization requires fixing routing temperature and using token-level co-activation rather than weight comparison.
- **The architectural intervention has no precedent**, so initial training instability is likely. Mitigation: ensure each per-head dim is a multiple of 8 for tensor-core efficiency; use a warmup schedule scaled per-head by √(head_dim); audit gradient norms per head in the first 1000 steps.
- **The mid-layer instability finding** from Bali et al. may itself be model-family-specific (their experiments use multiple optimizer/family combinations but small models); the architectural-heterogeneity benefit could be most visible *if and only if* mid-layer instability is real in your specific configuration, so include layer-wise stability as a primary outcome variable, not just a layer-averaged number.
- **Hymba's single-seed limitation is a real ceiling for that architecture** — analyzing whether attention-vs-Mamba head importance is stable across seeds requires re-pretraining, which at 1.5B is expensive. Reduce to a 100M–300M Hymba-style model trained on 10–30B tokens for a tractable multi-seed study.
- **The "specialization vs modularity" formal distinction proposed in §5 is the author's synthesis** — there is no single citable paper that defines both cleanly in this form. Csordás et al. 2021 is the closest; Filan et al. 2021 defines clusterability but not specialization; Olah/Voss defines specialization qualitatively. Treat the formalization as a contribution of the project rather than a citation.
- **Differential Transformer follow-up (arXiv:2505.16333)** is a direct warning: simply doubling head dim while halving head count did *not* reproduce Diff Transformer's specialization improvement. The user's heterogeneous-mix design is meaningfully different (varied not uniform), but the null result is a serious prior; budget for it.
