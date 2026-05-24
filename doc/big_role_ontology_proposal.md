# Big Role Ontology Proposal For Attention-Head Structure

Date: 2026-05-23

## Purpose

The current toy ontology is too small:

```text
local_copy: local_a, local_b
kv_lookup:  kv_a, kv_b
induction:  induction_short, induction_long
```

It is good for first evidence, but not enough for a convincing modularity
claim. The next stage should build a larger tree of functional roles, then test
the three project questions across that tree:

1. Structural role affinity: which structural head type gets a role?
2. Functional specialization: how concentrated is each role over heads?
3. Functional modularity: do related role families cluster together?

Results must be presented as:

```text
baseline result -> heterogeneous result -> interpretation
```

This proposal keeps the unit fixed as an ordinary attention head. It does not
switch the project to MoE experts, branch towers, or non-attention modules. The
goal is to ask whether differently shaped attention heads make some functional
roles more likely to occupy particular attention-head types, and whether that
role assignment becomes more specialized or more modular.

## Where The Current Roles Came From

The current six-role experiment was not meant to be the final ontology. It was
a minimal causal testbed built from roles that are already standard in the
attention-head literature:

| Current Role | Why It Is A Real Candidate Role | Main Prior Anchor |
|---|---|---|
| `local_a`, `local_b` | Previous-token and nearby-token attention are repeatedly observed in transformer heads. These roles are also easy to test causally with local-copy templates. | Elhage et al. 2021; Clark et al. 2019; Voita et al. 2019 |
| `kv_a`, `kv_b` | Attention is naturally a key-value lookup mechanism; many in-context retrieval tasks can be expressed as "find the query key, copy the paired value." | Transformer QK/OV framing in Elhage et al. 2021; retrieval-style toy tasks |
| `induction_short`, `induction_long` | Induction heads are a canonical mechanistic-interpretability role: attend to a previous occurrence and copy the following token. | Olsson et al. 2022; Elhage et al. 2021 |

The reason these were good first roles is that they give clean head-level
causal ablation measurements. The reason they are not enough is that the
ontology has only three families and six subroles. That is too small to support
a convincing claim about "functional modularity" as a family-level clustering
property.

## Literature Anchors

This ontology should be grounded in roles already discussed in the
attention-head and mechanistic-interpretability literature.

Key sources:

- Clark et al. 2019, "What Does BERT Look at?", BlackboxNLP/ACL Workshop:
  delimiter, fixed positional offset, broad attention, syntax, and coreference
  patterns in BERT heads.
  `https://aclanthology.org/W19-4828/`
- Voita et al. 2019, "Analyzing Multi-Head Self-Attention", ACL:
  linguistically interpretable and prune-resistant heads; commonly discussed
  roles include positional, syntactic, and rare-word attention.
  `https://aclanthology.org/P19-1580/`
- Olsson et al. 2022, "In-context Learning and Induction Heads", Transformer
  Circuits:
  induction heads and previous-token composition.
  `https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/`
- Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits",
  Transformer Circuits:
  QK/OV decomposition, previous-token heads, and induction-head mechanism.
  `https://transformer-circuits.pub/2021/framework/`
- Wang et al. 2022, "Interpretability in the Wild: a Circuit for Indirect
  Object Identification in GPT-2 small", arXiv:
  IOI circuit with 26 attention heads grouped into seven main classes.
  `https://arxiv.org/abs/2211.00593`
- McDougall et al. 2023, "Copy Suppression", arXiv:
  copy-suppression / negative-head behavior.
  `https://arxiv.org/abs/2310.04625`
- Htut et al. 2019, "Do Attention Heads in BERT Track Syntactic Dependencies?",
  arXiv:
  specialist heads for some dependency relation types, but no general parser
  head.
  `https://arxiv.org/abs/1911.12246`
- Pande et al. 2021, "The Heads Hypothesis", AAAI:
  role classification including syntactic, local, block, and delimiter roles,
  with attention to co-location of multiple roles in one head.
  `https://arxiv.org/abs/2101.09115`

## Proposed Role Tree

This tree intentionally mixes two kinds of roles:

1. Roles that can be built as clean synthetic causal tasks now.
2. Roles that probably require real pretrained language models or parsed text
   before they are meaningful.

The first group should drive the next toy experiment. The second group should
guide real-model validation and paper framing.

### A. Token Transport And Copying

These are the most directly compatible with toy causal tasks and decoder-only
LM probes.

| Family | Subroles | Toy Probe | Real-Model Probe | Source Motivation |
|---|---|---|---|---|
| Local copy | previous-token copy, separator-local copy, local-window copy | `[x, SEP, x]`, previous-token prediction | previous-token attention/logit effect | prior toy result; previous-token composition in Transformer Circuits |
| Key-value lookup | exact key-value retrieval, value-copy, multi-key lookup | `[k1,v1,...,qk,vq]` | factual/attribute retrieval prompt with in-context table | current toy ontology; retrieval-like attention |
| Duplicate-token detection | same-token previous occurrence, repeat membership, first-vs-second occurrence | repeated token and membership probes | duplicate-token attention in repeated-name/text prompts | IOI and induction literature |
| Induction | short induction, long induction, n-gram induction, selective induction | `[A,B,...,A] -> B` variants | repeated n-gram continuation in natural text | induction-head literature |
| Copy suppression | suppress repeated candidate, anti-copy, negative copy | repeated distractor where copied token should be wrong | copy-suppression prompts; negative-head logit effects | McDougall et al. 2023 |

### B. Positional And Boundary Roles

These roles are common in BERT and translation-head analyses and should be easy
to test with attention-pattern probes.

| Family | Subroles | Toy Probe | Real-Model Probe | Source Motivation |
|---|---|---|---|---|
| Fixed offset | previous token, next token, token two-back, sentence-local offset | fixed-offset prediction/control tasks | diagonal attention score at fixed offsets | Clark 2019; Voita 2019 |
| Delimiter/sink | BOS, SEP, CLS, newline, punctuation, document boundary | special-marker retrieval/sink tasks | attention to BOS/SEP/newline/punctuation | Clark 2019; Pande 2021 |
| Broad/global attention | uniform/broad context, global summary, segment average | summary marker requiring broad scan | entropy/broad-attention probes | Clark 2019 |
| Local-window attention | nearby-window aggregation, recency-only attention | local n-gram task | local attention concentration | Pande 2021 local/block roles |

### C. Syntax And Grammatical Relations

These are harder for toy tasks but important for English-language model roles.
They should be tested on parsed natural text, not only synthetic data.

| Family | Subroles | Toy Probe | Real-Model Probe | Source Motivation |
|---|---|---|---|---|
| Dependency heads | determiner-noun, adjective-noun, verb-object, preposition-object, subject-verb | synthetic grammar templates | Universal Dependencies relation probes | Clark 2019; Htut 2019 |
| Agreement | subject-verb number, pronoun agreement, reflexive agreement | template agreement tasks | logit/ablation effect on agreement prompts | syntactic probe literature |
| Phrase/chunk structure | noun phrase boundary, prepositional phrase, clause boundary | bracketed/chunk templates | parsed phrase-boundary probes | BERT syntax/head analyses |
| Coreference/entity linking | pronoun antecedent, repeated entity, alias/name linking | entity templates | coreference datasets/prompts | Clark 2019 |

### D. Entity And IOI-Style Circuit Roles

These are closest to causal mechanistic-interpretability role classes.

| Family | Subroles | Toy Probe | Real-Model Probe | Source Motivation |
|---|---|---|---|---|
| Name mover | copy correct entity/name to output | IOI-style name templates | IOI logit-difference patching | Wang et al. 2022 |
| Negative name mover | suppress wrong entity/name | IOI contrast templates | negative logit effect on distractor names | Wang et al. 2022 |
| S-inhibition | detect repeated subject and inhibit it | repeated-subject templates | IOI circuit intervention | Wang et al. 2022 |
| Backup name mover | redundant copy path when primary path ablated | ablation/backup tests | IOI backup-head patching | Wang et al. 2022 |
| Duplicate-token/name detector | detect repeated name or token | repeated-name templates | duplicate-token attention/ablation | Wang et al. 2022 |

### E. Suppression, Distractor, And Conflict Roles

These are important because modularity may require conflict, not just parallel
easy roles.

| Family | Subroles | Toy Probe | Real-Model Probe | Source Motivation |
|---|---|---|---|---|
| Distractor suppression | ignore wrong repeated token, ignore wrong key, ignore local trap | conflicting local/global lookup | logit suppression of distractor token | copy suppression and IOI negative heads |
| Anti-induction | repeated pattern where previous continuation is wrong | false-repeat task | negative induction/copy-suppression probes | McDougall et al. 2023 |
| Recency conflict | choose older value over recent distractor, or reverse | two-value conflict task | long-context retrieval with recent distractor | long-context retrieval concern |
| Calibration/self-repair | compensate after key head ablation | ablation with alternate path | self-repair after head ablation | copy suppression paper |

### F. Rare/Anchor/Topic And Content Selection

Some of these are less "head-only" and may involve MLPs, but they are still
worth probing as attention roles.

| Family | Subroles | Toy Probe | Real-Model Probe | Source Motivation |
|---|---|---|---|---|
| Rare-word/anchor attention | rare token, salient noun, proper noun, code identifier | rare-token retrieval task | attention to low-frequency tokens | Voita 2019 |
| Topic/document anchor | title token, section header, earlier topic word | topic marker retrieval | section-title/header attention | broad attention/anchor behavior |
| List/code format | list-item marker, indentation, quote/bracket matching | structured text templates | code/list punctuation probes | observed head-pattern work, practical LM roles |

## Proposed Ontology v2

The next experimental ontology should not try every role above immediately. It
should be larger than the current six-role ontology but still trainable.

Recommended v2 toy ontology:

```text
copy_transport:
  local_copy
  previous_token
  kv_lookup
  duplicate_token

induction:
  induction_short
  induction_long
  induction_ngram
  false_induction_control

position_boundary:
  bos_sink
  sep_sink
  fixed_offset_prev
  punctuation_boundary

suppression_conflict:
  distractor_suppression
  anti_copy
  wrong_key_suppression
  recency_conflict

entity_coreference:
  repeated_name_detection
  pronoun_antecedent
  simple_ioi_name_mover
  negative_name_control
```

This gives:

```text
5 families x 4 subroles = 20 roles
```

That is large enough to make modularity more meaningful but still small enough
for toy training and head ablation.

### Ontology v2 Priority Order

The proposed 20-role ontology should be implemented in two passes.

**Pass 1: synthetic and causal immediately.**

| Family | Subroles | Why First |
|---|---|---|
| `copy_transport` | `local_copy`, `previous_token`, `kv_lookup`, `duplicate_token` | Direct extension of the roles that already produced strong affinity/specialization. |
| `induction` | `induction_short`, `induction_long`, `induction_ngram`, `false_induction_control` | Tests whether induction remains less locked to the largest head once the ontology is larger. |
| `position_boundary` | `bos_sink`, `sep_sink`, `fixed_offset_prev`, `punctuation_boundary` | Common in BERT/head-role literature and easy to generate synthetically. |
| `suppression_conflict` | `distractor_suppression`, `anti_copy`, `wrong_key_suppression`, `recency_conflict` | Important because modularity may only appear when roles conflict instead of coexisting easily. |

**Pass 2: more language-like, likely needs real-model validation.**

| Family | Subroles | Why Second |
|---|---|---|
| `entity_coreference` | `repeated_name_detection`, `pronoun_antecedent`, `simple_ioi_name_mover`, `negative_name_control` | Closest to IOI and coreference literature, but toy templates can be brittle. |
| syntax/grammar extensions | dependency, agreement, phrase boundary | Requires parsed or carefully templated English; should not be overclaimed from toy token games. |
| topic/content anchors | title/header, rare word, section topic, code/list marker | Plausible LM roles, but harder to isolate causally without real text. |

## Experimental Design

The central paper-scale question becomes:

```text
Do heterogeneous attention-head structures change
1. which head type a role chooses,
2. how concentrated each role is over heads, and
3. whether related roles cluster together as a functional family?
```

The experiment should not claim ahead of time that heterogeneity must improve
all three. The current evidence suggests the first two are strong and the third
is still open.

### Phase 1: Toy Ontology v2

Configs:

```text
uniform4:          [32,32,32,32]
uniform2:          [64,64]
hetero4_best:      [16,64,16,32] or [16,32,64,16]
hetero2_best:      [48,80]
hetero2_extreme:   [16,112]
```

Metrics:

```text
role accuracy per role
role x head causal matrix
structural role affinity by head type
specialization and effective heads per role
within-family vs between-family similarity
family_gap and ARI
per-family modularity table
```

Result format:

| Question | Baseline | Hetero result | Interpretation |
|---|---|---|---|
| Affinity | top head/type counts under `uniform4` and `uniform2` | top head/type counts under hetero | whether roles prefer structural head types |
| Specialization | specialization/effective heads under baselines | same under hetero | whether roles concentrate |
| Modularity | family gap/ARI under baselines | same under hetero | whether role families cluster better |

### Phase 2: Natural-Text Probe Validation

Use existing pretrained ordinary-head models:

```text
Pythia seeds
GPT-2 small if useful for IOI compatibility
MultiBERTs for syntax/coreference heads
```

Probe families:

```text
fixed offset
delimiter/sink
induction/repeated n-gram
syntax dependencies
coreference/entity
copy suppression / IOI where model family supports it
```

This phase cannot directly test heterogeneous head dimensions in pretrained
uniform models, but it validates whether the role tree is meaningful in real
models.

### Phase 3: Train Small Heterogeneous LMs

After the role ontology is validated:

```text
train small uniform vs heterogeneous decoder-only LMs
evaluate the same role tree
test affinity/specialization/modularity under realistic text training
```

## Decision Criteria

### Strong Evidence For Structural Role Affinity

```text
roles repeatedly choose the same structural head type across seeds and layout
permutations, compared against baseline head-index distributions.
```

### Strong Evidence For Functional Specialization

```text
heterogeneous configs reduce effective heads and increase max_h p_r(h) relative
to both uniform4 and strong capacity baselines.
```

### Strong Evidence For Functional Modularity

```text
heterogeneous configs improve family_gap and ARI relative to both uniform4 and
uniform2 / hetero2 capacity baselines, and the effect holds across a larger role
ontology.
```

## Current Risk

The current evidence already suggests:

```text
capacity imbalance is enough to increase affinity and specialization.
```

But:

```text
capacity imbalance alone does not improve modularity over uniform2.
```

So the next proposal should not be "more width imbalance fixes modularity." The
better proposal is:

```text
larger role ontologies and harder conflict tasks are needed to test whether
heterogeneous head structure improves functional modularity.
```

## Immediate Next Step

Implement Toy Ontology v2 with about 20 roles and run a smoke test:

```text
uniform4 vs uniform2 vs hetero4_best vs hetero2_best
```

Do not move to a full expensive sweep until the smoke test shows:

```text
all role families are learnable;
ablation metrics are non-degenerate;
baseline-vs-hetero tables are interpretable.
```

## Paper-Scale Proposal

### Claim To Try To Establish

The current best possible claim is:

```text
Heterogeneous attention-head dimensions act as structural attractors for
functional role allocation: some roles repeatedly occupy the same head type
across seeds and layout permutations. This reliably changes role affinity and
specialization; whether it also improves functional modularity depends on the
role ontology, baselines, and task pressure.
```

That claim is strong because it does not require proving that every role cleanly
separates. It only requires showing that structure changes the distribution of
roles over heads in a reproducible way, then separately testing whether family
modularity follows.

### Minimum Convincing Experiment Package

| Package | Must Include | Why It Matters |
|---|---|---|
| Toy Ontology v2 smoke | 20 roles, `uniform4`, `uniform2`, best hetero4, best hetero2, 2-3 seeds | Checks whether the larger ontology is learnable and measurable. |
| Toy Ontology v2 main sweep | same configs, 5-10 seeds, all three metrics | Main evidence for affinity, specialization, and modularity. |
| Layout permutation control | move the largest head across positions | Separates head type from head index. |
| Head-count/capacity controls | `uniform2`, hetero2, matched total head dimension | Prevents confusing "fewer wider heads" with heterogeneity. |
| Per-role row inspection | full role x head causal matrices | Prevents aggregate metrics from hiding degenerate collapse. |
| Natural-text validation | pretrained ordinary-head models on induction, delimiter, syntax/coreference, IOI/copy suppression where feasible | Checks that the role tree is not just a toy artifact. |

### Success/Failure Interpretation

| Outcome | Interpretation |
|---|---|
| Hetero beats baselines on affinity only | Structural shape biases role placement, but roles may still be distributed or entangled. |
| Hetero beats baselines on affinity and specialization | Strong version of the current result: structure creates stable specialized role slots. |
| Hetero also beats baselines on family modularity | Strongest paper claim: structural heterogeneity can induce functional family organization. |
| Hetero loses to `uniform2` on modularity again | Still useful: specialization and modularity dissociate; capacity imbalance creates role affinity without family-level modularity. |
| Extreme hetero collapses roles onto one huge head | Important boundary condition: too much heterogeneity can hurt modularity. |

## Search Log

Queries used:

```text
Transformer attention heads functional roles induction heads previous token heads
BERT attention heads syntactic coreference positional separator attention heads
Voita 2019 specialized heads rare words positional syntactic ACL
IOI circuit name mover heads duplicate token heads S-inhibition heads
copy suppression heads transformer attention heads
Do Attention Heads in BERT Track Syntactic Dependencies
The heads hypothesis attention heads roles BERT
```

Included primary sources:

```text
Clark et al. 2019, BlackboxNLP/ACL Workshop
Voita et al. 2019, ACL
Olsson et al. 2022, Transformer Circuits
Elhage et al. 2021, Transformer Circuits
Wang et al. 2022, arXiv
McDougall et al. 2023, arXiv
Htut et al. 2019, arXiv
Pande et al. 2021, AAAI/arXiv
```
