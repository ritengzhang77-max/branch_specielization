# Role And Dataset Organization For Attention-Head Experiments

Date: 2026-05-23

## Core Answer

Yes: every experiment needs a dataset or probe scene for each role. The workflow
is:

```text
define role -> define scene/dataset -> train or probe model -> compute role x head causal matrix -> answer the three questions
```

The project should not only collect role names. Each role must be operational:
it needs token templates or natural-text examples, target positions, a scoring
rule, and controls.

## Hierarchy

Use this hierarchy throughout the project:

```text
ontology
  family
    role / subrole
      scene or dataset
        target positions
          metric rows
```

Definitions:

| Level | Meaning | Example |
|---|---|---|
| Ontology | A complete set of roles tested together in one model setting. | Toy Ontology v1, Toy Ontology v2 |
| Family | A group of related roles that should cluster if functional modularity is real. | `copy_transport`, `induction`, `position_boundary` |
| Role / subrole | A specific measurable function. | `kv_lookup`, `induction_long`, `bos_sink` |
| Scene / dataset | Concrete examples that require the role. | key-value table, repeated n-gram, BOS/delimiter template |
| Target positions | Tokens where loss or logits are measured. | query value token, second occurrence continuation, delimiter prediction |
| Metric row | One row in the role x head causal matrix. | ablation deltas for `kv_lookup` over all heads |

## How To Interpret The Previous Three

The previous experiment should be treated as one small ontology, not as three
unrelated tasks.

| Family | Subroles Used So Far | Status |
|---|---|---|
| `local_copy` | `local_a`, `local_b` | First family/class in Toy Ontology v1 |
| `kv_lookup` | `kv_a`, `kv_b` | Second family/class in Toy Ontology v1 |
| `induction` | `induction_short`, `induction_long` | Third family/class in Toy Ontology v1 |

So the old "three roles" are better described as:

```text
3 families x 2 subroles = 6 measured role rows
```

This matters because:

- structural role affinity is measured at the role/subrole level;
- specialization is measured at the role/subrole level;
- modularity is measured at the family level, because it asks whether related
  subroles cluster together.

## Can Old And Future Roles Be Talked About Together?

Yes, but only when the comparison is well-defined.

| Situation | Can We Analyze Together? | Reason |
|---|---|---|
| Same trained model, same head set, same mixed ontology dataset | Yes | Role x head matrix is shared, so ARI/family gap are meaningful. |
| Same architecture/config but separate datasets evaluated on the same model | Usually yes | Causal rows share the same heads, but dataset balance must be checked. |
| Different training runs with different ontology sizes | Partly | Compare summary metrics, but not one shared ARI. |
| Toy synthetic roles vs natural-text pretrained probes | No for one matrix; yes for paper narrative | They validate related ideas but are not one shared modularity test. |
| Attention-head roles vs MoE/router/branch roles | No unless explicitly approved | Unit changes, so the project claim changes. |

For the final paper, the clean structure is:

1. Toy Ontology v1: first evidence with 3 families / 6 rows.
2. Toy Ontology v2: main causal test with about 4-5 families / 16-20 rows.
3. Natural-text validation: separate evidence that the role tree corresponds to
   known language-model head roles.

## Dataset Requirements Per Role

Each role entry must specify:

| Field | Required Content |
|---|---|
| Role name | Stable label, e.g. `kv_lookup`. |
| Family | Parent cluster label, e.g. `copy_transport`. |
| Scene generator | Template or dataset source. |
| Positive target | What token/logit/loss position requires the role. |
| Negative/control target | A matched example where the shortcut should fail. |
| Metric | Accuracy, loss, logit difference, attention score, or ablation delta. |
| Head attribution | How to convert the role into a row over heads. |
| Intended setting | toy training, pretrained probe, natural text, or small hetero LM. |

Without those fields, a role is only a literature label, not yet an experiment.

## Proposed Toy Ontology v2

The next runnable experiment should use a synthetic multi-role dataset first.
That keeps every role in one shared head space.

### Smoke Test

Use 16 roles first:

| Family | Roles | Example Scene |
|---|---|---|
| `copy_transport` | `local_copy`, `previous_token`, `kv_lookup`, `duplicate_token` | local token copy, key-value table, duplicate membership |
| `induction` | `induction_short`, `induction_long`, `induction_ngram`, `false_induction_control` | repeated-token and repeated-ngram continuations |
| `position_boundary` | `bos_sink`, `sep_sink`, `fixed_offset_prev`, `punctuation_boundary` | marker, separator, offset, punctuation templates |
| `suppression_conflict` | `distractor_suppression`, `anti_copy`, `wrong_key_suppression`, `recency_conflict` | conflicting lookup or repeated distractor templates |

Why 16 first: these are the most synthetic-friendly families. They should be
learnable without relying on real English syntax or coreference.

### Full Toy Ontology v2

Then add 4 entity/coreference roles for 20 total:

| Family | Roles | Example Scene |
|---|---|---|
| `entity_coreference` | `repeated_name_detection`, `pronoun_antecedent`, `simple_ioi_name_mover`, `negative_name_control` | simple name/pronoun/IOI-style templates |

Why second: these are important for paper framing, but their toy versions can
be brittle. They should be added after the 16-role smoke test works.

## Which Questions Use Which Organization?

| Question | Unit Of Measurement | Minimum Role Organization Needed |
|---|---|---|
| Structural role affinity | role/subrole row | Any role with a causal head row. |
| Functional specialization | role/subrole row | Any role with non-degenerate ablation effects. |
| Functional modularity | family of related roles | At least several families with multiple subroles each. |
| Cross-seed stability | role-to-head or family-to-head mapping across seeds | Same ontology and config across seeds. |
| Natural-language validity | role family in pretrained models | Real-model probes or annotated natural text. |

This is why pairwise local-vs-induction separability is only a diagnostic. It
does not by itself prove modularity; modularity needs many roles with family
labels.

## Head-Dimension Configuration Rule

Future non-uniform configs should use all-distinct head dimensions. Uniform
baselines necessarily repeat dimensions, but heterogeneous configs should not.

Rules:

```text
same total attention dimension when comparing configs
same number of heads when that is the controlled comparison
all non-uniform head dimensions are distinct
all dimensions are multiples of 8
include layout permutations to separate head type from head index
```

Recommended matched-total-dim examples:

| Config Type | Head Dims | Use |
|---|---|---|
| `uniform4` | `[32,32,32,32]` | Four-head uniform baseline. |
| `uniform2` | `[64,64]` | Strong fewer/wider uniform baseline. |
| `hetero4_unique_mild` | `[16,24,40,48]` | All-distinct, moderate heterogeneity. |
| `hetero4_unique_64` | `[8,16,40,64]` | All-distinct one-large-head condition. |
| `hetero4_unique_extreme` | `[8,16,24,80]` | All-distinct extreme imbalance boundary test. |
| `hetero2_unique_mild` | `[48,80]` | Best current two-head non-uniform control. |
| `hetero2_unique_mid` | `[32,96]` | Wider two-head imbalance. |
| `hetero2_unique_extreme` | `[16,112]` | Collapse-risk boundary test. |

For layout controls, permute the non-uniform vector, for example:

```text
[8,16,40,64]
[64,8,16,40]
[8,64,16,40]
[8,16,64,40]
```

The old duplicated-small-head configs were useful as early one-large-head
controls, but the next phase should use all-distinct non-uniform dimensions.

## Planned Experiment Sizes

| Experiment | Roles | Configs | Seeds | Purpose |
|---|---:|---:|---:|---|
| Toy Ontology v2 smoke | 16 | 4-5 | 2-3 | Check learnability and metric quality. |
| Toy Ontology v2 main | 20 | 5-8 | 5-10 | Main affinity/specialization/modularity evidence. |
| Layout permutation control | 16 or 20 | 4 permutations of one vector | 5 | Test type-vs-index role assignment. |
| Natural-text validation | family-specific | pretrained models | existing seeds | Validate that the role tree maps onto known LM head roles. |

Do not run the expensive main sweep until the smoke test shows non-degenerate
role rows and interpretable baseline-vs-hetero tables.
