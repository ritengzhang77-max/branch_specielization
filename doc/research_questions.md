# Research Questions and Project Phases

## Core Framing

The cleanest framing is:

> Does **structural branch design** in transformer-style architectures induce
> **stable functional specialization** or **functional modularity** across random
> seeds?

This is an empirical question, not a design desideratum. The project should not
assume that functions ought to be separated, or that separation is always better.
The goal is to measure which outcomes actually occur:

- no reliable specialization;
- specialization without modularity;
- modularity without role specialization;
- specialization and modularity together.

Positive, negative, and mixed outcomes are all potentially informative.

Use the terms as follows:

- **Structural heterogeneity**: architectural differences among components before
  training, such as heterogeneous head dimensions, routed experts, attention-vs-SSM
  branches, or MoE attention experts.
- **Structural modularity**: weak coupling by construction or by learned weights,
  such as sparse routing, separate branches, or low cross-cluster edge weight.
- **Functional specialization**: one component disproportionately supports a
  task/function, such as induction copying, previous-token behavior, BOS attention,
  syntactic relations, or name moving.
- **Functional modularity**: specialized components are also functionally separable,
  meaning their computations are not strongly entangled with other components.

Avoid the phrase **structural specialization** as the main term. A component can be
structurally different or structurally isolated, but it is only specialized after
we observe what function it performs.

## Main Research Questions

### RQ1: Do Attention Heads Have Stable Functional Roles Across Seeds?

Given the same architecture and training recipe, do independently trained seeds
learn the same head functions?

Operational tests:

- raw head-index attention-pattern similarity;
- Hungarian-matched attention-pattern similarity;
- head-level specialization score `S(h, t)`;
- cross-seed consistency `C(h, t)`.

This is the Stage 1 baseline and should be tested first on Pythia and MultiBERTs.

### RQ2: Is Stability a Head-Index Property or a Relabeled-Role Property?

If the same function appears across seeds, does it appear in the same `(layer,
head)` slot, or only after optimal relabeling?

Operational tests:

- raw similarity vs Hungarian-matched similarity;
- matched similarity gap over random permutation baseline;
- optional Git-Re-Basin-style weight alignment as a robustness check.

This question separates strong universality from weak universality.

### RQ3: Does Specialization Coincide With Modularity?

When a head or branch is specialized for a function, is it also functionally
separable from other heads or branches? Or do multiple functions cohabit the same
component despite specialization?

Operational tests:

- specialization score `S(h, t)`;
- graph modularity / clusterability of head-head interaction graphs;
- Csordas-style mask IoU / IoMin;
- conditional mutual information between head outputs, if estimation is reliable;
- causal path-patching separability.

This is a core measurement axis, not a pass/fail criterion. Specialization
without modularity is possible and scientifically important, especially in
transformer residual-stream architectures.

### RQ4: Do Explicit Branch Architectures Change Specialization or Modularity Relative to Vanilla MHA?

Do MoE experts, routed attention experts, or SwitchHead-style attention experts
show different specialization or modularity patterns than ordinary attention
heads?

Operational tests:

- router entropy;
- expert co-activation;
- domain and vocabulary specialization;
- expert ablation effects;
- cross-seed or cross-checkpoint stability where seeds are unavailable.

This tests whether "branch" is a more natural unit for some functions than
ordinary attention heads, and whether branch structure changes specialization,
modularity, both, or neither.

### RQ5: Does Structural Heterogeneity Change Functional Specialization or Modularity?

Does an intervention such as heterogeneous per-head dimension make functional
roles more consistent across seeds? Separately, does it affect functional
modularity, or only specialization?

Candidate intervention:

```text
uniform baseline:       [64, 64, 64, 64, 64, 64, 64, 64]
heterogeneous variant:  [32, 32, 64, 64, 128, 128, 256, 256]
```

Primary outcomes:

- lower cross-seed variance of `S(h, t)`;
- larger Hungarian-matched similarity gap;
- change in clusterability/modularity, without assuming the direction in advance;
- stronger reuse of head-dimension slots for the same function.

Required controls:

- matched validation loss;
- matched total Q/K/V dimension or parameter budget;
- uniform-but-wider/fewer-head baseline to address the Differential Transformer
  follow-up warning.

## Phases

### Phase 0: Calibration

Reproduce a minimal attention-head stability result on Pythia.

Deliverables:

- fixed probe text set;
- attention extraction script;
- raw cross-seed similarity matrices;
- Hungarian-matched similarity report;
- layer-wise stability summary.

Decision criterion:

- If even a two-seed Pythia smoke test cannot produce stable attention matrices
  and sensible matched-vs-random gaps, debug infrastructure before expanding.

### Phase 1: Existing Multi-Seed Models

Analyze Pythia and MultiBERTs without retraining.

Deliverables:

- full seed-pair similarity tables;
- head-role probe results;
- ablation/path-patching results for a small task set;
- stability heatmaps.

Decision criterion:

- If cross-seed stability is already very high, focus on unstable mid-layer heads.
- If it is low but improves after matching, frame as weak universality.
- If it is low even after matching, the project becomes a negative result about
  head-role non-universality.

### Phase 2: Explicit Branch Architectures

Analyze OLMoE, SwitchHead, and other routed/expert architectures.

Deliverables:

- expert specialization and routing metrics;
- comparison against vanilla MHA head stability;
- within-model or cross-checkpoint proxy analysis where multi-seed checkpoints do
  not exist.

SwitchHead feasibility update:

- Use `RobertCsordas/moe_attention` for the official training framework.
- Use `RobertCsordas/switchhead` for the first local plug-in experiment.
- Do not use `RobertCsordas/moe` as the SwitchHead target; it is a related
  MoE-MLP codebase.
- A GPU smoke test of `SwitchHeadRope` passed locally; CPU is not suitable
  because the implementation uses Triton expert-projection kernels.
- The first tiny SwitchHead pilot solved the conflict-heavy task in 5/5 seeds
  but did not show meaningful role-aligned expert modularity: gate same-top
  expert was 1.00, causal same-top expert was 0.80, and causal expert
  distribution distance was only 0.0087.
- A 4-expert `moe_k=2` variant also solved the task but remained non-modular:
  causal same-top expert was 1.00 and single-expert ablation effects were tiny,
  suggesting redundancy across active experts rather than role separation.
- With a weak role-informed output-expert selection loss
  (`expert_supervision_weight=0.05`), the two-expert SwitchHead setup produced
  role-aligned causal expert modularity in 5/5 seeds: gate same-top expert was
  0.00, causal same-top expert was 0.00, routed expert match was 1.00, and
  causal expert distribution distance was 0.5675. This is induced modularity,
  not spontaneous modularity, because the auxiliary loss was active throughout
  training.
- A transient version with the same selector loss turned off after step 800
  preserved the split after 1200 further unsupervised steps: gate same-top
  expert was 0.00, causal same-top expert was 0.00, routed expert match was
  1.00, gate distance was 0.9645, and causal expert distribution distance was
  0.5664. This suggests induced expert routing can consolidate into persistent
  functional modularity.
- A selector-window sweep narrowed the provisional reliability boundary. End
  step 400 was partial (`routed expert match=0.80`), end step 425 had reliable
  gate splitting but one causal failure, and end step 450 was reliable in 5/5
  seeds (`routed expert match=1.00`). This suggests gate specialization can
  precede causal functional modularity.
- A direct checkpoint trajectory with selector pressure ending after step 450
  confirmed the ordering within training: meaningful reliable gate separation
  appeared by checkpoint 425, while causal same-top expert reached 0/5 and
  routed expert match reached 5/5 at checkpoint 500. This is the clearest local
  evidence so far for `structural routing cue -> gate specialization -> causal
  functional modularity`.
- A fixed-window weight sweep at end step 450 found that weights `0.02`, `0.03`,
  `0.04`, and `0.045` all solved the task but remained only `4/5` on routed
  expert match; `0.05` was the first tested weight with reliable 5/5 causal
  expert modularity. This narrows the claim: the structural cue must be strong
  enough as well as present long enough.
- A strength-duration check showed that longer selector pressure lowers the
  required weight. At end step 800, weight `0.02` remained partial, but `0.025`
  and `0.03` were reliable in 5/5 seeds. This supports a threshold model rather
  than a fixed minimum-weight model.
- A reversed-label control showed that the induced modules can follow the
  requested labels. With selector target local -> expert 1 and induction ->
  expert 0, the 800-step run was reliable in 5/5 seeds with the causal roles
  reversed. The 450-step reversed run was only 4/5, so the exact threshold is
  label/optimization sensitive.

### Phase 3: Architectural Intervention

Train uniform, heterogeneous-head-dimension, and explicitly routed/separated
toy or small-scale models across multiple seeds.

Deliverables:

- matched-loss training curves;
- specialization/stability comparison;
- modularity comparison, including the possibility of specialization without
  modularity;
- control for uniform wider/fewer heads.

Decision criterion:

- If heterogeneous head dimensions or routing do not produce a practically
  meaningful change in specialization, modularity, or their dissociation in
  pilot experiments, reframe the architectural-intervention claim.

Current toy-pilot status as of 2026-05-22:

- Heterogeneous head dimensions have produced stable capacity-attractor slots for
  functional specialization in toy key-value recall and induction-style tasks.
- In the local-vs-induction competition task, heterogeneous capacity affects
  which role occupies which slot, but does not by itself guarantee role-specific
  functional modularity.
- Separate branch towers without routing solve the task but remain causally
  entangled across roles.
- Oracle routing can produce clean branch-level functional modularity.
- Unconstrained learned routing, tested with position-based and token-based
  gates, solved the task but did not recover the oracle role split.
- Weak scored-position routing supervision with weight 0.05 made the token router
  recover near-oracle branch-level functional modularity in 5/5 seeds.
- A weak-token-router supervision sweep found a gate-vs-causality threshold:
  gate routed match reached 1.00 by weight 0.02, but causal routed role match
  reached 1.00 only at weight 0.05.
- Unlabeled entropy and load-balancing regularizers changed gate statistics but
  did not reliably produce role-aligned causal branch modularity.
- Bottlenecking each branch attention head from 64 dims to 16 dims did not rescue
  unlabeled modularity: unconstrained, balance-only, and entropy+balance
  bottlenecked routers all had same-top-branch rate 1.00 and routed role match
  0.00 across 5 seeds. Oracle routing still achieved routed role match 1.00,
  so the bottlenecked architecture can support modularity when routing is
  correct.
- A conflict-heavy `bidirectional_lookup` task made the same query token require
  predecessor lookup in the local role and successor lookup in the induction
  role. This still did not rescue unlabeled modularity: unconstrained,
  balance-only, and entropy+balance conflict-task routers all had
  same-top-branch rate 1.00 and routed role match 0.00 across 5 seeds. Weak
  labels and oracle routing both achieved routed role match 1.00.
- Annealed weak-label routing showed that brief role labels are not enough as a
  symmetry-breaking nudge. End steps 50, 100, 200, and 400 all solved the task
  but had same-top-branch rate 1.00 and routed role match 0.00. End step 800
  had routed role match 0.80 and branch distance 0.3337. End step 1200 had
  routed role match 1.00 and branch distance 0.7652, below the always-on weak
  label branch distance of 0.9773.
- Checkpointed training trajectories showed why end 400 failed. By step 400 the
  model had solved both roles and the router gate was role-aligned, but causal
  branch ablation still had routed role match 0.00 and branch distance 0.1525.
  With continued weak labels, causal routed role match reached 1.00 by step 600
  and branch distance reached 0.4996 by step 800. Removing labels at step 800
  preserved a weakened split, while removing at step 1200 preserved a stronger
  split.

The next Phase 3 question is therefore:

```text
Does the gate-before-causality lag also appear in less hand-designed routed
attention settings or in small real transformer tasks with role probes and
causal patching?
```

Phase 3 synthesis as of 2026-05-23:

- The answer to "does structure become function?" is conditional. Structural
  heterogeneity has produced stable functional specialization slots in the toy
  head-dimension interventions, but structural modularity alone has not
  reliably produced functional modularity.
- Separate branches, unconstrained learned routers, entropy/balance
  regularizers, branch bottlenecks, and direct role conflict all solved the toy
  tasks without reliably separating local and induction functions.
- Oracle routing and weak role-informative router supervision did produce
  branch-level functional modularity. The trajectory runs show a lag: the gate
  becomes role-aligned before causal branch separation appears.
- The next narrow toy check is a denser trajectory between steps 400 and 800 to
  locate the causal consolidation window. The next broader research step is to
  test whether this gate/probe-before-causality lag appears in a less
  hand-designed routed attention setting.
- The dense 400-800 trajectory refined the timing: after the task was solved,
  gate routed-role match was already 5/5 at step 400, causal routed-role match
  reached 5/5 at step 550, branch distance crossed 0.30 at step 600, and branch
  distance crossed 0.40 at step 750.

Initial real-transformer follow-up:

- A Pythia-14M checkpoint trajectory over seeds 1-3 found an analogous
  probe-before-causality pattern for repeat-match heads. The mean normalized
  repeat-match specialization of selected layer-0/1 heads rose by step 4000
  (`0.3675` vs `0.2590` at step0), while top-head ablation was still only
  `0.0556` loss-delta above random same-layer controls. The causal excess became
  much larger later: `1.3224` at step16000, `4.1506` at step64000, and `7.2982`
  at step143000.
- This does not show branch modularity in Pythia, but it supports the broader
  measurement warning: probe-defined specialization can precede causal
  importance, so specialization and causal tests must be reported separately.
- A Pythia-160M ordinary-phrase natural-repeat trajectory sharpened this warning
  on unmodified WikiText-103 examples. Probe specialization appeared by
  step4000, own-head causal importance was clear by step16000, and target-level
  aligned cross-seed transfer became robust by step64000.
- A standard-dataset naturalistic follow-up replaced synthetic `[x, SEP, x]`
  local-copy triples with WikiText repeated spans of the form
  `prefix + span + distractor + span`. Pythia-160M all-layer candidate pools
  across 9 seeds gave a positive but small transfer result: own-head excess
  `0.6458`, aligned transfer `0.0665` versus same-index transfer `-0.0170`,
  aligned-minus-same `0.0835`, target-level CI `[0.0334, 0.1343]`, and
  positive aligned-minus-same for 8/9 targets. A 128-example replication kept
  the effect nearly unchanged (`0.0816`, target CI `[0.0333, 0.1300]`), while a
  matched `step0` control was null (`0.0007`, target CI
  `[-0.0004, 0.0016]`). Pythia-410M all-layer candidate pools were weaker:
  own-head excess `0.2416`, aligned-minus-same `0.0455`, target-level CI
  `[-0.0190, 0.0894]`, and 8/9 target positives. A 128-example 410M replication
  weakened this to aligned-minus-same `0.0293` with target CI
  `[-0.0237, 0.0630]`, and target seed 6 remained a stable negative outlier.
  This means the synthetic result is not purely an arbitrary-token artifact, but
  the naturalistic validation is small and should be treated as supporting
  evidence, not the primary effect. A follow-up alignment-basis ablation changed
  this interpretation: task-span alignment on the repeated-span probe split
  raised 160M aligned-minus-same from `0.0835` to `0.5645`, with target CI
  `[0.3653, 0.8068]`, 9/9 target positives, and a null `step0` control. Thus
  the naturalistic role is much stronger when the matching representation is
  role-specific. The same task-span alignment rescued 410M:
  aligned-minus-same rose to `0.1544`, target CI `[0.0430, 0.2460]`, with 8/9
  target positives and a null `step0` control, although seed 6 remained a
  negative outlier. Larger 128/128 task-span replications remained positive for
  both 160M (`aligned-minus-same=0.4773`, target CI `[0.2829, 0.6852]`) and 410M
  (`aligned-minus-same=0.1158`, target CI `[0.0222, 0.1884]`).
- A stricter natural-repeat follow-up scanned unmodified WikiText windows for
  exact repeated 4-token n-grams. Pythia-160M all-layer candidate pools across 9
  seeds showed trained own-head causal importance (`own_top_excess=0.1588`,
  target CI `[0.0806, 0.2718]`, 9/9 positive) and a null `step0` control
  (`own_top_excess=-0.0001`). Generic Phase 0 alignment did not beat same-index
  transfer (`aligned=0.0448`, same-index `0.0464`, aligned-minus-same `-0.0016`,
  target CI `[-0.0548, 0.0360]`). But task-repeat alignment, using repeated-span
  attention vectors from the probe split, did recover transfer
  (`aligned=0.2361`, aligned-minus-same `0.1897`, target CI
  `[0.0737, 0.3140]`, 8/9 targets, 66/72 pairs), with a null `step0` control.
  This makes alignment basis a first-class methodological variable: generic
  attention-score matching can miss a weak natural functional role that
  role-specific matching recovers.

Alignment-basis summary as of 2026-05-23:

- Generic Phase 0 alignment works for high-signal synthetic local-copy
  (`aligned-minus-same=1.7838` for 160M, `1.6554` for 410M).
- For weak natural roles, task-specific alignment is decisive:
  - inserted WikiText repeated spans, 160M: generic `0.0835`, task-specific
    `0.5645`;
  - inserted WikiText repeated spans, 410M: generic `0.0455`, task-specific
    `0.1544`;
  - naturally occurring exact repeats, 160M: generic `-0.0016`,
    task-specific `0.1897`.
- 410M naturally occurring exact repeats were weaker: own top excess was
  positive (`0.0503`, target CI `[0.0068, 0.0873]`), but task-specific
  aligned-minus-same was only `0.0215` with target CI `[-0.0166, 0.0564]`.
- This means the alignment representation must be specified as part of the
  metric. Generic alignment is an unsupervised baseline; role-specific alignment
  is the better measurement for weak functional roles.
- A Pythia-160M follow-up over seeds 1-3 added checkpoint-specific raw-score
  alignment and source-head transfer. Repeat-match specialization rose by
  `step4000` (`0.4794`), while causal own-top excess over random controls was
  still negative or near zero (`-0.0180`). By `step16000`, causal excess was
  positive (`0.2324`) and aligned transfer beat same-index transfer by `0.1939`.
  At `step143000`, same-index source transfer was much weaker than aligned
  transfer (`0.3046` vs `1.1774`), with aligned transfer beating same-index in
  6/6 ordered seed pairs. This supports weak, relabeled cross-seed role
  universality for a causal repeat-match head role.

The next Phase 1 question is therefore:

```text
Does the Pythia-160M aligned-transfer result survive a broader seed set and a
larger alignment probe corpus?
```

Seed-9 selected-checkpoint follow-up:

- The aligned-transfer result survived the full Pythia-160M seed set at the
  selected checkpoints `step4000`, `step16000`, and `step143000`, using all 8
  fixed alignment probe texts. At the final checkpoint, aligned transfer was
  `1.0619` versus `0.2541` for same-index transfer, with aligned transfer
  better in `59/72` ordered source-target pairs.
- The timing story was revised. In the 3-seed pilot, `step4000` looked mostly
  probe-only. In the 9-seed run, `step4000` already had own-top causal excess
  `0.1659` and aligned-minus-same transfer `0.1731`, with aligned transfer
  better in `62/72` ordered source-target pairs.
- The better current claim is that probe specialization, causal importance, and
  aligned transfer all strengthen across training; aligned causal transfer is
  detectable by `step4000` and much larger by `step143000`.

The next Phase 1 question is now:

```text
Does alignment-based causal transfer generalize beyond repeat-match heads to a
second head-role task?
```

Second-role result:

- A local-copy / previous-token task was implemented using `[x, SEP, x]`
  triples. The probe is attention from `SEP` back to the previous `x`; the
  causal readout is next-token loss at `SEP`, where the target is copied `x`.
- A Pythia-160M two-seed final-checkpoint pilot on layer 3 found strong within
  seed causality but weak cross-seed transfer: own top excess over random was
  `2.1609`, same-index source transfer was `0.0033`, aligned source transfer was
  `0.1033`, and aligned-minus-same was `0.0999`.
- A larger all-source chunk completed for target seeds 1-3: own top excess over
  random was `2.8567`, same-index source transfer was `0.1901`, aligned source
  transfer was `1.5894`, aligned-minus-same was `1.3993`, and aligned transfer
  was better in `17/24` ordered source-target pairs.
- The full all-target chunked run is now complete across all 9 source seeds and
  all 9 target seeds. Over 72 ordered source-target pairs, aligned source
  transfer was `1.0137` versus `0.3142` for same-index transfer; aligned-minus
  same was `0.6995`, and aligned transfer was better in `40/72` pairs.
- The effect is target-conditional rather than uniform. Target seeds 1-3 had
  aligned-minus-same `1.3993`, target seeds 4-6 had `-0.0116`, and target seeds
  7-9 had `0.7107`.
- Target own-head causal excess strongly tracked aligned-minus-same transfer
  across target seeds (`r ~= 0.97`). This makes the next question sharper:
  probe-defined structural/role specialization can transfer functionally when
  the target seed actually uses that probed head causally, but a high probe score
  alone is not sufficient.
- A layer causal sweep showed that the weak layer-3 targets do have causal
  local-copy heads elsewhere. The best causal local-copy layer is layer 3 for
  seeds 1, 2, 3, 7, and 9; layer 2 for seed 4; and layer 4 for seeds 5, 6, and
  8. A fixed layers 2+4 transfer rule improves the weak targets but is worse
  than layer 3 globally (`aligned-minus-same=0.2441` vs `0.6995`), so the issue
  is not solved by picking one alternative fixed layer set.
- The cross-layer candidate-pool test was strongly positive. Selecting the top
  2 local-copy heads across layers 2-4 and aligning over the full 36-head
  candidate pool gave own-top excess `2.2896`, same-index transfer `0.4876`,
  aligned transfer `2.2714`, aligned-minus-same `1.7838`, and aligned transfer
  better in `66/72` ordered source-target pairs. At the target level,
  aligned-minus-same was positive for `9/9` seeds.
- The selected-checkpoint trajectory shows development rather than an
  initialization artifact: `step0` has no effect
  (`aligned-minus-same=-0.0004`), `step4000` is already positive (`0.4191`),
  `step16000` is stronger (`1.2037`), and the final checkpoint is strongest
  (`1.7838`).
- A Pythia-70M final-checkpoint check is weak/negative. Layers 1-3 gave
  own-top excess `0.0508` and aligned-minus-same `-0.0348`; all layers 0-5 gave
  own-top excess `0.2692` and aligned-minus-same `0.0810`. The likely issue is
  that 70M does not robustly implement the synthetic local-copy behavior, so
  functional transfer has little causal substrate.
- A Pythia-410M final-checkpoint check is strong/positive. Candidate layers 2-6
  gave own-top excess `4.1723`, same-index transfer `0.2562`, aligned transfer
  `1.9116`, aligned-minus-same `1.6554`, and aligned better `49/72` ordered
  pairs. Target-level aligned-minus-same was positive for all 9 seeds.
- Bootstrap/sign-test summaries make the model-size pattern cleaner: the
  target-level aligned-minus-same CI crosses zero for 70M (`[-0.1332, 0.2989]`)
  but is positive for 160M (`[1.3341, 2.3715]`) and 410M
  (`[1.0261, 2.2362]`).
- The 410M trajectory is nonmonotonic under the current candidate window:
  `step0` aligned-minus-same is `-0.0007`, `step4000` is `1.2062`,
  `step16000` is `3.4057`, and final is `1.6554`. This suggests the next
  larger-model analysis should test whether final-checkpoint role location
  shifts outside layers 2-6 or whether the synthetic local-copy behavior is
  redistributed late in training.

The next Phase 1 question is now:

```text
Can the cross-layer candidate-pool method be made robust enough for a paper
claim by adding naturalistic local-copy/induction probes and confidence
intervals across model sizes?
```

### Phase 4: Mechanistic Interpretation

If Phase 3 shows a signal, explain it mechanistically.

Deliverables:

- path-patching circuits for the key tasks;
- evidence about whether any structural slot consistently hosts any role, without
  assuming that small heads should host local roles or large heads should host
  global/induction roles;
- comparison of specialization and modularity axes.
