# Autonomous Sleep Research Log - Larger Head-Count Control

Date: 2026-05-23

## Scope Lock

- Unit of analysis: ordinary attention heads.
- Intervention: heterogeneous attention-head dimensions.
- Immediate concern from user: the previous Toy Ontology v2 models had only
  4 or 8 total head slots, which may be too few for sparse family-level
  modularity to appear.
- Next step: run a larger-head toy experiment before real-model validation.

## Start

- Local start time: 2026-05-23 23:19 PDT.
- Requested window: overnight while user sleeps.
- Stop condition: finish larger-head sweep plus, if feasible, initial
  real-model validation; otherwise stop after a documented blocker or
  meaningful failure-analysis point.

## Baseline From Previous Round

- `uniform4` and hetero4 configs used 4 heads per layer x 2 layers = 8 ordinary
  head slots.
- `uniform2` and hetero2 configs used 2 heads per layer x 2 layers = 4 ordinary
  head slots.
- This is enough to test structural role affinity, but may be too small for
  sparse ontology-level modularity.

## Planned Larger-Head Toy Control

- Use 8 heads per layer and 4 layers, giving 32 ordinary head slots.
- Keep Toy Ontology v2 fixed: 20 roles, 5 families.
- Compare:
  - `uniform8 = [48,48,48,48,48,48,48,48]`
  - `hetero8_unique_spread = [16,24,32,40,48,56,72,96]`
  - `hetero8_unique_extreme = [8,16,24,32,40,48,64,152]`
- All three configs have total attention dimension 384.
- Non-uniform configs use all-distinct multiples of 8.

## Failure-Analysis Rule

For weak or mixed results, do not immediately make a broad conclusion. Instead:

1. list plausible causes or confounds;
2. identify which are testable tonight;
3. run a targeted tweak/control if compute allows;
4. document whether the interpretation changes.

## First Completed Sweep

- Result root: `results/phase3_toy_role_ontology_v2_large_heads_1000_20260523`.
- Setting: 8 heads per layer x 4 layers = 32 ordinary head slots.
- All configs learned:
  - `uniform8` min role accuracy mean `0.996`;
  - `hetero8_unique_spread` min role accuracy mean `0.991`;
  - `hetero8_unique_extreme` min role accuracy mean `0.990`.
- Slot-level modularity did not improve:
  - `uniform8` family gap `0.048`, ARI `0.060`;
  - `hetero8_unique_spread` family gap `0.050`, ARI `0.051`;
  - `hetero8_unique_extreme` family gap `0.038`, ARI `0.025`.
- Structural affinity remained:
  - spread largest-top rate `0.50` vs 8-way chance `0.125`;
  - extreme largest-top rate `0.72` vs 8-way chance `0.125`.
- Specialization remained architecture-dependent:
  - `uniform8` specialization `0.517`, effective heads `9.12`;
  - `hetero8_unique_spread` specialization `0.488`, effective heads `8.02`;
  - `hetero8_unique_extreme` specialization `0.619`, effective heads `4.59`.

## Failure-Analysis Pass

The 32-slot sweep weakens the simple hypothesis that the earlier mixed
modularity result was only caused by having too few heads. Plausible causes:

1. Exact-slot family modularity may be diluted by extra layers; roles can choose
   different layer/head slots even if they share a structural type.
2. Family labels may be too coarse for the synthetic roles; structural affinity
   may be role-specific or difficulty-specific rather than family-specific.
3. Extreme heterogeneity may collapse many unrelated roles onto the largest
   head type, increasing specialization while reducing family separation.
4. Single-head ablation may under-credit distributed or redundant paths in a
   larger model.

Targeted tweak now running:

- Same 8-head configs, but 2 layers instead of 4.
- This gives 16 ordinary head slots, separating "more heads" from "more depth".

## Clean 16-Slot Control

- Initial 1000-step 16-slot run showed `uniform8` undertraining:
  - `uniform8` mean role accuracy `0.980`;
  - mean minimum-role accuracy `0.632`.
- Reran the same 16-slot grid for 2000 steps:
  `results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523`.
- Clean 2000-step result:
  - `uniform8`: family gap `0.132`, ARI `0.130`;
  - `hetero8_unique_spread`: family gap `0.079`, ARI `0.046`;
  - `hetero8_unique_extreme`: family gap `0.081`, ARI `0.058`.
- Structural affinity remained strong:
  - spread largest-top rate `0.62` vs chance `0.125`;
  - extreme largest-top rate `0.76` vs chance `0.125`.
- Specialization increased under heterogeneity:
  - `uniform8` specialization `0.514`, effective heads `5.314`;
  - spread specialization `0.609`, effective heads `3.923`;
  - extreme specialization `0.784`, effective heads `2.247`.

## Real-Model Validation

- Ran Pythia-160M-deduped attention-role probe.
- Initial float16 run produced NaNs in later layers, so it was treated as a
  failed probe and rerun in float32.
- Clean float32 root:
  `results/phase3_real_model_role_probe_pythia160m_deduped_float32_20260524`.
- Main role-probe results:
  - `repeat_match`: best `L0H7`, specialization `0.834`, effective heads
    `2.088`;
  - `previous_token`: best `L10H9`, specialization `0.332`, effective heads
    `9.169`;
  - `bos`: best `L8H6`, specialization `0.140`, effective heads `9.809`.
- Existing causal Pythia results were also summarized:
  - local-copy top heads: loss delta `2.519` vs random `0.229`;
  - natural repeated 8-gram top heads: loss delta `0.387` vs random `0.016`.

## Final Overnight Interpretation

More head slots do not rescue the claim that heterogeneous dimensions
automatically create family-level modularity. They do preserve or strengthen the
cleaner claims:

```text
structural role affinity is strong;
role-level specialization is strong;
family-level modularity is separate and not automatic.
```
