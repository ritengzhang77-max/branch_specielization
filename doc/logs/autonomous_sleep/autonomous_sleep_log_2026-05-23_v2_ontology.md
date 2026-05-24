# Autonomous Sleep Research Log - Toy Ontology v2

Date: 2026-05-23

## Scope Lock

- Unit of analysis: ordinary attention heads.
- Intervention: heterogeneous attention-head dimensions.
- Main artifact: Toy Ontology v2 with families, subroles, and concrete
  synthetic role scenes.
- Do not switch to MoE experts, branch towers, or SwitchHead units in this run.

## Start

- Local start time: 2026-05-23 18:28 PDT.
- Requested window: about 8 hours.
- Stop condition: finish the planned Toy Ontology v2 experiments, analyses,
  documentation, and presentation, or stop earlier only if the experiment needs
  a project-level decision from the user.

## Planned Work

1. Implement Toy Ontology v2 dataset/generator.
2. Run a small full-ontology learnability smoke.
3. Run the full 20-role main sweep.
4. Run layout permutation controls if the main sweep is clean.
5. Analyze baseline-vs-hetero results.
6. Write a comprehensive result memo and presentation deck.
7. Commit and push coherent checkpoints.

## Running Notes

- Implemented `scripts/toy_role_ontology_v2_head_dim_intervention.py`.
- Added 20 role rows across five families:
  - `copy_transport`;
  - `induction`;
  - `position_boundary`;
  - `suppression_conflict`;
  - `entity_coreference`.
- Added concrete token scenes, targets, and controls for every role.
- First 400-step full-ontology smoke:
  - most roles solved;
  - `kv_lookup` remained low, suggesting undertraining rather than a broken
    template.
- Longer 1200-step full-ontology smoke:
  - `uniform4` role accuracy mean `0.9983`, min `0.9688`;
  - `hetero4_unique_64` role accuracy mean `0.9998`, min `0.9961`.
- Main grid was first launched at 1200 steps, but one seed still had a
  non-negligible final train loss. I stopped that grid and restarted at 1600
  steps to reduce undertraining as a confound.

## Completed Work

- Finished the 20-role main sweep at 1600 steps:
  `results/phase3_toy_role_ontology_v2_full_1600_20260523`.
- Finished the layout permutation control:
  `results/phase3_toy_role_ontology_v2_layout_1600_20260523`.
- Wrote the comprehensive result memo:
  `doc/experiments/phase3/phase3_toy_role_ontology_v2.md`.
- Updated `todo.md` and `doc/provenance_log.md`.

## Main Findings

- Learnability:
  - all tested configs learned the 20-role ontology;
  - mean minimum-role accuracy stayed at or above `0.991`.
- Structural role affinity:
  - four-head hetero chance largest-top rate is `0.25`;
  - observed rates were `0.50`, `0.65`, and `0.82` as imbalance increased;
  - two-head hetero chance largest-top rate is `0.50`;
  - observed rates were `0.66`, `0.82`, and `0.94` as imbalance increased.
- Functional specialization:
  - `uniform4` specialization `0.733`, effective heads `2.400`;
  - `uniform2` specialization `0.684`, effective heads `2.166`;
  - best hetero4 specialization was `0.849`, effective heads `1.522`;
  - all hetero2 configs improved specialization over `uniform2`.
- Functional modularity:
  - `uniform4` family gap `0.153`, ARI `0.149`;
  - `uniform2` family gap `0.117`, ARI `0.108`;
  - `hetero2_unique_mid [32,96]` reached gap `0.151`, ARI `0.159`;
  - `hetero2_unique_extreme [16,112]` collapsed family clustering, with gap
    `0.050`, ARI `0.039`.

## Interpretation

The project should continue. The strongest current claim is:

```text
Heterogeneous ordinary attention-head dimensions induce structural role affinity
and increase role-level specialization in this toy setting.
```

The modularity claim should remain separate and cautious:

```text
Functional modularity can improve in some layouts, but heterogeneity alone does
not guarantee family-level modularity.
```
