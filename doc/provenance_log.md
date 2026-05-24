# Provenance Log

## 2026-05-21

- Initialized this research workspace as a git repository.
- Connected the local workspace to `https://github.com/ritengzhang77-max/branch_specielization.git`.
- Added `doc/metric_literature_review.md` summarizing proposed metrics, provenance,
  citation-count snapshots from OpenAlex, and trustworthiness ratings.
- Added `doc/research_questions.md` to clarify the central framing:
  structural branch design / structural heterogeneity as the intervention, stable
  functional specialization and functional modularity as the outcomes.
- Decided that "structural specialization" is not the best primary term. The
  cleaner language is:

```text
Does structural branch design induce stable functional specialization or
functional modularity?
```

- Environment check:
  - Python 3.10.15.
  - PyTorch 2.4.0+cu121 installed.
  - Transformers 4.51.0 installed.

- Added `scripts/attention_stability.py`, a Phase 0 script that computes
  cross-seed attention-pattern similarity, Hungarian-matched similarity, and a
  random-permutation baseline.
- Added `probes/phase0_probe_texts.txt`, a small fixed probe set for smoke tests.
- Ran a successful Pythia-14M three-seed infrastructure validation:
  - seeds: 1, 2, 3;
  - probe texts: 8;
  - raw same-index similarity mean: 0.5813;
  - Hungarian-matched similarity mean: 0.6828;
  - matched-minus-random mean: 0.1115.
- Recorded the smoke result in `doc/experiments/phase0/phase0_smoke_report.md`.
- Ran a successful Pythia-160M two-seed Phase 0 pilot:
  - seeds: 1, 2;
  - probe texts: 8;
  - raw same-index similarity mean: 0.7082;
  - Hungarian-matched similarity mean: 0.8220;
  - matched-minus-random mean: 0.0989.
- Added `doc/experiments/phase0/phase0_pythia160m_pilot.md`.
- Updated `scripts/attention_stability.py` to write `layer_summary.csv` in
  addition to pairwise layer metrics.
- Ran the all-seed Pythia-160M Phase 0 baseline:
  - seeds: 1 through 9;
  - revision: `step143000`;
  - seed pairs x layers: 36 x 12 = 432 layer-pair comparisons;
  - raw same-index similarity mean: 0.7127;
  - Hungarian-matched similarity mean: 0.8127;
  - matched-minus-random mean: 0.0998;
  - all layers had positive matched-minus-random gaps.
- Recorded the all-seed result in `doc/experiments/phase0/phase0_pythia160m_all_seed_baseline.md`.
- Noted an important limitation: current extraction compares Hugging Face
  returned attention probabilities, not pre-softmax raw attention scores.
- Updated `scripts/attention_stability.py` to support:
  - `--attention-representation raw_scores`, implemented for GPT-NeoX/Pythia by
    capturing pre-mask, pre-softmax scaled `QK^T` scores;
  - `--entry-mask causal`, which compares only valid lower-triangular causal
    token pairs by default.
- Ran a Pythia-160M seed 1 vs seed 2 raw-score pilot:
  - raw same-index similarity mean: 0.3342;
  - Hungarian-matched similarity mean: 0.6831;
  - matched-minus-random mean: 0.3444.
- Recorded this result in `doc/experiments/phase0/phase0_pythia160m_raw_score_pilot.md`.
- Ran the all-seed Pythia-160M raw-score baseline:
  - seeds: 1 through 9;
  - revision: `step143000`;
  - seed pairs x layers: 36 x 12 = 432 layer-pair comparisons;
  - raw same-index similarity mean: 0.3735;
  - Hungarian-matched similarity mean: 0.6692;
  - matched-minus-random mean: 0.2982.
- Recorded this result in
  `doc/experiments/phase0/phase0_pythia160m_raw_score_all_seed_baseline.md`.
- Created checkpoint deck:
  `presentations/2026-05-21-1535-raw-score-checkpoint/raw_score_checkpoint.pdf`.

## 2026-05-21 Phase 1 Role-Specialization Proxy

- Added `scripts/attention_role_specialization.py`.
- The script estimates simple head role scores for:
  - BOS attention;
  - previous-token attention;
  - synthetic repeat-match / induction-style attention.
- Ran Pythia-160M seeds 1 through 9 with revision `step143000`, using the
  all-seed raw-score Hungarian alignment from Phase 0.
- Aggregate role consistency:
  - BOS: raw distribution similarity 0.8317, aligned 0.8566, random 0.8310;
  - previous-token: raw 0.7531, aligned 0.8116, random 0.7531;
  - repeat-match: raw 0.5152, aligned 0.6191, random 0.5057.
- Key finding:
  - repeat-match is highly concentrated in layers 0 and 1;
  - layer 1 repeat-match mean max specialization is 0.8045;
  - layer 1 raw top-head match rate is 0.0833;
  - layer 1 aligned top-head match rate is 0.7778.
- Recorded the result in
  `doc/experiments/phase1/phase1_pythia160m_attention_role_specialization.md`.
- Created checkpoint deck:
  `presentations/2026-05-21-1553-role-specialization-checkpoint/role_specialization_checkpoint.pdf`.

## 2026-05-21 Phase 1 Repeat-Match Causal Ablation

- Added `scripts/repeat_match_ablation.py`.
- The script ablates selected Pythia/GPT-NeoX heads by zeroing per-head
  attention outputs before the attention output projection.
- Ran Pythia-160M seeds 1 through 9 with revision `step143000`.
- Evaluation used 64 shared synthetic repeated-token sequences of length
  `[x_1, ..., x_32, x_1, ..., x_32]`, scoring second-half continuation loss.
- Ablated top repeat-match heads from layers 0 and 1, one head per layer.
- Aggregate loss deltas:
  - own top repeat-match heads: 1.5244;
  - own random same-layer controls: 0.2403;
  - source heads transferred by raw-score alignment: 1.0538;
  - source heads transferred by same layer/head index: 0.2571.
- Paired source-transfer comparison:
  - aligned minus same-index loss delta mean: 0.7967;
  - aligned beat same-index in 59 of 72 target/source pairs.
- Key finding:
  - repeat-match heads are causally relevant on the synthetic task;
  - raw-score alignment transfers the causal role across seeds much better than
    same-index transfer.
- Recorded the result in
  `doc/experiments/phase1/phase1_repeat_match_ablation_pythia160m.md`.
- Created checkpoint deck:
  `presentations/2026-05-21-1603-repeat-match-ablation-checkpoint/repeat_match_ablation_checkpoint.pdf`.

## 2026-05-21 Phase 3 Toy Head-Dimension Intervention

- Added `scripts/toy_head_dim_intervention.py`.
- The script trains tiny decoder-only transformers on a synthetic key-value
  recall task:
  `[k_1, v_1, ..., k_8, v_8, k_q] -> v_q`.
- Compared matched total attention dimension configurations:
  - `uniform4`: `[32, 32, 32, 32]`;
  - `hetero4`: `[16, 16, 32, 64]`;
  - `uniform2`: `[64, 64]`;
  - position control `hetero4_64first`: `[64, 16, 16, 32]`.
- Used single-head ablation loss delta as the primary toy specialization score.
- All configs solved the task:
  - `uniform4` eval accuracy mean: 0.9992;
  - `hetero4` eval accuracy mean: 0.9980;
  - `uniform2` eval accuracy mean: 0.9996;
  - `hetero4_64first` eval accuracy mean: 1.0000.
- Main pilot result:
  - `uniform4` top specialization: 0.4414;
  - `hetero4` top specialization: 0.9741;
  - `uniform2` top specialization: 0.7704;
  - `hetero4_64first` top specialization: 0.9807.
- Causal top-head loss deltas:
  - `uniform4`: 0.1380;
  - `hetero4`: 1.6084;
  - `uniform2`: 0.9343;
  - `hetero4_64first`: 1.4581.
- Key structural result:
  - in `hetero4`, the top causal head was the 64-dim head at index 3 in 5/5
    seeds and both layers;
  - in `hetero4_64first`, the top causal head moved to the 64-dim head at index
    0 in 5/5 seeds and both layers.
- Interpretation:
  - in this toy setting, structural head-dimension heterogeneity induced stable
    functional specialization;
  - the position control suggests the role followed head dimension, not fixed
    head index.
- Recorded the result in `doc/experiments/phase3/phase3_toy_head_dim_intervention.md`.
- Created checkpoint deck:
  `presentations/2026-05-21-2248-toy-head-dim-intervention/toy_head_dim_intervention_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Induction Head-Dimension Intervention

- Added `scripts/toy_induction_head_dim_intervention.py`.
- The script trains tiny decoder-only transformers on repeated random-token
  sequences `[x_1, ..., x_16, x_1, ..., x_16]`, scoring second-half
  next-token prediction.
- Compared:
  - `uniform4`: `[32, 32, 32, 32]`;
  - `hetero4`: `[16, 16, 32, 64]`;
  - `uniform2`: `[64, 64]`;
  - `hetero4_64first`: `[64, 16, 16, 32]`.
- Used single-head ablation loss delta as the primary specialization score.
- All configs solved the task:
  - `uniform4` eval accuracy mean: 0.9993;
  - `hetero4` eval accuracy mean: 0.9991;
  - `uniform2` eval accuracy mean: 0.9993;
  - `hetero4_64first` eval accuracy mean: 0.9992.
- Main result:
  - `uniform4` top specialization: 0.5796;
  - `hetero4` top specialization: 0.9830;
  - `uniform2` top specialization: 0.6578;
  - `hetero4_64first` top specialization: 0.9882.
- Causal top-head loss deltas:
  - `uniform4`: 0.0587;
  - `hetero4`: 1.0195;
  - `uniform2`: 0.2989;
  - `hetero4_64first`: 1.5637.
- Structural slot result:
  - in `hetero4`, the top causal head was the 64-dim head at index 3 in both
    layers and 5/5 seeds;
  - in `hetero4_64first`, the top causal head moved to the 64-dim head at index
    0 in both layers and 5/5 seeds.
- Interpretation:
  - the toy structural-intervention result generalizes from key-value recall to
    an induction-style repeated-token task;
  - heterogeneous head dimensions appear to turn the high-capacity head into a
    stable causal role slot.
- Recorded the result in `doc/experiments/phase3/phase3_toy_induction_head_dim_intervention.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-0818-toy-induction-head-dim/toy_induction_head_dim_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Local-vs-Induction Competition

- Added `scripts/toy_competition_head_dim_intervention.py`.
- The script trains tiny decoder-only transformers on sequences with two scored
  regions:
  - local copy: `[x, SEP, x]`, scored at `SEP`;
  - global induction: `[y_1, ..., y_16, y_1, ..., y_16]`, scored on second-half
    continuation positions.
- Compared:
  - `uniform4`: `[32, 32, 32, 32]`;
  - `hetero4`: `[16, 16, 32, 64]`;
  - `uniform2`: `[64, 64]`;
  - `hetero4_64first`: `[64, 16, 16, 32]`.
- All configs learned both objectives:
  - `uniform4` local accuracy 1.0000, induction accuracy 0.9999;
  - `hetero4` local accuracy 1.0000, induction accuracy 0.9992;
  - `uniform2` local accuracy 1.0000, induction accuracy 0.9976;
  - `hetero4_64first` local accuracy 0.9999, induction accuracy 0.9986.
- Heterogeneity increased causal specialization concentration:
  - local top specialization: `uniform4` 0.4924, `hetero4` 0.8857,
    `hetero4_64first` 0.9187;
  - induction top specialization: `uniform4` 0.5999, `hetero4` 0.8188,
    `hetero4_64first` 0.7220.
- The clean small-local / large-global role partition did not appear:
  - `hetero4` local top dims `{"16": 1, "64": 4}`;
  - `hetero4` induction top dims `{"16": 1, "64": 4}`;
  - `hetero4_64first` local top dims `{"64": 5}`;
  - `hetero4_64first` induction top dims `{"16": 3, "64": 2}`.
- Interpretation:
  - heterogeneous head dimensions remain useful as permutation-symmetry-breaking
    stabilizers;
  - task competition and head layout affect which function occupies which slot;
  - the project should avoid claiming that head dimension alone automatically
    allocates semantic role classes.
- Recorded the result in
  `doc/experiments/phase3/phase3_toy_competition_head_dim_intervention.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-0845-toy-competition-head-dim/toy_competition_head_dim_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Competition Layout Permutations

- Added two named heterogeneous head-dimension presets:
  - `hetero4_64second`: `[16, 64, 16, 32]`;
  - `hetero4_64third`: `[16, 32, 64, 16]`.
- Updated `scripts/toy_competition_head_dim_intervention.py` to record top
  `(layer, head, dimension)` slot counts in config summaries.
- Added `scripts/analyze_competition_layout_permutations.py` to combine the
  original competition run and the new layout permutations.
- Ran the two new permutations with:
  - seeds: 1 through 5;
  - steps: 1200;
  - batch size: 128;
  - eval examples: 512;
  - local pairs: 8;
  - repeat length: 16.
- All new models learned both objectives:
  - `hetero4_64second`: local accuracy 1.0000, induction accuracy 1.0000;
  - `hetero4_64third`: local accuracy 1.0000, induction accuracy 0.9993.
- Combined across all four placements of the 64-dim head:
  - local role top head was 64-dim in 19/20 models;
  - induction role top head was 64-dim in 10/20 models;
  - local mean top specialization was 0.9180;
  - induction mean top specialization was 0.7388.
- Key interpretation:
  - local/previous-token behavior follows the 64-dim head as a stable
    high-capacity slot;
  - induction behavior is layout-sensitive and often occupies a 16-dim
    layer-0 slot;
  - heterogeneous head dimensions support a symmetry-breaking / slot-formation
    claim, not an automatic semantic role taxonomy claim.
- Recorded the result in
  `doc/experiments/phase3/phase3_toy_competition_layout_permutations.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-0858-layout-permutation/layout_permutation_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Competition Weight Sweep

- Added `scripts/analyze_competition_weight_sweep.py`.
- Tested the most diagnostic heterogeneous layout from the prior checkpoint:
  `hetero4_64second = [16, 64, 16, 32]`.
- Swept local objective weight while keeping induction weight at 1.0:
  - `local_weight = 0.00`;
  - `local_weight = 0.01`;
  - `local_weight = 0.10`;
  - `local_weight = 0.25`;
  - reused the existing `local_weight = 1.00` baseline.
- Each new condition used:
  - seeds: 1 through 5;
  - steps: 1200;
  - batch size: 128;
  - eval examples: 512;
  - local pairs: 8;
  - repeat length: 16.
- Main results:
  - `local_weight = 0.00`: induction top head was 64-dim in 5/5 seeds;
  - `local_weight = 0.01`: local and induction top heads were both 64-dim in
    5/5 seeds, with local accuracy 0.9963 and induction accuracy 0.9976;
  - `local_weight = 0.10`: induction top head was 64-dim in 2/5 seeds;
  - `local_weight = 0.25`: induction top head was 64-dim in 2/5 seeds;
  - `local_weight = 1.00`: induction top head was 64-dim in 1/5 seeds.
- Interpretation:
  - induction can occupy the 64-dim slot when local pressure is absent;
  - weak local pressure can cohabit the 64-dim slot;
  - stronger local pressure often displaces induction to 16-dim or 32-dim
    secondary slots;
  - the best current mechanism is capacity-slot competition, not fixed
    dimension-to-function semantics.
- Recorded the result in `doc/experiments/phase3/phase3_toy_competition_weight_sweep.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-0921-weight-sweep/weight_sweep_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Competition All-Layout Weight Sweep

- Added `scripts/analyze_competition_all_layout_weight_sweep.py`.
- Repeated the local-weight sweep across all four placements of the 64-dim head:
  - `hetero4_64first`: `[64, 16, 16, 32]`;
  - `hetero4_64second`: `[16, 64, 16, 32]`;
  - `hetero4_64third`: `[16, 32, 64, 16]`;
  - `hetero4`: `[16, 16, 32, 64]`.
- The full analysis grid combines 100 trained models:
  - 4 layouts;
  - 5 local weights: 0.00, 0.01, 0.10, 0.25, 1.00;
  - 5 seeds per layout/weight.
- Aggregate induction top-64 rates:
  - local weight 0.00: 19/20 models;
  - local weight 0.01: 19/20 models;
  - local weight 0.10: 12/20 models;
  - local weight 0.25: 5/20 models;
  - local weight 1.00: 10/20 models.
- Aggregate local top-64 rates:
  - local weight 0.00: 5/20 models;
  - local weight 0.01: 18/20 models;
  - local weight 0.10: 17/20 models;
  - local weight 0.25: 12/20 models;
  - local weight 1.00: 19/20 models.
- Key interpretation:
  - induction strongly occupies the 64-dim head when local pressure is absent or
    tiny;
  - moderate local pressure often displaces induction to secondary 16-dim or
    32-dim heads;
  - the pressure effect is not strictly monotonic, since equal weighting
    produces layout-specific mixed solutions;
  - the best mechanism is capacity-attractor slots plus task-pressure,
    layout-sensitive, and optimization-basin-sensitive role allocation.
- Recorded the result in
  `doc/experiments/phase3/phase3_toy_competition_all_layout_weight_sweep.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1135-all-layout-weight-sweep/all_layout_weight_sweep_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Competition Two-Attractor Test

- Added three two-48 heterogeneous presets:
  - `hetero4_two48_center`: `[16, 48, 48, 16]`;
  - `hetero4_two48_skip`: `[48, 16, 48, 16]`;
  - `hetero4_two48_front`: `[48, 48, 16, 16]`.
- Added `scripts/analyze_competition_two_attractor.py`.
- Ran the two-48 layouts at the strongest competition setting from the prior
  sweep:
  - local weight: 0.25;
  - induction weight: 1.0;
  - seeds: 1 through 5;
  - steps: 1200;
  - batch size: 128;
  - eval examples: 512.
- Added a `uniform2 = [64, 64]` control at the same local weight.
- Main results:
  - two-48 layouts solved the task, but local top heads were 48-dim in only
    5/15 models and induction top heads were 48-dim in only 4/15 models;
  - two-48 local and induction roles shared the same top slot in 12/15 models;
  - distinct max-dim role slots appeared in only 1/15 two-48 models;
  - `uniform2` put both roles on 64-dim heads by construction, but still shared
    the same exact top slot in 4/5 models.
- Key interpretation:
  - adding a second high-dimensional head does not automatically create modular
    role separation;
  - structural heterogeneity can stabilize functional specialization, but
    functional modularity requires more than differentiated capacity.
- Recorded the result in `doc/experiments/phase3/phase3_toy_competition_two_attractor.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1152-two-attractor/two_attractor_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Explicit Branch Isolation

- Added `scripts/toy_branch_isolation_intervention.py`.
- Added `scripts/analyze_branch_isolation.py`.
- Tested explicit two-branch transformer variants on the same local-vs-induction
  competition task:
  - `branch_sum`: two separate branch towers, both active at every scored
    position;
  - `oracle_route`: local scored positions use branch 0 and induction scored
    positions use branch 1.
- Both variants used:
  - two branch towers;
  - branch head dimensions `[64]`;
  - shared token embedding and unembedding;
  - separate attention/MLP towers;
  - local weight 0.25 and induction weight 1.0;
  - seeds 1 through 5;
  - 1200 steps.
- Both variants learned the task:
  - `branch_sum`: local accuracy 1.0000, induction accuracy 0.9998;
  - `oracle_route`: local accuracy 0.9999, induction accuracy 0.9987.
- Main modularity result:
  - `branch_sum`: same top branch rate 0.60, routed role match 0.20,
    branch-distribution distance 0.0411;
  - `oracle_route`: same top branch rate 0.00, routed role match 1.00,
    branch-distribution distance 1.0000.
- Branch ablation effects:
  - `branch_sum`: both branches support both roles;
  - `oracle_route`: ablating branch 0 hurts only local, and ablating branch 1
    hurts only induction.
- Key interpretation:
  - separate branch towers are not enough for modularity;
  - explicit routing can produce role-specific functional modularity in the toy
    setting;
  - the project should split the architectural claim into capacity
    heterogeneity for stable specialization and routing/separation for
    modularity.
- Recorded the result in `doc/experiments/phase3/phase3_toy_branch_isolation.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1230-branch-isolation/branch_isolation_checkpoint.pdf`.

## 2026-05-22 Framing Clarification

- Updated `doc/plan.md` and `doc/research_questions.md` to make the research
  framing explicitly neutral about whether functions should separate into
  different branches.
- Clarified that the project asks whether structural heterogeneity, branch
  design, or routing changes:
  - functional specialization;
  - cross-seed stability;
  - functional modularity;
  - the dissociation between specialization and modularity.
- Replaced directional language such as "higher modularity" or "cleaner
  specialization" with measurement-first language such as "change in
  modularity" and "different specialization or modularity patterns."
- Clarified that specialization without modularity, modularity without
  specialization, both together, or neither are all valid empirical outcomes.

## 2026-05-22 Phase 3 Toy Learned Branch Routing

- Extended `scripts/toy_branch_isolation_intervention.py` with learned routing
  variants:
  - `learned_position_router`: a softmax branch gate learned per sequence
    position;
  - `learned_token_router`: a softmax branch gate learned from the token and
    position representation at each position.
- Added gate diagnostics alongside causal branch-ablation diagnostics:
  - local and induction mean gate weights by branch;
  - local and induction gate entropy;
  - gate distribution distance between local and induction positions;
  - gate routed-role match rate.
- Ran the four-way comparison:
  - `branch_sum`;
  - `learned_position_router`;
  - `learned_token_router`;
  - `oracle_route`.
- All variants solved the local-vs-induction toy task:
  - `branch_sum`: local accuracy 1.0000, induction accuracy 0.9997;
  - `learned_position_router`: local accuracy 0.9999, induction accuracy
    0.9987;
  - `learned_token_router`: local accuracy 1.0000, induction accuracy 0.9989;
  - `oracle_route`: local accuracy 1.0000, induction accuracy 0.9988.
- Main modularity result:
  - `learned_position_router`: same top branch rate 1.00, routed role match
    0.00, branch-distribution distance 0.0665;
  - `learned_token_router`: same top branch rate 1.00, routed role match 0.00,
    branch-distribution distance 0.0006;
  - `oracle_route`: same top branch rate 0.00, routed role match 1.00,
    branch-distribution distance 1.0000.
- Gate behavior separated from causal modularity:
  - the position router kept gates near-uniform and high-entropy;
  - the token router often learned lower-entropy routing, but routed both roles
    through the same top branch in every seed.
- Key interpretation:
  - unconstrained learned routers can solve the toy task;
  - in this setup, they did not discover role-specific functional modularity;
  - explicit oracle routing remains a positive upper bound showing that
    branch-level functional modularity is possible.
- Recorded the result in `doc/experiments/phase3/phase3_toy_learned_router.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1249-learned-router/learned_router_checkpoint.pdf`.

## 2026-05-22 Phase 3 Toy Weakly Supervised Branch Routing

- Extended `scripts/toy_branch_isolation_intervention.py` with weakly
  supervised learned routing variants:
  - `weak_position_router`;
  - `weak_token_router`.
- The auxiliary routing loss is applied only at scored positions:
  - local scored positions target branch 0;
  - induction scored positions target branch 1.
- Added `gate_target_nll_mean` as a routing-target diagnostic.
- Ran the weak-router comparison with:
  - router supervision weight 0.05;
  - local weight 0.25;
  - induction weight 1.0;
  - seeds 1 through 5;
  - 1200 steps.
- Both weakly supervised variants solved the task:
  - `weak_position_router`: local accuracy 0.9999, induction accuracy 0.9988;
  - `weak_token_router`: local accuracy 0.9998, induction accuracy 0.9983.
- Main modularity result:
  - `weak_position_router`: same top branch rate 0.20, routed role match 0.80,
    branch-distribution distance 0.6446;
  - `weak_token_router`: same top branch rate 0.00, routed role match 1.00,
    branch-distribution distance 0.8957.
- Compared with the previous unconstrained-router checkpoint:
  - `learned_position_router` and `learned_token_router` had routed role match
    0.00;
  - weak supervision changed the outcome from entangled learned routing to
    near-oracle branch-level functional modularity, especially for the token
    router.
- Key interpretation:
  - a small scored-position routing objective is sufficient for learned
    functional modularity in this toy setup;
  - this is not evidence for spontaneous modularity, because the auxiliary loss
    directly names the desired role split.
- Recorded the result in `doc/experiments/phase3/phase3_toy_weak_router.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1300-weak-router/weak_router_checkpoint.pdf`.

## 2026-05-22 Phase 3 Weak Token-Router Supervision Sweep

- Added `scripts/analyze_weak_router_sweep.py`.
- Swept `weak_token_router` scored-position supervision weights:
  - 0.005;
  - 0.01;
  - 0.02;
  - 0.03;
  - 0.04;
  - 0.045;
  - reused the previous 0.05 checkpoint.
- Each new condition used:
  - seeds 1 through 5;
  - 1200 steps;
  - local weight 0.25;
  - induction weight 1.0.
- All weights solved the task, with induction accuracy at or above 0.9978.
- Main threshold result:
  - weight 0.005: routed role match 0.40, branch distance 0.2789;
  - weight 0.01: routed role match 0.20, branch distance 0.2050;
  - weight 0.02: routed role match 0.20, branch distance 0.2844;
  - weight 0.03: routed role match 0.40, branch distance 0.3910;
  - weight 0.04: routed role match 0.40, branch distance 0.4504;
  - weight 0.045: routed role match 0.80, branch distance 0.7459;
  - weight 0.05: routed role match 1.00, branch distance 0.8957.
- Key dissociation:
  - gate routed match reached 1.00 by weight 0.02;
  - causal routed role match stayed low until 0.045 and did not reach 1.00 until
    0.05.
- Key interpretation:
  - gate compliance is not sufficient for functional modularity;
  - the routing objective must be strong enough to reshape branch computations,
    not only gate probabilities;
  - induction routing is the limiting role.
- Recorded the result in `doc/experiments/phase3/phase3_toy_weak_router_sweep.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1354-weak-router-sweep/weak_router_sweep_checkpoint.pdf`.

## 2026-05-22 Phase 3 Unlabeled Router Regularization

- Extended `scripts/toy_branch_isolation_intervention.py` with unlabeled router
  regularization:
  - entropy minimization for sharper gates;
  - global branch-usage load balancing.
- Added global gate diagnostics:
  - global gate entropy;
  - global branch 0/1 usage;
  - global gate balance error.
- Added `scripts/analyze_unlabeled_router_regularization.py`.
- Tested `learned_token_router` without role-label routing loss under:
  - entropy weight 0.05;
  - balance weight 1.0;
  - entropy 0.05 plus balance 1.0;
  - entropy 0.10 plus balance 1.0.
- All unlabeled conditions solved the task, but none produced reliable
  role-aligned causal branch modularity:
  - entropy-only: routed role match 0.00, branch distance 0.0002;
  - balance-only: routed role match 0.00, branch distance 0.0033;
  - entropy 0.05 plus balance 1.0: routed role match 0.40, branch distance
    0.3082;
  - entropy 0.10 plus balance 1.0: routed role match 0.20, branch distance
    0.1484.
- Key negative controls:
  - entropy-only made gates sharp but collapsed both roles onto the same branch;
  - balance-only made global branch usage nearly 50/50 but still kept local and
    induction causally co-located.
- Key interpretation:
  - sharp routing and balanced routing are not the same as functional modularity;
  - generic unlabeled gate regularizers were not enough in this toy setting;
  - the next test should change task pressure or branch bottlenecks rather than
    only sweep more generic regularizer weights.
- Recorded the result in `doc/experiments/phase3/phase3_toy_unlabeled_router_regularization.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1551-unlabeled-router/unlabeled_router_checkpoint.pdf`.

## 2026-05-22 Phase 3 Bottlenecked Branch Routing

- Added `scripts/analyze_bottleneck_router_experiment.py`.
- Tested whether shrinking each branch attention head from 64 dims to 16 dims
  makes unlabeled routing pressure align with local-vs-induction functional
  roles.
- Ran 5 seeds each for:
  - unconstrained `learned_token_router`;
  - balance-only `learned_token_router`;
  - entropy 0.05 plus balance 1.0 `learned_token_router`;
  - weak-label `weak_token_router` with supervision weight 0.05;
  - `oracle_route`.
- All conditions solved the task.
- Main bottleneck result:
  - unconstrained: same-top-branch rate 1.00, routed role match 0.00, branch
    distance 0.0220;
  - balance-only: same-top-branch rate 1.00, routed role match 0.00, branch
    distance 0.0551;
  - entropy 0.05 plus balance 1.0: same-top-branch rate 1.00, routed role match
    0.00, branch distance 0.0004.
- Positive controls:
  - weak-label bottlenecked router: routed role match 0.80, branch distance
    0.6768;
  - oracle bottlenecked router: routed role match 1.00, branch distance 1.0000.
- Key interpretation:
  - the bottlenecked architecture can support clean modularity when routing is
    correct;
  - simple branch-capacity bottlenecks do not make generic unlabeled gate
    regularizers discover causal functional modularity in this setup;
  - the next decisive test should change task conflict, not only capacity or
    entropy/balance weights.
- Recorded the result in `doc/experiments/phase3/phase3_toy_bottleneck_router.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1609-bottleneck-router/bottleneck_router_checkpoint.pdf`.

## 2026-05-22 Phase 3 Conflict-Heavy Branch Routing

- Extended `scripts/toy_branch_isolation_intervention.py` with a
  `--task-variant bidirectional_lookup` option.
- The conflict task samples a prefix `[y_0, ..., y_15]` and scores the same query
  token under two incompatible role rules:
  - local role: `y_i -> y_{i-1}`;
  - induction role: `y_i -> y_{i+1}`.
- Added `scripts/analyze_conflict_router_experiment.py`.
- Ran 5 seeds each with 64-dim branch heads, equal local/induction weights, and
  1600 steps for:
  - unconstrained `learned_token_router`;
  - balance-only `learned_token_router`;
  - entropy 0.05 plus balance 1.0 `learned_token_router`;
  - weak-label `weak_token_router` with supervision weight 0.05;
  - `oracle_route`.
- All conditions solved the conflict task.
- Main conflict-task result:
  - unconstrained: same-top-branch rate 1.00, routed role match 0.00, branch
    distance 0.0237;
  - balance-only: same-top-branch rate 1.00, routed role match 0.00, branch
    distance 0.0283;
  - entropy 0.05 plus balance 1.0: same-top-branch rate 1.00, routed role match
    0.00, branch distance 0.2069.
- Positive controls:
  - weak-label conflict router: routed role match 1.00, branch distance 0.9773;
  - oracle conflict router: routed role match 1.00, branch distance 1.0000.
- Key interpretation:
  - direct predecessor-vs-successor role conflict is not enough for generic
    unlabeled gate pressure to discover causal role modularity;
  - role-informative routing pressure remains the reliable mechanism observed so
    far;
  - the next test should anneal weak labels to see whether the role signal is
    needed only early as a symmetry breaker or throughout training.
- Recorded the result in `doc/experiments/phase3/phase3_toy_conflict_router.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1626-conflict-router/conflict_router_checkpoint.pdf`.

## 2026-05-22 Phase 3 Annealed Weak Router Supervision

- Extended `scripts/toy_branch_isolation_intervention.py` with
  `--router-supervision-end-step`.
- The new flag makes weak router supervision active only for optimization steps
  `< N`; `-1` preserves the previous always-on behavior and `0` disables the
  auxiliary routing-label loss.
- Added `scripts/analyze_annealed_router_experiment.py`.
- Tested the conflict-heavy `bidirectional_lookup` task with 64-dim branch heads,
  equal local/induction weights, 1600 training steps, and weak-token-router
  supervision weight 0.05 active through:
  - step 50;
  - step 100;
  - step 200;
  - step 400;
  - step 800;
  - step 1200.
- All annealed conditions solved the task.
- Brief early labels did not persist as causal branch modularity:
  - end 50: same-top-branch rate 1.00, routed role match 0.00, branch distance
    0.0670;
  - end 100: same-top-branch rate 1.00, routed role match 0.00, branch distance
    0.0395;
  - end 200: same-top-branch rate 1.00, routed role match 0.00, branch distance
    0.0560;
  - end 400: same-top-branch rate 1.00, routed role match 0.00, branch distance
    0.0945.
- Longer role pressure produced partial or full persistence:
  - end 800: routed role match 0.80, branch distance 0.3337;
  - end 1200: routed role match 1.00, branch distance 0.7652.
- Reference points from the conflict checkpoint:
  - unlabeled entropy+balance: routed role match 0.00, branch distance 0.2069;
  - always-on weak label 0.05: routed role match 1.00, branch distance 0.9773;
  - oracle route: routed role match 1.00, branch distance 1.0000.
- Key interpretation:
  - role-informative pressure does not need to remain active until the final
    step for top-branch modularity to persist;
  - it must last through a large fraction of training in this setup;
  - continuous pressure gives much stronger causal separation than late removal.
- Recorded the result in `doc/experiments/phase3/phase3_toy_annealed_router.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1709-annealed-router/annealed_router_checkpoint.pdf`.

## 2026-05-22 Phase 3 Router Trajectory Checkpoint

- Extended `scripts/toy_branch_isolation_intervention.py` with
  `--trajectory-eval-steps`.
- The script now writes `trajectory_summary.csv` when intermediate optimizer
  update counts are requested.
- Added `scripts/analyze_router_trajectory_experiment.py`.
- Ran 5-seed trajectory sweeps for:
  - unlabeled entropy 0.05 plus balance 1.0;
  - weak labels through step 400;
  - weak labels through step 800;
  - weak labels through step 1200;
  - always-on weak labels.
- Evaluated at steps:
  `0, 50, 100, 200, 400, 401, 600, 800, 801, 1000, 1200, 1201, 1400, 1600`.
- Main mechanism result:
  - at step 400, task accuracy was near one and gate routed match was 1.00, but
    causal routed role match was 0.00 and branch distance was 0.1525;
  - with continued labels, causal routed role match reached 1.00 by step 600
    and branch distance reached 0.4996 by step 800;
  - removing labels at step 400 prevented causal consolidation;
  - removing labels at step 800 preserved the top-branch split in 4/5 seeds but
    branch distance decayed to 0.3337 by step 1600;
  - removing labels at step 1200 preserved routed role match 1.00 but branch
    distance decayed from 0.8701 to 0.7652;
  - always-on labels kept increasing branch distance to 0.9773.
- Key interpretation:
  - gate alignment is an intermediate state, not sufficient evidence of
    functional branch modularity;
  - causal branch modularity consolidates later under sustained role-aligned
    routing pressure.
- Recorded the result in `doc/experiments/phase3/phase3_toy_router_trajectory.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1745-router-trajectory/router_trajectory_checkpoint.pdf`.

## 2026-05-22 Phase 1 Pythia-14M Repeat-Match Checkpoint Trajectory

- Added `scripts/pythia_repeat_match_checkpoint_trajectory.py`.
- Tested whether an attention-role probe for repeat-match heads appears before
  those heads have a strong causal effect under head-output ablation.
- Ran Pythia-14M seeds 1, 2, and 3 at revisions:
  `step0`, `step64`, `step256`, `step1000`, `step4000`, `step16000`,
  `step64000`, and `step143000`.
- Selected the top repeat-match head in layers 0 and 1 from 64 synthetic probe
  sequences, then evaluated head-output ablation on 64 separate repeated-token
  evaluation sequences.
- Compared top-head ablation to 8 random same-layer controls per seed and
  checkpoint.
- Main result:
  - repeat-match specialization rose by step4000 (`0.3675` vs `0.2590` at
    step0);
  - top-head causal excess over random controls was still small at step4000
    (`0.0556`);
  - causal excess became larger at step16000 (`1.3224`), step64000 (`4.1506`),
    and step143000 (`7.2982`).
- Key interpretation:
  - this is a real-transformer analogue of the toy measurement warning;
  - attention-role specialization can precede strong causal importance;
  - it does not establish branch modularity in Pythia, because these are ordinary
    attention heads rather than routed branches.
- Recorded the result in
  `doc/experiments/phase1/phase1_pythia14m_repeat_match_checkpoint_trajectory.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1759-pythia-repeat-trajectory/pythia_repeat_trajectory_checkpoint.pdf`.

## 2026-05-22 Phase 1 Pythia-160M Repeat-Match Alignment Trajectory

- Added `scripts/pythia_repeat_match_alignment_trajectory.py`.
- Added `scripts/analyze_pythia_alignment_trajectory.py`.
- Tested whether repeat-match causal head roles transfer across seeds by raw
  head index or by checkpoint-specific raw-score alignment.
- Ran Pythia-160M seeds 1, 2, and 3 at revisions:
  `step0`, `step1000`, `step4000`, `step16000`, `step64000`, and `step143000`.
- For each seed and checkpoint:
  - selected the top repeat-match head in layers 0 and 1 from 64 synthetic
    repeated-token probe sequences;
  - built raw-score Hungarian head alignments using four natural probe texts;
  - ablated own selected heads, random same-layer controls, source heads by same
    index, and source heads by raw-score alignment;
  - measured second-half repeated-token continuation loss.
- Main trajectory result:
  - repeat-match probe specialization rose by `step4000` (`0.4794`);
  - causal own-top excess over random controls was still negative or near zero
    at `step4000` (`-0.0180`);
  - by `step16000`, own-top causal excess was positive (`0.2324`) and aligned
    transfer beat same-index transfer by `0.1939`;
  - at `step143000`, same-index source transfer was `0.3046`, aligned transfer
    was `1.1774`, and aligned-minus-same was `0.8728`.
- Paired transfer result:
  - aligned transfer beat same-index in 5/6 ordered seed pairs at `step16000`;
  - 5/6 at `step64000`;
  - 6/6 at `step143000`.
- Key interpretation:
  - repeat-match specialization again appears before strong causal transfer;
  - cross-seed causal role identity is much better captured by raw-score
    alignment than by raw head index;
  - this supports weak, relabeled role universality, not branch modularity in
    ordinary Pythia heads.
- Recorded the result in
  `doc/experiments/phase1/phase1_pythia160m_repeat_match_alignment_trajectory.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1902-pythia160m-alignment-trajectory/pythia160m_alignment_trajectory_checkpoint.pdf`.

## 2026-05-22 Phase 1 Pythia-160M Seed-9 Selected Checkpoints

- Added `scripts/analyze_pythia_seed9_alignment_selected.py`.
- Scaled the Pythia-160M repeat-match alignment-transfer analysis to all 9
  official seeds at selected checkpoints:
  `step4000`, `step16000`, and `step143000`.
- Used all 8 fixed probe texts for raw-score alignment.
- Ran each selected checkpoint as a separate one-checkpoint job after an earlier
  all-checkpoint job was interrupted by unrelated GPU contention before writing
  results.
- Main result:
  - step4000: aligned transfer `0.1682`, same-index transfer `-0.0049`,
    aligned-minus-same `0.1731`, aligned better in 62/72 ordered seed pairs;
  - step16000: aligned transfer `0.3958`, same-index transfer `0.0785`,
    aligned-minus-same `0.3173`, aligned better in 64/72 pairs;
  - step143000: aligned transfer `1.0619`, same-index transfer `0.2541`,
    aligned-minus-same `0.8078`, aligned better in 59/72 pairs.
- Interpretation:
  - the aligned-transfer claim survives the full Pythia-160M seed set;
  - same-index transfer remains weak relative to raw-score alignment;
  - the timing story is revised because causal aligned transfer is already
    detectable by step4000 in the 9-seed run, not only at later checkpoints.
- Recorded the result in
  `doc/experiments/phase1/phase1_pythia160m_seed9_alignment_selected_checkpoints.md`.
- Created checkpoint deck:
  `presentations/phase1/2026-05-22-2032-pythia160m-seed9-alignment/outputs/pythia160m_seed9_alignment_checkpoint.pdf`.

## 2026-05-22 Autonomous Sleep Research: Local-Copy Pilot

- Read and followed `/home/gavin/.codex/skills/autonomous-sleep-research/SKILL.md`.
- Recorded autonomous block start and planned stop in
  `doc/logs/autonomous_sleep/autonomous_sleep_log_2026-05-22.md`.
- Added `scripts/pythia_local_copy_alignment.py`, a local-copy /
  previous-token contrast task:
  - sequence pattern `[x, SEP, x]`;
  - probe score: attention from `SEP` to previous `x`;
  - causal readout: next-token loss at `SEP`, where target is copied `x`;
  - same-index vs raw-score-aligned source-head transfer.
- Completed a Pythia-14M two-seed smoke run; the task was not reliable at 14M
  (`own_top_excess_over_random=-0.1005`).
- Completed a Pythia-160M two-seed final-checkpoint pilot on layer 3:
  - selected local-copy specialization `0.3142`;
  - own top excess over random `2.1609`;
  - same-index source transfer `0.0033`;
  - aligned source transfer `0.1033`;
  - aligned-minus-same `0.0999`.
- Completed a larger all-source target chunk:
  - source seeds 1-9;
  - target seeds 1-3;
  - layer 3;
  - final checkpoint `step143000`;
  - own top excess over random `2.8567`;
  - same-index source transfer `0.1901`;
  - aligned source transfer `1.5894`;
  - aligned-minus-same `1.3993`;
  - aligned better in 17/24 ordered source-target pairs.
- Interpretation:
  - local-copy heads can be strongly causal within a seed;
  - with all source seeds, raw-score alignment transfers local-copy causal heads
    much better than same index for target seeds 1-3;
  - the result is still target-partial, not a full all-seed claim.
- Attempted the next chunk, target seeds 4-6, but current GPU jobs were
  interrupted during probing before target rows completed. The script has
  partial-output writes and `--target-seeds` chunking for the next attempt.
- Added `scripts/analyze_local_copy_chunks.py` to merge completed local-copy
  target chunks as they become available.
- Recorded the pilot in `doc/experiments/phase1/phase1_pythia160m_local_copy_pilot.md`.

## 2026-05-22 - Local-copy target seeds 4-6 completed

- Completed the Pythia-160M final-checkpoint local-copy target chunk for target
  seeds 4-6 under the same settings as the target seeds 1-3 chunk.
- Result directory:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets4_6/`.
- Transfer summary over 24 ordered source-target pairs:
  - same-index source transfer: `0.1700`;
  - aligned source transfer: `0.1584`;
  - aligned-minus-same: `-0.0116`;
  - aligned better count: `11/24`.
- Seed summary:
  - selected specialization mean: `0.3254`;
  - own top excess over random: `0.0420`.
- This is a mixed/negative chunk for local-copy causal transfer: probe
  specialization remains visible, but the target's own selected heads are barely
  causal above random for seeds 4-6.
- Re-ran `scripts/analyze_local_copy_chunks.py`; with target chunks 1-3 and 4-6
  merged, the combined result over target seeds 1-6 has aligned-minus-same
  `0.6939` over 48 ordered source-target pairs.

## 2026-05-22 - Local-copy all-target result completed

- Completed the final Pythia-160M final-checkpoint local-copy target chunk for
  target seeds 7-9 under the same settings.
- Result directory:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_targets7_9/`.
- Target seeds 7-9 transfer summary over 24 ordered source-target pairs:
  - same-index source transfer: `0.5826`;
  - aligned source transfer: `1.2933`;
  - aligned-minus-same: `0.7107`;
  - aligned better count: `12/24`.
- Re-ran `scripts/analyze_local_copy_chunks.py` with all target chunks present.
- Combined all-target result directory:
  `results/phase1_pythia160m_local_copy_alignment_seed9_layer3_combined/`.
- Full all-target transfer summary over 72 ordered source-target pairs:
  - selected local-copy specialization: `0.3262`;
  - own top excess over random: `1.6072`;
  - same-index source transfer: `0.3142`;
  - aligned source transfer: `1.0137`;
  - aligned-minus-same: `0.6995`;
  - aligned better count: `40/72`.
- Target-level heterogeneity:
  - target seeds 1-3: aligned-minus-same `1.3993`;
  - target seeds 4-6: aligned-minus-same `-0.0116`;
  - target seeds 7-9: aligned-minus-same `0.7107`.
- Follow-up diagnostic: target own-head causal excess and aligned-minus-same
  transfer correlate strongly across target seeds (`r=0.9664`; target sign count
  `7/9`, two-sided sign `p=0.1797`). This supports a conditional result:
  raw-score alignment transfers local-copy function when the target seed
  actually uses the selected probe head causally, but probe specialization alone
  is not sufficient.
- Updated `scripts/analyze_local_copy_chunks.py` to write
  `target_diagnostic_summary.csv` into the combined result directory.

## 2026-05-22 - Local-copy layer-selection follow-up

- Added `scripts/pythia_local_copy_layer_causal_sweep.py`.
- Ran all-layer local-copy causal sweeps with the same synthetic task and
  same-layer random-head controls.
- Weak layer-3 targets were not missing local-copy function; they used different
  layers:
  - seed 4: best L2H10, excess `1.4051`, layer-3 excess `-0.3294`;
  - seed 5: best L4H6, excess `2.0489`, layer-3 excess `0.1554`;
  - seed 6: best L4H4, excess `1.8105`, layer-3 excess `-0.0029`;
  - seed 8: best L4H6, excess `1.8033`, layer-3 excess `-0.1622`.
- Remaining targets were genuinely layer-3 seeds:
  - seed 1: L3H2, excess `1.9411`;
  - seed 2: L3H4, excess `3.8196`;
  - seed 3: L3H5, excess `2.6416`;
  - seed 7: L3H5, excess `2.9073`;
  - seed 9: L3H5, excess `2.4525`.
- Added `scripts/analyze_local_copy_layer_sweeps.py` and combined the layer
  sweeps in
  `results/phase1_pythia160m_local_copy_layer_sweep_combined/`.
- Ran cross-seed transfer with top local-copy heads from layers 2 and 4:
  - weak targets 4, 5, 6, 8:
    `own_top_excess=1.8528`, `aligned-minus-same=0.3222`, aligned better
    `19/32`;
  - other targets 1, 2, 3, 7, 9:
    `own_top_excess=0.1331`, `aligned-minus-same=0.1816`, aligned better
    `21/40`;
  - combined all targets:
    `own_top_excess=0.8974`, `aligned-minus-same=0.2441`, aligned better
    `40/72`.
- The fixed layers 2+4 rule is not globally better than the layer-3 result
  (`aligned-minus-same=0.6995`). It fixes different seeds while harming the
  original layer-3 seeds.
- Wrote the follow-up memo:
  `doc/experiments/phase1/phase1_pythia160m_local_copy_layer_selection.md`.

## 2026-05-22 - Cross-layer candidate-pool local-copy transfer

- Added `scripts/pythia_local_copy_candidate_pool_alignment.py`.
- Ran candidate-pool transfer on Pythia-160M final checkpoint:
  - seeds: 1-9;
  - target seeds: 1-9;
  - candidate layers: 2, 3, 4;
  - selected heads: top 2 local-copy probe heads per seed across the full
    candidate pool;
  - alignment: Hungarian matching over the full 36-head raw-score candidate
    pool, allowing cross-layer head mappings.
- Result directory:
  `results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2/`.
- Full all-target transfer summary over 72 ordered source-target pairs:
  - own top excess over random: `2.2896`;
  - same-index source transfer: `0.4876`;
  - cross-layer aligned source transfer: `2.2714`;
  - aligned-minus-same: `1.7838`;
  - aligned better count: `66/72`.
- Per-target aligned-minus-same was positive for `9/9` target seeds
  (two-sided sign `p=0.0039`); pair-level aligned-better sign test is about
  `7.3e-14`.
- This is the strongest current local-copy result. It supports the updated
  framing: functional specialization is stable across seeds after role-level
  relabeling, but the raw structural layer/head slot can shift across nearby
  layers.
- Wrote the candidate-pool memo:
  `doc/experiments/phase1/phase1_pythia160m_local_copy_candidate_pool.md`.

## 2026-05-22 - Candidate-pool local-copy checkpoint trajectory

- Ran the cross-layer candidate-pool local-copy experiment at additional
  Pythia-160M checkpoints: `step0`, `step4000`, and `step16000`.
- Added `scripts/analyze_candidate_pool_trajectory.py`.
- Combined trajectory summary:
  `results/phase1_pythia160m_local_copy_candidate_pool_trajectory/`.
- Trajectory:
  - `step0`: own top excess `-0.0004`, aligned-minus-same `-0.0004`, aligned
    better `34/72`;
  - `step4000`: own top excess `0.4339`, aligned-minus-same `0.4191`, aligned
    better `66/72`;
  - `step16000`: own top excess `1.9006`, aligned-minus-same `1.2037`, aligned
    better `66/72`;
  - `step143000`: own top excess `2.2896`, aligned-minus-same `1.7838`,
    aligned better `66/72`.
- Interpretation: local-copy candidate-pool transfer is not an initialization
  artifact. It is already detectable by `step4000` and grows in causal magnitude
  through training.

## 2026-05-22 - Pythia-70M local-copy candidate-pool check

- Ran Pythia-70M final-checkpoint candidate-pool checks with all 9 seeds.
- Layers 1-3 candidate pool:
  `results/phase1_pythia70m_local_copy_candidate_pool_layers1_3_top2/`.
  - own top excess over random: `0.0508`;
  - same-index transfer: `0.1463`;
  - aligned transfer: `0.1115`;
  - aligned-minus-same: `-0.0348`;
  - aligned better count: `35/72`.
- All-layer candidate pool:
  `results/phase1_pythia70m_local_copy_candidate_pool_all_layers_top2/`.
  - own top excess over random: `0.2692`;
  - same-index transfer: `0.1043`;
  - aligned transfer: `0.1854`;
  - aligned-minus-same: `0.0810`;
  - aligned better count: `41/72`.
- Interpretation: unlike Pythia-160M, Pythia-70M does not robustly implement the
  synthetic local-copy causal role. This should be treated as a capacity/task
  caveat rather than a contradiction of the 160M cross-layer alignment result.

## 2026-05-22 - Pythia-410M local-copy candidate-pool check

- Ran Pythia-410M final-checkpoint candidate-pool check with all 9 seeds.
- Candidate pool:
  - layers: 2-6;
  - selected heads: top 2 local-copy probe heads per seed;
  - alignment: Hungarian matching over the full cross-layer raw-score candidate
    pool;
  - transfer pairs: 72 ordered source-target pairs.
- Result directory:
  `results/phase1_pythia410m_local_copy_candidate_pool_layers2_6_top2/`.
- Transfer summary:
  - own top excess over random: `4.1723`;
  - same-index transfer: `0.2562`;
  - cross-layer aligned transfer: `1.9116`;
  - aligned-minus-same: `1.6554`;
  - aligned better count: `49/72`.
- Per-target aligned-minus-same was positive for all 9 target seeds, though seed
  3 was essentially neutral (`0.0004`).
- Interpretation: the local-copy candidate-pool result generalizes upward to
  410M. Combined with the 70M weak result, this suggests a capacity threshold for
  robust synthetic local-copy causal roles.

## 2026-05-22 - Transfer significance summaries

- Added `scripts/analyze_transfer_significance.py`.
- Ran bootstrap confidence intervals and exact two-sided sign tests for:
  - `results/phase1_pythia70m_local_copy_candidate_pool_all_layers_top2/`;
  - `results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2/`;
  - `results/phase1_pythia410m_local_copy_candidate_pool_layers2_6_top2/`.
- Target-level aligned-minus-same summaries:
  - 70M: mean `0.0810`, bootstrap CI `[-0.1332, 0.2989]`, sign `p=0.1797`;
  - 160M: mean `1.7838`, bootstrap CI `[1.3341, 2.3715]`,
    sign `p=0.0039`;
  - 410M: mean `1.6554`, bootstrap CI `[1.0261, 2.2362]`,
    sign `p=0.0039`.
- Pair-level aligned-minus-same summaries:
  - 70M: mean `0.0810`, bootstrap CI `[-0.1165, 0.2619]`, sign `p=0.2888`;
  - 160M: mean `1.7838`, bootstrap CI `[1.5049, 2.0556]`,
    sign `p=7.3e-14`;
  - 410M: mean `1.6554`, bootstrap CI `[1.0527, 2.2303]`,
    sign `p=0.0029`.

## 2026-05-22 - Pythia-410M candidate-pool trajectory

- Ran Pythia-410M candidate-pool checks for `step0`, `step4000`, and
  `step16000` using the same layers 2-6, top-2 setup as the final-checkpoint
  run.
- Result directories:
  - `results/phase1_pythia410m_local_copy_candidate_pool_layers2_6_top2_step0/`;
  - `results/phase1_pythia410m_local_copy_candidate_pool_layers2_6_top2_step4000/`;
  - `results/phase1_pythia410m_local_copy_candidate_pool_layers2_6_top2_step16000/`;
  - combined trajectory:
    `results/phase1_pythia410m_local_copy_candidate_pool_trajectory/`.
- Trajectory:
  - `step0`: own top excess `-0.0009`, aligned-minus-same `-0.0007`,
    aligned better `23/72`;
  - `step4000`: own top excess `1.3363`, aligned-minus-same `1.2062`,
    aligned better `72/72`;
  - `step16000`: own top excess `4.1083`, aligned-minus-same `3.4057`,
    aligned better `71/72`;
  - `step143000`: own top excess `4.1723`, aligned-minus-same `1.6554`,
    aligned better `49/72`.
- Interpretation: 410M transfer is absent at initialization and strongly
  positive at all selected trained checkpoints, but it is not monotonic under the
  fixed layers 2-6 candidate window. The result peaks at `step16000` and becomes
  less pairwise consistent by final checkpoint.

## 2026-05-22 - Naturalistic local-copy probe design

- Wrote `doc/experiments/phase1/naturalistic_local_copy_probe_design.md`.
- The memo proposes a repeated-natural-span task:
  `prefix + span + distractor + span`, scored on next-token prediction in the
  second span occurrence.
- It specifies metrics, initial Pythia-160M settings, decision rules, and an
  implementation plan.
- Reason for writing a design memo before running: the naturalistic branch needs
  corpus/provenance choices, and a tiny ad hoc corpus would risk creating a
  misleading negative or positive result.

## 2026-05-22 - Sleep checkpoint summary

- Wrote `doc/logs/autonomous_sleep/autonomous_sleep_checkpoint_summary_2026-05-22.md`.
- The checkpoint memo summarizes repeat-match, fixed-layer local-copy,
  layer-selection, cross-layer candidate-pool, checkpoint trajectory, and
  model-size results.
- It identifies the next best experiment as a naturalistic local-copy/induction
  probe to test whether the current synthetic `[x, SEP, x]` result generalizes.

## 2026-05-23 - Naturalistic WikiText repeated-span candidate-pool probe

- Added `scripts/pythia_naturalistic_span_candidate_pool_alignment.py`.
- Implemented a standard-dataset repeated-span task using Hugging Face
  `wikitext`, config `wikitext-2-raw-v1`, train split.
- Task construction:
  - `prefix + span + distractor + span`;
  - prefix length 16, span length 12, distractor length 24;
  - attention probe scores attention from the second span occurrence to matching
    first-span positions;
  - causal readout is next-token loss over the second span occurrence.
- Added boundary filtering after the first smoke test:
  - exclude EOS tokens in sampled windows;
  - require the first span token to decode as starting with whitespace/newline.
- Result directories:
  - smoke before boundary filtering:
    `results/debug_pythia14m_naturalistic_span_candidate_pool/`;
  - smoke after boundary filtering:
    `results/debug_pythia14m_naturalistic_span_candidate_pool_v2/`;
  - Pythia-160M 3-seed narrow layers 2-4:
    `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed3/`;
  - Pythia-160M 3-seed all layers:
    `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed3_all_layers/`;
  - Pythia-160M 9-seed all layers:
    `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers/`;
  - Pythia-410M 3-seed narrow layers 2-6:
    `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed3_layers2_6/`;
  - Pythia-410M 3-seed all layers:
    `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed3_all_layers/`;
  - Pythia-410M 9-seed all layers:
    `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed9_all_layers/`.
- Main all-seed results:
  - 160M all layers: own top excess `0.6458`, aligned-minus-same `0.0835`,
    pair CI `[0.0216, 0.1430]`, target CI `[0.0334, 0.1343]`, target positives
    8/9;
  - 410M all layers: own top excess `0.2416`, aligned-minus-same `0.0455`,
    pair CI `[-0.0014, 0.0873]`, target CI `[-0.0190, 0.0894]`, target positives
    8/9.
- Interpretation: naturalistic repeated spans preserve a small positive
  cross-seed role-alignment signal in 160M and a weaker suggestive signal in
  410M. The result is much smaller than synthetic local-copy, so the paper claim
  should present it as external-validity support, not as the primary effect.
- Full memo: `doc/experiments/phase1/phase1_naturalistic_span_candidate_pool.md`.

## 2026-05-23 - Naturalistic 160M larger-sample replication and step0 control

- Reran Pythia-160M all-layer WikiText repeated-span candidate-pool alignment
  with 128 probe sequences and 128 evaluation sequences.
- Result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers_n128/`.
- Result:
  - own top excess over random: `0.6060`;
  - same-index transfer: `-0.0281`;
  - aligned transfer: `0.0534`;
  - aligned-minus-same: `0.0816`;
  - pair CI: `[0.0237, 0.1379]`, pair sign `p=0.0063`;
  - target CI: `[0.0333, 0.1300]`, target sign `p=0.0391`;
  - target positives: 8/9.
- Ran matched `step0` initialization control with the same 128/128 setup.
- Result directory:
  `results/phase1_pythia160m_naturalistic_span_candidate_pool_seed9_all_layers_step0_n128/`.
- Result:
  - own top excess over random: `-0.0005`;
  - same-index transfer: `-0.0006`;
  - aligned transfer: `0.0000`;
  - aligned-minus-same: `0.0007`;
  - pair CI: `[-0.0006, 0.0020]`, pair sign `p=0.2888`;
  - target CI: `[-0.0004, 0.0016]`, target sign `p=0.1797`.
- Interpretation: the small naturalistic 160M effect replicated almost exactly
  after doubling the sample count and is absent at initialization. This supports
  a training-created weak aligned-transfer signal.

## 2026-05-23 - Naturalistic 410M larger-sample replication

- Reran Pythia-410M all-layer WikiText repeated-span candidate-pool alignment
  with 128 probe sequences and 128 evaluation sequences.
- Result directory:
  `results/phase1_pythia410m_naturalistic_span_candidate_pool_seed9_all_layers_n128/`.
- Result:
  - own top excess over random: `0.1809`;
  - same-index transfer: `-0.0046`;
  - aligned transfer: `0.0247`;
  - aligned-minus-same: `0.0293`;
  - pair CI: `[-0.0102, 0.0636]`, pair sign `p=0.0444`;
  - target CI: `[-0.0237, 0.0630]`, target sign `p=0.0391`;
  - target positives: 8/9.
- Target seed 6 remained a stable negative outlier:
  - 64-example run: aligned-minus-same `-0.1853`;
  - 128-example run: aligned-minus-same `-0.1659`.
- Interpretation: the 410M naturalistic result is weak and heterogeneous. The
  positive sign pattern persists, but the larger-sample effect size is smaller
  and bootstrap intervals cross zero. This should not be treated as a strong
  positive replication of the 160M naturalistic result.

## 2026-05-23 - Naturally occurring repeat-ngram probe

- Added `scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`.
- This stricter naturalistic probe scans unmodified WikiText windows for exact
  repeated n-grams instead of inserting a second span.
- Default construction:
  - 96-token windows;
  - exact repeated 4-token n-gram;
  - stride 8;
  - at least 8 intervening tokens beyond the repeated span;
  - skip EOS-containing windows;
  - require first repeated token to start on a whitespace/newline boundary.
- Smoke result:
  `results/debug_pythia14m_natural_repeat_ngram_candidate_pool/`.
- Pythia-160M 3-seed pilot:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed3/`.
  - own top excess: `0.0956`;
  - aligned-minus-same: `0.0321`;
  - target CI for aligned-minus-same: `[-0.0074, 0.0633]`.
- Pythia-160M all-seed final checkpoint:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed9/`.
  - own top excess: `0.1588`;
  - own top excess target CI: `[0.0806, 0.2718]`;
  - own top excess positive for 9/9 targets;
  - same-index transfer: `0.0464`;
  - aligned transfer: `0.0448`;
  - aligned-minus-same: `-0.0016`;
  - target CI for aligned-minus-same: `[-0.0548, 0.0360]`;
  - target positives: 6/9.
- Pythia-160M all-seed `step0` control:
  `results/phase1_pythia160m_natural_repeat_ngram_candidate_pool_seed9_step0/`.
  - own top excess: `-0.0001`;
  - aligned-minus-same: `-0.0004`;
  - target CI for aligned-minus-same: `[-0.0030, 0.0019]`.
- Interpretation: naturally occurring exact repeats show trained own-head causal
  importance, absent at initialization, but generic Phase 0 alignment does not
  show an aligned-transfer advantage over same-index transfer.
- Added `--alignment-source task_repeat` to
  `scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`.
- Reran Pythia-160M all-seed final checkpoint with task-specific alignment:
  `results/phase1_pythia160m_natural_repeat_ngram_task_alignment_seed9/`.
  - own top excess: `0.1588`;
  - same-index transfer: `0.0464`;
  - task-repeat aligned transfer: `0.2361`;
  - aligned-minus-same: `0.1897`;
  - pair CI: `[0.0908, 0.2843]`, pair sign `p=7.3e-14`;
  - target CI: `[0.0737, 0.3140]`, target sign `p=0.0391`;
  - aligned-minus-same positive for 8/9 targets;
  - aligned better in 66/72 source-target pairs.
- Reran matched `step0` task-specific alignment control:
  `results/phase1_pythia160m_natural_repeat_ngram_task_alignment_seed9_step0/`.
  - own top excess: `-0.0001`;
  - task-repeat aligned-minus-same: `-0.0033`;
  - target CI: `[-0.0060, -0.0006]`.
- Revised interpretation: the stricter natural-repeat task supports cross-seed
  role transfer only when the alignment basis is role-specific. Generic
  attention-score matching can miss a real but weak functional role.
- Full memo: `doc/experiments/phase1/phase1_natural_repeat_ngram_candidate_pool.md`.

## 2026-05-23 - Inserted natural-span task-specific alignment

- Added `--alignment-source task_span` to
  `scripts/pythia_naturalistic_span_candidate_pool_alignment.py`.
- This aligns heads using repeated-span attention vectors from the probe split
  instead of generic Phase 0 texts; causal readout remains on held-out evaluation
  spans.
- Pythia-160M all-seed final checkpoint:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers/`.
  - own top excess: `0.6458`;
  - same-index transfer: `-0.0170`;
  - task-span aligned transfer: `0.5475`;
  - aligned-minus-same: `0.5645`;
  - pair CI: `[0.4501, 0.6855]`, pair sign `p=4.6e-16`;
  - target CI: `[0.3653, 0.8068]`, target sign `p=0.0039`;
  - target positives: 9/9;
  - aligned better count: 68/72.
- Matched Pythia-160M `step0` task-span alignment control:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers_step0/`.
  - own top excess: `0.0002`;
  - aligned-minus-same: `0.0003`;
  - target CI: `[-0.0008, 0.0015]`.
- Pythia-160M 128/128 task-span alignment replication:
  `results/phase1_pythia160m_naturalistic_span_task_alignment_seed9_all_layers_n128/`.
  - own top excess: `0.6060`;
  - task-span aligned transfer: `0.4492`;
  - aligned-minus-same: `0.4773`;
  - pair CI: `[0.3682, 0.5899]`;
  - target CI: `[0.2829, 0.6852]`;
  - target positives: 8/9.
- Interpretation: the inserted WikiText repeated-span task has a much stronger
  cross-seed transfer signal when alignment is computed on role-specific
  repeated-span attention vectors. Generic Phase 0 alignment underestimates this
  weak natural role.
- Pythia-410M all-seed final checkpoint with task-span alignment:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers/`.
  - own top excess: `0.2416`;
  - same-index transfer: `0.0007`;
  - task-span aligned transfer: `0.1551`;
  - aligned-minus-same: `0.1544`;
  - pair CI: `[0.0939, 0.2075]`, pair sign `p=2.6e-05`;
  - target CI: `[0.0430, 0.2460]`, target sign `p=0.0391`;
  - target positives: 8/9;
  - aligned better count: 54/72.
- Matched Pythia-410M `step0` task-span alignment control:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers_step0/`.
  - own top excess: `-0.0008`;
  - aligned-minus-same: `-0.0003`;
  - target CI: `[-0.0005, 0.0000]`.
- Pythia-410M 128/128 task-span alignment replication:
  `results/phase1_pythia410m_naturalistic_span_task_alignment_seed9_all_layers_n128/`.
  - own top excess: `0.1809`;
  - task-span aligned transfer: `0.1112`;
  - aligned-minus-same: `0.1158`;
  - pair CI: `[0.0650, 0.1604]`;
  - target CI: `[0.0222, 0.1884]`;
  - target positives: 8/9.
- Revised 410M interpretation: 410M was weak under generic Phase 0 matching, but
  role-specific task-span alignment recovers a positive naturalistic transfer
  effect. Seed 6 remains a negative outlier, so the 410M effect is more
  heterogeneous than 160M.

## 2026-05-23 - Alignment-basis summary

- Wrote `doc/experiments/phase1/phase1_alignment_basis_summary.md`.
- Consolidated the emerging methodological result:
  - synthetic local-copy is high-signal and transfers under generic Phase 0
    alignment;
  - inserted natural repeated spans and naturally occurring exact repeats are
    weaker roles whose cross-seed transfer is strongly underestimated by generic
    alignment;
  - task-specific alignment over repeated-position attention vectors recovers
    held-out transfer for both inserted repeated spans and exact natural repeats.
- Updated framing:
  `alignment representation is part of the measured phenomenon, not only a
  technical detail`.

## 2026-05-23 - 410M naturally occurring repeat-ngram check

- Ran Pythia-410M all-seed naturally occurring exact-repeat task-specific
  alignment:
  `results/phase1_pythia410m_natural_repeat_ngram_task_alignment_seed9/`.
- Result:
  - own top excess: `0.0503`;
  - own top excess target CI: `[0.0068, 0.0873]`;
  - own top excess positives: 8/9;
  - same-index transfer: `0.0173`;
  - task-repeat aligned transfer: `0.0388`;
  - aligned-minus-same: `0.0215`;
  - pair CI: `[-0.0120, 0.0461]`, pair sign `p=0.0005`;
  - target CI: `[-0.0166, 0.0564]`, target sign `p=1.0`;
  - target positives: 5/9.
- Ran matched `step0` control:
  `results/phase1_pythia410m_natural_repeat_ngram_task_alignment_seed9_step0/`.
  - own top excess: `-0.0013`;
  - aligned-minus-same: `-0.0022`;
  - target CI: `[-0.0042, -0.0002]`.
- Interpretation: 410M naturally occurring exact repeats show a weak
  training-created own-head causal signal, but not a clean target-level
  aligned-transfer effect. This is weaker than the 160M exact-repeat result.

## 2026-05-23 - Synthetic local-copy task-specific alignment

- Added `--alignment-source task_local_copy` to
  `scripts/pythia_local_copy_candidate_pool_alignment.py`.
- The new alignment source matches heads using attention to the local-copy
  repeated-value positions on the probe split. Causal loss is still evaluated on
  held-out synthetic sequences.
- Smoke-tested the path with Pythia-14M seeds 1-2:
  `results/debug_pythia14m_local_copy_task_alignment/`.
- Ran Pythia-160M all-seed final checkpoint:
  `results/phase1_pythia160m_local_copy_task_alignment_layers2_4_top2/`.
  - own top excess: `2.2896`;
  - same-index transfer: `0.4876`;
  - task-local aligned transfer: `2.4469`;
  - aligned-minus-same: `1.9593`;
  - pair CI: `[1.6902, 2.2171]`, pair sign `p=1.2e-14`;
  - target CI: `[1.5948, 2.5006]`, target sign `p=0.0039`;
  - target positives: 9/9;
  - aligned better count: 66/72.
- Ran matched Pythia-160M `step0` control:
  `results/phase1_pythia160m_local_copy_task_alignment_layers2_4_top2_step0/`.
  - own top excess: `-0.0004`;
  - aligned-minus-same: `0.0000`;
  - target CI: `[-0.0004, 0.0003]`.
- Ran Pythia-410M all-seed final checkpoint:
  `results/phase1_pythia410m_local_copy_task_alignment_layers2_6_top2/`.
  - own top excess: `4.1723`;
  - same-index transfer: `0.2562`;
  - task-local aligned transfer: `4.0299`;
  - aligned-minus-same: `3.7737`;
  - pair CI: `[3.2483, 4.2896]`, pair sign `p=7.3e-14`;
  - target CI: `[2.4658, 4.8657]`, target sign `p=0.0391`;
  - target positives: 8/9;
  - aligned better count: 66/72.
- Ran matched Pythia-410M `step0` control:
  `results/phase1_pythia410m_local_copy_task_alignment_layers2_6_top2_step0/`.
  - own top excess: `-0.0009`;
  - aligned-minus-same: `-0.0004`;
  - target CI: `[-0.0007, -0.0001]` at negligible absolute scale.
- Interpretation: synthetic local-copy remains the high-signal upper-bound
  task. Generic Phase 0 alignment is already strong, but task-local alignment
  recovers even stronger 410M final-checkpoint transfer, suggesting the generic
  matching representation can miss part of the role-level relabeling even when
  the role is causally large.

## 2026-05-23 - WikiText-103 exact 8-gram repeat check

- Scanned a larger WikiText-103 token stream to support longer naturally
  occurring exact repeats.
- Candidate-count scan on 500024 tokens from `wikitext-103-raw-v1`, first 20000
  train rows, context length 128, min gap 8:
  - exact 5-gram candidates: 3839;
  - exact 6-gram candidates: 2049;
  - exact 7-gram candidates: 980;
  - exact 8-gram candidates: 524.
- Ran Pythia-160M all-seed exact 8-gram task-repeat alignment with 128 probe and
  128 evaluation windows, sampled without replacement:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128/`.
  - own top excess: `0.3718`;
  - own top target CI: `[0.1494, 0.6444]`;
  - own top positives: 9/9;
  - same-index transfer: `0.0334`;
  - task-repeat aligned transfer: `0.3155`;
  - aligned-minus-same: `0.2820`;
  - pair CI: `[0.1822, 0.3918]`, pair sign `p=4.2e-11`;
  - target CI: `[0.0995, 0.5164]`, target sign `p=0.0391`;
  - target positives: 8/9;
  - aligned better count: 63/72.
- Ran matched Pythia-160M `step0` task-repeat control:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128_step0/`.
  - own top excess: `-0.0010`;
  - aligned-minus-same: `-0.0018`;
  - target CI: `[-0.0035, -0.0001]` at negligible absolute scale.
- Ran Pythia-160M all-seed exact 8-gram generic Phase 0 comparison:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_phase0_alignment_seed9_n128/`.
  - own top excess: `0.3718`;
  - same-index transfer: `0.0334`;
  - generic aligned transfer: `0.0397`;
  - aligned-minus-same: `0.0063`;
  - pair CI: `[-0.0424, 0.0426]`;
  - target CI: `[-0.0253, 0.0344]`;
  - target positives: 5/9.
- Interpretation: longer exact repeats on a larger standard corpus strengthen
  the 160M natural-repeat result. The role is causally stronger than in the
  WikiText-2 4-gram task, but generic Phase 0 alignment remains neutral; the
  positive transfer appears under task-repeat matching.

## 2026-05-23 - 410M WikiText-103 exact 8-gram repeat check

- Ran Pythia-410M all-seed exact 8-gram task-repeat alignment with 128 probe and
  128 evaluation windows, sampled without replacement:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128/`.
  - repeat candidate count: 491;
  - own top excess: `0.0580`;
  - own top target CI: `[0.0059, 0.1088]`;
  - own top positives: 8/9;
  - same-index transfer: `0.0193`;
  - task-repeat aligned transfer: `0.0571`;
  - aligned-minus-same: `0.0378`;
  - pair CI: `[0.0065, 0.0612]`, pair sign `p=6.5e-07`;
  - target CI: `[-0.0042, 0.0708]`, target sign `p=0.0391`;
  - target positives: 8/9;
  - aligned better count: 57/72.
- Ran matched Pythia-410M `step0` task-repeat control:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128_step0/`.
  - own top excess: `0.0004`;
  - aligned-minus-same: `-0.0002`;
  - target CI: `[-0.0019, 0.0013]`.
- Ran Pythia-410M all-seed exact 8-gram generic Phase 0 comparison:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_phase0_alignment_seed9_n128/`.
  - own top excess: `0.0580`;
  - same-index transfer: `0.0193`;
  - generic aligned transfer: `0.0171`;
  - aligned-minus-same: `-0.0022`;
  - pair CI: `[-0.0303, 0.0183]`;
  - target CI: `[-0.0333, 0.0196]`;
  - target positives: 6/9.
- Interpretation: moving to WikiText-103 exact 8-grams improves 410M relative
  to the earlier WikiText-2 exact 4-gram task, but 410M still remains much
  weaker and more heterogeneous than 160M. Generic Phase 0 alignment is neutral;
  task-repeat alignment is directionally positive with target-level uncertainty.

## 2026-05-23 - Natural-repeat heterogeneity inspection

- Wrote `doc/experiments/phase1/phase1_natural_repeat_heterogeneity.md`.
- Main finding: the weak 410M exact-repeat result is partly driven by
  same-index outliers and weak target own-head causality, not simply by
  aligned heads failing.
- Most important outlier:
  - 410M exact 8-gram, target seed 6, source seed 3:
    same-index transfer `0.8055`, task-repeat aligned transfer `0.0599`,
    aligned-minus-same `-0.7456`;
  - the same pattern appears in the exact 4-gram run:
    same-index transfer `0.9588`, task-repeat aligned transfer `0.0577`,
    aligned-minus-same `-0.9011`.
- Target seed 4 has negative own-head excess on 410M exact 8-grams (`-0.0958`),
  so there is little causal role for cross-seed transfer to recover.
- Interpretation: report aligned transfer and same-index transfer separately.
  Aligned-minus-same is useful, but can understate role transfer when a raw
  same-index source seed is unusually good.

## 2026-05-23 - Natural-repeat semantic category counts

- Added `scripts/analyze_natural_repeat_categories.py`.
- The script can classify either a written `example_rows.csv` preview or
  reconstruct the full deterministic evaluation set from a result directory.
- Full evaluation-set counts:
  - 160M WikiText-2 exact 4-gram, 64 eval examples:
    ordinary phrase 27, tokenizer markup 17, proper-name-like 8,
    numeric/date 8, quoted/title 3, all-caps/initialism 1;
  - 160M WikiText-103 exact 8-gram, 128 eval examples:
    ordinary phrase 35, tokenizer markup 32, proper-name-like 25,
    numeric/date 20, quoted/title 16;
  - 410M WikiText-103 exact 8-gram, 128 eval examples:
    tokenizer markup 35, ordinary phrase 29, numeric/date 25,
    quoted/title 21, proper-name-like 18.
- Interpretation: natural exact repeats are semantically mixed. The next
  filtering experiment should stratify by category or baseline predictability
  before making model-size claims.

## 2026-05-23 - Ordinary-phrase exact 8-gram filtered check

- Added `--span-primary-category` to
  `scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`.
- Ran Pythia-160M WikiText-103 exact 8-gram ordinary-phrase task-repeat
  alignment:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`.
  - token stream length: `1000066`;
  - ordinary-phrase candidate windows: `147`;
  - sampled 64 probe plus 64 evaluation windows, without replacement;
  - own top excess: `0.3133`;
  - same-index transfer: `0.0248`;
  - task-repeat aligned transfer: `0.2500`;
  - aligned-minus-same: `0.2252`;
  - pair CI: `[0.1510, 0.3070]`, pair sign `p=4.6e-16`;
  - target CI: `[0.1096, 0.3776]`, target sign `p=0.0391`;
  - target positives: 8/9;
  - aligned better count: 68/72.
- Ran matched `step0` task-repeat control:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0/`.
  - own top excess: `0.0009`;
  - aligned-minus-same: `0.0012`;
  - target CI: `[-0.0005, 0.0028]`.
- Ran generic Phase 0 comparison:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_phase0_alignment_seed9_n64/`.
  - own top excess: `0.3133`;
  - generic aligned transfer: `0.0384`;
  - aligned-minus-same: `0.0137`;
  - target CI: `[-0.0044, 0.0305]`.
- Interpretation: the ordinary-phrase filtered task preserves the 160M
  task-specific transfer result and keeps generic Phase 0 matching neutral.
- Ran Pythia-410M WikiText-103 exact 8-gram ordinary-phrase task-repeat
  alignment:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`.
  - token stream length: `1000066`;
  - ordinary-phrase candidate windows: `140`;
  - sampled 64 probe plus 64 evaluation windows, without replacement;
  - own top excess: `0.0559`;
  - own top target CI: `[0.0189, 0.0949]`;
  - same-index transfer: `0.0102`;
  - task-repeat aligned transfer: `0.0429`;
  - aligned-minus-same: `0.0327`;
  - pair CI: `[0.0112, 0.0506]`, pair sign `p=2.4e-06`;
  - target CI: `[0.0027, 0.0599]`, target sign `p=0.0391`;
  - target positives: 8/9;
  - aligned better count: 56/72.
- Ran matched Pythia-410M `step0` task-repeat control:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step0/`.
  - own top excess: `0.0010`;
  - aligned-minus-same: `0.0009`;
  - target CI: `[-0.0003, 0.0022]`.
- Ran Pythia-410M generic Phase 0 comparison:
  `results/phase1_pythia410m_wikitext103_natural_repeat_8gram_ordinary_phase0_alignment_seed9_n64/`.
  - generic aligned transfer: `0.0124`;
  - aligned-minus-same: `0.0022`;
  - pair CI: `[-0.0164, 0.0169]`;
  - target CI: `[-0.0191, 0.0182]`.
- Interpretation update: category filtering gives the first clean target-level
  positive 410M exact-repeat transfer result, though the effect remains small
  compared with 160M. Generic Phase 0 alignment remains neutral.

## 2026-05-23 - Phase 1 paper-facing claims memo

- Wrote `doc/experiments/phase1/phase1_paper_claims_and_methods.md`.
- Consolidated the current Phase 1 claim:
  functional repeat/copy roles are stable across seeds after role-level
  relabeling, but weak natural roles require role-specific alignment features.
- The memo defines the two-stage metric:
  - generic Phase 0 alignment as the task-agnostic baseline;
  - role-specific alignment as the held-out role-stability measurement.
- It also records the main caveat: these results establish functional
  specialization and cross-seed role stability, not functional modularity.

## 2026-05-23 - Phase 3 structural-to-functional synthesis

- Wrote `doc/experiments/phase3/phase3_structural_to_functional_synthesis.md`.
- Consolidated the current answer to the user's reframed research question:

```text
Structural heterogeneity can stabilize functional specialization slots.
Structural branch/routing design can support functional modularity, but the
current toy evidence shows reliable functional modularity only when
role-informative routing pressure is supplied long enough for causal branch
computations to consolidate.
```

- Important negative boundary:
  separate branches, unconstrained learned routing, entropy/load-balancing
  regularizers, branch bottlenecks, and conflict-heavy lookup did not reliably
  produce spontaneous role-aligned causal modularity.
- Important positive boundary:
  oracle routing and weak scored-position routing labels did produce
  branch-level functional modularity.
- Next narrow experiment selected: rerun the weak-token-router `end800`
  trajectory with denser evaluation between steps 400 and 800 to locate the
  causal consolidation window after gate alignment.

## 2026-05-23 - Dense router consolidation-window trajectory

- Added `scripts/analyze_router_consolidation_window.py`.
- Ran the weak-token-router conflict-task trajectory with supervision active
  through step 800 and dense checkpoints:
  `0, 400, 450, 500, 550, 600, 650, 700, 750, 800`.
- Output:
  `results/phase3_toy_trajectory_consolidation_end800/`.
- Analysis:
  `results/phase3_toy_trajectory_consolidation_end800_analysis/`.
- Wrote `doc/experiments/phase3/phase3_toy_router_consolidation_window.md`.
- Main result:
  - first solved checkpoint with gate routed-role match 5/5: step 400;
  - first solved checkpoint with causal routed-role match 5/5: step 550;
  - first solved checkpoint with branch distance >= 0.30: step 600;
  - first solved checkpoint with branch distance >= 0.40: step 750;
  - final step 800 branch distance: `0.4996`.
- Interpretation: role-aligned routing gates precede causal branch modularity by
  roughly 150 optimizer steps in this setup, and separation strength continues
  growing after the top-branch split appears.

## 2026-05-23 - Pythia-160M ordinary natural-repeat checkpoint trajectory

- Added `scripts/analyze_natural_repeat_checkpoint_trajectory.py`.
- Ran Pythia-160M seeds 1-9 on the WikiText-103 ordinary-phrase exact 8-gram
  task-repeat alignment setup at `step4000`, `step16000`, and `step64000`.
- Combined these with the existing matched `step0` control and final
  `step143000` run in:
  `results/phase1_pythia160m_wikitext103_ordinary_repeat_trajectory/`.
- Wrote `doc/experiments/phase1/phase1_pythia160m_ordinary_repeat_checkpoint_trajectory.md`.
- Main trajectory:
  - step0: probe `0.0077`, own excess `0.0009`, aligned-minus-same `0.0012`;
  - step4000: probe `0.1115`, own excess `0.0205`, aligned-minus-same `0.0154`;
  - step16000: probe `0.1481`, own excess `0.1756`, aligned-minus-same `0.0460`;
  - step64000: probe `0.1576`, own excess `0.1693`,
    aligned-minus-same `0.1174`, target CI `[0.0387, 0.2099]`;
  - step143000: probe `0.1623`, own excess `0.3133`,
    aligned-minus-same `0.2252`, target CI `[0.1096, 0.3776]`.
- Interpretation: ordinary natural-repeat specialization appears early,
  own-head causal importance becomes clear by step16000, and robust target-level
  aligned transfer appears later, by step64000.

## 2026-05-23 - SwitchHead feasibility check

- Verified that the roadmap's earlier `github.com/RobertCsordas/moe` SwitchHead
  target is stale/wrong for this project. That repo is for a related MoE-MLP
  paper, not SwitchHead attention.
- Cloned ignored local checkouts:
  - `.tools/moe_attention/`, commit `7169ad3`;
  - `.tools/switchhead/`, commit `0bb2f61`;
  - `.tools/csordas_moe/`, commit `6b175aa`, retained only as the stale-target
    comparison.
- Confirmed `switchhead` imports locally and a GPU smoke forward pass through
  `SwitchHeadRope` succeeds with output shape `(2, 16, 32)`.
- CPU smoke fails because the SwitchHead implementation uses Triton CVMM kernels
  that require GPU tensors.
- Wrote `doc/side_branches/switchhead/switchhead_followup_feasibility.md`.
- Updated `doc/plan.md` and `doc/research_questions.md` to use
  `RobertCsordas/moe_attention` for training code and `RobertCsordas/switchhead`
  for the first local plug-in experiment.

## 2026-05-23 - SwitchHead toy competition pilot

- Added `scripts/toy_switchhead_competition.py`.
- Smoke-tested expert ablation and fixed an indexing bug where advanced
  indexing zeroed a copy rather than the original expert rows.
- Ran a 5-seed one-layer SwitchHead pilot:
  `results/phase3_toy_switchhead_competition_seed5_steps2000/`.
- Setup:
  - `SwitchHeadRope`;
  - one layer;
  - two heads;
  - two experts;
  - `moe_k=1`;
  - conflict-heavy `bidirectional_lookup`;
  - 2000 training steps.
- Result:
  - local accuracy: `1.0000`;
  - induction accuracy: `1.0000`;
  - gate same top expert: `1.00`;
  - causal same top expert: `0.80`;
  - routed expert match: `0.20`;
  - gate distribution distance: `0.0032`;
  - causal expert distribution distance: `0.0087`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_competition.md`.
- Interpretation: this SwitchHead pilot solves the task but does not
  spontaneously separate local and induction into different experts.
- Ran a 4-expert, `moe_k=2` variant:
  `results/phase3_toy_switchhead_competition_seed5_e4k2_steps2000/`.
- Result:
  - local accuracy: `1.0000`;
  - induction accuracy: `1.0000`;
  - gate same top expert: `0.80`;
  - causal same top expert: `1.00`;
  - routed expert match: `0.00`;
  - gate distribution distance: `0.0083`;
  - causal expert distribution distance: `0.0486`;
  - local/induction top expert loss deltas: about `0.024`.
- Interpretation update: adding more active experts produced redundancy rather
  than role separation.
- Added weak expert-selection supervision to
  `scripts/toy_switchhead_competition.py`.
- Ran a 5-seed weak role-informed SwitchHead run:
  `results/phase3_toy_switchhead_competition_weak_w005_seed5_steps2000/`.
- Setup:
  - same one-layer, two-head, two-expert SwitchHead model;
  - `moe_k=1`;
  - `expert_supervision_weight=0.05`;
  - auxiliary selector loss active for the full 2000-step run.
- Result:
  - local accuracy: `1.0000`;
  - induction accuracy: `1.0000`;
  - gate same top expert: `0.00`;
  - causal same top expert: `0.00`;
  - routed expert match: `1.00`;
  - gate distribution distance: `0.9982`;
  - causal expert distribution distance: `0.5675`;
  - local/induction top expert loss deltas: about `7.0`.
- Interpretation update: weak role-informative selector pressure is sufficient
  to make SwitchHead attention experts become role-aligned causal modules in
  this toy setup. This is a positive result for induced modularity, but not a
  spontaneous-modularity result because the auxiliary loss remains active.
- Ran the same weak-selector setup with transient pressure ending after step 800:
  `results/phase3_toy_switchhead_competition_weak_w005_end800_seed5_steps2000/`.
- Result:
  - local accuracy: `1.0000`;
  - induction accuracy: `1.0000`;
  - gate same top expert: `0.00`;
  - causal same top expert: `0.00`;
  - routed expert match: `1.00`;
  - gate distribution distance: `0.9645`;
  - causal expert distribution distance: `0.5664`;
  - local/induction top expert loss deltas: about `7.0`.
- Interpretation update: SwitchHead's induced role-aligned expert split persists
  after the auxiliary selector loss is removed for the final 1200 training
  steps. This makes the result stronger than active-loss compliance.
- Ran a selector-window sweep with end steps 400, 425, 450, 500, 600, and 800,
  plus the no-supervision and full-supervision baselines.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_selector_window_sweep.md`.
- Main sweep result:
  - no supervision: routed expert match `0.20`, gate distance `0.0032`, causal
    distance `0.0087`;
  - end step 400: routed expert match `0.80`, gate distance `0.3030`, causal
    distance `0.3290`;
  - end step 425: gate same-top `0.00`, but routed expert match still `0.80`;
  - end step 450: routed expert match `1.00`, gate distance `0.4476`, causal
    distance `0.4240`;
  - end step 800: routed expert match `1.00`, gate distance `0.9645`, causal
    distance `0.5664`.
- Interpretation update: in this toy setup, gate separation can precede fully
  reliable causal expert modularity; persistent 5/5 causal modularity appears
  between selector end steps 425 and 450.

## 2026-05-23 - SwitchHead checkpoint trajectory

- Added `--trajectory-eval-steps` to `scripts/toy_switchhead_competition.py`.
- During implementation, a smoke test exposed a SwitchHead RoPE cache issue:
  trajectory evaluation under `torch.inference_mode()` can cache inference
  tensors and break subsequent training. The script now uses `torch.no_grad()`
  in evaluation paths so training can safely resume after checkpoint analysis.
- Ran a 5-seed trajectory with `expert_supervision_weight=0.05` and
  `expert_supervision_end_step=450`:
  `results/phase3_toy_switchhead_trajectory_w005_end450_seed5_steps2000/`.
- Checkpoints:
  `0, 100, 200, 300, 400, 425, 450, 500, 600, 800, 1000, 1500, 2000`.
- Milestones:
  - meaningful reliable gate split: checkpoint 425;
  - causal same-top expert `0/5`: checkpoint 500;
  - routed expert match `5/5`: checkpoint 500;
  - mean local and induction accuracy both `1.0`: checkpoint 1500.
- Boundary seed: seed 4 has gate split at checkpoints 425 and 450, but the
  causal local role does not move to expert 0 until checkpoint 500.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_checkpoint_trajectory.md`.
- Interpretation update: direct within-run evidence supports the ordering
  `gate specialization -> causal functional modularity`.

## 2026-05-23 - SwitchHead selector-weight sweep

- Ran a fixed-window selector-weight sweep with
  `expert_supervision_end_step=450`.
- Tested weights `0.02`, `0.03`, `0.04`, `0.045`, with the earlier no-supervision
  and `0.05` runs as baselines.
- Result:
  - weight `0.00`: routed expert match `0.20`, gate distance `0.0032`, causal
    distance `0.0087`;
  - weight `0.02`: routed expert match `0.80`, gate distance `0.0443`, causal
    distance `0.1060`;
  - weight `0.03`: routed expert match `0.80`, gate distance `0.1068`, causal
    distance `0.1946`;
  - weight `0.04`: routed expert match `0.80`, gate distance `0.2571`, causal
    distance `0.2888`;
  - weight `0.045`: routed expert match `0.80`, gate distance `0.3483`, causal
    distance `0.3460`;
  - weight `0.05`: routed expert match `1.00`, gate distance `0.4476`, causal
    distance `0.4240`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_weight_sweep.md`.
- Interpretation update: at the shortest reliable 450-step window, the first
  tested selector weight that reliably induces 5/5 causal expert modularity is
  `0.05`. Smaller weights still influence routing but do not cross the
  reliability threshold in this seed set.

## 2026-05-23 - SwitchHead strength-duration tradeoff

- Ran longer-window lower-weight selector runs:
  - `expert_supervision_weight=0.02`, `expert_supervision_end_step=800`;
  - `expert_supervision_weight=0.025`, `expert_supervision_end_step=800`;
  - `expert_supervision_weight=0.03`, `expert_supervision_end_step=800`.
- Results:
  - weight `0.02`, end step 800: routed expert match `0.80`, gate distance
    `0.3864`, causal distance `0.3405`;
  - weight `0.025`, end step 800: routed expert match `1.00`, gate distance
    `0.5823`, causal distance `0.4834`;
  - weight `0.03`, end step 800: routed expert match `1.00`, gate distance
    `0.7632`, causal distance `0.5528`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_strength_duration_tradeoff.md`.
- Interpretation update: selector duration can compensate for selector strength.
  The reliable weight boundary moves from between `0.045` and `0.05` at end step
  450 to between `0.02` and `0.025` at end step 800.

## 2026-05-23 - SwitchHead expert-label control

- Parameterized selector targets in `scripts/toy_switchhead_competition.py` with
  `--local-target-expert` and `--induction-target-expert`.
- Ran reversed target controls:
  - local -> expert 1;
  - induction -> expert 0.
- Results:
  - reversed target, weight `0.05`, end step 450: routed expert match `0.80`,
    gate distance `0.3936`, causal distance `0.4014`;
  - reversed target, weight `0.05`, end step 800: routed expert match `1.00`,
    gate distance `0.9646`, causal distance `0.5609`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_label_control.md`.
- Interpretation update: under sufficient cue duration, the causal roles follow
  the requested expert labels. The 450-step asymmetry cautions that the exact
  threshold is optimization- and label-assignment-sensitive.
- Updated `doc/experiments/phase3/phase3_structural_to_functional_synthesis.md` to incorporate the
  SwitchHead spontaneous negative result, induced positive result,
  gate-before-causality trajectory, strength-duration threshold, and
  reversed-label control.

## 2026-05-23 - Two-layer SwitchHead follow-up

- Fixed `scripts/toy_switchhead_competition.py` so `expert_scores.csv` reports
  gate metrics per layer rather than duplicating an across-layer average.
- Reran authoritative two-layer SwitchHead conditions:
  - spontaneous:
    `results/phase3_toy_switchhead_2layer_spontaneous_seed5_steps2000_v2/`;
  - induced:
    `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_v2/`.
- Spontaneous result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - gate same-top expert `1.00`;
  - causal same-top expert `0.80`;
  - routed expert match `0.20`;
  - gate distance `0.0017`;
  - causal distance `0.1617`.
- Induced result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - gate same-top expert `0.00`;
  - causal same-top expert `0.00`;
  - routed expert match `1.00`;
  - gate distance `0.7066`;
  - causal distance `0.6148`.
- Layer localization:
  - spontaneous local top was `L1E0` in 5/5 seeds; induction top was `L1E0` in
    4/5 seeds;
  - induced local top was `L1E0` in 5/5 seeds; induction top was `L1E1` in 5/5
    seeds.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_two_layer.md`.
- Interpretation update: extra SwitchHead depth does not create spontaneous role
  modularity, but induced causal expert modularity survives and localizes to the
  later layer.

## 2026-05-23 - Layer-specific SwitchHead supervision

- Added `--expert-supervision-layers` to
  `scripts/toy_switchhead_competition.py`.
- Ran two-layer layer-specific induced conditions:
  - layer 0 only:
    `results/phase3_toy_switchhead_2layer_supervise_l0_w005_end800_seed5_steps2000/`;
  - layer 1 only:
    `results/phase3_toy_switchhead_2layer_supervise_l1_w005_end800_seed5_steps2000/`.
- Results:
  - layer 0 only: gate same-top `0.00`, causal same-top `0.60`, routed match
    `0.40`, gate distance `0.4862`, causal distance `0.2499`;
  - layer 1 only: gate same-top `0.20`, causal same-top `0.20`, routed match
    `0.80`, gate distance `0.4155`, causal distance `0.5791`;
  - both layers: gate same-top `0.00`, causal same-top `0.00`, routed match
    `1.00`, gate distance `0.7066`, causal distance `0.6148`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_layer_specific_supervision.md`.
- Interpretation update: upstream gate splitting alone is not enough. Direct
  role-aligned pressure on the later causal layer is much closer to sufficient,
  but both-layer pressure is the robust condition in this setup.
- Updated `doc/experiments/phase3/phase3_structural_to_functional_synthesis.md` with the two-layer
  and layer-specific SwitchHead results.

## 2026-05-23 - SwitchHead selector-type control

- Added `--expert-supervision-selector output|value|both` to
  `scripts/toy_switchhead_competition.py`.
- Added value-selector metrics to `model_summary.csv` and `expert_scores.csv`.
- Ran one-layer selector controls at weight `0.05`, end step `800`:
  - output only:
    `results/phase3_toy_switchhead_selector_output_w005_end800_seed5_steps2000/`;
  - value only:
    `results/phase3_toy_switchhead_selector_value_w005_end800_seed5_steps2000/`;
  - both:
    `results/phase3_toy_switchhead_selector_both_w005_end800_seed5_steps2000/`.
- Results:
  - output only: local acc `1.0000`, induction acc `1.0000`, routed match
    `1.00`, output gate distance `0.9645`, value gate distance `0.0134`, causal
    distance `0.5663`;
  - value only: local acc `0.9752`, induction acc `1.0000`, routed match `0.00`,
    output gate distance `0.0049`, value gate distance `0.9299`, causal distance
    `0.0667`;
  - both: local acc `0.9506`, induction acc `1.0000`, routed match `1.00`,
    output gate distance `0.6978`, value gate distance `0.7860`, causal distance
    `0.6327`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_selector_type.md`.
- Interpretation update: output selector pressure is the clean sufficient path.
  Value selector pressure alone can split value routing but does not produce
  causal role modularity; adding it to output pressure can hurt optimization.

## 2026-05-23 - Two-layer SwitchHead selector-type control

- Ran selector-type controls with supervision restricted to layer 1 in a
  two-layer SwitchHead model:
  - output only:
    `results/phase3_toy_switchhead_2layer_selector_output_l1_w005_end800_seed5_steps2000/`;
  - value only:
    `results/phase3_toy_switchhead_2layer_selector_value_l1_w005_end800_seed5_steps2000/`;
  - both:
    `results/phase3_toy_switchhead_2layer_selector_both_l1_w005_end800_seed5_steps2000/`.
- Results:
  - output only: local acc `1.0000`, induction acc `1.0000`, routed match
    `1.00`, output gate distance `0.4301`, value gate distance `0.0063`, causal
    distance `0.6143`;
  - value only: local acc `1.0000`, induction acc `1.0000`, routed match `0.80`,
    output gate distance `0.0046`, value gate distance `0.4197`, causal distance
    `0.5522`;
  - both: local acc `1.0000`, induction acc `1.0000`, routed match `0.80`,
    output gate distance `0.2764`, value gate distance `0.3518`, causal distance
    `0.7956`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_two_layer_selector_type.md`.
- Interpretation update: value selector pressure is not null at the causal layer
  in a two-layer model, but output selector pressure remains the cleaner
  sufficient cue.

## 2026-05-23 - SwitchHead expanded-seed robustness

- Ran seeds 6-10 for the one-layer output-selector induced condition:
  `results/phase3_toy_switchhead_selector_output_w005_end800_seed6_10_steps2000/`.
- Result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - routed expert match `1.00`;
  - output gate distance `0.9609`;
  - value gate distance `0.0194`;
  - causal expert distance `0.5721`.
- Ran seeds 6-10 for the two-layer all-layer output-selector induced condition:
  `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed6_10_steps2000/`.
- Result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - routed expert match `1.00`;
  - output gate distance `0.7777`;
  - causal expert distance `0.5551`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_seed_robustness.md`.
- Interpretation update: the core output-selector induced-modularity result is
  robust across 10/10 one-layer seeds and 10/10 two-layer seeds. The two-layer
  causal localization to `L1E0` for local and `L1E1` for induction also holds in
  all 10 tested seeds.
- Ran spontaneous negative controls on seeds 6-10:
  - one-layer:
    `results/phase3_toy_switchhead_spontaneous_seed6_10_steps2000/`;
  - two-layer:
    `results/phase3_toy_switchhead_2layer_spontaneous_seed6_10_steps2000/`.
- Negative-control result:
  - one-layer spontaneous seeds 6-10: routed match `0.20`, gate distance
    `0.0061`, causal distance `0.0086`;
  - two-layer spontaneous seeds 6-10: routed match `0.00`, gate distance
    `0.0041`, causal distance `0.1209`.
- Updated `doc/experiments/phase3/phase3_toy_switchhead_seed_robustness.md`.
- Interpretation update: the negative spontaneous result also survives the
  expanded seed set.

## 2026-05-23 - SwitchHead expert-swap interventions

- Added `--run-swap-interventions` to
  `scripts/toy_switchhead_competition.py`.
- Added frozen-model expert-row swap interventions for SwitchHead value
  projections (`v`), output projections (`o`), value selectors (`sel_v`), and
  output selectors (`sel_o`), plus paired and full relabeling controls.
- Added source-aligned value-selector metrics:
  `source_value_gate_distribution_distance`,
  `source_value_gate_local_top_expert`, and related per-expert fields.
- Ran the one-layer output-selector induced condition with the swap grid:
  `results/phase3_toy_switchhead_swap_interventions_w005_end800_seed5_steps2000/`.
- Baseline replicated the induced result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - routed expert match `1.00`;
  - output gate distance `0.9645`;
  - causal expert distance `0.5664`;
  - query-position value gate distance `0.0134`;
  - source-position value gate distance `0.0029`.
- Swap results:
  - `swap_v`: local/induction accuracy `0.0804/0.0706`;
  - `swap_value_selector`: `0.0804/0.0706`;
  - `swap_v_and_value_selector`: `1.0000/1.0000`;
  - `swap_o`: `1.0000/1.0000`;
  - `swap_output_selector`: `1.0000/1.0000`;
  - `swap_o_and_output_selector`: `1.0000/1.0000`;
  - `swap_all`: `1.0000/1.0000`.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_swap_interventions.md`.
- Updated `doc/experiments/phase3/phase3_structural_to_functional_synthesis.md`,
  `doc/experiments/phase3/phase3_toy_switchhead_selector_type.md`,
  `doc/research_questions.md`, and `doc/plan.md`.
- Interpretation update: output-selector pressure remains the clean sufficient
  training cue, but the frozen inference-time code is fragile on the value side.
  Coherent full relabeling is harmless; mismatching the value selector and value
  projection destroys both roles.

## 2026-05-23 - Two-layer SwitchHead expert-swap interventions

- Added layer-specific swap support to `scripts/toy_switchhead_competition.py`:
  - `--swap-intervention-layers`;
  - `--swap-intervention-layer-groups`, accepting values like `0 1 all`.
- Ran preliminary separate layer-0 and layer-1 two-layer swap checks:
  - `results/phase3_toy_switchhead_2layer_swap_l0_w005_end800_seed5_steps2000/`;
  - `results/phase3_toy_switchhead_2layer_swap_l1_w005_end800_seed5_steps2000/`.
- Then ran the cleaner grouped comparison on the same trained seed set:
  `results/phase3_toy_switchhead_2layer_swap_groups_w005_end800_seed5_steps2000/`.
- Baseline grouped result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - routed expert match `1.00`;
  - local top `L1E0` and induction top `L1E1` in 5/5 seeds;
  - output gate distance `0.7275`;
  - causal expert distance `0.6360`;
  - source-position value gate distance `0.0019`.
- Layer-specific swap result:
  - layer-0 `swap_v`: local/induction accuracy `0.8913/0.9995`;
  - layer-0 `swap_o`: `0.9254/1.0000`;
  - layer-1 `swap_v`: `0.6117/0.5892`;
  - layer-1 `swap_o`: `1.0000/1.0000`;
  - all-layer `swap_v`: `0.5156/0.5743`;
  - all-layer `swap_o`: `0.9237/1.0000`;
  - paired value-side relabeling and full relabeling restored `1.0000/1.0000`
    for all layer groups.
- Wrote `doc/experiments/phase3/phase3_toy_switchhead_two_layer_swap_interventions.md`.
- Updated `doc/experiments/phase3/phase3_structural_to_functional_synthesis.md`,
  `doc/experiments/phase3/phase3_toy_switchhead_swap_interventions.md`,
  `doc/research_questions.md`, and `doc/plan.md`.
- Interpretation update: the main two-role value-side codebook localizes to
  layer 1, while layer 0 still has local-supporting expert-label structure that
  single top-component ablations understate.

## 2026-05-23 - Attention-weighted value-gate diagnostic

- Added attention-weighted value-gate metrics to
  `scripts/toy_switchhead_competition.py`:
  - `attended_value_gate_distribution_distance`;
  - `attended_value_gate_local_top_expert`;
  - `attended_value_gate_induction_top_expert`;
  - per-expert attended value-gate means.
- The diagnostic reconstructs causal attention weights from each block's Q/K
  projections and RoPE transform, then weights source-token value-selector
  distributions by the attention paid from local and induction query positions.
- Smoke-tested the metric in `results/debug_switchhead_attended_value_gate/`.
- Ran the one-layer induced condition:
  `results/phase3_toy_switchhead_attended_value_gate_w005_end800_seed5_steps2000/`.
- Result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - routed expert match `1.00`;
  - output gate distance `0.9645`;
  - query-position value gate distance `0.0134`;
  - source-position value gate distance `0.0029`;
  - attended value-gate distance `0.0091`.
- Updated `doc/experiments/phase3/phase3_toy_switchhead_swap_interventions.md`,
  `doc/experiments/phase3/phase3_structural_to_functional_synthesis.md`,
  `doc/research_questions.md`, and `doc/plan.md`.
- Interpretation update: the value-side swap fragility is not explained by
  simple marginal or attention-weighted local-vs-induction value-expert usage.
  It is better described as an internal expert-codebook or basis-consistency
  relation.

## 2026-05-23 - SwitchHead checkpoint saving

- Added `--save-final-checkpoints` and `--checkpoint-dir` to
  `scripts/toy_switchhead_competition.py`.
- Checkpoints save:
  - `model_state_dict`;
  - serializable args;
  - layout tensors;
  - seed;
  - train step.
- Smoke-tested checkpoint saving:
  `results/debug_switchhead_checkpoint_save/checkpoints/model_seed1.pt`.
- Ran a two-layer induced checkpoint-saving pass:
  `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed5_steps2000_checkpoints/`.
- This pass saved checkpoints for seeds 1-5, but should not be used as the
  canonical 5/5 induced set: seed 3 solved the task but had local and induction
  both top at `L1E0`, giving routed match `0.80` overall.
- Ran a seed-3 retry:
  `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed3_checkpoint_retry/`.
- The retry succeeded with local top `L1E0`, induction top `L1E1`, routed match
  `1.00`, and saved
  `results/phase3_toy_switchhead_2layer_induced_w005_end800_seed3_checkpoint_retry/checkpoints/model_seed3.pt`.
- Interpretation update: checkpoint saving works. Because SwitchHead/Triton
  training is not fully deterministic, future canonical checkpoint sets should
  either save during the exact successful run or rerun failed seeds explicitly.

## 2026-05-23 - SwitchHead checkpoint loading

- Added `--load-final-checkpoints` to `scripts/toy_switchhead_competition.py`.
- Loading skips training, restores `model_seed{seed}.pt` from `--checkpoint-dir`
  or `output_dir/checkpoints`, and then runs the normal analysis/swap pipeline.
- Updated checkpoint saves to store CPU state dict tensors.
- Updated checkpoint loads to use `torch.load(..., weights_only=True)`.
- Smoke-tested checkpoint loading from:
  `results/debug_switchhead_checkpoint_save/checkpoints/model_seed1.pt`.
- Load smoke output:
  `results/debug_switchhead_checkpoint_load_weights_only/`.
- The loaded smoke reproduced the saved four-step debug metrics exactly, with no
  PyTorch pickle warning.
- Used the checkpoint loader for an end-to-end loaded swap run on the validated
  seed-3 retry checkpoint:
  `results/phase3_toy_switchhead_2layer_seed3_loaded_swap_groups/`.
- Loaded seed-3 swap result:
  - baseline local/induction accuracy `1.0000/1.0000`;
  - layer-1 `swap_v`: `0.4812/0.5081`;
  - layer-1 `swap_o`: `1.0000/1.0000`;
  - layer-1 `swap_v_and_value_selector`: `1.0000/1.0000`;
  - all-layer `swap_v`: `0.4517/0.4995`;
  - all-layer `swap_o`: `0.6890/0.9946`;
  - all-layer `swap_all`: `1.0000/1.0000`.
- Interpretation update: the checkpoint loader supports the intended no-retrain
  swap workflow, and the loaded validated checkpoint preserves the same
  value-side codebook pattern.

## 2026-05-23 - SwitchHead checkpoint parameter diagnostics

- Added `scripts/switchhead_checkpoint_parameter_diagnostics.py`.
- The script loads saved checkpoints and computes expert-0 vs expert-1 cosine
  similarity and relative L2 distance for `v`, `o`, `sel_v`, and `sel_o`, per
  layer and head.
- Ran on the validated seed-3 retry checkpoint:
  `results/phase3_toy_switchhead_seed3_parameter_diagnostics/`.
- Ran on the successful seeds 1, 2, 4, and 5 from the two-layer checkpoint set:
  `results/phase3_toy_switchhead_seed1245_parameter_diagnostics/`.
- Main seed-1/2/4/5 summary:
  - layer-0 `o` cosine `0.2628`, relative L2 `1.2160`;
  - layer-0 `v` cosine `0.0659`, relative L2 `1.3665`;
  - layer-1 `o` cosine `0.8295`, relative L2 `0.5868`;
  - layer-1 `v` cosine `0.2884`, relative L2 `1.1936`.
- Interpretation update: layer-1 output projection experts are much more similar
  than layer-1 value projection experts, matching the swap result where layer-1
  `swap_o` is tolerated but layer-1 `swap_v` is destructive. Layer-0 output
  experts are less similar, consistent with layer-0 output swaps hurting local
  performance.

## 2026-05-23 - One-layer SwitchHead checkpoint parameter diagnostics

- Saved one-layer induced checkpoints:
  `results/phase3_toy_switchhead_1layer_induced_w005_end800_seed5_steps2000_checkpoints/`.
- The checkpoint-saving run reproduced the canonical one-layer induced result:
  - local accuracy `1.0000`;
  - induction accuracy `1.0000`;
  - routed expert match `1.00`;
  - output gate distance `0.9645`;
  - causal expert distance `0.5663`;
  - attended value-gate distance `0.0091`.
- Ran parameter diagnostics:
  `results/phase3_toy_switchhead_1layer_parameter_diagnostics/`.
- Summary:
  - layer-0 `o` cosine `0.7887`, relative L2 `0.6514`;
  - layer-0 `v` cosine `0.0451`, relative L2 `1.3816`;
  - layer-0 `sel_o` cosine `-0.8963`;
  - layer-0 `sel_v` cosine `-0.0199`.
- Interpretation update: the one-layer parameter geometry matches the swap
  result directly. Output projections are similar enough that output-side swaps
  are tolerated, while value projections are highly distinct and value-side
  mismatches are destructive.

## 2026-05-23 - Ordinary attention-head specialization/modularity sweep

- Scope correction: this run uses ordinary attention heads as the unit of
  analysis, not SwitchHead experts, MoE experts, or branch towers.
- Added explicit head-level modularity metrics to
  `scripts/toy_competition_head_dim_intervention.py`:
  `role_distribution_tv_distance` and `role_distribution_overlap`, computed
  from flattened local-vs-induction causal ablation distributions over
  `(layer, head)` slots.
- Ran a 10-seed, nine-layout matched-budget sweep:
  `results/phase3_toy_competition_head_dim_modularity_sweep_20260523/`.
- Added the memo:
  `doc/experiments/phase3/phase3_attention_head_specialization_modularity_sweep.md`.
- Updated the direction lock:
  `doc/project_direction_attention_heads_primary.md`.
- Main specialization result: in the one-64 heterogeneous layouts, the local
  causal role followed the 64-dim head slot in `40/40` models:
  `[16,16,32,64] -> L1H3`, `[64,16,16,32] -> L1H0`,
  `[16,64,16,32] -> L1H1`, and `[16,32,64,16] -> L1H2`, each in `10/10`
  seeds.
- Main pairwise-separability result: `hetero4` improved local-vs-induction role
  separation over `uniform4` (`TV=0.528` vs `0.398`), but `uniform2` also had
  high separation (`TV=0.511`). Interpretation update: heterogeneous head
  dimensions strongly stabilize functional specialization slots, while
  two-role separability is a separate outcome that appears in some layouts but
  is not automatic or uniquely caused by heterogeneity. Full ontology-level
  functional modularity remains untested until the project evaluates many
  roles/subroles and asks whether related roles cluster together across heads.

## 2026-05-23 - Ordinary attention-head role ontology sweep

- Added the v2 framing memo:
  `doc/project_plan_v2_attention_head_structure.md`.
- Added `scripts/toy_role_ontology_head_dim_intervention.py`.
- The script keeps the unit as ordinary attention heads and trains one tiny
  decoder-only model on six roles in three families: local-copy (`local_a`,
  `local_b`), KV lookup (`kv_a`, `kv_b`), and induction (`induction_short`,
  `induction_long`).
- Ran a 5-seed, six-layout matched-budget sweep:
  `results/phase3_toy_role_ontology_head_dim_20260523/`.
- Added the memo:
  `doc/experiments/phase3/phase3_toy_role_ontology_head_dim.md`.
- Main structural role-affinity result: across the four one-64 heterogeneous
  layouts, local-copy and KV-lookup subroles chose the 64-dim structural type in
  `80/80` cases. Induction subroles split across 64/32/16, with
  `induction_short` choosing 64 in `9/20` and `induction_long` choosing 32 in
  `12/20`.
- Main specialization result: one-64 heterogeneous layouts increased mean role
  specialization over `uniform4` (`0.663` to `0.723` vs `0.449`) and reduced
  effective heads (`2.08` to `2.30` vs `4.15`).
- Main ontology-modularity result: hetero layouts can improve family clustering
  over `uniform4`, but `uniform2` is a strong baseline. `hetero4_64second`
  reached family gap `0.607`, ARI `0.889`; `uniform4` had gap `0.511`, ARI
  `0.586`; `uniform2` had gap `0.653`, ARI `1.000`.
- Interpretation update: structural role affinity is now the cleanest term for
  the user's core claim. Heterogeneous heads strongly bias role-to-head-type
  assignment and increase specialization. Full functional modularity remains a
  separate controlled question because fewer/wider uniform heads can also
  produce strong role-family clustering.

## 2026-05-23 - Hetero2 role ontology control

- Ran the missing two-head heterogeneous control against `uniform2 [64,64]`:
  `results/phase3_toy_role_ontology_hetero2_20260523/`.
- Configs:
  - `uniform2 = [64,64]`
  - `[32,96]`
  - `[48,80]`
  - `[16,112]`
- Added memo:
  `doc/experiments/phase3/phase3_toy_role_ontology_hetero2.md`.
- Main baseline-vs-experiment table:
  - `uniform2`: specialization `0.636`, effective heads `2.25`, family gap
    `0.653`, ARI `1.000`.
  - `[32,96]`: specialization `0.695`, effective heads `1.86`, family gap
    `0.480`, ARI `0.889`.
  - `[48,80]`: specialization `0.756`, effective heads `1.91`, family gap
    `0.601`, ARI `1.000`.
  - `[16,112]`: specialization `0.702`, effective heads `1.76`, family gap
    `0.327`, ARI `0.667`.
- Structural role-affinity result: local-copy and KV-lookup roles chose the
  larger hetero2 head in `60/60` cases.
- Interpretation update: hetero2 gives a clean matched-head-count result for
  stronger structural role affinity and stronger specialization than `uniform2`,
  but it does not improve ontology-level modularity over `uniform2`. The best
  hetero2 modularity setting, `[48,80]`, ties ARI but has lower family gap than
  `uniform2`. Extreme imbalance `[16,112]` appears to collapse all roles onto
  the huge head and harms family clustering.

## 2026-05-23 - Big role ontology proposal

- Added the proposal:
  `doc/big_role_ontology_proposal.md`.
- Purpose:
  - expand beyond the six-role toy ontology;
  - keep ordinary attention heads as the unit;
  - make the next modularity experiment meaningful by testing many role
    families and subroles.
- Literature anchors used:
  - Clark et al. 2019 on BERT delimiter, positional, broad, syntax, and
    coreference attention patterns;
  - Voita et al. 2019 on specialized/prune-resistant positional, syntactic, and
    rare-word heads;
  - Elhage et al. 2021 and Olsson et al. 2022 on QK/OV framing,
    previous-token heads, and induction heads;
  - Wang et al. 2022 on IOI circuit head classes;
  - McDougall et al. 2023 on copy-suppression / negative heads;
  - Htut et al. 2019 on syntactic-dependency specialist heads;
  - Pande et al. 2021 on broader head-role taxonomy and role co-location.
- Proposed Toy Ontology v2:
  - `copy_transport`;
  - `induction`;
  - `position_boundary`;
  - `suppression_conflict`;
  - `entity_coreference`.
- Proposed scale:
  - five families;
  - four subroles per family;
  - about 20 roles total.
- Next planned artifact:
  - implement a Toy Ontology v2 smoke test before any full expensive sweep;
  - compare `uniform4`, `uniform2`, all-distinct hetero4 configs such as
    `[8,16,40,64]` or `[16,24,40,48]`, and all-distinct hetero2 configs such
    as `[48,80]`;
  - report baseline-vs-hetero tables for affinity, specialization, and
    modularity.

## 2026-05-23 - Document and role/task organization

- Added `doc/README.md` to define the document layout.
- Added `doc/role_task_organization.md` to make the experiment organization
  explicit:
  - ontology;
  - family;
  - role/subrole;
  - scene or dataset;
  - target positions;
  - metric rows.
- Clarified that the previous local/KV/induction setup is Toy Ontology v1:
  three families with two subroles each, not three unrelated tasks.
- Clarified that every role must have a dataset or probe scene before it can be
  used in an experiment.
- Moved time-variant reports out of the root `doc/` folder:
  - phase reports to `doc/experiments/phase0/`,
    `doc/experiments/phase1/`, and `doc/experiments/phase3/`;
  - autonomous logs to `doc/logs/autonomous_sleep/`;
  - SwitchHead side-branch notes to `doc/side_branches/switchhead/`.
- Updated internal markdown references to the moved report paths.
- Updated future non-uniform head-dimension policy:
  - uniform baselines may repeat dimensions;
  - non-uniform configs should use all-distinct dimensions;
  - dimensions should be multiples of 8;
  - total attention dimension should stay matched when comparing configs.

## 2026-05-23 - Toy Ontology v2 full 20-role sweep

- Added `scripts/toy_role_ontology_v2_head_dim_intervention.py`.
- Added `scripts/analyze_role_ontology_v2.py`.
- Added memo:
  `doc/experiments/phase3/phase3_toy_role_ontology_v2.md`.
- Scope:
  - unit is ordinary attention heads only;
  - 20 role rows;
  - five role families: `copy_transport`, `induction`,
    `position_boundary`, `suppression_conflict`, and `entity_coreference`;
  - all non-uniform configs use all-distinct head dimensions;
  - total attention dimension is matched at 128.
- Main full-sweep command:
  `python scripts/toy_role_ontology_v2_head_dim_intervention.py --role-set v2_full --configs uniform4 uniform2 hetero4_unique_mild hetero4_unique_64 hetero4_unique_extreme hetero2_unique_mild hetero2_unique_mid hetero2_unique_extreme --seeds 1 2 3 4 5 --steps 1600 --batch-size 128 --eval-examples 512 --output-dir results/phase3_toy_role_ontology_v2_full_1600_20260523`
- Main result root:
  `results/phase3_toy_role_ontology_v2_full_1600_20260523`.
- Layout control root:
  `results/phase3_toy_role_ontology_v2_layout_1600_20260523`.
- Main learnability result:
  - all configs learned the expanded ontology;
  - mean minimum-role accuracy stayed at or above `0.991`.
- Main structural role-affinity result:
  - heterogeneous four-head chance largest-top rate is `0.25`;
  - observed largest-top rates were `0.50`, `0.65`, and `0.82` for
    `[16,24,40,48]`, `[8,16,40,64]`, and `[8,16,24,80]`;
  - heterogeneous two-head chance largest-top rate is `0.50`;
  - observed rates were `0.66`, `0.82`, and `0.94` for `[48,80]`,
    `[32,96]`, and `[16,112]`.
- Main specialization result:
  - `uniform4` specialization `0.733`, effective heads `2.400`;
  - `uniform2` specialization `0.684`, effective heads `2.166`;
  - best hetero4 specialization was `[8,16,24,80]` at `0.849`, effective
    heads `1.522`;
  - all hetero2 configs beat `uniform2` on specialization.
- Main modularity result:
  - `uniform4` family gap `0.153`, ARI `0.149`;
  - `uniform2` family gap `0.117`, ARI `0.108`;
  - `hetero2_unique_mid [32,96]` was best in the main grid on ARI
    (`0.159`) and near-best on family gap (`0.151`);
  - `hetero2_unique_extreme [16,112]` had low family gap `0.050` and ARI
    `0.039`, showing specialization can collapse without useful modularity.
- Layout control:
  - permuted `[8,16,40,64]` across four head-index placements;
  - local copy selected the moved 64-dim type in `20/20` cases;
  - wrong-key suppression and recency conflict selected it in `19/20` cases;
  - KV lookup selected it in `17/20` cases;
  - induction and some boundary roles were less locked to 64.
- Interpretation update:
  - the project should continue;
  - current strongest claim is structural role affinity plus role-level
    specialization;
  - functional modularity is real as a question but should not yet be claimed
    as automatic from heterogeneous head dimensions.

## 2026-05-24 - Larger head-count control and real-model role validation

- Added larger-head presets to
  `scripts/toy_role_ontology_v2_head_dim_intervention.py`:
  - `uniform8 = [48,48,48,48,48,48,48,48]`;
  - `hetero8_unique_spread = [16,24,32,40,48,56,72,96]`;
  - `hetero8_unique_extreme = [8,16,24,32,40,48,64,152]`.
- Extended `scripts/analyze_role_ontology_v2.py` with
  dimension-level family-modularity tables.
- Added memo:
  `doc/experiments/phase3/phase3_large_head_count_and_real_validation.md`.
- User concern tested:
  - previous Toy Ontology v2 configs had only 4 or 8 total ordinary head slots;
  - maybe modularity was weak because there were too few heads.
- 32-slot sweep:
  - result root:
    `results/phase3_toy_role_ontology_v2_large_heads_1000_20260523`;
  - setting: 8 heads per layer x 4 layers;
  - `uniform8` family gap `0.048`, ARI `0.060`;
  - `hetero8_unique_spread` family gap `0.050`, ARI `0.051`;
  - `hetero8_unique_extreme` family gap `0.038`, ARI `0.025`;
  - largest-head top rates were `0.50` and `0.72` for spread/extreme vs
    8-way chance `0.125`.
- 16-slot 2000-step sweep:
  - result root:
    `results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523`;
  - setting: 8 heads per layer x 2 layers;
  - `uniform8` family gap `0.132`, ARI `0.130`;
  - `hetero8_unique_spread` family gap `0.079`, ARI `0.046`;
  - `hetero8_unique_extreme` family gap `0.081`, ARI `0.058`;
  - largest-head top rates were `0.62` and `0.76` for spread/extreme vs
    8-way chance `0.125`.
- Failure-analysis result:
  - more heads did not rescue the family-modularity claim;
  - heterogeneity still strongly supports structural role affinity and
    specialization;
  - family-level modularity remains mixed and is not automatically created by
    heterogeneous head dimensions.
- Real-model validation:
  - ran Pythia-160M-deduped, revision `step143000`, float32 attention-role
    probe:
    `results/phase3_real_model_role_probe_pythia160m_deduped_float32_20260524`;
  - `repeat_match` had strong attention specialization: best layer/head
    `L0H7`, specialization `0.834`, effective heads `2.088`;
  - `previous_token` was measurable but more distributed: best layer/head
    `L10H9`, specialization `0.332`;
  - `bos` was diffuse: best specialization `0.140`;
  - initial float16 run produced NaNs in later-layer attention summaries, so it
    was not used.
- Existing causal Pythia result roots reused:
  - `results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2`;
  - `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128`.
- Real-model interpretation:
  - local-copy and repeat/induction roles are measurable and causally important
    in ordinary pretrained heads;
  - this validates the role-measurement side of the project, but does not test
    heterogeneous head dimensions because Pythia heads are uniform.

## 2026-05-24 - Ontology alignment metric replaces ARI as main modularity diagnostic

- User raised a conceptual issue: ontology families are predefined hypotheses,
  not guaranteed discovered clusters, so ARI is too brittle as the primary
  modularity metric.
- Updated `scripts/analyze_role_ontology_v2.py` to emit:
  - `ontology_alignment_table.csv`;
  - `ontology_alignment_per_seed.csv`;
  - `role_ontology_neighbor_table.csv`;
  - `dimension_ontology_alignment_table.csv`;
  - `dimension_ontology_alignment_per_seed.csv`.
- Added metric documentation:
  `doc/ontology_alignment_metric.md`.
- Updated framework/report docs:
  - `doc/three_question_framework_attention_heads.md`;
  - `doc/experiments/phase3/phase3_large_head_count_and_real_validation.md`.
- Re-ran analysis on:
  - `results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523`;
  - `results/phase3_toy_role_ontology_v2_large_heads_1000_20260523`;
  - `results/phase3_toy_role_ontology_v2_full_1600_20260523`.
- Clean 16-slot result with the new metric:
  - `uniform8`: family gap `0.132`, ontology alignment `0.130`,
    shuffled-label p `0.173`;
  - `hetero8_unique_spread`: family gap `0.079`, ontology alignment `0.087`,
    shuffled-label p `0.192`;
  - `hetero8_unique_extreme`: family gap `0.081`, ontology alignment `0.087`,
    shuffled-label p `0.099`.
- Interpretation:
  - the stronger hetero specialization result still holds because accuracy is
    matched or better;
  - ontology-level functional modularity remains only modest and uneven;
  - use family gap plus ontology alignment and per-role neighbor margins, not
    ARI, in main result tables.

## 2026-05-24 - Ontology refinement analysis for modularity claim

- Added `scripts/analyze_ontology_refinement.py`.
- Added memo:
  `doc/experiments/phase3/phase3_ontology_refinement_modularity.md`.
- Tested whether alternative, defensible ontology views make non-uniform heads
  look more modular without retraining:
  - original v2 family labels;
  - coarse task-primitive 3-way ontology;
  - finer mechanism-group ontology;
  - mechanism multilabel ontology;
  - label-free clusterability controls.
- Clean 16-slot result:
  - under original v2 labels, `uniform8` still had higher ontology alignment
    (`0.130`) than hetero spread/extreme (`0.087`, `0.087`);
  - under task-primitive 3-way labels, hetero improved over uniform
    (`uniform8=0.004`, spread `0.034`, extreme `0.063`), but the effect was
    small and not robust enough for a main claim;
  - hetero had higher raw silhouette clusterability (`0.674`/`0.693` vs
    `0.647`), but lower pairwise role separation, so separation-adjusted
    clusterability still favored uniform.
- 32-slot control stayed mixed:
  - some ontology choices favored hetero spread;
  - task-primitive and multilabel views favored uniform.
- Interpretation:
  - post-hoc ontology refinement does not yet prove non-uniform functional
    modularity;
  - the fair next step is a predeclared Toy Ontology v3 with families designed
    as repeated variants of the same algorithmic primitive.

## 2026-05-24 - Toy Ontology v3 algorithmic-family run

- Added Toy Ontology v3 role set to
  `scripts/toy_role_ontology_v2_head_dim_intervention.py`.
- V3 has 5 families x 4 roles, with each family designed as repeated variants
  of one algorithmic primitive:
  - `local_offset`;
  - `key_value_lookup`;
  - `sequence_induction`;
  - `boundary_anchor`;
  - `conflict_suppression`.
- Ran smoke:
  `results/phase3_toy_role_ontology_v3_smoke_20260524`.
- Ran main 5-seed comparison:
  `results/phase3_toy_role_ontology_v3_main_2000_20260524`.
- Added memo:
  `doc/experiments/phase3/phase3_toy_role_ontology_v3_algorithmic.md`.
- Main result:
  - all configs reached near-perfect accuracy;
  - `uniform8`: ontology alignment `0.139`, specialization `0.660`,
    effective heads `3.586`;
  - `hetero8_unique_spread`: ontology alignment `0.142`, specialization
    `0.670`, effective heads `3.174`;
  - `hetero8_unique_extreme`: ontology alignment `0.193`, specialization
    `0.783`, effective heads `2.073`.
- Interpretation:
  - v3 is the strongest modularity evidence so far;
  - extreme hetero improves ontology alignment by about `+0.055` over uniform
    and wins 4/5 seeds on that metric;
  - spread hetero improves separation-adjusted label-free clusterability in 5/5
    seeds;
  - the effect is promising but not yet a final proof because family-gap gains
    are small and some families still weaken under heterogeneity.

## 2026-05-24 - Main metric cleanup

- Updated `doc/ontology_alignment_metric.md`,
  `doc/experiments/phase3/phase3_toy_role_ontology_v3_algorithmic.md`, and
  `todo.md` to reflect the user's metric preference.
- Largest-head/top-dimension rate is no longer a headline metric. It remains a
  raw structural-affinity diagnostic only, because presenting it centrally can
  misleadingly imply "bigger head is better."
- Family Gap is no longer a main modularity metric. In the current binary-family
  setting it measures the same underlying signal as ontology alignment:
  same-family pairs should have higher head-usage similarity than
  different-family pairs. It is retained only as an appendix/simple-effect-size
  check.
- Main modularity metrics going forward:
  - ontology alignment;
  - separation-adjusted clusterability.
- Added the canonical metric-system document:
  `doc/current_metric_system.md`.
