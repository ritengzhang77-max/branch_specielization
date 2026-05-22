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
- Recorded the smoke result in `doc/phase0_smoke_report.md`.
- Ran a successful Pythia-160M two-seed Phase 0 pilot:
  - seeds: 1, 2;
  - probe texts: 8;
  - raw same-index similarity mean: 0.7082;
  - Hungarian-matched similarity mean: 0.8220;
  - matched-minus-random mean: 0.0989.
- Added `doc/phase0_pythia160m_pilot.md`.
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
- Recorded the all-seed result in `doc/phase0_pythia160m_all_seed_baseline.md`.
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
- Recorded this result in `doc/phase0_pythia160m_raw_score_pilot.md`.
- Ran the all-seed Pythia-160M raw-score baseline:
  - seeds: 1 through 9;
  - revision: `step143000`;
  - seed pairs x layers: 36 x 12 = 432 layer-pair comparisons;
  - raw same-index similarity mean: 0.3735;
  - Hungarian-matched similarity mean: 0.6692;
  - matched-minus-random mean: 0.2982.
- Recorded this result in
  `doc/phase0_pythia160m_raw_score_all_seed_baseline.md`.
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
  `doc/phase1_pythia160m_attention_role_specialization.md`.
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
  `doc/phase1_repeat_match_ablation_pythia160m.md`.
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
- Recorded the result in `doc/phase3_toy_head_dim_intervention.md`.
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
- Recorded the result in `doc/phase3_toy_induction_head_dim_intervention.md`.
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
  `doc/phase3_toy_competition_head_dim_intervention.md`.
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
  `doc/phase3_toy_competition_layout_permutations.md`.
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
- Recorded the result in `doc/phase3_toy_competition_weight_sweep.md`.
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
  `doc/phase3_toy_competition_all_layout_weight_sweep.md`.
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
- Recorded the result in `doc/phase3_toy_competition_two_attractor.md`.
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
- Recorded the result in `doc/phase3_toy_branch_isolation.md`.
- Created checkpoint deck:
  `presentations/2026-05-22-1230-branch-isolation/branch_isolation_checkpoint.pdf`.
