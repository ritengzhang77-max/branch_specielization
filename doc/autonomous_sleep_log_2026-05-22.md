# Autonomous Sleep Research Log

Start: 2026-05-22 20:38:35 PDT

Planned stop: 2026-05-23 08:38:35 PDT

## Initial Plan

The current strongest Phase 1 result is that raw-score alignment transfers a
repeat-match causal head role across all 9 Pythia-160M seeds better than raw
same-index transfer. The next decisive question is whether this generalizes
beyond repeat-match heads.

Immediate experiment:

```text
Test a second causal head-role task: local-copy / previous-token behavior.
```

Task definition:

- Input pattern: `[x, SEP, x]` repeated across a sequence.
- Probe site: attention from the `SEP` position back to the previous `x`.
- Causal readout: next-token loss at the `SEP` position, where the target token
  is the copied previous token `x`.
- Cross-seed question: do source-selected local-copy heads transfer into target
  seeds better by raw-score alignment than by the same raw layer/head index?

This is a direct contrast to repeat-match. Prior role-probe notes suggested that
plain previous-token attention is more distributed than repeat-match, so a weak
or negative result would also be informative.

## 2026-05-22 20:40-21:25 PDT

- Added `scripts/pythia_local_copy_alignment.py`.
- The script implements a local-copy / previous-token synthetic task using
  `[x, SEP, x]` triples.
- Added partial-output support and `--target-seeds` chunking after long GPU jobs
  were interrupted before final writes.
- Completed a Pythia-14M two-seed smoke run:
  - baseline loss `27.17`;
  - own top excess over random `-0.1005`;
  - aligned-minus-same `-0.0338`.
- Completed a Pythia-160M two-seed pilot:
  - baseline loss `10.3996`;
  - own top excess over random `2.1609`;
  - same-index source transfer `0.0033`;
  - aligned source transfer `0.1033`;
  - aligned-minus-same `0.0999`.
- Attempted all-9-seed Pythia-160M local-copy runs on GPU and CPU. Under current
  machine contention these were interrupted before enough target-seed rows were
  completed. This is recorded as an infrastructure blocker, not as a model
  result.
- Wrote pilot report:
  `doc/phase1_pythia160m_local_copy_pilot.md`.

## 2026-05-22 21:25-21:45 PDT

- GPU 0 became stable enough to complete a larger local-copy target chunk.
- Ran Pythia-160M final checkpoint with all 9 source seeds and target seeds 1-3
  on layer 3.
- Result over 24 ordered source-target pairs:
  - own top excess over random: `2.8567`;
  - same-index source transfer: `0.1901`;
  - aligned source transfer: `1.5894`;
  - aligned-minus-same: `1.3993`;
  - aligned better count: `17/24`.
- This changes the local-copy interpretation: the weak two-seed transfer was not
  representative once all source seeds were available. The completed target
  chunk suggests local-copy may be a second positive causal alignment-transfer
  role, but target seeds 4-9 are still incomplete.
- Retried target seeds 4-6, but the job was interrupted during probing before
  target rows were written.
- Added `scripts/analyze_local_copy_chunks.py` so completed target chunks can be
  merged as they become available.

## Later Sleep-Block Checkpoint: Local-Copy Targets 4-6

- Committed and pushed the local-copy chunk combiner checkpoint:
  `336bab2 Add local-copy chunk combiner`.
- Retried the Pythia-160M local-copy chunk for target seeds 4-6 on GPU 3.
- The chunk completed over 24 ordered source-target pairs:
  - own top excess over random: `0.0420`;
  - same-index source transfer: `0.1700`;
  - aligned source transfer: `0.1584`;
  - aligned-minus-same: `-0.0116`;
  - aligned better count: `11/24`.
- Interpretation: target seeds 4-6 are not like target seeds 1-3. The selected
  layer-3 local-copy heads have similar probe specialization, but weak own-seed
  causal importance, so aligned transfer also has little effect.
- Re-ran the chunk combiner. Across target seeds 1-6:
  - own top excess over random: `1.4493`;
  - same-index source transfer: `0.1800`;
  - aligned source transfer: `0.8739`;
  - aligned-minus-same: `0.6939`;
  - aligned better count: `28/48`.
- Started the final local-copy target chunk for target seeds 7-9.

## Later Sleep-Block Checkpoint: Local-Copy All-Target Result

- Completed the final Pythia-160M local-copy target chunk for target seeds 7-9.
- Result over 24 ordered source-target pairs:
  - own top excess over random: `1.9231`;
  - same-index source transfer: `0.5826`;
  - aligned source transfer: `1.2933`;
  - aligned-minus-same: `0.7107`;
  - aligned better count: `12/24`.
- Re-ran the chunk combiner with all target chunks present. Full all-target
  local-copy result over 72 ordered source-target pairs:
  - selected specialization mean: `0.3262`;
  - own top excess over random: `1.6072`;
  - same-index source transfer: `0.3142`;
  - aligned source transfer: `1.0137`;
  - aligned-minus-same: `0.6995`;
  - aligned better count: `40/72`.
- Target-level heterogeneity is the key result:
  - target seeds 1-3: aligned-minus-same `1.3993`;
  - target seeds 4-6: aligned-minus-same `-0.0116`;
  - target seeds 7-9: aligned-minus-same `0.7107`.
- Target own-head causal excess strongly tracks alignment-transfer benefit
  across target seeds (`r=0.9664`; target sign count `7/9`, two-sided sign
  `p=0.1797`). This suggests that local-copy is a positive but conditional
  second role: alignment transfers the role when the target seed actually uses
  the probed head causally.
- Updated `scripts/analyze_local_copy_chunks.py` to write
  `target_diagnostic_summary.csv` for this target-level heterogeneity check.

## Later Sleep-Block Checkpoint: Layer-Selection Follow-Up

- Added `scripts/pythia_local_copy_layer_causal_sweep.py` to test the top
  local-copy probe head in every layer against same-layer random controls.
- Ran the layer sweep first on weak layer-3 targets 4, 5, 6, and 8:
  - seed 4 best causal head: L2H10, excess `1.4051`;
  - seed 5 best causal head: L4H6, excess `2.0489`;
  - seed 6 best causal head: L4H4, excess `1.8105`;
  - seed 8 best causal head: L4H6, excess `1.8033`.
- Ran a layers 2+4 cross-seed transfer test on weak targets 4, 5, 6, and 8.
  It rescued own-head causality (`own_top_excess=1.8528`) and produced positive
  but modest aligned transfer over same index (`aligned-minus-same=0.3222`,
  aligned better `19/32`).
- Ran the complementary layers 2+4 transfer chunk for targets 1, 2, 3, 7, and
  9. It was much weaker than layer 3 on these targets
  (`own_top_excess=0.1331`, `aligned-minus-same=0.1816`).
- Combined the layers 2+4 chunks. Across all targets, layers 2+4 has
  `aligned-minus-same=0.2441` over 72 ordered source-target pairs, worse than
  the layer-3 all-target result (`0.6995`).
- Ran the all-layer causal sweep for the remaining targets:
  - seeds 1, 2, 3, 7, and 9 are genuinely layer-3 local-copy seeds.
- Added `scripts/analyze_local_copy_layer_sweeps.py` and wrote the layer
  selection memo:
  `doc/phase1_pythia160m_local_copy_layer_selection.md`.
- Current interpretation: local-copy functional specialization exists in all
  seeds, but its structural layer is seed-dependent. Fixed-slot specialization
  is therefore too brittle; the next test should use a cross-layer candidate
  pool rather than a single chosen layer.

## Later Sleep-Block Checkpoint: Cross-Layer Candidate Pool

- Added `scripts/pythia_local_copy_candidate_pool_alignment.py`.
- Ran a cross-layer candidate-pool local-copy transfer test:
  - candidate layers: 2, 3, 4;
  - selected heads per seed: top 2 by local-copy probe score across the
    candidate pool;
  - alignment: Hungarian matching over the full 36-head cross-layer raw-score
    candidate pool;
  - target seeds: all 9;
  - ordered source-target transfer pairs: 72.
- Result:
  - own top excess over random: `2.2896`;
  - same-index source transfer: `0.4876`;
  - cross-layer aligned source transfer: `2.2714`;
  - aligned-minus-same: `1.7838`;
  - aligned better count: `66/72`.
- Per-target aligned-minus-same was positive for `9/9` seeds. Pair-level
  two-sided sign `p ~= 7.3e-14`; target-level sign `p=0.0039`.
- Wrote the candidate-pool memo:
  `doc/phase1_pythia160m_local_copy_candidate_pool.md`.
- Current interpretation: the local-copy role is functionally stable across
  seeds after cross-layer role relabeling. The earlier weak result was caused by
  fixed structural-slot assumptions, not absence of a reusable function.

## Later Sleep-Block Checkpoint: Candidate-Pool Trajectory

- Ran the same cross-layer candidate-pool local-copy experiment at earlier
  Pythia-160M checkpoints.
- `step0` control:
  - own top excess over random: `-0.0004`;
  - same-index transfer: `0.0001`;
  - aligned transfer: `-0.0003`;
  - aligned-minus-same: `-0.0004`;
  - aligned better count: `34/72`.
- `step4000`:
  - own top excess over random: `0.4339`;
  - same-index transfer: `0.0631`;
  - aligned transfer: `0.4822`;
  - aligned-minus-same: `0.4191`;
  - aligned better count: `66/72`.
- `step16000`:
  - own top excess over random: `1.9006`;
  - same-index transfer: `0.3281`;
  - aligned transfer: `1.5318`;
  - aligned-minus-same: `1.2037`;
  - aligned better count: `66/72`.
- Final `step143000` remains strongest:
  - own top excess over random: `2.2896`;
  - aligned-minus-same: `1.7838`;
  - aligned better count: `66/72`.
- Added `scripts/analyze_candidate_pool_trajectory.py` and wrote
  `results/phase1_pythia160m_local_copy_candidate_pool_trajectory/`.
- Current interpretation: local-copy candidate-pool transfer is absent at
  initialization, detectable by `step4000`, and grows in causal magnitude
  through training.

## Later Sleep-Block Checkpoint: Pythia-70M Model-Size Check

- Ran Pythia-70M final-checkpoint candidate-pool checks to test model-size
  generalization.
- Layers 1-3, top 2:
  - own top excess over random: `0.0508`;
  - same-index transfer: `0.1463`;
  - aligned transfer: `0.1115`;
  - aligned-minus-same: `-0.0348`;
  - aligned better count: `35/72`.
- All layers 0-5, top 2:
  - own top excess over random: `0.2692`;
  - same-index transfer: `0.1043`;
  - aligned transfer: `0.1854`;
  - aligned-minus-same: `0.0810`;
  - aligned better count: `41/72`.
- Interpretation: 70M does not robustly instantiate the synthetic local-copy
  causal role. This is a capacity/task caveat rather than strong evidence
  against the alignment method.

## Later Sleep-Block Checkpoint: Pythia-410M Model-Size Check

- Ran Pythia-410M final-checkpoint candidate-pool check:
  - candidate layers: 2-6;
  - selected heads per seed: top 2 by local-copy probe score;
  - target seeds: all 9;
  - ordered source-target transfer pairs: 72.
- Result:
  - own top excess over random: `4.1723`;
  - same-index transfer: `0.2562`;
  - cross-layer aligned transfer: `1.9116`;
  - aligned-minus-same: `1.6554`;
  - aligned better count: `49/72`.
- Per-target aligned-minus-same was positive for all 9 seeds, though seed 3 was
  essentially neutral (`0.0004`).
- Interpretation: the strong local-copy candidate-pool result is not unique to
  Pythia-160M. The model-size story is now: 70M weak, 160M strong, 410M strong.
