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

## 2026-05-22 21:45-22:05 PDT

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

## 2026-05-22 22:05-22:15 PDT

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
