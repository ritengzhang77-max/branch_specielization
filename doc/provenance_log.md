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
