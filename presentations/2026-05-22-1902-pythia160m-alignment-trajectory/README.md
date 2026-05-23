# Pythia-160M Repeat-Match Alignment Trajectory

Date: 2026-05-22

## Purpose

This deck summarizes the Pythia-160M checkpoint trajectory testing whether
repeat-match causal head roles transfer across seeds by same head index or by
raw-score alignment.

## Primary Artifacts

- Main PDF: `pythia160m_alignment_trajectory_checkpoint.pdf`
- LaTeX source: `pythia160m_alignment_trajectory_checkpoint.tex`

## Local Copied Data

- `data/revision_summary.csv`
- `data/condition_summary.csv`
- `data/transfer_pair_summary.csv`
- `data/revision_seed_summary.csv`
- `data/alignment_rows.csv`
- `data/ablation_results.csv`
- `figures/repeat_match_alignment_trajectory.png`

## Original Data Roots

- `results/phase1_pythia160m_repeat_match_alignment_trajectory/`

## Source Code

- `scripts/pythia_repeat_match_alignment_trajectory.py`
- `scripts/analyze_pythia_alignment_trajectory.py`
- Reused helpers from:
  `scripts/attention_stability.py`,
  `scripts/pythia_repeat_match_checkpoint_trajectory.py`, and
  `scripts/repeat_match_ablation.py`.

## Reproduction Commands

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/pythia_repeat_match_alignment_trajectory.py \
  --model-size 160m \
  --seeds 1 2 3 \
  --revisions step0 step1000 step4000 step16000 step64000 step143000 \
  --layers 0,1 \
  --top-k-per-layer 1 \
  --random-controls 4 \
  --probe-sequences 64 \
  --eval-sequences 64 \
  --repeat-length 32 \
  --batch-size 8 \
  --alignment-num-texts 4 \
  --alignment-batch-size 2 \
  --random-permutations 100 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase1_pythia160m_repeat_match_alignment_trajectory

python3 -u scripts/analyze_pythia_alignment_trajectory.py

python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1902-pythia160m-alignment-trajectory/pythia160m_alignment_trajectory_checkpoint.tex
```

## Notes

The main result is that aligned transfer becomes much stronger than same-index
transfer after `step16000`, reaching a `0.8728` loss-delta advantage at the
final checkpoint with aligned transfer better in all six ordered seed pairs.
