# Pythia-160M Seed-9 Alignment Transfer

Date: 2026-05-22

## Purpose

This deck summarizes the all-seed Pythia-160M selected-checkpoint follow-up for
repeat-match causal head-role transfer.

## Primary Artifacts

- Main PDF: `outputs/pythia160m_seed9_alignment_checkpoint.pdf`
- LaTeX source: `outputs/pythia160m_seed9_alignment_checkpoint.tex`

## Local Copied Data

- `data/revision_summary.csv`
- `data/condition_summary.csv`
- `data/transfer_pair_summary.csv`
- `data/summary.json`
- `figures/seed9_alignment_selected_trajectory.png`

## Original Data Roots

- `results/phase1_pythia160m_repeat_match_alignment_seed9_step4000/`
- `results/phase1_pythia160m_repeat_match_alignment_seed9_step16000/`
- `results/phase1_pythia160m_repeat_match_alignment_seed9_step143000/`
- `results/phase1_pythia160m_repeat_match_alignment_seed9_selected/`

## Source Code

- `scripts/pythia_repeat_match_alignment_trajectory.py`
- `scripts/analyze_pythia_alignment_trajectory.py`
- `scripts/analyze_pythia_seed9_alignment_selected.py`

## Reproduction Commands

Run the one-checkpoint trajectory script for `step4000`, `step16000`, and
`step143000` with seeds 1-9, then run:

```bash
python3 -u scripts/analyze_pythia_alignment_trajectory.py \
  --input-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step4000
python3 -u scripts/analyze_pythia_alignment_trajectory.py \
  --input-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step16000
python3 -u scripts/analyze_pythia_alignment_trajectory.py \
  --input-dir results/phase1_pythia160m_repeat_match_alignment_seed9_step143000
python3 -u scripts/analyze_pythia_seed9_alignment_selected.py

python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/phase1/2026-05-22-2032-pythia160m-seed9-alignment/outputs/pythia160m_seed9_alignment_checkpoint.tex
```

## Notes

The first all-checkpoint run was interrupted by unrelated GPU contention before
writing results. The reported result uses three completed one-checkpoint runs
with identical metric settings.
