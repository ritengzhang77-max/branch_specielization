# Pythia Repeat-Match Checkpoint Trajectory

Date: 2026-05-22

## Purpose

This deck summarizes a real-transformer checkpoint trajectory testing whether a
repeat-match attention-role probe appears before strong causal head-ablation
effects in Pythia-14M.

## Primary Artifacts

- Main PDF: `pythia_repeat_trajectory_checkpoint.pdf`
- LaTeX source: `pythia_repeat_trajectory_checkpoint.tex`

## Local Copied Data

- `data/revision_summary.csv`
- `data/revision_seed_summary.csv`
- `data/probe_head_scores.csv`
- `data/ablation_results.csv`
- `figures/repeat_match_checkpoint_trajectory.png`

## Original Data Roots

- `results/phase1_pythia14m_repeat_match_checkpoint_trajectory/`

## Source Code

- `scripts/pythia_repeat_match_checkpoint_trajectory.py`
- Reused helpers from:
  `scripts/attention_role_specialization.py`,
  `scripts/repeat_match_ablation.py`, and
  `scripts/attention_stability.py`.

## Reproduction Commands

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/pythia_repeat_match_checkpoint_trajectory.py \
  --model-size 14m \
  --seeds 1 2 3 \
  --revisions step0 step64 step256 step1000 step4000 step16000 step64000 step143000 \
  --layers 0,1 \
  --top-k-per-layer 1 \
  --random-controls 8 \
  --probe-sequences 64 \
  --eval-sequences 64 \
  --repeat-length 32 \
  --batch-size 8 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase1_pythia14m_repeat_match_checkpoint_trajectory

python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1759-pythia-repeat-trajectory/pythia_repeat_trajectory_checkpoint.tex
```

## Notes

This is a pilot on Pythia-14M, not a final Pythia-160M result. The main
interpretation uses top-head excess loss delta over random same-layer controls,
because some random head ablations improve synthetic repeated-token loss at
intermediate checkpoints.
