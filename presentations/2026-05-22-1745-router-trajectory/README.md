# Router Trajectory Checkpoint

Date: 2026-05-22

## Purpose

This deck summarizes the checkpointed trajectory experiment that measures when
role-aligned gates and causally separable branch computations appear during
training.

## Primary Artifacts

- Main PDF: `router_trajectory_checkpoint.pdf`
- LaTeX source: `router_trajectory_checkpoint.tex`

## Local Copied Data

- `data/trajectory_by_step.csv`
- `data/trajectory_rows.csv`
- `figures/router_trajectory_metrics.png`

## Original Data Roots

- Analysis output:
  `results/phase3_toy_router_trajectory_analysis/`
- Per-condition runs:
  `results/phase3_toy_trajectory_end400/`
  `results/phase3_toy_trajectory_end800/`
  `results/phase3_toy_trajectory_end1200/`
  `results/phase3_toy_trajectory_always_label/`
  `results/phase3_toy_trajectory_unlabeled_entropy_balance/`

## Source Code

- Training/evaluation script:
  `scripts/toy_branch_isolation_intervention.py`
- Aggregation/plot script:
  `scripts/analyze_router_trajectory_experiment.py`

## Reproduction Commands

```bash
python3 -u scripts/analyze_router_trajectory_experiment.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1745-router-trajectory/router_trajectory_checkpoint.tex
```

## Notes

This is a checkpoint deck. The main decision is that the toy mechanism is now
clear enough to move toward a less hand-designed routed attention setting rather
than continuing broad toy sweeps.
