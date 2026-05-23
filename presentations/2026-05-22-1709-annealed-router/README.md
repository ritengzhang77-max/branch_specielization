# Annealed Weak Router Supervision Checkpoint

Date: 2026-05-22

## Purpose

This deck summarizes the Phase 3 toy experiment testing whether weak
role-informative routing labels are needed only early as a symmetry breaker, or
whether they must remain active through most of training for branch-level
functional modularity to persist.

## Primary Artifacts

- Main PDF: `annealed_router_checkpoint.pdf`
- LaTeX source: `annealed_router_checkpoint.tex`

## Local Copied Data

- `data/annealed_summary.csv`
- `data/annealed_with_references.csv`
- `figures/annealed_router_comparison.png`

## Original Data Roots

- Analysis output:
  `results/phase3_toy_annealed_router_analysis/`
- Annealed per-seed runs:
  `results/phase3_toy_conflict_wide64_anneal_label_end*_seed*/`
- Reference conflict-task runs:
  `results/phase3_toy_conflict_router_analysis/conflict_summary.csv`

## Source Code

- Training/evaluation script:
  `scripts/toy_branch_isolation_intervention.py`
- Aggregation/plot script:
  `scripts/analyze_annealed_router_experiment.py`

## Reproduction Commands

```bash
python3 -u scripts/analyze_annealed_router_experiment.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1709-annealed-router/annealed_router_checkpoint.tex
```

## Notes

The deck is a checkpoint artifact, not a final paper figure set. Its purpose is
to record the decision-relevant result and the next experiment: checkpointed
training trajectories around label-removal times.
