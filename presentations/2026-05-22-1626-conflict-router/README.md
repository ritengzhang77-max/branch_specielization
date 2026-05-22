# Conflict Router Checkpoint Deck

## Purpose

This deck summarizes the Phase 3 conflict-heavy branch-routing checkpoint. The
checkpoint asks whether direct predecessor-vs-successor role conflict makes
unlabeled entropy/load-balancing routing pressure discover branch-level
functional modularity.

## Primary Artifacts

- Open first: `conflict_router_checkpoint.pdf`
- Source: `conflict_router_checkpoint.tex`

## Local Copied Data

- `data/conflict_summary.csv`
- `data/conflict_with_standard_refs.csv`
- `figures/conflict_router_comparison.png`

## Original Data Roots

- Analysis output:
  `results/phase3_toy_conflict_router_analysis/`
- Per-seed experiment outputs:
  `results/phase3_toy_conflict_wide64_unlabeled_unconstrained_seed*/`
  `results/phase3_toy_conflict_wide64_unlabeled_balance1_seed*/`
  `results/phase3_toy_conflict_wide64_unlabeled_entropy005_balance1_seed*/`
  `results/phase3_toy_conflict_wide64_weak_label005_seed*/`
  `results/phase3_toy_conflict_wide64_oracle_route_seed*/`

## Reproduction Commands

```bash
python3 -u scripts/analyze_conflict_router_experiment.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1626-conflict-router/conflict_router_checkpoint.tex
```

## Notes

The PDF was compiled from the Beamer source with the local LaTeX deck compiler.
Representative slides were rendered with `pdftoppm` and inspected for table and
label readability.
