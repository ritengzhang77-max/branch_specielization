# Toy Ontology v2 Checkpoint Deck

Date: 2026-05-23

## Purpose

This deck summarizes the Toy Ontology v2 ordinary-attention-head experiment:
20 synthetic role rows, five role families, all-distinct heterogeneous
head-dimension configs, uniform baselines, and layout controls.

## Primary Artifacts

- Open first after compilation:
  `outputs/toy_ontology_v2_checkpoint.pdf`
- Source:
  `outputs/toy_ontology_v2_checkpoint.tex`

## Source Code

- `scripts/toy_role_ontology_v2_head_dim_intervention.py`
- `scripts/analyze_role_ontology_v2.py`

## Copied Data

- `data/full_model_metric_table.csv`
- `data/full_affinity_table.csv`
- `data/full_family_metric_table.csv`
- `data/full_role_top_dim_counts.csv`
- `data/layout_model_metric_table.csv`
- `data/layout_affinity_table.csv`
- `data/layout_role_top_dim_counts.csv`
- `data/role_dataset_examples.json`
- `data/phase3_toy_role_ontology_v2.md`

## Figures

- `figures/full_config_metric_bars.png`
- `figures/full_family_specialization_heatmap.png`
- `figures/layout_config_metric_bars.png`

## Original Data Roots

- `results/phase3_toy_role_ontology_v2_full_1600_20260523`
- `results/phase3_toy_role_ontology_v2_layout_1600_20260523`

## Reproduction

Compile the deck with:

```bash
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/phase3/2026-05-23-2130-toy-ontology-v2/outputs/toy_ontology_v2_checkpoint.tex
```

The exact experiment commands are listed in
`data/phase3_toy_role_ontology_v2.md`.

## Notes

The deck is a checkpoint summary. The comprehensive written memo is
`doc/experiments/phase3/phase3_toy_role_ontology_v2.md`.
