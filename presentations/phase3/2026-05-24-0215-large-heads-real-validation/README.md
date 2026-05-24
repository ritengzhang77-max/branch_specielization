# Larger Head-Count And Real-Model Validation Deck

Date: 2026-05-24

## Purpose

This deck summarizes the overnight follow-up requested by the user:

- answer how many heads were in the previous Toy Ontology v2 experiment;
- test whether more ordinary attention-head slots rescue family-level
  modularity;
- diagnose mixed or failed results with targeted controls;
- run an initial real-model ordinary-head role validation.

## Primary Artifacts

- Primary PDF:
  `outputs/large_heads_real_validation_checkpoint.pdf`
- Beamer source:
  `outputs/large_heads_real_validation_checkpoint.tex`

## Copied Data

- `data/large32_model_metric_table.csv`
- `data/large32_affinity_table.csv`
- `data/large32_dimension_modularity_table.csv`
- `data/large16_model_metric_table.csv`
- `data/large16_affinity_table.csv`
- `data/large16_dimension_modularity_table.csv`
- `data/pythia160m_deduped_layer_role_summary.csv`
- `data/pythia160m_local_copy_revision_summary.csv`
- `data/pythia160m_natural_repeat_revision_summary.csv`
- `data/phase3_large_head_count_and_real_validation.md`

## Original Data Roots

- `results/phase3_toy_role_ontology_v2_large_heads_1000_20260523`
- `results/phase3_toy_role_ontology_v2_large_heads_2layer_2000_20260523`
- `results/phase3_real_model_role_probe_pythia160m_deduped_float32_20260524`
- `results/phase1_pythia160m_local_copy_candidate_pool_layers2_4_top2`
- `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_task_alignment_seed9_n128`

## Reproduction

Compile with:

```bash
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/phase3/2026-05-24-0215-large-heads-real-validation/outputs/large_heads_real_validation_checkpoint.tex
```

Experiment commands are listed in
`data/phase3_large_head_count_and_real_validation.md`.
