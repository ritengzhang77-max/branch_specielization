# Branch Specialization in Attention-Based Architectures

This repository supports a research project on whether transformer attention heads,
MoE experts, and routed attention experts behave like "branches" in the branch
specialization literature.

The central question is:

> Does structural branch design in attention architectures induce stable functional
> specialization or functional modularity across random seeds?

The project starts with post-hoc measurements on existing multi-seed models
(Pythia and MultiBERTs), then tests architectural interventions such as
heterogeneous per-head dimensions.

## Repository Layout

- `doc/plan.md`: full project roadmap.
- `doc/metric_literature_review.md`: metric provenance, citation snapshot, and
  trustworthiness review.
- `doc/research_questions.md`: clarified framing, research questions, and phases.
- `doc/provenance_log.md`: running log of project decisions, searches, and artifacts.
- `scripts/`: reproducible analysis scripts.

## First Milestone

Reproduce a minimal version of the attention-head stability baseline:

1. Load two or more Pythia seeds.
2. Run the same probe texts through each seed.
3. Extract per-layer, per-head attention matrices.
4. Compute cross-seed head similarity.
5. Compare raw head-index similarity with Hungarian-matched similarity.

