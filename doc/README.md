# Document Layout

This folder now separates permanent paper-planning documents from time-variant
experiment reports.

## Permanent Planning Documents

These stay in `doc/` because they are expected to inform the final paper:

- `plan.md`
- `research_questions.md`
- `project_direction_attention_heads_primary.md`
- `project_plan_v2_attention_head_structure.md`
- `three_question_framework_attention_heads.md`
- `baseline_comparisons_three_questions.md`
- `metric_literature_review.md`
- `big_role_ontology_proposal.md`
- `role_task_organization.md`
- `provenance_log.md`

## Time-Variant Reports

Past experiment reports live under:

```text
doc/experiments/phase0/
doc/experiments/phase1/
doc/experiments/phase3/
```

Autonomous-work logs live under:

```text
doc/logs/autonomous_sleep/
```

Side-branch research that is not the current ordinary-attention-head direction
lives under:

```text
doc/side_branches/
```

Rule of thumb: if a document states a durable definition, framing, ontology,
metric, or paper plan, keep it in `doc/`. If it reports one run, one checkpoint,
one dated sweep, or a superseded side branch, put it in a subfolder.
