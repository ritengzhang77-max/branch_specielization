# Branch Specialization Research TODO

Last updated: 2026-05-23

## Level 0: Non-Negotiable Scope

- [ ] Keep the primary unit as ordinary attention heads unless the user explicitly approves a side experiment.
- [ ] Present results as baseline -> experiment -> interpretation.
- [ ] Keep the three questions separate:
  - structural role affinity: which structural head type gets a role?
  - functional specialization: how concentrated is each role over heads?
  - functional modularity: do related role families cluster together across heads?
- [ ] When uncertain about a change in unit, claim, or experiment family, stop and ask.

## Level 1: Active Experiment

- [x] Run two-head non-uniform controls against the strong `uniform2` baseline.
  - Baseline: `uniform2 = [64, 64]`
  - Hetero2 controls: `[32, 96]`, `[48, 80]`, `[16, 112]`
  - Same total attention head dimension: `128`
  - Same number of heads: `2`
  - Same role ontology: `local_a`, `local_b`, `kv_a`, `kv_b`, `induction_short`, `induction_long`
- [x] Summarize hetero2 results with a baseline comparison table:
  - affinity: top head/type counts
  - specialization: specialization and effective heads
  - modularity: family gap and ARI
- [x] Update provenance and relevant plan docs after the hetero2 run.

### Hetero2 Result Snapshot

- Baseline: `uniform2 [64,64]`
  - specialization `0.636`
  - effective heads `2.25`
  - family gap `0.653`
  - ARI `1.000`
- Hetero2:
  - `[32,96]`: specialization `0.695`, family gap `0.480`, ARI `0.889`
  - `[48,80]`: specialization `0.756`, family gap `0.601`, ARI `1.000`
  - `[16,112]`: specialization `0.702`, family gap `0.327`, ARI `0.667`
- Interpretation:
  - hetero2 strengthens structural role affinity and specialization;
  - hetero2 does not beat `uniform2` on ontology-level modularity.

## Level 2: Immediate Follow-Ups

- [ ] Add more matched head-dimension controls.
  - `[24, 24, 40, 40]`
  - `[16, 48, 16, 48]`
  - `[8, 40, 40, 40]`
  - goal: separate large-head capacity from heterogeneity shape.
- [ ] Add more subroles inside the existing ontology.
  - more local-copy variants
  - more KV-lookup variants
  - more induction variants, especially because induction did not simply prefer the 64-dim head
- [ ] Inspect per-role rows, not only aggregate metrics, before drawing the next conclusion.
- [ ] Decide whether to pause toy capacity sweeps and move to broader task families, because hetero2 suggests capacity imbalance alone does not improve modularity over `uniform2`.

## Level 3: Broader Toy Task Expansion

- [ ] Try more different task families, not only variants of the current local/KV/induction ontology.
  - suppression/distractor tasks
  - positional/BOS/SEP tasks
  - copy-with-conflict tasks
  - longer-range retrieval tasks
  - compositional lookup tasks
- [ ] Build an expanded role ontology and rerun:
  - `uniform4`
  - `uniform2`
  - best hetero4 controls
  - best hetero2 controls
- [ ] Test whether family clustering remains stable as the ontology gets larger.

## Level 4: Real-Model Validation

- [ ] Translate the strongest toy probes to real pretrained ordinary heads.
  - Pythia local-copy probes
  - Pythia induction/repeated-ngram probes
  - Pythia KV-like retrieval probes if feasible
  - MultiBERTs follow-up if Pythia signal is clean
- [ ] Report real-model results with the same table format:
  - baseline/head-index distribution
  - structural role affinity if structural variation exists
  - specialization
  - family modularity

## Level 5: Paper Framing

- [ ] Keep the current best claim narrow:
  - heterogeneous head dimensions create structural role affinity and increase specialization in toy ordinary-head models.
- [ ] Do not claim full functional modularity unless hetero beats strong capacity/head-count baselines.
- [ ] Use `structural role affinity` as the central vocabulary for the user's intended claim.
