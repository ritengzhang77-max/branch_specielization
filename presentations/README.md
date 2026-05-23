# Presentation Index

This directory contains checkpoint decks for the branch-specialization research
project. Each dated subdirectory is a self-contained snapshot with source,
compiled PDF, copied data, and provenance notes when available.

## 2026-05-22

- `2026-05-22-1902-pythia160m-alignment-trajectory/`: Pythia-160M
  repeat-match alignment trajectory. The main result is that causal
  repeat-match transfer across seeds is weak by same head index but strong after
  raw-score alignment, especially at the final checkpoint.
- `2026-05-22-1626-conflict-router/`: conflict-heavy branch-routing checkpoint.
  The main result is that direct predecessor-vs-successor role conflict did not
  make unlabeled entropy/load-balancing routing discover causal branch
  modularity, while weak labels and oracle routing did.
- `2026-05-22-1709-annealed-router/`: annealed weak-router supervision
  checkpoint. The main result is that brief early role labels did not persist,
  while labels through 75% of training produced full top-branch role separation
  after label removal, but weaker causal separation than always-on labels.
- `2026-05-22-1745-router-trajectory/`: router trajectory checkpoint. The main
  result is a gate-before-causality lag: role-aligned gates appeared by step
  400, but causal branch modularity appeared only after continued weak labels to
  about step 600.
- `2026-05-22-1759-pythia-repeat-trajectory/`: Pythia-14M repeat-match
  checkpoint trajectory. The main result is an analogous probe-before-causality
  lag in a real pretrained transformer: repeat-match specialization rises before
  top-head ablations become strongly worse than random same-layer controls.
