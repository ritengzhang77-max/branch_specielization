# Presentation Index

This directory contains checkpoint decks for the branch-specialization research
project. Each dated subdirectory is a self-contained snapshot with source,
compiled PDF, copied data, and provenance notes when available.

## 2026-05-22

- `2026-05-22-1626-conflict-router/`: conflict-heavy branch-routing checkpoint.
  The main result is that direct predecessor-vs-successor role conflict did not
  make unlabeled entropy/load-balancing routing discover causal branch
  modularity, while weak labels and oracle routing did.
- `2026-05-22-1709-annealed-router/`: annealed weak-router supervision
  checkpoint. The main result is that brief early role labels did not persist,
  while labels through 75% of training produced full top-branch role separation
  after label removal, but weaker causal separation than always-on labels.
