# Phase 1: Pythia-160M Attention-Role Specialization Proxies

Date: 2026-05-21

This run adds the first task-proxy specialization measurement on top of the Phase
0 attention-score alignment. It is not causal patching yet. It estimates simple
attention-role specialization scores `S(h,t)` from attention probabilities, then
compares those role distributions across seeds by raw head index and by the
independently computed raw-attention-score Hungarian alignment.

## Command

```bash
python3 -u scripts/attention_role_specialization.py \
  --model-size 160m \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --revision step143000 \
  --probe-file probes/phase0_probe_texts.txt \
  --max-length 64 \
  --batch-size 2 \
  --synthetic-repeat-sequences 32 \
  --synthetic-repeat-length 32 \
  --alignment-summary results/phase0_pythia160m_seed1_to_9_step143000_raw_scores/summary.json \
  --random-permutations 200 \
  --output-dir results/phase1_pythia160m_attention_role_specialization
```

## Roles

- **BOS**: average attention mass to token position 0.
- **Previous-token**: average attention mass from position `i` to `i-1`.
- **Repeat-match**: on synthetic repeated-token sequences `[x_1 ... x_n x_1 ... x_n]`,
  average attention mass from the second occurrence of `x_i` to the first
  occurrence of `x_i`. This is an induction-style attention-pattern proxy, not a
  full causal induction-head test.

For each layer and seed, scores are normalized over heads:

```text
S(h,t) = score(h,t) / sum_h score(h,t)
```

## Aggregate Cross-Seed Consistency

| Role | Raw distribution similarity | Aligned distribution similarity | Random distribution similarity | Raw top-head match | Aligned top-head match |
|---|---:|---:|---:|---:|---:|
| BOS | 0.8317 | 0.8566 | 0.8310 | 0.0787 | 0.1296 |
| Previous-token | 0.7531 | 0.8116 | 0.7531 | 0.1157 | 0.2824 |
| Repeat-match | 0.5152 | 0.6191 | 0.5057 | 0.1019 | 0.2338 |

Interpretation:

- BOS is high-similarity but mostly uninformative because random permutation is
  almost equally high. The role is diffuse rather than specialized.
- Previous-token improves under raw-score alignment, but the distribution remains
  fairly broad.
- Repeat-match has the clearest role-specific alignment signal. Raw consistency
  is near random overall, but aligned consistency is meaningfully higher.

## Layer-Level Specialization

The repeat-match role is highly concentrated in the earliest layers:

| Role | Layer | Mean max `S(h,t)` | Mean effective heads |
|---|---:|---:|---:|
| Repeat-match | 0 | 0.7728 | 2.84 |
| Repeat-match | 1 | 0.8045 | 2.26 |
| Repeat-match | 2 | 0.4588 | 5.63 |
| Repeat-match | 3 | 0.5669 | 4.61 |
| Previous-token | 3 | 0.2215 | 10.31 |
| BOS | 11 | 0.1300 | 10.60 |

The BOS and previous-token roles are much more distributed. Repeat-match is the
first role in this project that looks like a real branch-specialization target:
one or two heads dominate in early layers.

## Layer-Level Cross-Seed Consistency for Repeat-Match

| Layer | Raw similarity | Aligned similarity | Random similarity | Raw top match | Aligned top match |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3047 | 0.5346 | 0.1863 | 0.2500 | 0.5556 |
| 1 | 0.1519 | 0.6883 | 0.1541 | 0.0833 | 0.7778 |
| 2 | 0.3228 | 0.4829 | 0.3257 | 0.0833 | 0.3889 |
| 3 | 0.2908 | 0.4432 | 0.2809 | 0.1111 | 0.3056 |
| 4 | 0.4041 | 0.4306 | 0.4083 | 0.0556 | 0.0278 |
| 5 | 0.4714 | 0.5102 | 0.4759 | 0.1111 | 0.1944 |
| 6 | 0.4624 | 0.5009 | 0.4605 | 0.1111 | 0.0833 |
| 7 | 0.7209 | 0.7235 | 0.7135 | 0.1111 | 0.1389 |
| 8 | 0.8020 | 0.8119 | 0.7943 | 0.1111 | 0.1389 |
| 9 | 0.8342 | 0.8408 | 0.8327 | 0.0278 | 0.1111 |
| 10 | 0.7528 | 0.7641 | 0.7638 | 0.0556 | 0.0000 |
| 11 | 0.6640 | 0.6982 | 0.6720 | 0.1111 | 0.0833 |

Layers 0-1 are the important result. Repeat-match is concentrated, raw top-head
identity is unstable, and aligned top-head identity is much more stable.

Layer 1 is especially clean:

```text
raw top-head match rate:     0.0833
aligned top-head match rate: 0.7778
```

This is exactly the weak-universality pattern the project is looking for.

## Main Finding

The strongest current claim is:

> Pythia-160M learns a concentrated early-layer repeat-match attention role
> across seeds, but the responsible head index is unstable. Raw attention-score
> alignment recovers the corresponding role across seeds.

This directly supports the project framing:

```text
functional specialization exists;
same-index stability is weak;
permutation-aligned stability is substantially stronger.
```

## Caveats

- This is an attention-pattern role proxy, not a causal ablation/path-patching
  result.
- The repeat-match metric is induction-style but not a full induction-copying
  behavioral test.
- The synthetic repeated-token setup may overemphasize simple token matching.
- BOS is not a good specialization target in this setup because it is too diffuse.

## Next Decisive Experiment

Run causal validation for the early-layer repeat-match heads:

1. For each seed, identify top repeat-match heads in layers 0-1.
2. Ablate or patch those heads on a repeated-token next-token prediction task.
3. Compare performance drop against random heads and against raw-score-aligned
   corresponding heads across seeds.

This would upgrade the result from "attention-role specialization" to causal
functional specialization.

