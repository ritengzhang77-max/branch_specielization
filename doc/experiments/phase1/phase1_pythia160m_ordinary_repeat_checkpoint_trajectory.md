# Phase 1: Pythia-160M Ordinary Natural-Repeat Checkpoint Trajectory

Date: 2026-05-23

This checkpoint extends the natural-repeat result over training time. The final
checkpoint result showed that ordinary-phrase WikiText-103 exact 8-gram repeats
support task-specific cross-seed role transfer in Pythia-160M. This run asks
when that effect appears during training.

## Question

The tested developmental question is:

```text
Does a natural ordinary-phrase repeat role become causally important and
cross-seed transferable early in training, or only after later training?
```

This is the natural-data analogue of the synthetic repeat/local-copy trajectory
results. It is also a bridge to the Phase 3 router result: probe/gate metrics can
be positive before causal functional structure is strong.

## Setup

Task:

```text
WikiText-103 train, naturally occurring exact 8-token repeats,
ordinary-phrase spans only.
```

Shared settings:

```text
model = Pythia-160M seeds 1-9
candidate layers = all 12 layers
top_k_total = 2
probe/eval windows = 64/64, sampled without replacement
context length = 128
span length = 8
repeat candidates = 147 from 1,000,066 tokens
alignment source = task_repeat
```

Checkpoints:

```text
step0, step4000, step16000, step64000, step143000
```

Representative command:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py \
  --model-size 160m \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --revision step64000 \
  --candidate-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --top-k-total 2 \
  --random-controls 4 \
  --probe-sequences 64 \
  --eval-sequences 64 \
  --context-length 128 \
  --span-length 8 \
  --min-gap 8 \
  --window-stride 8 \
  --span-primary-category ordinary_phrase \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --dataset-split train \
  --dataset-rows 50000 \
  --min-token-stream 1000000 \
  --alignment-source task_repeat \
  --alignment-num-texts 8 \
  --alignment-batch-size 8 \
  --random-permutations 100 \
  --batch-size 8 \
  --device cuda \
  --dtype float32 \
  --output-dir results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step64000
```

Trajectory aggregation:

```bash
python3 -u scripts/analyze_natural_repeat_checkpoint_trajectory.py \
  --output-dir results/phase1_pythia160m_wikitext103_ordinary_repeat_trajectory
```

## Results

| Checkpoint | Probe spec. | Own top - random | Same-index | Task-aligned | Aligned - same | Target CI for aligned - same | Target positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| step0 | 0.0077 | 0.0009 | -0.0000 | 0.0012 | 0.0012 | [-0.0005, 0.0028] | 7/9 |
| step4000 | 0.1115 | 0.0205 | 0.0280 | 0.0434 | 0.0154 | [-0.0152, 0.0448] | 5/9 |
| step16000 | 0.1481 | 0.1756 | 0.0733 | 0.1193 | 0.0460 | [-0.0446, 0.1336] | 6/9 |
| step64000 | 0.1576 | 0.1693 | 0.0174 | 0.1349 | 0.1174 | [0.0387, 0.2099] | 8/9 |
| step143000 | 0.1623 | 0.3133 | 0.0248 | 0.2500 | 0.2252 | [0.1096, 0.3776] | 8/9 |

Pair-level aligned-minus-same comparison:

| Checkpoint | Pair mean | Pair CI | Aligned better |
|---|---:|---:|---:|
| step0 | 0.0012 | [-0.0008, 0.0033] | 41/72 |
| step4000 | 0.0154 | [-0.0060, 0.0350] | 48/72 |
| step16000 | 0.0460 | [-0.0312, 0.1153] | 52/72 |
| step64000 | 0.1174 | [0.0663, 0.1683] | 64/72 |
| step143000 | 0.2252 | [0.1510, 0.3070] | 68/72 |

## Interpretation

This is a positive developmental result, but it is slower and weaker than the
synthetic local-copy trajectory.

Supported:

```text
natural ordinary-repeat role probes appear before robust cross-seed causal
transfer.
```

At step4000, selected-head specialization is already far above initialization
(`0.1115` vs `0.0077`), but own-head causal excess is weak and target-level
aligned-minus-same remains uncertain.

Supported:

```text
own-head causal importance appears before robust target-level aligned transfer.
```

By step16000, own-head causal excess has a positive bootstrap interval
(`[0.0601, 0.3205]`), but target-level aligned-minus-same still crosses zero.
By step64000, aligned transfer becomes target-level positive
(`[0.0387, 0.2099]`) and remains stronger at final.

This gives a three-stage picture:

1. **Probe role appears**: by step4000.
2. **Own causal role strengthens**: clear by step16000.
3. **Cross-seed aligned transfer becomes robust**: clear by step64000 and final.

## Project-Level Update

The natural-repeat trajectory strengthens the paper's measurement story:

```text
Functional specialization, own causal importance, and cross-seed transferable
role identity are related but not synchronous.
```

This mirrors the Phase 3 router mechanism:

```text
gate/probe alignment can precede causal functional structure.
```

For the paper, the result should be framed as:

```text
synthetic copy/repeat tasks show early strong transfer, while unmodified
ordinary-phrase repeats develop more gradually and become robust only later in
training.
```

## Caveats

- The analysis uses task-repeat alignment, not generic Phase 0 alignment. The
  final-checkpoint generic alignment result is neutral for this task.
- The effect is target-heterogeneous at intermediate checkpoints.
- This is still a head-output ablation metric, not full path patching.
- Result folders are ignored by git; the docs and commands record how to
  reproduce them locally.

## Artifacts

- Main script:
  `scripts/pythia_natural_repeat_ngram_candidate_pool_alignment.py`
- Trajectory aggregator:
  `scripts/analyze_natural_repeat_checkpoint_trajectory.py`
- Output directories:
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step4000/`
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step16000/`
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64_step64000/`
  `results/phase1_pythia160m_wikitext103_natural_repeat_8gram_ordinary_task_alignment_seed9_n64/`
- Combined trajectory:
  `results/phase1_pythia160m_wikitext103_ordinary_repeat_trajectory/`
