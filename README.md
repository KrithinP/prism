# PRISM: Controlled RGB-T Fusion on KAIST

PRISM is a validation-first architecture study of visible/thermal pedestrian
detection. It compares unimodal YOLOX-S-derived baselines, literal early and
middle fusion graphs, late probabilistic fusion, twelve feature
operators/controls, and controlled modality degradation.

The corrected study has not been trained in this checkout. Earlier 16-epoch
test-set numbers are quarantined as exploratory and must not appear as final
evidence.

## What is ready

- canonical KAIST evaluator and differential parity harness;
- official-test release firewall and hashed manifests;
- one-command KAIST download, conversion, split, and audit;
- corrected YOLOX-derived graph/loss/SimOTA/BatchNorm semantics;
- KAIST ignore-region masking during training;
- early, true one-shot, multi-scale, and late fusion;
- capacity, duplicated-input, and nested-at-initialization controls;
- ten mechanism operators plus two exact parameter-matched concat controls;
- five-seed validation-only matrix runner and frozen winner handoff;
- durable attempt-level provenance with exact commands, environment/config/data
  hashes, non-overwriting logs, atomic checkpoints, and per-frame detections;
- modality-dropout/misalignment robustness arms;
- intervention, representation, targeted attribution, sequence-bootstrap, and
  profiling scripts;
- one fail-closed post-hoc runner with predeclared model/seed coverage.

## Protocol

```text
train labels:     MSDS sanitized, 7,601 pairs
selection:        sequence-disjoint train-cv / val-cv
final test:       improved test annotations, 2,252 pairs
input:            640x512
training:         from scratch, exactly 40 epochs
seeds:            0,1,2,3,4
primary metric:   validation Reasonable All MR^-2
test access:      frozen manifest only
```

The base is a **YOLOX-S-derived detector graph under a study-specific recipe**,
not an exact reproduction of the official 300-epoch YOLOX-S training recipe.

## Setup

```sh
uv sync --extra dev --python 3.11
source .venv/bin/activate

make prepare-kaist
make certify-evaluator
make test
make preflight
make data-stats
```

The PyTorch source is configured for CUDA 12.8. Verify the wheel supports the
training GPU before starting the matrix.

## Run

```sh
make study-all
```

This runs the stages serially because each winner is frozen before the next
block starts. It finishes with profiling, five-seed interventions,
representation dumps, quantitative Grad-CAM/Integrated-Gradients analysis, and
paired sequence bootstraps. Artifacts go under `outputs/`.

## Read first

- [EXPLANATION.md](EXPLANATION.md): architecture, equations, matrix, exact commands
- [docs/LITERATURE.md](docs/LITERATURE.md): extended primary-source review
- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md): frozen execution plan
- [TODO.md](TODO.md): remaining gates and evidence requirements
- [data/README.md](data/README.md): exact dataset sources and layout

## Core claim rule

A fused model does not count as a win because it beats one weak modality. It must
be evaluated against both unimodal models, width controls, duplicated-input
branches, nested-at-initialization controls, early/late fusion, ProbEn, five training
seeds, compute, and degradation behavior.

Attention maps, gates, CKA, and effective rank are diagnostics. Causal claims
about model reliance require fixed-weight input interventions and uncertainty.
