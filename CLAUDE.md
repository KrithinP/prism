# PRISM: Controlled RGB-Thermal Fusion

This repository is an empirical architecture study of visible/thermal pedestrian
detection on KAIST. Read `EXPLANATION.md`, `docs/EXPERIMENT_PLAN.md`, and
`TODO.md` before changing the protocol.

## Current Status (2026-08-06)

The corrected, validation-first implementation is ready for execution on the
training machine. It has not been trained in this checkout. Earlier 8/16-epoch
runs selected or evaluated architecture choices on the official test set; their
numbers and figures are quarantined as exploratory history in `docs/RESULTS.md`
and `docs/textbook/`. They are not scientific evidence for the corrected study.

The current system includes:

- a YOLOX-S-derived CSPDarknet/PAFPN/decoupled-head base;
- YOLOX-consistent BatchNorm, IoU loss, SimOTA, and NMS semantics;
- KAIST ignore-region handling in both training and evaluation;
- early, one-shot, multi-scale middle, and late fusion graphs;
- ten fusion mechanisms, two parameter-matched operator controls, and
  capacity/nestedness controls;
- deterministic robustness interventions and representation diagnostics;
- five-seed, validation-only architecture selection;
- a fail-closed official-test release firewall;
- a canonical evaluator differentially certified against the pinned MLPD code.

## Frozen Protocol

```text
train annotations: MSDS sanitized, 7,601 pairs
selection:         sequence-disjoint train-cv / val-cv
final test:        improved test annotations, 2,252 pairs
input:             640x512
initial training:  from scratch, exactly 40 epochs
seeds:             0,1,2,3,4
primary metric:    validation Reasonable All MR^-2
test access:       frozen manifest only
```

Initial experiments deliberately fix secondary decisions such as epochs. Horizon
saturation is a later confirmation after the major architecture and operator
decisions are frozen.

The graph is YOLOX-S-derived, but the 40-epoch from-scratch recipe is specific to
this controlled study. Do not describe it as an exact reproduction of the
official 300-epoch YOLOX recipe.

## Execution

```sh
uv sync --extra dev --python 3.11
source .venv/bin/activate

make prepare-kaist
make certify-evaluator
make test
make preflight
make study-discovery
make data-stats
make study-operators
make study-robustness
make summary
```

`make study-all` performs the experimental stages in that order and then runs
the predeclared profiling, intervention, representation, attribution, and
sequence-bootstrap suite. Do not bypass preflight for a full run.

## Scientific Rules

- Never use official test data for training, hyperparameters, architecture
  choice, checkpoint choice, threshold choice, or qualitative example choice.
- Every core learned arm uses all five frozen seeds and exactly 40 epochs.
- Block B reports early and late fusion, but selects the Block C template only
  from feature-fusion graphs B2-B10.
- Block C holds that graph fixed while changing only the fusion operator.
- Block D is a separate robustness-training study, not a replacement for clean
  architecture selection.
- A fusion win must be compared with both unimodal baselines, wider unimodal
  controls, duplicated-input branches, late ProbEn, compute, and degradation.
- Report raw per-seed and per-frame values before summaries. Never silently omit
  failed seeds.
- Attention, gates, CKA, and effective rank are descriptive. Model-behavior
  causal claims require fixed-checkpoint interventions with paired uncertainty.
- The nested residual controls equal their own anchor feature at initialization
  and contain that function in the hypothesis class. They do not guarantee SGD
  will preserve an independently trained unimodal detector.

## Dataset Facts

KAIST uses beam-splitter, synchronized RGB/thermal capture, so the original
120 Hz/60 Hz asynchronous deployment problem is not directly identified by this
benchmark. Residual weak alignment still exists. Controlled spatial shifts are
therefore a robustness experiment, not a claim that KAIST reproduces the target
rig's timing and parallax.

Use only the exact downloads and checks in `scripts/prepare_kaist.py` and
`data/README.md`. Do not substitute KAIST-Paired or train02 while comparing
against this protocol; those define different sample/annotation protocols and
their paper numbers are not directly comparable.

## Authoritative Files

- `EXPLANATION.md`: formulation, graph equations, arm definitions, interpretation
- `docs/LITERATURE.md`: extended primary-source literature review
- `docs/EXPERIMENT_PLAN.md`: frozen execution and statistical plan
- `TODO.md`: remaining evidence gates
- `data/README.md`: exact data sources, hashes, and layout
- `src/prism/protocol.py`: machine-enforced frozen constants
- `docs/RESULTS.md`: historical quarantine plus future corrected ledger

The fetched code under `external/MLPD-eval/` is ignored third-party reference
material. `scripts/pin_mlpd_eval.py` fetches the exact audited commit and
regenerates the tracked manifest. Do not format or modify the fetched copy;
changes invalidate provenance and evaluator parity.

## Environment

The known training target is an RTX 5070 Ti Laptop with 12 GB. Blackwell
`sm_120` requires CUDA 12.8-or-newer PyTorch wheels; this is configured in
`pyproject.toml`. Confirm CUDA availability through `scripts/preflight.py`.
