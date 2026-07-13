# PRISM — Middle-Fusion RGB-Thermal Detection

An empirical architecture study: **where and how should two sensor streams be fused inside a
detector?** PRISM builds a two-stream YOLOX on the KAIST multispectral pedestrian benchmark,
then ablates the fusion design — fusion *stage*, fusion *operator*, and robustness to
*misalignment* — and reports proper metrics against the published literature.

The question it exists to answer: **does middle fusion actually earn its place over a good
single-modality detector or a simple late-fusion ensemble** — and if so, which fusion operator,
at what compute cost, and how gracefully does it fail when the two streams stop lining up?

## Status

Phase 0 — setup and literature. Nothing trained yet.

## The dataset

[KAIST Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/)
(Hwang et al., CVPR 2015). Captured through a **beam splitter**: RGB and LWIR share an optical
axis and are hardware-synced. That matters — KAIST has *no* parallax, FOV mismatch, or frame-rate
skew. It is the clean control. Real rigs are not like this, so PRISM injects misalignment
deliberately (see `docs/`) rather than pretending the benchmark already has it.

Evaluation is **log-average miss rate (MR⁻²)** over FPPI ∈ [10⁻², 10⁰] on the *reasonable*
subset, reported All / Day / Night — not mAP. Uses the sanitized train + improved test
annotations, without which results are not comparable to any published table.

## Quickstart

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
cp .env.example .env.local   # then fill in WANDB_API_KEY
```

GPU note: this repo targets an RTX 5070 Ti Laptop (Blackwell, `sm_120`). That requires PyTorch
built against **CUDA 12.8 or newer** — earlier wheels fail with "no kernel image is available".

## Definition of done

Nothing is "working" until:
1. The end-to-end slice runs: one KAIST image pair → two-stream model → detections → an MR number.
2. Single-modality baselines (RGB-only, thermal-only) are trained and match published ranges.
3. Any fusion variant is reported against *both* baselines, on the same seeds, with params/FLOPs.
   A fusion model that does not beat the better single modality is a **negative result, reported
   as such** — not a bug to be hidden.

## Layout

- `docs/LITERATURE.md` — the paper trail, the fusion-operator taxonomy, the evaluation protocol
- `docs/CURRICULUM.md` — the ground-up study plan behind this work
- `docs/textbook/` — the LaTeX textbook written as the project goes
- `src/prism/` — model, data, fusion ops, eval
- `configs/` — one config per ablation cell
- `scripts/` — data prep, training, evaluation entry points
