# PRISM — Middle-Fusion RGB-Thermal Detection

An empirical architecture study: **where and how should two sensor streams be fused inside a
detector?** PRISM builds a two-stream YOLOX on the KAIST multispectral pedestrian benchmark,
then ablates the fusion design — fusion *stage*, fusion *operator*, and robustness to
*misalignment* — and reports proper metrics against the published literature.

The question it exists to answer: **does middle fusion actually earn its place over a good
single-modality detector or a simple late-fusion ensemble** — and if so, which fusion operator,
at what compute cost, and how gracefully does it fail when the two streams stop lining up?

## Status

**Phase 1 complete; Phase 2 (fusion-depth sweep) running.** All models trained **from scratch**
(no pretraining), lr 0.01, 16 epochs, **5 seeds** each, evaluated on the improved-test annotations.
Absolute numbers sit in the from-scratch regime (~35% vs ~19% for pretrained methods); the study's
claims are *internal deltas*, which are regime-independent. Full ledger in `docs/RESULTS.md`; the
build is narrated chapter-by-chapter in `docs/textbook/`.

| Model | MR⁻² All | Day | Night |
|---|---|---|---|
| thermal-only | 36.1 ± 1.3 | 43.9 | 20.6 |
| RGB-only | 52.5 ± 1.5 | 46.3 | 65.9 |
| **fusion (F-cat-345)** | **27.7 ± 1.9** | **32.7 ± 2.3** | **17.5 ± 1.4** |

Fusion beats the better single camera by **−8.4 / −11.2 / −3.1** (All/Day/Night); All and Day clear
2σ decisively, Night is borderline. Controls rule out the two usual false positives: it beats
*both* single cameras in daytime (genuine cross-modal benefit, not a single-modality collapse), and
a **parameter-matched** single-stream control (thermal widened to fusion's 14M params) stays at
36.1 — so the win is from *combining cameras*, not model size. RGB's night-blindness (65.9 vs 20.6)
confirms the multispectral premise.

## The dataset

[KAIST Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/)
(Hwang et al., CVPR 2015). 95k color-thermal pairs, 640×480, 20 Hz, captured from a vehicle
through a **beam splitter**.

The alignment story is the whole reason this project is interesting, and it has two halves:

- **By design, KAIST is aligned.** The beam splitter makes the cameras co-axial (no large-baseline
  parallax) and the pairs are hardware-captured at a single 20 Hz rate (no frame-rate skew).
- **In practice, it is only *weakly* aligned.** AR-CNN (Zhang et al., ICCV 2019) showed the same
  pedestrian occupies different pixel positions in RGB vs thermal, and released **KAIST-Paired**
  per-modality annotations to quantify it. Essentially the entire published literature trains
  through this residual shift without mentioning it.

So PRISM treats misalignment robustness as a first-class ablation axis, with two instruments:
**measure** the shift that is already in the data (KAIST-Paired), and **inject** controlled extra
shift/scale/temporal skew to sweep the axis. Fusion operators that assume exact pixel
correspondence should degrade; ones that pool or gate should not.

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
