# PRISM — Middle-Fusion RGB-Thermal Detection

This repo sits inside Krithin's AI Operating System (root AIOS at `../../CLAUDE.md`). The root
instructions still apply — voice, model routing, working style. This file adds only what's
specific to PRISM.

## What this is

An empirical architecture study on **middle (halfway) fusion** of RGB + thermal for object
detection, on a **YOLOX** base, benchmarked on **KAIST**. Assigned by Soham as a research-method
exercise: learn how architectural research is actually done — hypothesis, controls, ablation
grid, honest metrics — rather than reasoning about fusion from an armchair.

Target: **report + reproducible repo in 4–6 weeks** (started 2026-07-13).

## Status (2026-07-30)

**Phase 1 nearly complete — fusion works.** Full pipeline end-to-end: CSPDarknet → PAFPN →
decoupled head → YOLOX (8.94M) → loss + SimOTA → training loop → protocol splits → MR⁻² (full
scale/occlusion suite). Two-stream fusion (`TwoStreamYOLOX`, 13.84M) built and validated.
**39 tests green.**

**Recipe locked:** lr = 0.01 (sweep), 16 epochs (calibration showed from-scratch plateaus fast;
~35% is the data-limited ceiling, modern methods reach ~19% via pretraining). Real annotations
installed (see below). Results in `docs/RESULTS.md`; the textbook (`docs/textbook/`, 11 chapters)
narrates the whole build.

**Headline results (improved-test, from scratch, 5 seeds unless noted):**
- thermal-only: All 36.1±1.3 / Day 43.9 / Night 20.6±1.2
- RGB-only:     All 52.5±1.5 / Day 46.3 / Night 65.9±0.9 (near-blind at night)
- **F-cat-345 fusion (seed 0):** All **27.3** / Day **31.6** / Night **18.7** — beats thermal by
  8.8/12.3/1.9, past the ~2.5 noise bar; beats BOTH cameras in Day → real cross-modal benefit,
  not the APEX single-modality trap. 5-seed noise floor pending.

**From-scratch bugs found + regression-tested:** (1) missing prior-bias init → all-background
objectness ≈18,000 → SGD NaN (fix: bias = −log((1−π)/π), π=0.01); the AdamW overfit test missed
it, SGD exposed it. (2) SimOTA's BCE is AMP-unsafe → run the cost in fp32, autocast off.
(3) divergence guard on a smoothed EMA (a single-batch limit false-fired at step 22).

**Next:** F-cat-345 seeds 1–4 (fusion noise floor) → width-matched control → evaluator
verification → **Phase 2: fusion-depth sweep** (dark3/4/5/all, `fuse_stages` already supports it).

### Annotations (resolved 2026-07-29 — now leaderboard-comparable)

The real sets are installed and wired: **`annotations-sanitized-train`** (Li et al./MSDS-RCNN,
7,601 frames — the field's standard training set, listed in `imageSets/train-sanitized.txt`) and
**`annotations-improved-test`** (Liu et al. 2016, covering all 2,252 test frames). Converted from
bbGt text to XML by `scripts/convert_kaist_bbgt.py`; `_SPLITS` points at them. (The old
`annotations-xml-new{,-sanitized}` dirs were a byte-identical duplicate — left in place,
unreferenced.) Numbers are now internally valid AND comparable to the published leaderboard, though
absolute levels stay in the from-scratch regime.

**Locked training directives (Soham, see decisions log):** train **from scratch** (small model +
small data); **5 seeds on every experiment** (mean±std, 2σ bar); small **LR + LR-decay sweep**
before locking the recipe; report the **full standard KAIST metric suite** (MR⁻² Reasonable
All/Day/Night primary; scale/occlusion subsets secondary; **mAP a cross-check only** — it hid the
Apollyon modality-collapse). Recipe held fixed across all experiments once set.

Docs kept in sync: textbook `docs/textbook/` (Ch.1–5), paper skeleton `docs/paper/main.tex`,
visual trace `docs/PIPELINE_TRACE.md`, build-order study list `docs/STUDY_MODEL_PHASE.md`.

## Working agreement (binding)

- **R1 — Krithin types all implementation code.** Claude provides reference code, reviews, and
  explains. Claude makes direct edits to implementation code only on explicit command. Repo
  scaffolding, configs, docs, and the textbook are Claude's to write.
- **R2 — Theory before build.** Claude delivers the phase's theory brief / curriculum gate before
  any code for that phase is written. Krithin reads it first. No exceptions — the whole point of
  this project is the learning, not the artifact.
- **R3 — Claude authors the LaTeX textbook** (`docs/textbook/`) and all documentation directly,
  in the same session anything textbook-worthy happens (a result, a failure + its lesson, a
  decision, a first success).
  - *Textbook format (binding):* the book uses the `subfiles` package. `main.tex` holds the one
    shared preamble; every chapter starts `\documentclass[../main.tex]{subfiles}` so it compiles
    **both** on its own and as part of the book, with zero preamble duplication. Keep listings
    language-agnostic (no `language=XML`) so any snippet compiles on stock Overleaf/TeX Live.
    Krithin works chapter-by-chapter in Overleaf: he uploads/zip-imports `docs/textbook/` and each
    chapter recompiles by itself. Do not write a chapter that only compiles via `main.tex`.

## The facts that will bite you

- **KAIST is beam-splitter aligned *by design*, but only weakly aligned *in practice*.** Get this
  distinction right; an earlier version of this file got it wrong.
  - *By design:* Hwang et al. used a beam splitter, so the two cameras are co-axial — there is no
    large-baseline parallax — and pairs are hardware-captured at 20 Hz, so there is no frame-rate
    skew. Krithin's 120fps-RGB / 60fps-thermal problem and his parallax-from-physical-separation
    problem are **the Apollyon rig's, not the benchmark's**.
  - *In practice:* the pairs are **not strictly aligned**. AR-CNN (Zhang et al., ICCV 2019) showed
    the same pedestrian sits at different pixel positions in RGB vs thermal, and released
    **KAIST-Paired** per-modality annotations to quantify it. The whole field trains through this
    residual shift silently.
  - *Consequence for PRISM:* misalignment robustness is a real axis with **two** instruments —
    **measure** the shift that is already there (KAIST-Paired), and **inject** controlled extra
    shift/scale/temporal skew to sweep the axis under control. Injection alone would have been
    simulating a problem the data already has.
- **The metric is MR⁻²**, log-average miss rate over FPPI ∈ [10⁻², 10⁰] on the *reasonable*
  subset, split All / Day / Night. Lower is better. It is **not** mAP.
- **Annotations matter more than the model.** Use the sanitized train + improved test annotation
  sets the field standardized on. Training or evaluating on the raw 2015 annotations silently
  makes every number incomparable to every published result.
- **Thermal-only is a strong baseline on KAIST** — it beats RGB-only overall and dominates at
  night. Any fusion model must be reported against it. "Fusion" that merely matches thermal-only
  is a null result, and on APEX exactly this failure hid behind a clean daytime mAP of 0.99.
- **"No attention" is too coarse a rule.** SE/CBAM-style channel and spatial gating cost a few
  percent of FLOPs and belong in the grid. What's ruled out by onboard compute is *dense
  cross-attention* over full feature maps. Grade every variant on params + FLOPs + latency.

## Definition of done (per experiment)

An experiment counts only when: it ran on the sanitized/improved annotations; it reports MR⁻² on
All/Day/Night; it reports params and FLOPs; it is compared against RGB-only and thermal-only on
the same seeds; and it is logged in W&B project `prism` and in the results ledger.

Negative results get written down. That is the deliverable, not a failure of it.

## Environment

RTX 5070 Ti Laptop, 12 GB, **Blackwell sm_120** → requires PyTorch on **CUDA 12.8+** wheels.
Older builds fail with "no kernel image is available for execution".

## Where things live

- `docs/LITERATURE.md` — paper trail, fusion taxonomy, evaluation protocol
- `docs/CURRICULUM.md` — the ground-up study plan
- `docs/PLAN.md` — phased build plan and the ablation grid
- `docs/RESULTS.md` — the results ledger (every run, including the bad ones)
- `docs/textbook/` — LaTeX textbook (R3)
- `src/prism/` — implementation (Krithin's, per R1)

Decisions and AIOS-level context live at the AIOS root (`decisions/log.md`), not here.
