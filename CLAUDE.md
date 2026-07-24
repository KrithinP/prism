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

## Status (2026-07-24)

Phase 0 + Phase 1 done. **The detector trains from scratch and the metric pipeline is wired
end-to-end.** blocks → CSPDarknet → PAFPN → decoupled head → YOLOX (**8.94M params**) → decoder →
loss + SimOTA → training loop → protocol splits → MR⁻² evaluation. **32 tests green**
(`make test`). Full 36GB KAIST verified + extracted at `data/raw/kaist-cvpr15`.

Verified smoke (thermal, 300 pedestrian frames, 20 epochs): loss 42.2→1.10, obj 37.4→0.16,
iou 0.70→0.12, no NaN; the trained model puts boxes on the pedestrians
(`docs/figures/trained_before_after.png`). That is *memorization*, not detection — the gate only
proves the loop learns.

**Two from-scratch bugs found and fixed here (both regression-tested):**
1. **Missing prior-probability bias init.** Head obj/cls biases started at 0 → σ(0)=0.5 → every
   cell screamed "object" → all-background objectness loss ≈ **18,000** → SGD diverged to NaN.
   Fix: init to −log((1−π)/π), π=0.01 (≈−4.595). The overfit-one-image test missed it because it
   used **AdamW** (self-normalizes gradient scale); **SGD** on background batches exposed it.
   *Lesson: a green suite proves the cases you tested, not correctness.*
2. **AMP-unsafe BCE.** SimOTA's internal `F.binary_cross_entropy` on sigmoided probabilities is
   forbidden under autocast; the cost matrix now runs in fp32 with autocast disabled.

**Next: LR sweep (`scripts/sweep_lr.py`) → lock the recipe → from-scratch RGB-only and
thermal-only baselines × 5 seeds → MR⁻² with the noise floor.** Then Phase 2 (two-stream fusion).

### ⚠ Annotation comparability (verified 2026-07-24 — read before quoting any number)

`annotations-xml-new-sanitized` on this disk is a **byte-identical copy** of
`annotations-xml-new` (95,324 XMLs each, zero differing files). We hold **one** annotation set
under two names, so the train=sanitized / test=improved mapping in `kaist_split()` is a **no-op**.

- **Internal comparisons stay valid** — every experiment shares the same annotations, so fusion
  A-vs-B, the misalignment sweep, and the 5-seed noise floor are unaffected. Proceed.
- **Cross-paper comparison is NOT valid yet**, and neither is the open evaluator verification
  (which works by reproducing a published number). Needs the real **sanitized-train** (Li et
  al./MSDS-RCNN) and **improved-test** (Liu et al. 2018) annotation sets — download links in the
  decisions log (2026-07-24). Installing them is a one-line change in `_SPLITS`.

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
