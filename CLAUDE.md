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

## The facts that will bite you

- **KAIST is beam-splitter aligned.** RGB and LWIR share an optical axis, hardware-synced, 20 fps.
  There is no parallax, no FOV mismatch, no frame-rate skew in this data. The 120fps/60fps +
  parallax problem is the *Apollyon rig's*, not the benchmark's. PRISM therefore **injects**
  misalignment (spatial shift/scale, temporal skew) as a deliberate ablation axis.
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
