# PRISM — Phased Build Plan & Ablation Grid

Companion to `LITERATURE.md` (ground truth on the field) and `CURRICULUM.md` (the study
schedule; its weekly theory gates are the R2 gates referenced below). Read those first.
This document is the execution plan: what gets built, in what order, which experiments run,
what they cost, and what we predict they show — written down **before** any of them run.

Started 2026-07-13. Target: report + reproducible repo in 4–6 weeks.

---

## 0. The thesis of this plan

LITERATURE.md §7.2 is the load-bearing fact: **on aligned data, the fusion operator barely
matters.** The published spread across a 100× fusion-cost range is ~2 MR points; annotation
cleanup alone was worth ~4. If this project were just "sweep operators on clean KAIST," the
honest outcome would be a flat table and the lesson "architecture research is noise-chasing."

So the plan is built around a different question:

> **Operators are interchangeable when pixels correspond. When do they stop being
> interchangeable — and which ones fail first?**

The experiment design therefore has two arms with opposite expected outcomes:

1. **The control arm (Phases 2–3):** the clean-KAIST stage & operator sweep. **Pre-registered
   prediction: mostly null** — most operators within 2σ of concat+1×1. This arm exists to prove
   we can measure honestly, to establish the noise floor, and to replicate/update Liu et al.
   2016 on an anchor-free base (which nobody has published — LITERATURE §6.6, §7.2).

2. **The headline arm (Phase 4): misalignment.** Take every trained operator and degrade the
   RGB↔thermal correspondence in controlled ways — spatial shift, scale mismatch, and
   **temporal skew** (essentially unstudied; LITERATURE §7.8 calls a simulated-skew curve novel
   and publishable). Prediction: the flat clean-data table **fans out**. Operators that assume
   per-pixel correspondence at high resolution (sum/max at stride 8) collapse; operators that
   pool, gate, or fuse deep degrade gracefully. This is where architecture choices actually
   have consequences — and it maps 1:1 onto the Apollyon rig (120/60 fps skew, FOV mismatch,
   parallax), which KAIST's beam-splitter geometry doesn't have.

Two instruments for the misalignment axis (per `CLAUDE.md`):
- **Measure** the weak misalignment KAIST already has (KAIST-Paired, AR-CNN ICCV 2019).
- **Inject** controlled extra shift / scale / temporal skew on top.

Crucial economy: **eval-time injection is nearly free** (one eval pass ≈ 10–15 min; no
retraining), while **train-time robustness costs a full retrain**. The grid exploits this: the
big misalignment sweep is eval-only over checkpoints we already have from Phases 2–3; only 3
selected configs get the expensive jitter-retrain arm.

---

## 1. Fixed experimental protocol (locked in Phase 0, then never touched)

Per CURRICULUM Track 5 rule 5: decide once, write down, freeze. Changing any of this
mid-study invalidates every row already run.

| Item | Value |
|---|---|
| Base detector | **YOLOX-S** (CSPDarknet + PAFPN + decoupled anchor-free head, SimOTA) |
| Input resolution | **640×512** (the field's evaluation resolution; LITERATURE §1.5.7) |
| Train set | **Sanitized train annotations (Li et al. BMVC 2018), 7,601 images** |
| Test set | **Improved test annotations (Liu et al. 2018), 2,252 images**, Reasonable setting |
| Evaluator | **MLPD `evaluation_script.py`** + its `KAIST_annotation.json` — one script for every number ever reported |
| Metric | MR⁻² (LAMR), FPPI ∈ [10⁻², 10⁰], IoU>0.5, Caltech ignore rules; **All / Day / Night**, always |
| Schedule | **60 epochs** from COCO-pretrained YOLOX-S weights (both streams init from the same weights; thermal replicated to 3ch), SGD + cosine, 5-epoch warmup, **Mosaic+MixUp off for the last 15 epochs**, identical for every row |
| Batch / precision | **batch 8, AMP (bf16)**. Expected peak ~9–10 GB on the 12 GB card. Fallback if OOM: batch 4 + grad-accum 2 (note: BN stats then come from micro-batches of 4 — acceptable, but record it; do NOT silently change batch between rows) |
| Augmentation | Geometric augs (mosaic layout, crop, scale, flip, affine): sampled **once**, applied identically to both modalities (implemented by stacking RGB+T into one 6-channel array through the aug pipeline). Photometric (HSV) on **RGB only**; thermal gets its own mild intensity-jitter/noise. MixUp: same λ, same partner pair. Enforced by the correspondence unit test (Phase 0). |
| Thermal preprocessing | KAIST thermal is 8-bit; normalize with **thermal train statistics** (ProbEn precedent: their thermal mean 135.438 mattered). Fixed global normalization, NOT per-frame min/max (train/test distribution bug — CURRICULUM Track 4B). |
| Score threshold at eval | 0.01 (FPPI sweep needs support down to 10⁻²) |
| Tracking | W&B project `prism`: full config, git hash, params, GFLOPs, MR All/Day/Night, MR–FPPI curve, latency, fixed 8-image qualitative panel. Every run also appended to `docs/RESULTS.md`. |
| Per-experiment reporting | MR⁻² All/Day/Night (mean±std, seed count stated), params (M), GFLOPs @640×512, latency (ms, see §5/Phase 5), vs RGB-only and thermal-only on same seeds — the `CLAUDE.md` definition of done |

**Hyperparameter search (LR, weight decay, mosaic strength) happens ONCE, in Phase 1, on the
`F-cat-345` baseline only, then the winning recipe is frozen for every row.** COCO-tuned
Mosaic/MixUp strength on 7.6k images is untested territory (LITERATURE §6.6); budget 3–4 short
probe runs for this and expect to weaken MixUp. Sweeps are for this step only — never for
ablation rows.

---

## 2. Compute budget — the honest arithmetic

**Hardware:** one RTX 5070 Ti Laptop (12 GB, sm_120), torch 2.11+cu128 (verified). ~30 GB RAM.
Human time ~15 h/week; GPU runs unattended overnight.

**Per-run cost estimates** (7,601 imgs, batch 8 → ~950 iters/epoch × 60 epochs ≈ 57k iters):

| Run type | Est. throughput | Est. wall-clock |
|---|---|---|
| Single-stream YOLOX-S (RGB-only, thermal-only, early-fusion) | ~6–7 it/s | **~2.5 h** |
| Two-stream YOLOX-S (any cheap fusion op) | ~4 it/s | **~4 h** |
| Width-scaled single-stream (param-matched control) | ~4 it/s | **~4 h** |
| Two-stream + cross-attention @P5 | ~3 it/s | **~5 h** |
| One eval pass (2,252 imgs + MR computation) | — | **~0.2 h** |

These are estimates; **Phase 0's thin slice measures the real number** and if a full run comes
out >6 h, the cut list in §10 activates immediately.

**Envelope:** ~10 h/night × ~6 nights/week × 4 build-weeks ≈ **240 GPU-h** realistically
schedulable. The grid below totals **≈190 GPU-h (≈175 must-run)** — ~25% headroom for failed
runs, the hyperparameter probe, and re-runs after bug fixes. That headroom is not optional;
Phase 1 *will* consume some of it re-running after the first thermal-normalization bug.

Overnight discipline: every night from Week 3 onward should have a queued run. A simple
`queue.txt` + shell loop that trains configs serially and fires the eval sweep afterward is
part of the Phase 1 build.

---

## 3. Phases

Each phase: **R2 theory gate → thin end-to-end slice → scale up → exit criteria.** Gates are
the corresponding CURRICULUM week gates; they are hard gates.

### Phase 0 — Data lockdown, verified evaluator, harness (Weeks 1–2, ~15 GPU-h)

**Theory gate (CURRICULUM Weeks 1–2):** YOLOX paper + the 5 source files (`yolox.py`,
`darknet.py`, `yolo_pafpn.py`, `yolo_head.py`, `losses.py`); OTA §1,3; Halfway Fusion (all);
AR-CNN §1, 3.1–3.2; thermal physics Ch.1. Gate checks: draw dark3/4/5 shapes from memory and
mark the fusion attach points; explain `dynamic_k_matching`; show 5 KAIST pairs whose RGB and
thermal boxes visibly don't line up.

**Krithin builds (order matters):**
1. **Environment + trust baseline.** YOLOX repo running on cu128; reproduce the published
   YOLOX-S COCO mAP on a val subset with the released checkpoint. You must trust the base
   before modifying it.
2. **Dataset acquisition.** Full KAIST (set00–11 — the **full-rate videos**, not just sampled
   frames: Phase 4's temporal-skew instrument needs frames t±k around each test frame; tens of
   GB, start the download day 1). Sanitized train + improved test annotations (links in
   LITERATURE §1.3 / CalayZhou resource repo). **KAIST-Paired** from the AR-CNN repo.
   - *Verify step:* KAIST-Paired is documented as a **train-set** relabeling. Check whether
     the AR-CNN repo also ships paired **test** annotations. If yes → Phase 4 can stratify the
     test set by measured misalignment. If no → the measurement instrument becomes "shift
     statistics measured on train" + qualitative test examples; the strata figure (Fig 5)
     downgrades accordingly. Record the answer in `docs/DATA.md`.
3. **`docs/DATA.md`** — the lockdown document: annotation sets, frame-sampling rule
   (train = the 7,601 sanitized frames; test = the standard 2,252), image size, thermal
   normalization constants (computed from train), day/night split source. Written once,
   frozen.
4. **Dataloader** returning aligned (RGB, thermal, boxes) triplets; **visualize 50 random
   pairs with boxes on both modalities and actually look at them.**
5. **The MR⁻² evaluator, VERIFIED.** Wire the MLPD eval script + a YOLOX-output→KAIST-format
   converter. **Verification is non-negotiable: reproduce a published number before trusting
   any of your own.** Cheapest path: run the eval script on a released detection file (MLPD
   and MS-DETR repos ship result files / reference detections) and match the published MR to
   within ~0.1. If no detection file is runnable, run a released checkpoint (MLPD or MBNet)
   and match its published MR to within a fraction of a point. Until this passes, no training
   result means anything (LITERATURE §1.5 lists the 8 silent ways evaluators go wrong).
6. **Augmentation-correspondence unit test.** Assert: a marked pixel in RGB and its thermal
   counterpart land at the same coordinates after the full geometric pipeline (incl. mosaic
   and mixup); photometric augs change RGB only; thermal-noise aug changes thermal only. Runs
   in CI/pre-commit. This test is the guardrail for the single most likely
   silently-fatal bug in the project.
7. **Thin end-to-end slice:** fine-tune single-stream YOLOX-S on **thermal only, 500 train
   images, 10 epochs**, evaluate through the full MR pipeline, log to W&B. This proves
   loader → train → detect → convert → MR⁻² end-to-end and measures real it/s for the §2
   budget. It does not need to produce a good number; it needs to produce *a* number.

**Reference-code guidance (R1 — pointers, not finished code):** MLPD repo for the evaluator
and KAIST JSON; YOLOX `yolox/data/data_augment.py` + mosaic wrapper for where to intervene
with the 6-channel trick; YOLOX `tools/eval.py` for the output format to convert.

**Exit criteria:** evaluator reproduces a published number (show the comparison in
RESULTS.md); correspondence test green; DATA.md frozen; thin slice ran end-to-end overnight;
measured it/s recorded and §2 budget re-checked.

---

### Phase 1 — Controls + noise floor (Week 3, ~28 GPU-h + ~8 h hyperparameter probe)

**Theory gate (CURRICULUM Week 3):** Lipton & Steinhardt (all, slowly); Picard; Bouthillier
§2–3; ATSS §3 with methodology eyes; W&B quickstart.

**Krithin builds:**
1. The four single-stream / control configs (C-rgb, C-thm, C-early — trivial YOLOX variants).
2. **ProbEn-style late fusion** (C-late): detection-level Bayesian fusion of the trained
   C-rgb and C-thm detectors — logit summation for scores, variance-weighted box averaging,
   graceful handling of single-modality detections (ProbEn §3). **No training cost**; it's a
   ~200-line post-processing script over saved detections. It must be the real ProbEn recipe,
   not NMS-merge — though also report plain NMS-merge, since ProbEn showed even that hits
   10.78 with Faster R-CNN singles (LITERATURE §2) and it's free.
3. **Width-scaled single-stream control** (C-wide): thermal-only (the stronger modality)
   YOLOX with width multiplier chosen so params/GFLOPs ≈ the two-stream fusion model
   (roughly YOLOX-S→M-ish width ~0.67–0.75; compute exactly once F-cat-345's cost is known).
   This is the row that separates "fusion works" from "2× params works."
4. **The two-stream baseline** `F-cat-345` (concat+1×1 at P3/P4/P5, independent streams) —
   the first fusion code. Build it as the *general* two-stream backbone with a swappable
   fusion-module interface and a config-selected fusion stage set; every later config is a
   plug-in to this. Thin slice first: 500 imgs / 10 epochs, check activation histograms at
   the fusion junctions per stream (Karpathy Lec-3 exercise; >2× scale mismatch = bug).
5. Hyperparameter probe (3–4 short runs on F-cat-345), then **freeze the recipe**.
6. **Noise floor: train F-cat-345 with 4 seeds.** σ of MR-All over those 4 runs is THE
   noise floor. Also 3 seeds each on C-rgb and C-thm (they are comparison anchors in the
   definition of done). Log per-stream gradient norms during fusion training (modality-
   imbalance tripwire, LITERATURE §6.5).
7. The overnight queue runner.

**Exit criteria (all must pass):**
- Noise floor stated: "F-cat-345 MR = X ± σ over 4 seeds." Every later claim needs |Δ| > 2σ.
- Sanity ordering holds: **thermal-only ≪ RGB-only at Night; RGB-only ≤ thermal-only at
  Day** (ProbEn's pattern: 7.76 vs 27.42 Night). If it doesn't → thermal normalization bug;
  stop and fix.
- Controls table filled (see §5) with params/GFLOPs columns.
- F-cat-345 beats min(C-rgb, C-thm) on All, Day, AND Night — or the discrepancy is
  investigated before proceeding (bug first, finding second).

---

### Phase 2 — Axis A: where to fuse (Week 4, ~40 GPU-h)

**Theory gate (CURRICULUM Week 4):** Prince Ch. 11; Karpathy Lecture 3 + histogram exercise.

The modern re-run of Liu et al. 2016 on an anchor-free base — which does not exist in the
literature (LITERATURE §7.4). Operator held fixed at concat+1×1 (one variable per row).
Configs: fuse at P3 only / P4 only / P5 only / all three. For single-level configs the other
levels feed the PAFPN from **the thermal stream** (the stronger single modality — this choice
is itself recorded; RGB-fed variant is a cut nice-to-have).

**Krithin builds:** the stage-selection config plumbing on the Phase-1 two-stream backbone
(mostly done by construction), plus per-level GFLOPs accounting.

**Exit criteria:** the 4-row stage table with σ marked and an explicit statement of which
rows clear 2σ ("none" is an acceptable and expected answer for parts of it); activation
histograms at each fusion junction; a written paragraph comparing to Liu et al. 2016 —
does halfway still win in 2026?

---

### Phase 3 — Axis B/C/E: the operator sweep + sharing + dropout (Week 5, ~58 GPU-h must)

**Theory gate (CURRICULUM Week 5):** CFT §3; MBNet §3 re-read; Szeliski §2.1 + §8.1.

Fusion stage fixed to Phase 2's winner (expected: multi-level P345; if nothing clears 2σ,
default to P345 — the field's standard — and say so). Sweep the operator taxonomy of
LITERATURE §3. Implementation order = cheapest first, so the harness is debugged before the
attention module: sum → max → gate → SE-cross → DMAF → cross-attention.

**Krithin builds:** each operator as a plug-in module (interface: (F_rgb, F_t, C) → F_fused,
C channels out so PAFPN is untouched). Reference-code guidance: MBNet repo for DMAF; GAFF
paper Fig. 2 for the SE-cross design (per-stream SE + cross-modal residual guidance —
implement the ~GAFF-lite version, no mask supervision); CFT repo for the attention block
(restrict to P5, stride 32, where quadratic cost is tolerable — LITERATURE §3(i)).
Also: modality dropout (zero one stream's features with p=0.15 during training) and the
shared-backbone variant (one backbone, per-modality BN — LITERATURE §7.6, unstudied).

**Exit criteria:** operator table complete with params/GFLOPs deltas explicit; the
pre-registered predictions of §7 scored against outcomes; one honest paragraph of negative
results distinguishing "didn't work" from "didn't clear 2σ".

---

### Phase 4 — THE HEADLINE: misalignment (Weeks 5–6 overlapping, ~24 GPU-h train + ~22 h eval)

**Theory gate:** AR-CNN re-read (§3.1–3.2); the parallax/temporal-skew arithmetic from
CURRICULUM Track 4A written out for the Apollyon rig (baseline b, focal f → disparity at
3/10/30 m; 8.3 ms skew × target speed → px). These numbers go in the report's motivation.

Three instruments, in cost order:

**4a. Measure what's already there (free, eval/analysis only).**
From KAIST-Paired: the distribution of RGB↔T box offsets (histogram, px). If paired test
annotations exist (Phase 0 verify step): stratify the 2,252 test images by measured offset
magnitude and report per-stratum MR for each operator — misalignment sensitivity measured on
*natural* misalignment, no injection. If not: report the train-set offset distribution as
context calibrating the injection sweep ("KAIST's natural shift is ~N px; our sweep spans
0–16 px").

**4b. Inject at eval time (nearly free — no retraining).**
For every Phase 2/3 checkpoint, run the eval with a controlled perturbation applied to the
**thermal input only** (GT stays anchored to RGB — matching the rig, where RGB is reference):
- **Spatial shift:** translate thermal by 0 / 2 / 4 / 8 / 16 px (diagonal; fill by edge-pad).
- **Temporal skew:** pair RGB test frame t with thermal frame **t−k from the full-rate 20 Hz
  video**, k ∈ {1, 2, 3, 5} → 50 / 100 / 150 / 250 ms of skew. This is *real* scene motion
  (ego-motion + pedestrian motion), not a synthetic warp — the novel, publishable curve
  (LITERATURE §7.8).
- **Scale/FOV mismatch:** center-scale thermal by ±5% (the FOV-mismatch proxy).
Budget control: full 10-condition sweep on seed-1 checkpoints of all 8 operators (~16 h);
seed-2 checkpoints only at {shift 8, skew 2} to attach error bars to the money figure
(~4 h). All scripted, all overnight.

**4c. Train for robustness (the expensive arm — 3 configs only).**
Retrain **sum, concat+1×1, and SE-cross** (the cheap/default/gated representatives) with
MLPD-style **unpaired geometric jitter** (independent small random shifts between streams,
±[0,8] px, during training) — 3 configs × 2 seeds × 4 h = 24 h — then re-run the 4b sweep on
them. This gives the train-clean/eval-shifted vs train-jittered/eval-shifted comparison: does
cheap augmentation buy what expensive operators buy? (MLPD's 7.58 with plain concat says
probably yes — LITERATURE §4.2.) Nice-to-have: one concat retrain with *temporal-skew*
augmentation (random k∈{0,1,2} pairing from full-rate train video).

Also in this phase: **blackout eval** (thermal stream zeroed for the whole image — the FFC/
NUC failure mode) on F-cat-345 vs F-cat-345-drop vs max-fusion, testing whether modality
dropout and OR-gate operators keep the model alive on RGB alone.

**Krithin builds:** the eval-time perturbation wrapper (a dataloader shim — shift/scale is
~50 lines; temporal skew needs the full-rate-video frame indexer from Phase 0); the unpaired-
jitter training aug (flag on the existing aug pipeline — and note it must *bypass* the
correspondence unit test via an explicit opt-in, so the test still guards against accidental
misalignment).

**Exit criteria:** the degradation curves exist (MR vs shift, MR vs skew — one line per
operator, jitter-trained variants dashed); the operator ranking under misalignment is stated
and explained **from geometry** (which ops assume pixel correspondence; how a k-px image
shift becomes k/stride feature cells); the natural-misalignment measurement is reported.

---

### Phase 5 — Efficiency, Pareto front, report (Week 6, ~5 GPU-h)

**Theory gate (CURRICULUM Week 6):** TensorRT quantization docs; Lipton & Steinhardt §3.2
re-read *while writing*.

- **Latency:** measured on the 5070 Ti laptop, PyTorch AMP/fp16, batch 1, 100+ runs after
  warmup, for every (stage, operator) trained config. Stated plainly in the report as a
  **desktop-GPU proxy, not Jetson numbers** (CURRICULUM Track 6 forbids pretending
  otherwise). TensorRT FP16/INT8 export + `trtexec` on the laptop: **stretch goal**; INT8
  MR-after-quantization column only if time allows (it's the differentiator row, but it is
  the correct thing to cut before any accuracy experiment).
- The Pareto figure and the full grouped results table.
- **The report**, ATSS-structured: equalize → isolate → attribute → propose. Every abstract
  claim points at a row. Negative results at equal prominence. Prediction scorecard (§7)
  included verbatim with outcomes.

**Exit criteria:** the five items of CURRICULUM Week 6's final gate.

---

## 4. The Ablation Grid

Naming: `C-*` = controls, `F-<op>-<stages>` = fusion configs, `-jit` = trained with unpaired
jitter, `-drop` = modality dropout. Eval conditions: `E-clean`, `E-shift{2,4,8,16}`,
`E-skew{1,2,3,5}`, `E-scale±5`, `E-blackout`, `E-strata`.

### Axis A — where to fuse (operator fixed: concat+1×1)

| Config | Fusion point(s) | Seeds | GPU-h | Priority | Note |
|---|---|---|---|---|---|
| F-cat-345 | P3+P4+P5 | 4 | 16 | **MUST** | Baseline + noise floor donor |
| F-cat-3 | P3 only (stride 8) | 2 | 8 | **MUST** | Most misalignment-exposed |
| F-cat-4 | P4 only (stride 16) | 2 | 8 | **MUST** | Liu et al.'s "halfway" analogue |
| F-cat-5 | P5 only (stride 32) | 2 | 8 | **MUST** | Predicted most shift-robust |
| F-cat-2 | stride 4 | — | — | CUT | Max compute, max misalignment pain, no literature support (§10) |

### Axis B — the operator (stage fixed to Phase-2 winner, default P345)

| Config | Operator (LITERATURE §3) | Extra params | Seeds | GPU-h | Priority | Why it's in |
|---|---|---|---|---|---|---|
| F-sum-345 | element-wise sum (a) | 0 | 2 | 8 | **MUST** | Zero-param floor; strict pixel-correspondence assumption — predicted to fail first under shift |
| F-max-345 | element-wise max (a) | 0 | 2 | 8 | **MUST** | Per-cell OR-gate; the blackout-robustness hypothesis |
| F-cat-345 | concat + 1×1 (b) | 2C² | (4, above) | 0 | **MUST** | The workhorse everything must beat |
| F-gate-345 | learned spatial gate (d), GFD-SSD-style | ~2 small convs | 2 | 8 | **MUST** | Minimal "trust which modality, where" |
| F-se-345 | SE-cross / GAFF-lite (e) | ~2C²/r ×2 | 2 | 8 | **MUST** | Best published cost/accuracy point (GAFF 6.48/3.46N) |
| F-dmaf-345 | differential SE (g), MBNet DMAF | SE-level | 2 | 8 | **MUST** | Gate on the difference signal |
| F-xattn-5 | dense cross-attention @P5 only (i) | large | 2 | 10 | **MUST** | The expensive ceiling; the "global correspondence" robustness claim to test |
| F-illum-345 | illumination-gated (f) | tiny subnet | 1 | 4 | nice | Interpretable global gate; predicted ≈ F-gate (§7.5 of LITERATURE says the comparison is open) — single-seed, flagged |
| F-prod-345 | element-wise product | 0 | — | — | CUT | AND-gate; predicted collapse at night (kills single-modality detections) — prediction recorded, run not spent |

### Axis C — weight sharing

| Config | Sharing | Seeds | GPU-h | Priority |
|---|---|---|---|---|
| (all above) | two independent streams | — | — | default (the field's universal choice) |
| F-share-345 | one backbone, per-modality BN, concat fusion | 1 | 4 | nice | ~45% backbone-memory saving question (LITERATURE §7.6); single-seed, flagged |

### Axis D — misalignment (the star)

**Eval-only (no training; runs over existing checkpoints):**

| Sweep | Conditions | Applied to | Eval passes | GPU-h | Priority |
|---|---|---|---|---|---|
| Spatial shift | 2/4/8/16 px on thermal | all 8 Axis-B ops + F-cat-3/-4/-5, seed 1 | ~44 | ~9 | **MUST** |
| Temporal skew | thermal frame t−k, k=1/2/3/5 (50–250 ms) | all 8 Axis-B ops, seed 1 | ~32 | ~7 | **MUST** — the novel curve |
| Scale/FOV | thermal ±5% | Axis-B ops, seed 1 | ~16 | ~3 | nice |
| Error bars | shift 8 + skew 2 only | seed-2 checkpoints | ~20 | ~4 | **MUST** |
| Natural strata | KAIST-Paired measurement (+ test strata if annotations exist) | analysis | — | ~0 | **MUST** |

**Train-time arm (retrains):**

| Config | Training change | Seeds | GPU-h | Priority |
|---|---|---|---|---|
| F-sum-345-jit | unpaired jitter ±8 px | 2 | 8 | **MUST** |
| F-cat-345-jit | unpaired jitter ±8 px | 2 | 8 | **MUST** |
| F-se-345-jit | unpaired jitter ±8 px | 2 | 8 | **MUST** |
| F-cat-345-tjit | temporal-skew aug (k∈{0,1,2}) | 1 | 4 | nice |
| (each -jit) | re-run shift+skew eval sweep | — | ~6 | **MUST** |

### Axis E — modality dropout

| Config | Change | Seeds | GPU-h | Priority |
|---|---|---|---|---|
| F-cat-345-drop | p=0.15 single-stream feature dropout | 2 | 8 | **MUST** |
| E-blackout eval | thermal zeroed, on F-cat-345 / -drop / F-max-345 | — | ~1 | **MUST** — the FFC/NUC deployment story |

### Budget roll-up

| Block | MUST GPU-h | nice GPU-h |
|---|---|---|
| Phase 0 (slices, checkpoint reproduction) | 15 | — |
| Phase 1 controls + noise floor + HP probe | 36 | — |
| Phase 2 stage sweep | 40 | — |
| Phase 3 operator sweep + dropout | 58 | 8 (illum, share) |
| Phase 4 misalignment (train + eval) | 44 | 7 (scale, tjit) |
| Phase 5 latency | 5 | — |
| **Total** | **≈198** | +15 |

Against the ~240 h envelope: fits, with headroom that will be consumed by reality. If Phase
0's measured it/s says runs cost >5 h, activate §10 cuts immediately (first casualties:
F-illum, F-share, scale sweep, seed 2 on F-cat-3/-4/-5).

---

## 5. The Controls — and why each is strong, not strawman

| Config | What | Why it's a strong control |
|---|---|---|
| C-rgb | RGB-only YOLOX-S, 3 seeds | Same backbone, schedule, augs as fusion rows. Anchor for the Day column. |
| C-thm | thermal-only YOLOX-S, 3 seeds | **The real bar.** Thermal ≥ RGB overall on KAIST and dominates Night (ProbEn: 7.76 vs 27.42). A fusion model that merely matches C-thm is a null result wearing a costume — the documented APEX failure mode. |
| C-early | 6-channel input, single backbone, 2 seeds | The genuine early-fusion design (input-concat), trained identically — not a crippled variant. ProbEn measured early fusion *below* RGB-only; we test whether that replicates. |
| C-late | **ProbEn §3 recipe** over C-rgb + C-thm: logit-sum score fusion + variance-weighted box fusion + missing-detection handling. Plus plain NMS-merge as its own row. | The literature's uncomfortable fact: ProbEn beat every learned mid-fusion method in 2022. A weak late-fusion control would make the whole study decorative (CURRICULUM Track 5 rule 1). Costs no training. Its ~2× inference cost is reported in the latency column — that cost IS the engineering argument for mid-fusion. |
| C-wide | width-scaled single-stream thermal YOLOX at params/GFLOPs ≈ F-cat-345, 2 seeds | The parameter-matched control. Two-stream ≈ 2× backbone params; "fusion beats single-stream" is otherwise indistinguishable from "2× params beats 1× params." **Mandatory** (CURRICULUM Track 5 rule 2). Width multiplier computed after F-cat-345's exact cost is known. |

Acceptance test for any fusion claim: MR_fused < min over controls on **All, Day, and Night
simultaneously**, beyond 2σ, at matched schedule — with C-wide and C-late in the comparison
set, not just the singles.

---

## 6. Seed / noise-floor protocol

- **Noise floor:** σ = std of MR-All over **4 seeds of F-cat-345** (Phase 1). Written on a
  sticky note. Expected σ ≈ 0.3–0.8 MR based on test-set size (~1,500 Reasonable pedestrians);
  if σ > 1.0, stop and investigate (schedule too short, eval nondeterminism, aug bug) before
  running any ablation row.
- **Seeds per row:** 4 on the baseline; 3 on C-rgb/C-thm; **2 on every MUST grid row**;
  1 on nice-to-have rows, which are printed in italics and never used for claims.
- **The 2σ rule:** a delta counts as a result only if |Δmean| > 2σ. Everything else is
  reported as "within noise" — which for the clean operator sweep is the *expected* outcome,
  not a failure.
- Report **mean ± std and seed count** in every table; seed count in every caption.
- A seed controls: weight init, data order, aug RNG. Data split is fixed (not a variance
  source, per protocol lockdown).
- Misalignment curves: seed-1 full sweep, seed-2 at two anchor conditions → error bars on the
  money figure without doubling its cost. Deltas *between operators under shift* must clear
  the same 2σ bar before the ranking is claimed.

This alone exceeds the rigor of the literature being cited (LITERATURE §7.3: no error bars
anywhere in the canonical trail).

---

## 7. Pre-registered predictions (written 2026-07-13, before any run)

Scored in the final report, hit or miss. This section may not be edited after Phase 1 starts
— corrections go in the report, not here.

| # | Prediction | Falsified if |
|---|---|---|
| P1 | Sanity ordering: C-thm ≪ C-rgb at Night (gap > 10 MR); C-rgb ≤ C-thm at Day. | Ordering fails → it's a bug (thermal normalization), not a finding, until proven otherwise. |
| P2 | C-early underperforms or matches the better single (ProbEn replication). | C-early beats both singles by >2σ. |
| P3 | F-cat-345 beats min(C-rgb, C-thm) on all three columns by >2σ, with Night driving the gain and **Day the closest call** (GAFF traded Day for Night; MBNet's imbalance mechanism). | Fusion ≤ best single anywhere → investigate imbalance (gradient-norm logs) before accepting. |
| P4 | **Clean operator sweep is mostly null:** all cheap ops (sum…DMAF) within ~1 MR of concat; at most one or two rows clear 2σ. The 100×-cost F-xattn-5 gains ≤1–2 MR over F-se-345 (LITERATURE §3(i)). | ≥3 operators separate by >2σ on clean data → the "operator doesn't matter when aligned" thesis is wrong and the report's framing changes. That would itself be a strong finding. |
| P5 | Stage sweep: P345 ≥ best single level; among singles P4 ≥ P3, P5; clean-data differences small (≤2σ for most pairs). | P3-only wins clean by >2σ. |
| P6 | C-wide closes most of the fusion gap at **Day** but not at **Night** — i.e. genuine fusion value is concentrated where modalities are complementary. Fusion still beats C-wide overall by >2σ. | C-wide matches fusion on All+Night → headline negative: "2× params, not fusion" — reported at full prominence. |
| P7 | C-late (ProbEn) matches or beats our best mid-fusion on MR, at ~2× latency. Mid-fusion wins the Pareto plot, not the accuracy column. | Mid-fusion beats ProbEn on MR by >2σ at equal training budget. |
| P8 | **Shift sweep (train-clean): the flat table fans out.** Degradation ranking, steepest first: F-sum-3-class ops at high res (F-sum, F-max, F-cat at P3) → F-cat-345 → gated ops (F-gate, F-se, F-dmaf) → F-cat-5 → F-xattn-5 (most robust; global correspondence). By 8 px, spread between best and worst operator exceeds 5× the clean-data spread. | Curves stay parallel/flat to 16 px → misalignment robustness doesn't differentiate operators at KAIST scale, and the rig-relevance claim must retreat to the temporal-skew and blackout results. |
| P9 | **Temporal skew:** monotone degradation in k; at k=2 (100 ms) degradation is measurable (>2σ) for per-pixel ops; Night degrades more than Day (thermal carries more of the night decision, and it's the stream being skewed); ranking of operators ≈ P8's. | No measurable degradation even at k=5 (250 ms) → 20 Hz vehicle scenes move too little; report says so and the skew instrument is declared insensitive at this frame rate (still a publishable calibration point). |
| P10 | **Jitter-training flattens the curves:** F-cat-345-jit at 8 px shift loses <⅓ of what F-cat-345 loses, at a clean-data cost ≤1σ. Cheap augmentation buys most of what expensive operators buy (the MLPD lesson). | Jitter training costs >2σ clean or fails to flatten → augmentation is not the free lunch the literature implies. |
| P11 | Blackout: F-cat-345 collapses toward (or below) C-rgb-level; F-cat-345-drop and F-max-345 survive near C-rgb level. Dropout's clean-data cost ≤1σ. | Dropout costs >2σ clean, or doesn't help under blackout. |
| P12 | F-share-345 (if run) loses 1–3 MR vs independent streams — sharing is not free. | Within 2σ of F-cat-345 → embedded-relevant positive finding (45% backbone memory for free). |
| P13 | Latency: the two-stream backbone dominates cost; all cheap fusion ops within ~10% latency of each other; F-xattn-5 measurably slower; FLOPs ranking ≠ latency ranking somewhere in the table (concat's memory traffic). | — (measurement, not hypothesis; recorded for the Pareto discussion) |

The point of this section: when the results arrive, the report says "we predicted X, observed
Y" — not a story reverse-engineered from whatever the tables happened to show. Prediction
misses are reported as prominently as hits; several of these (P4, P6, P7) are *expected*
partial-nulls, and their value is exactly that we said so in advance.

---

## 8. The figure list (the report is built around these)

1. **THE MONEY FIGURE — misalignment degradation curves.** MR-All vs injected spatial shift
   (0–16 px), one line per operator; train-clean solid, jitter-trained dashed; 2σ band shaded;
   KAIST's measured natural shift (from KAIST-Paired) marked as a vertical line on the x-axis.
2. **Temporal-skew curves.** MR vs skew k (in ms), per operator, Day/Night split as two
   panels. The novel plot — nothing like it exists in the literature (LITERATURE §7.8).
3. **Pareto front: MR-All vs measured latency** for every (stage, operator) pair + C-late +
   C-wide; front drawn; point size = params. The deployment-decision figure.
4. **Controls ladder.** MR All/Day/Night for C-rgb / C-thm / C-early / C-late / C-wide /
   best-fusion, with error bars — the honest "does fusion pay at all, against strong
   controls" summary.
5. **Natural misalignment measured.** KAIST-Paired offset histogram; if paired test
   annotations exist, per-stratum MR by operator (natural-misalignment sensitivity).
6. **Blackout/dropout panel** (or appendix): MR under thermal blackout for F-cat-345 /
   -drop / F-max — the FFC-failure deployment story. Appendix: activation histograms at the
   fusion junction; the fixed 8-image qualitative panel across configs.

---

## 9. Risks and tripwires

| Risk | How it manifests | Tripwire / mitigation |
|---|---|---|
| **Thermal normalization bug** (the #1 killer) | Fusion ≈ RGB-only; thermal stream dead; night numbers bad | Phase-1 hard gate: C-thm must crush C-rgb at Night. Activation histograms at fusion junction per stream; >2× scale mismatch = stop. Fixed global thermal stats in DATA.md. |
| **Evaluator wrong** | Everything shifts by points; incomparable to literature | Phase-0 hard gate: reproduce a published MR from released detections/checkpoint to <0.1 before any training. Never touched afterward; all numbers through the one script. |
| **Augmentation breaks correspondence** | Mid-training misalignment injected silently; fusion underperforms mysteriously | The correspondence unit test in CI; visualize mosaic'd pairs; jitter aug behind an explicit opt-in flag so the test still guards defaults. |
| **SimOTA × modality imbalance** | Fused model converges to "thermal detector + dead weight"; matches C-thm exactly | Log per-stream gradient norms from run 1; tripwire = fused within 1σ of C-thm on all three columns. Response: check RGB-stream learning (freeze-vs-finetune probe) before blaming the operator. |
| **Seed noise swamps deltas** | Ablation table is weather | 4-seed noise floor before any row; 2σ rule; if σ>1.0 MR, fix training stability first (longer schedule / EMA) — do not proceed into the grid. |
| **VRAM OOM on two-stream** | Runs die overnight | Phase-0 slice measures peak memory; fallback batch 4 + accum 2, recorded, applied uniformly to ALL rows (never mixed batch sizes across rows). |
| **Runs cost more than estimated** | Grid doesn't fit in 4 weeks | Phase-0 slice measures real it/s; §10 cut order pre-committed; overnight queue keeps utilization high. |
| **Mosaic/MixUp mistuned for 7.6k images** | Baseline underfits/overfits; every row inherits it | Phase-1 HP probe (3–4 short runs) before freezing; monitor val MR every 5 epochs; expect weaker MixUp + the 15-epoch no-aug tail. |
| **Temporal-skew instrument confounded** | Skewed thermal frame differs in *content*, not just alignment (ego-motion moves whole scene) | That's partly the point (it's the real rig failure mode) — but report ego-motion vs target-motion contributions; sanity-check on static scenes (parked sequences) where skew should cost ~0. GT stays RGB-anchored, stated explicitly. |
| **KAIST-Paired test annotations don't exist** | Strata figure impossible | Phase-0 verify step; documented fallback (train-set statistics calibrate the injection axis). No downstream dependency on the strata figure. |
| **Late-fusion control accidentally strawman** | Study decorative | Implement ProbEn's actual §3 recipe; validate by checking our C-late beats plain NMS-merge (ProbEn's own internal ordering). |
| **Week-6 time crunch** | Report unwritten | Report skeleton (tables with empty cells, figure stubs) created in Phase 1; RESULTS.md ledger filled continuously; the report is assembled, not written, in Week 6. |

---

## 10. Pre-committed cut list (in order — cut from the top when behind)

1. TensorRT INT8 + MR-after-INT8 column (stretch already).
2. F-illum-345, F-share-345, F-cat-345-tjit (the single-seed nice-to-haves; predictions P12
   etc. remain recorded as untested).
3. Scale/FOV ±5% eval sweep.
4. Seed 2 on F-cat-3/-4/-5 (stage sweep drops to single-seed with explicit caveat).
5. F-max-345 and F-dmaf-345 (keep sum, cat, gate, se, xattn as the operator spine).
6. Shift sweep trimmed to {4, 8} px; skew to {2, 5}.

Never cut, in any scenario: the verified evaluator; the noise floor; C-thm/C-wide/C-late
controls; the shift and skew eval sweeps on {F-sum, F-cat, F-se, F-xattn}; at least one
jitter retrain (F-cat-345-jit). Those are the project.

Already cut, deliberately (recorded so they aren't re-litigated): stride-4 fusion (cost,
misalignment pain, no literature support); element-wise product (AND-gate night collapse is
predictable from §3(a) — prediction recorded instead of a run); LLVIP/CVC-14/ROTX-MP
generalization rows (each is a fine follow-up; none fits the 4-week envelope — noted in the
report's future-work section, with the LLVIP shrink-test called out as the sharpest one);
CIoU loss row (orthogonal to the fusion question); thermal-stream pretraining study
(LITERATURE §7.9 — a project of its own).

---

*This plan is executable by someone who remembers nothing about how it was made: protocol in
§1, budget in §2, build order in §3, every run named and costed in §4–5, the decision rule in
§6, what we expect in §7, and what to do when it goes wrong in §9–10.*
