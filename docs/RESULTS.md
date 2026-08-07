# PRISM Archived Exploratory Results

> **QUARANTINED: not admissible as corrected-study evidence.** These runs used
> 8/16-epoch recipe selection and repeated improved-test evaluation, predate the
> final evaluator parity fixes, used the invalid legacy `fuse_stages` graph, and
> predate the YOLOX/ignore-region training corrections. The raw values are
> preserved to document why the repository was rebuilt. Do not combine them with
> `outputs/ledger.jsonl`, use them to select an architecture, or cite their deltas
> as regime-independent.

Every run that produced a number, including the bad ones. MR⁻² = log-average miss rate
on the Reasonable subset, **lower is better**. All models from scratch (no pretraining),
lr = 0.01, 16 epochs, batch 8, evaluated on the **improved-test** annotations (Liu et al.).
Training set = **sanitized-train** (Li et al. / MSDS-RCNN), 7,601 frames.

The prose below is the historical interpretation written before the audit. It is
not the current scientific conclusion. See `../EXPLANATION.md`,
`EXPERIMENT_PLAN.md`, and `LITERATURE.md`.

## The noise floor (why deltas need error bars)

Same model, same data, only the seed differs → MR wobbles by ~1–2 points. **A result
counts only if it beats the baseline by more than 2σ (~2.5 points).**

## Phase 1 — single-camera baselines (5 seeds each)

| Model | Params | MR⁻² All | Day | Night | Notes |
|---|---|---|---|---|---|
| thermal-only | 8.94M | **36.1 ± 1.3** | 43.9 ± 1.4 | **20.6 ± 1.2** | stronger modality overall |
| RGB-only | 8.94M | **52.5 ± 1.5** | 46.3 ± 1.8 | **65.9 ± 0.9** | near-blind at night |

Sanity ordering holds: RGB collapses at night (65.9), thermal dominates it (20.6) — a
45-point gap, the multispectral premise confirmed. Audit (5 checks) passed; see Ch.9.

## Recipe-selection runs (single seed — choosing hyperparameters, not claiming results)

**LR sweep** (thermal-only, 8 epochs, 1 seed each) — chose the learning rate:

| lr | MR⁻² All | note |
|---|---|---|
| **0.01** | **36.6** | winner (locked) |
| 0.005 | 39.0 | |
| 0.0025 | 38.7 | |
| 0.00125 | 43.2 | |
| 0.02 / 0.04 | diverged / unstable | see Ch.8 (the "cliff") |

**Epoch calibration** (thermal-only, lr 0.01, 1 seed, checkpoints along the way) — chose 16 epochs:

| epochs | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|
| MR⁻² All | 34.4 | 35.6 | 34.7 | 35.4 | 36.8 |

Flat within noise → from-scratch **plateaus by ~8 epochs** (data-limited, not undertrained:
training loss kept falling while test MR did not). Locked **16 epochs** (past the plateau,
cheap). This is why the absolute ceiling is ~35%. `docs/figures/epoch_calibration_thermal.png`.

## Phase 1 — first fusion model (F-cat-345, concat @ dark3/4/5, 13.84M = 1.55×)

Per-seed MR⁻² (improved-test):

| seed | All | Day | Night |
|---|---|---|---|
| 0 | 27.28 | 31.64 | 18.73 |
| 1 | 24.68 | 29.43 | 15.17 |
| 2 | 28.78 | 34.43 | 16.85 |
| 3 | 30.37 | 35.89 | 18.86 |
| 4 | 27.32 | 32.10 | 17.76 |
| **mean ± std** | **27.69 ± 1.88** | **32.70 ± 2.25** | **17.47 ± 1.36** |

**vs thermal-only (36.1 / 43.9 / 20.6):** All **−8.4**, Day **−11.2**, Night **−3.1**.
All and Day clear 2σ decisively; **Night is borderline** (Δ 3.1 vs combined 2σ ≈ 3.7) —
report as modest/near-significant, don't oversell (RGB adds little in the dark, as expected).
Fusion beats *both* single cameras in Day (neither reaches ~33 alone) → genuine cross-modal
benefit, **not** the APEX single-modality collapse. Verified visually
(`docs/figures/fusion_vs_baselines_grid.png`). Scale breakdown (seed 0): near 14.4 / medium 37.2 / far 73.0.

## Phase 1 — width-matched control (does param count explain the fusion win?)

Thermal-only widened to **base_ch=42 (14.06M ≈ fusion's 13.84M)**, 5 seeds:

| Model | Params | MR⁻² All | Day | Night |
|---|---|---|---|---|
| thermal-only | 8.94M | 36.1 ± 1.3 | 43.9 | 20.6 |
| **width-matched thermal** | 14.06M | **36.06 ± 1.14** | 43.48 ± 1.18 | 21.15 ± 1.51 |
| fusion (F-cat-345) | 13.84M | **27.69 ± 1.88** | 32.70 | 17.47 |

**Verdict: the fusion win is NOT parameters.** Widening a single camera to fusion's budget
moved MR by ~0 (36.1 → 36.1). Fusion, at the same budget, is **8.4 points better**. The win
comes from *combining cameras*, not model size. **Phase 1 complete.**

## Historical Pending State

- [ ] Evaluator verification vs a published method's released detections
- [ ] The historical depth sweep was retired; literal corrected graphs are Block B
  in `scripts/run_experiments.py`.
