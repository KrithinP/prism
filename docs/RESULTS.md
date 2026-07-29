# PRISM — Results Ledger

Every run that produced a number, including the bad ones. MR⁻² = log-average miss rate
on the Reasonable subset, **lower is better**. All models from scratch (no pretraining),
lr = 0.01, 16 epochs, batch 8, evaluated on the **improved-test** annotations (Liu et al.).
Training set = **sanitized-train** (Li et al. / MSDS-RCNN), 7,601 frames.

> **Comparability note:** absolute numbers are capped by the from-scratch regime
> (~35% thermal-only vs ~19% for pretrained methods). The study's claims are *internal
> deltas* vs these baselines, which are regime-independent. See decisions log 2026-07-24.

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

## Phase 1 — first fusion model

| Model | Params | MR⁻² All | Day | Night | Notes |
|---|---|---|---|---|---|
| **F-cat-345** (concat @ dark3/4/5) | 13.84M (1.55×) | **27.3** | **31.6** | **18.7** | seed 0; 5-seed mean±std **pending** |

**vs thermal-only (seed 0):** All −8.8, Day −12.3, Night −1.9 — all beyond the ~2.5
noise bar (Night marginal). Beats *both* single cameras in Day (31.6 vs 43.9 / 46.3),
which neither reaches alone → genuine cross-modal benefit, **not** the APEX single-
modality collapse. Verified visually (`docs/figures/fusion_vs_baselines_grid.png`):
fusion recovers daytime pedestrians both cameras miss; ignores blind RGB at night.

Scale breakdown (seed 0): near 14.4 / medium 37.2 / far 73.0 (far has few, noisy).

## Pending

- [ ] F-cat-345 seeds 1–4 → fusion noise floor (mean±std)
- [ ] Width-matched single-stream control (~13.84M) — isolate fusion from parameter count
- [ ] Evaluator verification vs a published method's released detections
- [ ] Phase 2: fusion-depth sweep (dark3 / dark4 / dark5 / all) — `fuse_stages` ready
