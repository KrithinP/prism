# PRISM — Orientation to the Rebuild (read this first)

*A human-readable map of what Soham's `d5944ed` commit changed and why. The
authoritative specs are `EXPLANATION.md`, `docs/EXPERIMENT_PLAN.md`, and
`TODO.md`; this file exists so you can hold the whole thing in your head before
you open them. Written 2026-08-07.*

---

## 1. The one sentence

We had a result that *looked* like a fusion win; Soham rebuilt the project so
that a fusion win would be **believable** — and in doing so showed that our old
result was not yet admissible evidence.

The rebuild is one idea applied everywhere: **an experiment only counts if every
alternative explanation for the number has been ruled out in advance.** His
voice note called this "the ablation should differentiate the causal reason."
This commit is that principle turned into code, data discipline, and an
experiment matrix.

---

## 2. Why he rebuilt — the three foundational faults

Everything downstream follows from these three. They are the reason our numbers
were quarantined, not edited.

### Fault 1 — We selected our architecture *on the test set* (the cardinal sin)

We trained, looked at the 2,252-frame improved-test MR, tuned (lr, epochs,
fusion stage), and looked again. Once you use the test set to *choose* anything,
its number is no longer an honest estimate of unseen performance — you have
fitted to it. Every published KAIST table is off-limits to compare against ours
because ours was contaminated.

**The fix:** a hard firewall. Selection now happens on a **sequence-disjoint
`train-cv` / `val-cv`** split carved out of the training set. The official test
set is *unreachable from `prism.train`* and only opens through
`--official-test` + a frozen release manifest, once, at the very end. See
`src/prism/protocol.py`, `scripts/freeze_release.py`.

### Fault 2 — Our "fusion graphs" were not the graphs we named

This is the one that directly kills the Phase-2 depth sweep I was about to
resume. `TwoStreamYOLOX(fuse_stages=(4,))` did **not** mean "the two streams
meet at dark4." It meant: keep both backbones fully separate, concat-fuse at
dark4, and **sum-fuse at dark3 and dark5 anyway** (the silent `SumFuse`
fallback). So *every* neck input still carried both modalities in *every*
config. Changing the "meeting stage" only swapped one operator while the graph
stayed the same. That is why the depth table looked flat — **not** because
"aligned data makes depth irrelevant," but because all three configs were
essentially the same multi-scale graph. Our conclusion was an artifact.

**The fix:** fusion is now four *literal, separately-tested* graph families —
early / true one-shot / multi-scale / late — each one unambiguous, each with
graph-level tests (parameter-ownership, modality-dependence, zero-gradient,
hook-counted fusion calls). See §4.

### Fault 3 — Our MR evaluator was never proven equal to the real one

Tiny choices (clip vs ignore boundary boxes, `>` vs `>=` at IoU 0.5, tie
ordering, FPPI reference points) move KAIST MR by points. Ours was "provisional"
— it *resembled* the metric. A result is meaningless if the ruler is unverified.

**The fix:** the canonical evaluator is now **differentially certified** against
the pinned MLPD reference implementation on its five released detection files and
its published table, with a fail-closed certificate (`scripts/parity_check.py`,
`external/evaluator_manifest.json`). This is exactly the "verify the evaluator
against a published method's detections" loose end from Phase 1 — he closed it.

---

## 3. The frozen protocol (the constants nobody may touch)

Machine-enforced in `src/prism/protocol.py`. A change to any of these is not a
tweak — it is a new experimental block with its own ledger record.

| Field | Frozen value | Why it's frozen |
|---|---|---|
| Train labels | MSDS sanitized, 7,601 frames | comparable to the field |
| Selection | sequence-disjoint train-cv / val-cv | no test-set leakage |
| Test | improved 2,252 frames, release-gated | honest final number |
| Init | from scratch | small-model/small-data regime, on purpose |
| Epochs | **exactly 40** (was 16) | a *control*, not a convergence claim |
| Seeds | 0,1,2,3,4 | 5-seed noise floor on everything |
| Metric | validation Reasonable **All** MR⁻² | one predeclared selection number |

The 40 epochs are not "the right amount of training." They are a *fixed budget*
so that when graph A beats graph B, the cause is the graph and not that one got
trained longer. Horizon saturation is a separate later study.

---

## 4. The four fusion graph families (§4 replaces our one fuzzy one)

| Family | What it is | Formula sketch |
|---|---|---|
| **Early** | concat RGB+thermal at the input, one backbone | `y = H(N(E(concat(RGB,T))))` |
| **One-shot** | streams private until level L, then **one** shared suffix | `Z_L = F(R_L,T_L)`, then shared `S_{L+1..5}` |
| **Multi-scale** | both backbones independent, fuse at a *selected* set; unselected levels use an **anchor adapter, never a silent sum** | `Z_l = F(R_l,T_l)` if `l∈S` else `A_l(anchor_l)` |
| **Late** | two full detectors, merge detections (union-NMS, ProbEn) | `D = merge(D_r, D_t)` |

The distinction between "one-shot" and "multi-scale" is the whole point of Fault
2: they are genuinely different computations, and our old code conflated them.

---

## 5. The experiment matrix — this is the "clear ablations" he asked for

Four blocks, each isolating exactly one causal factor. Read the "Question"
column as "the alternative explanation this arm rules out."

**Block A — baselines & controls** (does anything *other than fusion* explain a gain?)
| Arm | Rules out |
|---|---|
| A1/A2 RGB / thermal unimodal | the baselines fusion must beat |
| A3/A4 **wider** RGB / thermal | "it's just more parameters" |
| A5/A6 RGB+RGB, T+T dual branch | "it's just having two branches" |
| A7/A8 nested-residual anchor | "a two-stream net can't even preserve one modality" |

**Block B — where do the streams meet?** operator fixed to concat, only the graph moves (B1 early, B2–B4 one-shot, B5–B10 multi-scale, B11/B12 late, B13 anchor flip). Winner frozen to a file before Block C.

**Block C — which operator?** graph fixed to Block B's winner, 12 operators C1–C12 — average, concat, scalar/channel gates, GAFF-/DMAF-inspired attention, learned alignment, cross-attention, quality gate — **including C11/C12 which are dumb concat networks built to match the parameter count of the expensive operators**, so "it won because it's bigger" is controlled *per operator*.

**Block D — robustness training** (modality dropout, thermal shift). A separate question; it does **not** overwrite the clean winner.

Every arm: 5 seeds, raw per-seed values reported first, a win requires beating
**both** baselines with the upper 95% interval below zero. "Beats the weak
modality" is explicitly not a win.

---

## 6. What is now dead, and what our new job is

### Quarantined (do not cite, do not build on)
- The headline table (thermal 36.1 / fusion 27.7 / −8.4). Contaminated by Fault 1.
- The Phase-2 depth sweep (dark3 ≈ 28.1, dark4 ≈ 27.4, "flat"). Invalid by Fault 2.
- Textbook Ch. 8–10 result numbers. Preserved as history in `docs/RESULTS.md`
  under a QUARANTINED banner, kept only to document *why* we rebuilt.

### What survives
- The *pipeline skeleton* (YOLOX, SimOTA, data loading) — corrected, not discarded.
- The *lessons*: prior-bias init, AMP-safe BCE, the divergence guard, the
  driver-mismatch trap. Those were real and stay.
- The *premise*: RGB is night-blind (65.9 vs 20.6). That ordering is still true.

### The change in *our* path
1. **Compute exploded.** Roughly 8+11+12+3 ≈ 34 training arms × 5 seeds × 40
   epochs ≈ **~170 runs at ~2.5 h each ≈ 400+ GPU-hours**, before diagnostics.
   One suspending laptop cannot finish this in 12 days. **Cloud / multi-GPU is
   now mandatory, not optional.** This is the biggest practical consequence.
2. **Our role shifts.** Soham wrote the implementation. The R1 "Krithin types
   all code" agreement is effectively suspended for PRISM. Our value is now:
   *understand every arm, run the matrix correctly, read the diagnostics, and
   write the analysis.* That is the harder and more valuable half anyway.
3. **`git pull` and run the gates first.** `make prepare-kaist → certify-evaluator
   → test → preflight → data-stats`. Nothing trains until preflight passes.

---

## 7. The through-line to remember

Soham did not add rigor as decoration. Every piece answers a *specific way the
old result could have been a lie*: leakage (firewall), mislabeled graphs (literal
families), an unproven ruler (evaluator parity), capacity confound (width +
param-matched controls), branch-count confound (duplicate-input arms), luck
(5 seeds + intervals), and "we can't even explain it" (interventions before
attributions). That list *is* the project now.
