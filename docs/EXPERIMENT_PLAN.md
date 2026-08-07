# PRISM Experiment Plan

This is the executable order for the corrected study. Architecture details and
formulas are in `../EXPLANATION.md`; paper support is in `LITERATURE.md`.

## 1. Frozen protocol

| Field | Frozen value |
|---|---|
| Images | Official KAIST paired RGB/LWIR archive |
| Train annotations | MSDS sanitized, 7,601 frames |
| Selection | Sequence-disjoint `train-cv` / `val-cv` |
| Test | Improved 2,252-frame annotations, release-gated |
| Input | Native 640x512 |
| Detector | YOLOX-S-derived, one pedestrian class |
| Initialization | From scratch |
| Epochs | Exactly 40 |
| Seeds | 0, 1, 2, 3, 4 |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Base LR | 0.01 |
| Batch | 8 |
| Schedule | 5% linear warmup, cosine decay |
| Weight decay | 5e-4, excluding BN/bias |
| AMP | Enabled |
| EMA | Disabled |
| Core augmentation | Paired HFlip; RGB-only brightness/contrast |
| Primary selection metric | Validation Reasonable All MR, lower is better |
| Confidence floor | 0.001 |
| NMS IoU | 0.65 |
| Evaluator | PRISM canonical, differentially certified to pinned MLPD |

No core arm may tune these fields independently. A change creates a new block.

## 2. Preconditions

The other machine must complete:

```sh
make prepare-kaist
make certify-evaluator
make test
make preflight
make data-stats
```

Preflight must pass. `--skip-audits`, `--allow-no-cuda`, partial data, and a stale
parity certificate are not valid for full training.

## 3. Block A: baselines and controls

All five seeds:

| ID | Inputs/graph | Purpose |
|---|---|---|
| A1 | RGB unimodal | RGB baseline |
| A2 | thermal unimodal | thermal baseline |
| A3 | RGB, `base_ch=42` | wider capacity control |
| A4 | thermal, `base_ch=42` | wider capacity control |
| A5 | RGB+RGB, dual concat-345 | duplicate-input branch control |
| A6 | T+T, dual concat-345 | duplicate-input branch control |
| A7 | paired nested residual, T anchor | nested-at-initialization control |
| A8 | paired nested residual, RGB anchor | nested-at-initialization control |

Acceptance:

- all five raw values exist;
- no failed seed is silently omitted;
- all checkpoint metadata hashes match the arm;
- parameter/MAC/latency differences are reported.

## 4. Block B: interaction graph

Concat is held fixed for learned fusion arms.

| ID | Graph |
|---|---|
| B1 | early six-channel input fusion |
| B2 | one-shot meet dark3, thermal anchor |
| B3 | one-shot meet dark4, thermal anchor |
| B4 | one-shot meet dark5, thermal anchor |
| B5 | multi-scale `{3}` |
| B6 | multi-scale `{4}` |
| B7 | multi-scale `{5}` |
| B8 | multi-scale `{3,4}` |
| B9 | multi-scale `{4,5}` |
| B10 | multi-scale `{3,4,5}` |
| B11 | late union-NMS |
| B12 | late ProbEn |
| B13 | validation winner repeated with RGB anchor |

Selection:

1. Run B1-B10 for all five seeds.
2. Run B11/B12 from paired A1/A2 seeds.
3. Report B1, but rank only the feature-fusion graphs B2-B10 by five-seed
   mean validation All MR. Early input fusion has no intermediate graph on which
   the Block C operators or B13 anchor intervention can be defined.
4. Freeze `outputs/selection/block_b_winner.json`.
5. Run B13. B13 is an anchor-sensitivity analysis, not an opportunity to
   overwrite the frozen graph selection.

Early and late methods are reported alongside the learned graph comparison but
do not become the Block C graph because Block C requires an intermediate graph.

## 5. Block C: fusion operator

The frozen Block B graph is held fixed.

| ID | Operator |
|---|---|
| C1 | average |
| C2 | concat projection |
| C3 | global scalar mixture |
| C4 | sample/channel mixture |
| C5 | anchored gated residual |
| C6 | GAFF-inspired attention |
| C7 | DMAF-inspired differential |
| C8 | dense learned alignment |
| C9 | bounded cross-attention |
| C10 | sample-adaptive quality gate |
| C11 | concat network parameter-matched to C9 cross-attention |
| C12 | spatial concat network parameter-matched to C8 alignment |

All twelve run for all five seeds. One-seed screening may be used only as a smoke
test with `--limit`; it cannot remove an arm or support ranking.

Selection:

1. Rank C1-C12 by five-seed mean validation All MR.
2. Freeze `outputs/selection/block_c_winner.json`.
3. Report the number of operators searched and treat close best-of-twelve
   differences as exploratory.

## 6. Block D: robustness training

Use the frozen Block C architecture:

| ID | Training intervention |
|---|---|
| D1 | drop exactly one modality with probability 0.2 |
| D2 | translate thermal by uniform integer x/y offsets in `[-8,8]` pixels |
| D3 | both D1 and D2 |

All five seeds. Report both clean validation MR and the full intervention suite.
A robustness arm does not replace the clean winner merely because it wins under
corruption; the clean/corrupt tradeoff is the result.

## 7. Post-training behavioral experiments

Run on every seed used in a claim:

```text
paired
RGB train-mean / zero
thermal train-mean / zero
both train-mean / zero
RGB brightness {0.75,0.50,0.25}
RGB blur {5,9,15}
RGB grayscale
thermal contrast {0.75,0.50,0.25}
thermal blur {5,9,15}
thermal Gaussian noise, sigma={0.05,0.10,0.20} of range
thermal x shifts {-16,-8,-4,+4,+8,+16}
thermal y shifts {-16,-8,-4,+4,+8,+16}
```

The train means come only from `train-cv`. Noise is deterministic per frame and
condition. Shifts zero-pad and never wrap.

Report:

- raw per-frame detections/GT, MR, and detection count for each condition;
- MR-point change from paired input;
- all five seed-specific condition values;
- symmetric shift curves;
- clean versus D1-D3 curves.

These are fixed-model input interventions. They support claims about model
behavior, not physical-world sensor causality.

## 8. Representation and interpretability experiments

Implemented raw-first diagnostics:

- per-frame/level RGB, thermal, fused RMS;
- pooled-feature RGB/T cosine;
- gate/mask/offset/attention distributions;
- full pooled vectors;
- linear CKA;
- effective rank.

Implemented attribution and optimization diagnostics:

- GT-linked fixed pre-NMS candidate identity;
- train-mean Integrated Gradients plus branch Grad-CAM at the fusion point;
- deletion/insertion and pointing/inside-box metrics;
- learned-head and full-model parameter-randomization checks;
- raw maps before summaries;
- per-branch weight, gradient, and gradient-to-weight norms through training.
- identical train-cv-to-val-cv frozen ridge probes for branch/fused
  person-background information and sub-cell center information.

Still required for stronger mechanism claims:

- a fixed, stratified probe population;
- exact optimizer update-to-weight ratios;
- label-randomization retraining if saliency becomes a central claim.

Gate or attention plots alone are not an explanation.

## 9. Statistical analysis

### 9.1 Training seeds

Always show the raw five seed values first. Then report:

- mean;
- sample standard deviation;
- median;
- two-sided Student-t 95% interval;
- paired seed differences versus A1 and A2 with their raw values and interval.

For a general fusion-win claim, the candidate must beat both baselines. The
upper 95% interval bound of each paired MR difference should be below zero.
Otherwise report the result as uncertain or negative.

### 9.2 Validation sequences

KAIST frames are temporally dependent. Use paired sequence bootstrap:

- unit: complete `setXX/VYYY`;
- identical sampled sequences for candidate and baseline;
- replacement at sequence level;
- store every replicate;
- report percentile interval as a separate source of uncertainty.

Do not combine sequence-bootstrap and seed intervals into an invented single
confidence statement.

### 9.3 Multiple comparisons

Block B and C rankings search many models. The primary comparisons against A1
and A2 are confirmatory only after the winner and analysis are frozen.
Operator-to-operator best-of-twelve differences remain exploratory unless a new
held-out confirmation is run.

## 10. Run order

```sh
make study-all
```

Equivalent explicit commands:

```sh
PYTHONPATH= .venv/bin/python scripts/run_experiments.py --block A
PYTHONPATH= .venv/bin/python scripts/run_experiments.py --block B
PYTHONPATH= .venv/bin/python scripts/run_late_fusion_matrix.py
PYTHONPATH= .venv/bin/python scripts/select_winner.py \
  --out outputs/selection/block_b_winner.json
PYTHONPATH= .venv/bin/python scripts/run_experiments.py --arm B13 \
  --winning-graph-manifest outputs/selection/block_b_winner.json

PYTHONPATH= .venv/bin/python scripts/run_experiments.py --block C \
  --winning-graph-manifest outputs/selection/block_b_winner.json
PYTHONPATH= .venv/bin/python scripts/select_winner.py \
  --arms C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 \
  --out outputs/selection/block_c_winner.json

PYTHONPATH= .venv/bin/python scripts/run_experiments.py --block D \
  --winning-graph-manifest outputs/selection/block_c_winner.json
PYTHONPATH= .venv/bin/python scripts/summarize_results.py
PYTHONPATH= .venv/bin/python scripts/run_posthoc_studies.py --suite all
```

### 10.1 Provenance and logging contract

The durable append-only ledger is `outputs/ledger.jsonl`. Every arm/seed retry
gets a globally unique `attempt_id` and an immutable directory:

```text
outputs/experiments/<arm>_s<seed>/attempts/<attempt_id>/
  attempt_manifest.json
  attempt_result.json
  training_history.jsonl
  train.stdout.log
  train.stderr.log
  val.stdout.log
  val.stderr.log
  val_metrics.json
  raw_frames/val-cv.frames.jsonl
```

Before a subprocess starts, the runner fsyncs an `attempt_started` ledger row
and an atomic start manifest. The manifest records exact argv, UTC timestamp,
protocol/config/source hashes, Git SHA and dirty-diff hash, allowlisted
determinism variables, Python/package/CUDA/cuDNN/GPU details, and hashes of the
lockfile, architecture config, split, data, and evaluator manifests. Arbitrary
environment variables are never serialized.

Stdout, stderr, and training history are attempt-specific and opened without
overwrite permission, so retries cannot destroy earlier evidence. Training
history contains step losses/LR/foreground count/AMP scale/gradient diagnostics,
epoch means and throughput, peak CUDA memory, validation events, resume events,
and atomic checkpoint hashes. Checkpoints include optimizer, scaler, RNG,
protocol, source, config, attempt, and validation state.

Validation writes both aggregate metrics and a hashed per-frame JSONL containing
post-NMS boxes/scores plus the exact evaluator-shaped ground truth. A terminal
ledger row records phase durations/return codes, log hashes, checkpoint and
metric hashes, error traceback, and final status. A hard kill leaves a visible
unmatched start event; the next same-arm/seed attempt explicitly marks it
abandoned. Selection, summaries, and post-hoc analysis fail closed on unresolved
attempts and never hide a later failed retry behind an earlier success.

Ledger appends are process-locked and fsynced. Same-arm/seed runners use a
non-blocking process lock. JSON manifests and checkpoints use atomic replace.
Post-hoc and late-fusion subprocesses use the same unique-attempt,
stdout/stderr, input-hash, output-hash, and lifecycle-event discipline.

## 11. Official-test rule

Before test:

1. Freeze the selected validation report.
2. Freeze checkpoint hashes.
3. Freeze split/evaluator/parity hashes.
4. Commit the exact code, or record an explicit dirty exception.
5. Freeze hypotheses, intervention definitions, and statistics.

Then evaluate each predeclared model/seed through a release manifest and export
reference-format detections. Score the export with the pinned reference
evaluator. Do not return to model selection after reading test numbers.

## 12. Success and negative results

A fusion arm is successful only if it provides a useful accuracy/compute/
robustness tradeoff against:

- both unimodal baselines;
- wider unimodal controls;
- duplicate-input dual branches;
- nested-at-initialization anchor controls;
- early and late fusion;
- ProbEn.

If it does not, the result is still scientifically useful. The diagnostics
should determine whether evidence supports destructive mixing, optimization
imbalance, collapse, misalignment, shortcut learning, insufficient horizon, or
none of the above. "More information hurts" is not an acceptable conclusion
without that distinction.
