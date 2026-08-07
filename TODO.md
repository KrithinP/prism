# PRISM RGB-T Fusion Research Runbook

## 0. Purpose and Status

This document is the execution specification for turning the current exploratory
PRISM experiments into a controlled study of RGB-thermal pedestrian detection on
KAIST. It covers evaluator certification, data protocol, model definitions,
ablation design, statistical analysis, causal interventions, interpretability, and
reproducibility.

Nothing described here is implemented merely because it appears in this file. A
checkbox may be marked complete only when the corresponding code, command, artifact,
and acceptance evidence exist.

### Current status

- Existing RGB-only, thermal-only, and `F-cat-345` results are exploratory.
- Existing runs that used the official test set for any decision are not admissible
  as final benchmark evidence.
- The canonical evaluator and parity harness have been repaired against the pinned
  MLPD implementation. The reported five-file differential certificate must be
  regenerated and preserved on the execution machine before training.
- The old `fuse_stages` graph is quarantined. Literal early, one-shot, and
  multi-scale graphs are implemented and graph-tested.
- The YOLOX-derived base now matches the audited YOLOX-S graph, BatchNorm,
  `1-IoU^2` loss, and current center-only SimOTA semantics. The fixed 40-epoch
  recipe remains intentionally different from official YOLOX training.
- KAIST ignore regions now suppress negative objectness supervision rather than
  being deleted and learned as background.
- Data-dependent audits, all 40-epoch runs, intervention outputs, and official-test
  release artifacts remain to be produced on the other machine.
- No new training should begin until `make prepare-kaist`,
  `make certify-evaluator`, `make test`, and `make preflight` pass there.

### Primary scientific question

Under a fixed YOLOX detector, fixed training budget, and fixed data protocol:

> Which computational graph and fusion operator extract complementary information
> from RGB and thermal inputs without allowing one modality to corrupt, dominate, or
> shortcut the other?

### Secondary questions

1. At which feature depth should the modalities first interact?
2. Is one-shot fusion, multi-scale injection, progressive interaction, or decision
   fusion the strongest controlled family?
3. When fusion fails, is the failure caused by evaluator error, data error,
   optimization conflict, representation collapse, destructive feature mixing,
   spatial misalignment, or inadequate capacity?
4. Do apparent attention or saliency patterns agree with causal input
   interventions?
5. Does the selected model remain useful when one modality is shifted, degraded,
   missing, or stale?

### Explicit non-goals for the architecture-discovery phase

- Pretraining versus from-scratch comparisons.
- Hyperparameter saturation for the best possible leaderboard score.
- Real-camera calibration, asynchronous 120/60 Hz deployment, or different-FOV
  rectification.
- Claims about airborne objects or non-pedestrian classes.
- Selecting a model from the official test set.

These are outside the current matrix. The current study is a controlled KAIST
architectural investigation, and all of its arms start from random
initialization.

## 1. Locked Experimental Decisions

- [x] Use the audited YOLOX-S-derived graph as the common detector family.
- [x] Initialize every model from scratch.
- [x] Train every architecture-discovery run for exactly **40 epochs**.
- [x] Use the same train/validation examples, image size, preprocessing,
  augmentations, batch construction, optimizer family, learning-rate schedule,
  weight decay, EMA policy, precision mode, and seed list for all comparable arms.
- [x] Use validation for architecture selection and every other decision.
- [x] Keep the official improved-annotation test set unreachable until a finalist and
  analysis plan are frozen.
- [x] Report epoch 40 as the primary checkpoint for the architecture study. If a
  validation-best checkpoint is also reported, predeclare the selection metric and
  report it only as a secondary analysis.
- [x] Never silently retry only failed or unfavorable seeds. Document all failed,
  diverged, interrupted, or OOM runs.

Add this comment beside the future epoch configuration:

> Initial experiments fix secondary choices such as epoch count so that major
> architectural decisions can be compared cleanly. Training duration and other
> optimization choices can be saturated after the fusion design is fixed.

### Definition of a comparable run

Two arms are directly comparable only if all fields below agree except for the
declared intervention:

```text
dataset revision
train sequence list
validation sequence list
annotation revision
image preprocessing
augmentation draws or seed policy
input resolution
batch size and accumulation
optimizer and schedule
epoch count
initialization policy
random seed
evaluation implementation and commit
confidence floor and NMS configuration
```

If batch size or hardware forces a change, record it as a separate experimental
block rather than adding it to the same mean and standard deviation.

## 2. Phase 0: Preserve and Audit Existing Evidence

### 2.1 Artifact collection

- [ ] Copy every available checkpoint from the training machine.
- [ ] Copy raw stdout/stderr, local ledger/history, system metrics, any external
  tracker exports, tensorboard events, and run summaries.
- [ ] Copy the exact config used by each run, not only the current config.
- [ ] Copy raw per-image detections before NMS, after NMS, and after evaluator-format
  conversion where available.
- [ ] Copy optimizer, scheduler, AMP scaler, and EMA state where saved.
- [ ] Record the repository Git SHA, dirty diff, package lock/`pip freeze`, CUDA,
  cuDNN, GPU model, driver, and hostname.
- [ ] Hash all copied artifacts and preserve the hash manifest.

### 2.2 Historical run ledger

Create one row per run with at least:

```text
run_id
date
status
scientific_status: exploratory | admissible | invalid
seed
model family
fusion graph
fusion operator
fusion levels
anchor modality
width/depth multiplier
parameter count
FLOPs
train/validation/test split hashes
annotation hash
epochs completed
checkpoint evaluated
data settings
optimizer settings
evaluator commit/config
All/Day/Night MR
artifact paths
notes
```

- [ ] Label any test-guided run `exploratory-test-contaminated`.
- [ ] Label runs without recoverable config or split metadata
  `exploratory-unreproducible`.
- [ ] Do not delete invalid runs; they remain useful for debugging and
  hypothesis-generation.
- [ ] Do not combine historical and corrected runs in the same statistical summary.

### Phase 0 exit gate

- [ ] Every reported historical number maps to a run ID and recoverable artifact, or
  is explicitly marked unrecoverable.
- [ ] The exploratory results are preserved before evaluator or model code changes.

## 3. Phase 1: Certify the KAIST Evaluator

### 3.1 Exact reference bundle

For evaluator certification, download the MLPD KAIST evaluation bundle:

<https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection>

The full KAIST image archive is not required for evaluator certification. The MLPD
repository already contains:

```text
evaluation_script/KAIST_annotation.json
evaluation_script/evaluation_script.py
evaluation_script/coco.py
evaluation_script/cocoeval.py
evaluation_script/state_of_arts/MLPD_result.txt
evaluation_script/state_of_arts/ARCNN_result.txt
evaluation_script/state_of_arts/CIAN_result.txt
evaluation_script/state_of_arts/MSDS-RCNN_result.txt
evaluation_script/state_of_arts/MBNet_result.txt
```

Download and pin it:

```sh
PYTHONPATH= .venv/bin/python scripts/pin_mlpd_eval.py --clone
```

The script fetches the exact audited commit declared in its source, rejects a
different checkout, excludes generated cache files, and writes the tracked
manifest. `external/MLPD-eval/` itself is fetched state and is ignored by Git.

- [x] Record the returned SHA in a machine-readable evaluator manifest.
- [x] Never depend on an unpinned remote branch for a reported score.
- [x] Preserve an untouched copy of the pinned source.

### 3.2 Isolated reference environment

The released evaluator is historical code. Build a small isolated environment
instead of modifying it until its bundled result runs:

```sh
python -m venv .venv-mlpd-eval
source .venv-mlpd-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<1.24" matplotlib
```

The conservative NumPy pin avoids removed historical aliases such as `np.float`.
Record the exact resolved versions. If the evaluator works unchanged on newer
versions, document that separately; do not patch the reference first and then call
it a reproduction.

### 3.3 Reproduce bundled detections

From `external/MLPD-eval/evaluation_script`:

```sh
python evaluation_script.py \
  --annFile KAIST_annotation.json \
  --rstFile state_of_arts/MLPD_result.txt \
            state_of_arts/ARCNN_result.txt \
            state_of_arts/CIAN_result.txt \
            state_of_arts/MSDS-RCNN_result.txt \
            state_of_arts/MBNet_result.txt \
  --evalFig mlpd_reference.jpg
```

The repository table reports the following values as cross-checks:

| Method | All MR | Day MR | Night MR |
|---|---:|---:|---:|
| CIAN | 14.12 | 14.77 | 11.13 |
| MSDS-RCNN | 11.34 | 10.53 | 12.94 |
| AR-CNN | 9.34 | 9.94 | 8.38 |
| MBNet | 8.13 | 8.28 | 7.86 |
| MLPD VGG16 | 7.58 | 7.95 | 6.95 |
| MLPD ResNet50 | 7.61 | 8.36 | 6.35 |

The bundled `MLPD_result.txt` must be associated with the correct MLPD row by its
actual output; do not assume which backbone produced it.

- [ ] Save command, stdout, stderr, figure, environment, source SHA, annotation hash,
  and detection-file hashes.
- [ ] Require the bundled detections to reproduce the repository values to displayed
  precision.
- [ ] Investigate any mismatch before evaluating PRISM outputs.

### 3.4 Canonical KAIST semantics

The reference implementation fixes:

```text
IoU threshold: 0.5
maximum detections per frame: 1000
FPPI reference points:
  0.0100, 0.0178, 0.0316, 0.0562, 0.1000,
  0.1778, 0.3162, 0.5623, 1.0000
Reasonable height range: >=55 px
Reasonable occlusion: none or partial
image boundary range: [5, 5, 635, 507]
Day images: first 1455 sorted test IDs
Night images: remaining sorted test IDs
```

Important matching semantics:

- Regular ground truth uses intersection-over-union.
- Ignored/crowd ground truth uses intersection divided by detection area.
- Regular ground truth can match only once.
- An ignored region can absorb multiple detections.
- Non-ignored ground truth is considered before ignored ground truth.
- Detections are sorted stably by descending confidence.
- MR is the geometric mean of miss rates sampled at the nine FPPI points.

These are not implementation trivia. A detector near ignored people, image
boundaries, or tied confidence scores can receive a different TP/FP sequence when
any one rule changes.

### 3.5 PRISM export contract

Export one detection per CSV line:

```text
image_id,x,y,width,height,score
```

- `image_id` is one-based in the TXT file; the reference loader subtracts one.
- Coordinates use top-left `x,y,width,height`.
- Use the exact image-ID mapping in `KAIST_annotation.json`.
- Do not assume the dataloader order equals benchmark order.
- Do not round coordinates or scores unless the reference format requires it.
- Do not apply an undocumented confidence threshold.
- Record whether NMS is class-aware, its IoU threshold, and whether it occurs before
  export.

- [x] Create an explicit `frame_key -> benchmark_image_id` table.
- [x] Assert one-to-one coverage of all 2,252 test frames, including frames with zero
  detections.
- [x] Assert finite coordinates/scores and positive width/height.
- [x] Assert that exported image IDs lie in the annotation set.
- [x] Save the exporter version and output hash.

### 3.6 Synthetic evaluator unit cases

Construct small deterministic fixtures for:

- [x] One exact detection on one regular ground-truth box: one TP.
- [x] One detection below IoU 0.5: one FP and one FN.
- [x] Two detections on one regular box: first TP, second FP.
- [x] Several detections inside one ignored region: all absorbed/ignored.
- [x] Detection overlapping regular and ignored boxes: regular match takes priority.
- [x] Detection with high intersection-over-detection-area but low ordinary IoU on an
  ignored region: ignored under canonical semantics.
- [x] Boxes at 54 and 55 pixels height: boundary behavior is exact.
- [x] Boxes just inside and outside the benchmark image boundary.
- [x] Identical confidence scores: stable ordering is reproduced.
- [x] No detections, no ground truth, and ground truth with no detections.
- [x] Confidence floor too high to reach FPPI=1: evaluation fails loudly.
- [ ] Correct All/Day/Night subset sizes.

Each fixture must compare the local evaluator's intermediate TP/FP/ignore sequence
with the pinned evaluator, not merely its final MR.

### 3.7 Differential parity study

Run both evaluators on at least:

```text
MLPD_result.txt
ARCNN_result.txt
MBNet_result.txt
one PRISM unimodal detection file
one PRISM fusion detection file
```

Compare:

```text
number of scored GT boxes
number of ignored GT boxes
detections retained
detections absorbed by ignore regions
per-detection TP/FP/ignore label
stable confidence ordering
cumulative TP and FP
FPPI curve
miss-rate curve
nine sampled miss rates
All/Day/Night final MR
```

Acceptance:

- [ ] Exact agreement on counts and per-detection labels for deterministic fixtures.
- [ ] Exact agreement on the nine sampled miss rates within floating-point tolerance.
- [x] At most 0.1 MR-point disagreement for All, Day, and Night on every released
  detection file.
- [ ] Lowering the export confidence floor further does not change MR.

If parity fails:

1. Treat MLPD as the authoritative final scorer.
2. Find the first detection at which sorted TP/FP/ignore sequences differ.
3. Reduce that case into a synthetic test.
4. Fix the local implementation without modifying the reference.
5. Re-run the entire parity suite.
6. Retire the local evaluator if exact semantics remain uncertain.

An independent secondary cross-check can later use MS-DETR, which also releases
KAIST evaluation code and detections:
<https://github.com/YinghuiXing/MS-DETR>.

### Phase 1 exit gate

- [x] The pinned reference reproduces its bundled published results.
- [x] PRISM can export valid reference-format detections.
- [x] The local evaluator passes the synthetic and released-file parity criteria, or
  is formally retired.
- [ ] All historical raw detections have been re-scored with the canonical evaluator.

## 4. Phase 2: Freeze the Data and Selection Protocol

### 4.1 Dataset identity

- [ ] Record the exact KAIST archive source, download date, file count, and archive
  checksum.
- [ ] Record the paired-training annotation source and commit/archive checksum.
- [ ] Record the improved test annotation hash.
- [ ] Verify visible/LWIR pair existence and dimensions for every selected frame.
- [ ] Detect duplicates, missing images, unreadable images, and orphan annotations.
- [ ] Produce a deterministic dataset audit report before training.

### 4.2 Sequence-disjoint validation

Nearby frames from the same video are highly correlated. Random frame splitting
would leak backgrounds, pedestrians, and temporal context.

- [ ] Split at `set/video` granularity, never by individual frame.
- [ ] Preserve complete videos in exactly one of train or validation.
- [ ] Make the validation split large enough to contain meaningful Day and Night,
  scale, and occlusion strata.
- [ ] Report train/validation counts by frames, boxes, Day/Night, scale, occlusion,
  and video.
- [ ] Publish sorted sequence lists and their SHA256 hashes.
- [ ] Freeze the split before architecture comparison.
- [ ] Never move a hard validation sequence into training during model selection.

The official 2,252-frame improved-annotation test set remains completely separate.

### 4.3 Annotation and preprocessing audit

- [ ] Render at least 100 deterministic train/validation pairs with annotations over
  both modalities.
- [ ] Verify box conversion from source coordinates through resize, letterbox, crop,
  flip, and mosaic transformations.
- [ ] Confirm that geometric augmentation is identical for paired modalities.
- [ ] Confirm that modality-specific photometric augmentation never alters the other
  modality accidentally.
- [ ] Audit thermal channel loading, dtype, normalization, dynamic range, clipping,
  and channel replication.
- [ ] Audit RGB color order (`RGB` versus `BGR`) and normalization.
- [ ] Measure paired-image registration residuals on annotated pedestrian centers and
  edges where possible.
- [ ] Record the fraction of boxes partially outside the image or excluded by
  benchmark rules.

### 4.4 Test-set firewall

- [ ] Training scripts do not load test annotations or test images.
- [ ] Hyperparameter sweeps cannot invoke the test evaluator.
- [ ] Validation is the only metric visible during model development.
- [ ] Official test evaluation is a separate command requiring a frozen run manifest.
- [ ] Log every official test invocation with timestamp, model hash, and reason.
- [ ] Do not use test Day/Night breakdown to choose a model.

### Phase 2 exit gate

- [ ] Dataset, annotations, and split hashes are frozen.
- [ ] Pairing and coordinate audits pass.
- [ ] Test data is unreachable from the training/model-selection path.
- [ ] A reviewer can reproduce the train/validation membership from committed lists.

## 5. Phase 3: Freeze the 40-Epoch Training Recipe

### 5.1 Configuration fields to freeze

- [ ] Input resolution and aspect-ratio policy.
- [ ] Effective batch size and gradient accumulation.
- [ ] Optimizer, base LR, momentum/betas, and weight decay.
- [ ] LR warmup and schedule over exactly 40 epochs.
- [ ] YOLOX no-augmentation/final-stage policy.
- [ ] Mosaic, mixup, flip, scale, crop, and color augmentation parameters.
- [ ] Thermal-specific augmentation policy.
- [ ] BatchNorm/SyncBN behavior.
- [ ] AMP and gradient-scaling policy.
- [ ] EMA configuration and whether EMA or raw weights are primary.
- [ ] Gradient clipping.
- [ ] Objectness, box, and class loss weights.
- [ ] Label assignment settings.
- [ ] Dataloader workers, deterministic flags, and seed propagation.
- [ ] NMS and confidence floor used for validation.

### 5.2 Seed policy

- [ ] Predeclare five exact seeds and use them for every core arm.
- [x] Seed Python, NumPy, PyTorch CPU, every CUDA device, dataloader workers, samplers,
  and augmentation generators.
- [ ] Prefer common random numbers across paired arms where the implementation permits
  it: the same seed should produce the same sample ordering and comparable
  augmentations.
- [ ] Record nondeterministic CUDA operations and quantify repeat-run variance on one
  seed if strict determinism is not practical.

### 5.3 Training health logging

Log at least per epoch, and more frequently where affordable:

```text
total loss
box/objectness/class losses
learning rate
gradient norm
AMP scale and skipped steps
GPU memory
examples/second
validation MR and recall
per-branch activation norms
per-branch gradient norms for fusion models
```

- [x] Fail loudly on NaN/Inf loss, empty loaders, and missing required modality.
  The data audit checks invalid boxes before full training.
- [x] Save periodic and epoch-40 raw checkpoints. EMA is frozen off; an
  initialization checkpoint is intentionally omitted to avoid duplicating every
  large run artifact.
- [x] Save model, optimizer, AMP scaler, RNG, dataloader-generator, epoch, and
  global-step state needed to resume without changing the schedule.

### 5.4 Recipe sanity check

Before the full matrix:

- [ ] Run one RGB-only seed and one thermal-only seed through 40 epochs.
- [ ] Confirm neither diverges and both exceed a predeclared nontrivial baseline.
- [ ] Confirm train and validation curves are recorded completely.
- [ ] Confirm epoch-40 evaluation is deterministic when repeated.
- [ ] Confirm resource estimates for the full matrix.

Do not tune the recipe separately per modality or fusion arm during the controlled
study. Any later saturation is a new phase with its own protocol.

### Phase 3 exit gate

- [ ] One immutable architecture-study config exists.
- [ ] Five seeds and checkpoint-selection rules are frozen.
- [ ] Training and validation commands no longer access test data.
- [ ] Sanity runs complete without pipeline errors.

## 6. Phase 4: Define Fusion Mathematically and Computationally

Let:

```text
x_r: RGB input
x_t: thermal input
R_l = B_r^l(R_{l-1}), R_0 = x_r
T_l = B_t^l(T_{l-1}), T_0 = x_t
F_l: fusion operator at level l
A_l: single-modality adapter/projection
N: YOLOX PAFPN neck
H: YOLOX decoupled detection head
```

For every model, publish a graph diagram and a table containing input/output shapes,
stride, channels, ownership, and whether each tensor is influenced by RGB, thermal,
or both.

### 6.1 Unimodal baselines

```text
y_r = H(N(B_r(x_r)))
y_t = H(N(B_t(x_t)))
```

Required controls:

- [x] Standard RGB-only YOLOX.
- [x] Standard thermal-only YOLOX.
- [x] A wider unimodal model approximately matching the parameter/FLOP cost of the
  principal fusion model.
- [x] A duplicated-backbone unimodal control where both branches receive the same
  modality, to separate multimodal information from extra capacity.

The duplicated-input controls are important:

```text
RGB+RGB: two branches, both receive x_r
T+T: two branches, both receive x_t
```

If RGB+RGB or T+T improves as much as RGB+T, the gain is primarily capacity or
ensembling, not complementary modalities.

### 6.2 Early fusion

```text
Z_0 = P([x_r; x_t])
y = H(N(B_s(Z_0)))
```

- Thermal is resized/represented under an explicitly documented policy.
- The projection maps four input channels to the standard stem width.
- Compare concat projection with simple four-channel stem only if both are defined
  and named separately.
- Record whether thermal is one channel or replicated.

Early fusion is a diagnostic baseline, not the presumed final architecture.

### 6.3 True one-shot middle fusion

At meeting level `L`:

```text
R_L = B_r^{1:L}(x_r)
T_L = B_t^{1:L}(x_t)
Z_L = F_L(R_L, T_L)
Z_{L+1:5} = B_s^{L+1:5}(Z_L)
```

After `F_L`, there is exactly one shared stream. It must never split again.

YOLOX PAFPN requires multiple backbone scales. For a thermal-anchored one-shot model:

```text
Meet at dark3:
  neck inputs = [Z_3, Z_4, Z_5]

Meet at dark4:
  neck inputs = [A_3(T_3), Z_4, Z_5]

Meet at dark5:
  neck inputs = [A_3(T_3), A_4(T_4), Z_5]
```

Here RGB information first enters at the declared meeting level; earlier neck skips
remain thermal-only. This makes the depth claim literal. Repeat the critical result
with RGB anchoring if conclusions may depend on the anchor:

```text
dark4 RGB anchor = [A_3(R_3), Z_4, Z_5]
dark5 RGB anchor = [A_3(R_3), A_4(R_4), Z_5]
```

- [x] Implement dark3, dark4, and dark5 meeting graphs.
- [x] Use one fusion operation only.
- [x] Ensure dark4/dark5 shared suffixes are structurally identical after meeting.
- [x] Report the anchor modality prominently.
- [ ] Test anchor sensitivity for the selected depth.

### 6.4 Multi-scale neck fusion

Both complete backbones remain independent:

```text
R_3,R_4,R_5 = B_r(x_r)
T_3,T_4,T_5 = B_t(x_t)

Z_l = F_l(R_l,T_l), if l in S
Z_l = A_l(T_l),     if l not in S  [thermal anchor]
y = H(N(Z_3,Z_4,Z_5))
```

Required subsets:

```text
{3}
{4}
{5}
{3,4}
{4,5}
{3,4,5}
```

Optional `{3,5}` may be added for completeness but is not needed before the ordered
subsets are understood.

This family answers which feature scales should receive complementary information.
It does not answer where the streams first become one.

### 6.5 Progressive or bidirectional interaction

The streams remain distinct:

```text
R'_l = R_l + G_{t->r}(T_l, R_l)
T'_l = T_l + G_{r->t}(R_l, T_l)
```

Later blocks consume `R'_l` and `T'_l`. Define:

- interaction direction: thermal-to-RGB, RGB-to-thermal, or bidirectional;
- interaction levels;
- whether weights are shared;
- final aggregation before the neck/head.

Report this as a separate family.

### 6.6 Late/decision fusion

```text
D_r = detector_r(x_r)
D_t = detector_t(x_t)
M = associate(D_r, D_t)
y = aggregate(M)
```

Required late-fusion baselines:

- [x] Union plus NMS.
- [ ] Confidence averaging after box association.
- [x] ProbEn score fusion.
- [ ] ProbEn uncertainty-weighted box fusion where supported.

Specify association IoU, unmatched-box handling, score calibration, and NMS.

### 6.7 Why the current depth sweep is invalid

The current `TwoStreamYOLOX(fuse_stages=(4,))` behavior is:

```text
Z_3 = R_3 + T_3
Z_4 = Conv1x1([R_4;T_4])
Z_5 = R_5 + T_5
```

Every neck input contains both modalities. The model changes the operator at dark4
while retaining fusion at dark3 and dark5. It does not represent streams meeting at
dark4.

- [x] Rename/quarantine all results produced by this implementation.
- [x] Do not use them to claim an optimal meeting depth.
- [ ] They may later be reported as mixed-operator multi-scale fusion if the graph is
  otherwise verified.

### 6.8 Graph-level tests

For each architecture:

- [x] Tensor-shape test for every supported input resolution.
- [x] Forward/backward finite-value test.
- [x] Parameter ownership test: private prefix versus shared suffix.
- [x] Modality-dependence test: perturb RGB and assert only declared downstream
  tensors change.
- [x] Zero-gradient test: excluded modality has no path to an anchor-only level.
- [x] Hook-based test counting the exact number and location of fusion calls.
- [x] Serialization test preserving graph configuration.
- [ ] Parameter, FLOP, activation-memory, and latency report.

### Phase 4 exit gate

- [x] Every implemented experiment name maps to one unambiguous computational graph.
- [x] True one-shot and multi-scale fusion are implemented and tested separately.
- [x] No non-selected level silently combines modalities.
- [ ] Capacity differences are measured and controlled.

## 7. Phase 5: Standardize Fusion Operators

Let adapters place features in the same shape:

```text
r = P_r(R)
t = P_t(T)
```

All operators must state normalization, activation, initialization, output width,
residual path, and added parameters/FLOPs.

### 7.1 Operator ladder

1. **Average**

   ```text
   Z = (r + t) / 2
   ```

   Zero-parameter equal-trust baseline.

2. **Concatenation plus projection**

   ```text
   Z = Act(Norm(Conv1x1([r;t])))
   ```

   Primary learned linear-mixing anchor.

3. **Global scalar mixture**

   ```text
   alpha = sigmoid(a)
   Z = alpha*r + (1-alpha)*t
   ```

   Initialize `alpha=0.5`. Log its learned value.

4. **Channel mixture**

   ```text
   alpha_c = sigmoid(MLP(GAP([r;t])))
   Z = alpha_c*r + (1-alpha_c)*t
   ```

5. **Spatial/channel gated residual**

   ```text
   g = sigmoid(G([r;t]))
   Z = r + g*t
   ```

   Repeat with thermal anchor `Z=t+g*r` when anchor dependence matters.

   A separate exact-nesting control uses `Z=anchor+alpha*complement`, with
   unconstrained `alpha=0` at initialization and no post-normalization. It belongs
   in Block A, not the capacity-matched operator ranking.

6. **GAFF/CBAM-style attention**

   Guided channel and spatial masks. Save masks, but do not call them causal
   explanations without interventions.

7. **MBNet/DMAF-style differential fusion**

   Use `r-t` or an equivalent differential representation to estimate channel
   reweighting.

8. **Alignment-aware fusion**

   Predict offsets and sample/warp the complementary feature before fusion. Regularize
   and log offsets.

9. **Cross-attention ceiling**

   ```text
   Q = W_q r
   K = W_k t
   V = W_v t
   Z = r + softmax(QK^T/sqrt(d))V
   ```

   Restrict to low-resolution levels and report real memory/latency.

10. **Sample-adaptive quality-conditioned fusion**

    Estimate one per-image trust coefficient from RGB/thermal feature means and
    standard deviations. Compare with the fixed global scalar and per-channel gate.
    Explicit validity metadata remains a later deployment extension.

### 7.2 Operator fairness

- [x] Keep output width identical.
- [x] Use matched normalization and activation for all ranked Block C operators.
- [ ] Report added capacity.
- [x] Include exact parameter-matched concat controls for cross-attention and
  learned alignment.
- [x] Verify gates do not saturate at initialization.
- [ ] Log modality-specific weight norms and gate distributions during training.
- [x] Use the same frozen fusion graph while comparing operators.

### Phase 5 exit gate

- [ ] Every operator passes shape, gradient, serialization, and finite-value tests.
- [ ] Concat is established as the simple learned anchor.
- [ ] Expensive operators have compute/capacity controls.

## 8. Phase 6: Controlled Experiment Matrix

Avoid a full Cartesian product initially. Use sequential questions so each stage
changes one factor.

### 8.1 Block A: unimodal and capacity controls

| ID | Inputs | Architecture | Purpose |
|---|---|---|---|
| A1 | RGB | standard YOLOX | visible baseline |
| A2 | Thermal | standard YOLOX | thermal baseline |
| A3 | RGB | wider YOLOX | parameter/FLOP control |
| A4 | Thermal | wider YOLOX | parameter/FLOP control |
| A5 | RGB+RGB | dual branch, concat-345 | duplicated-input capacity control |
| A6 | T+T | dual branch, concat-345 | duplicated-input capacity control |
| A7 | RGB+T | nested residual-345, thermal anchor | nested-at-initialization control |
| A8 | RGB+T | nested residual-345, RGB anchor | nested-at-initialization control |

- [ ] Five seeds per arm.
- [ ] Use these to quantify gains due merely to extra parameters/branches.

### 8.2 Block B: fusion-family comparison

Use concat projection as the common fusion operator.

| ID | Family | Variant |
|---|---|---|
| B1 | Early | input concat projection |
| B2 | One-shot | meet dark3, thermal anchor |
| B3 | One-shot | meet dark4, thermal anchor |
| B4 | One-shot | meet dark5, thermal anchor |
| B5 | Multi-scale | fuse `{3}` |
| B6 | Multi-scale | fuse `{4}` |
| B7 | Multi-scale | fuse `{5}` |
| B8 | Multi-scale | fuse `{3,4}` |
| B9 | Multi-scale | fuse `{4,5}` |
| B10 | Multi-scale | fuse `{3,4,5}` |
| B11 | Late | union plus NMS |
| B12 | Late | ProbEn |

- [ ] Five seeds for core learned arms.
- [ ] Late fusion must use paired unimodal seeds or a predeclared cross-seed pairing.
- [ ] Compare within family before comparing across unequal compute budgets.
- [ ] Repeat the winning eligible feature graph with RGB anchoring.
- [x] Keep B1 as a required comparator but exclude it from the Block C template
  selection; only B2-B10 expose an intermediate graph and an anchor.

### 8.3 Block C: operator comparison

Choose one graph using validation only, then hold it fixed:

| ID | Operator |
|---|---|
| C1 | average |
| C2 | concat projection |
| C3 | scalar mixture |
| C4 | channel mixture |
| C5 | gated residual |
| C6 | GAFF-style attention |
| C7 | DMAF-style differential fusion |
| C8 | alignment-aware fusion |
| C9 | low-resolution cross-attention |
| C10 | sample-adaptive quality gate |
| C11 | concat network parameter-matched to C9 |
| C12 | spatial concat network parameter-matched to C8 |

Run all operators and controls for five seeds. A `--limit` smoke may catch broken
implementations, but it cannot remove an arm, support ranking, or enter a paper.

### 8.4 Block D: robustness additions

After selecting a clean-pair architecture:

- [x] Implement modality dropout training as D1/D3.
- [x] Implement shift/misalignment augmentation as D2/D3.
- [x] Implement deterministic thermal contrast/noise evaluation interventions.
- [x] Implement deterministic RGB illumination/blur evaluation interventions.
- [x] Implement sample-adaptive feature-quality conditioning as C10.
- [ ] Add explicit sensor-validity metadata when moving beyond paired KAIST.
- [ ] Optional consistency loss between full and degraded-modality predictions.

These runs answer a different robustness question and must not replace the clean
architecture controls.

### 8.5 Required output per run

```text
epoch-40 checkpoint
raw checkpoint (EMA is frozen off)
training log
validation detections at a very low score floor
validation All MR
full MR-FPPI points
recall at FPPI={0.1, 1.0}
loss curves
parameter count
FLOPs
peak training/inference memory
batch-1 and deployment-batch latency
per-image prediction table
run manifest and artifact hashes
```

### 8.6 Selection rule

- [ ] Predeclare one primary validation metric: Reasonable All log-average MR.
- [ ] Treat Day/Night MR, robustness, latency, and memory as secondary constrained
  metrics.
- [ ] Select a family first, then an operator; do not repeatedly search the same
  validation set without tracking the number of comparisons.
- [ ] Freeze finalist configuration and analysis code before official test.
- [ ] Evaluate the official test once for the frozen primary model set, not after
  every training run.

### Phase 6 exit gate

- [ ] All core arms use the frozen protocol and five predeclared seeds.
- [ ] Every summary is backed by raw per-seed and per-image artifacts.
- [ ] Model selection is reproducible from validation data alone.

## 9. Phase 7: Diagnose Why More Information Can Hurt

Fusion has access to a superset of the unimodal inputs, but a finite learned system is
not guaranteed to exploit that superset. More input variables enlarge the hypothesis
and optimization spaces. A model can learn a worse function because of destructive
mixing, shortcut correlations, finite capacity, misalignment, unequal feature scale,
or conflicting gradients.

### 9.1 Hypothesis H1: implementation or evaluator error

Prediction:

- Fusion looks worse because boxes, frame IDs, NMS, resize, or evaluator semantics
  differ between arms.

Tests:

- [x] Evaluator parity suite implementation.
- [x] Identical postprocessing tests on injected synthetic predictions.
- [ ] Trace a fixed image from dataloader through transforms, decoding, NMS, export,
  and evaluator for every arm.
- [ ] Confirm RGB/T/fusion use identical frame lists and ground truth.

Falsification:

- Pipelines and intermediate labels match exactly, yet fusion remains worse.

### 9.2 Hypothesis H2: additional capacity is not the benefit

Prediction:

- Dual-branch unimodal controls match RGB-T performance, or widening a unimodal
  network gives the same improvement.

Tests:

- [ ] RGB+RGB, T+T, and wider unimodal controls.
- [ ] Parameter/FLOP-matched comparisons.

Interpretation:

- RGB-T versus standard unimodal is insufficient to claim complementarity.

### 9.3 Hypothesis H3: scale/statistics mismatch causes destructive mixing

Prediction:

- RGB and thermal feature norms/distributions differ sharply before fusion.
- Sum or unnormalized concat produces saturation or abrupt distribution changes.
- Normalized projections or residual gating improve stability.

Tests:

- [ ] Log per-channel means, variances, norms, sparsity, kurtosis, and saturation
  before/after every fusion block.
- [ ] Compare sum, normalized concat, and gated residual.
- [ ] Inspect BatchNorm running statistics separately by branch.

### 9.4 Hypothesis H4: modality dominance or gradient conflict

Prediction:

- One branch receives much larger gradients.
- RGB and thermal gradients on shared parameters have negative cosine similarity.
- One branch becomes uninformative even though its unimodal model works.

Tests:

- [ ] Per-branch gradient norm over training.
- [ ] Gradient cosine similarity on shared/fusion parameters.
- [ ] Loss change after freezing each branch temporarily in diagnostic runs.
- [ ] Frozen-feature linear probes by epoch.
- [ ] Compare balanced/gated fusion and DMAF.

### 9.5 Hypothesis H5: representational collapse or rank bottleneck

Prediction:

- Fused representations lose effective rank or become nearly identical to one
  branch.
- Decodable pedestrian information decreases after fusion.

Tests:

- [x] Effective-rank summaries with all pooled feature vectors preserved.
- [x] RGB-thermal CKA at fused levels.
- [ ] Per-branch and fused linear-probe detection/classification/localization scores.
- [ ] Feature covariance condition number.
- [ ] Compare concat bottleneck widths and residual fusion.

### 9.6 Hypothesis H6: spatial misalignment corrupts local correspondence

Prediction:

- Fusion degradation grows with measured RGB-T shift and at high-resolution fusion
  levels.
- Artificial shift hurts concat more than alignment-aware or late fusion.

Tests:

- [ ] Measure per-frame alignment proxy.
- [x] Controlled shifts in `x/y` over a symmetric grid.
- [ ] Compare dark3 versus dark5 sensitivity.
- [ ] Compare concat, deformable/alignment-aware, and late fusion.
- [ ] Stratify results by pedestrian size because a fixed pixel shift is proportionally
  larger for small boxes.

### 9.7 Hypothesis H7: shortcut learning from Day/Night

Prediction:

- The model learns scene illumination as a modality-selection shortcut.
- Modality benefit reverses in unusual illumination/thermal-contrast conditions.

Tests:

- [ ] Stratify beyond Day/Night using measured RGB luminance and thermal contrast.
- [ ] Cross strata analogous to ROTO/RXTO/ROTX where data permits.
- [ ] Counterfactual brightness/contrast interventions.
- [ ] Gate/attention distributions by stratum.

### 9.8 Hypothesis H8: optimization budget is insufficient

Prediction:

- Fusion is still improving at epoch 40 while unimodal arms have plateaued.

Tests after architecture selection only:

- [ ] Compare learning curves and validation slope near epoch 40.
- [ ] Extend the frozen best unimodal and fusion configurations to a longer
  predeclared horizon.
- [ ] Do not retrospectively mix the extended runs into the 40-epoch architecture
  matrix.

### Diagnostic decision table

For every hypothesis, record:

```text
prediction
measurement
result
confidence interval
supports | contradicts | inconclusive
next intervention
```

Do not tell a causal story from the aggregate MR result alone.

## 10. Phase 8: Causal Modality-Use Experiments

### 10.1 Unit of analysis

Use a fixed ground-truth object or a consistently matched prediction as the unit of
prediction-level analysis. Associate predictions across interventions using a
predeclared GT-centered matching rule, not whichever high-confidence box appears.

For each object, store:

```text
frame_id
ground_truth_id
condition
matched_prediction_id
score
box
IoU
center error
TP/FP/FN
RGB contrast
thermal contrast
occlusion
size
Day/Night
alignment proxy
```

### 10.2 Core interventions

Evaluate every selected trained seed without retraining under:

| Code | RGB input | Thermal input | Purpose |
|---|---|---|---|
| RT | paired | paired | observed fused behavior |
| R0 | paired | null | RGB-dependent behavior |
| 0T | null | paired | thermal-dependent behavior |
| 00 | null | null | model bias/background response |
| RsT | shuffled | paired | destroy RGB correspondence |
| RTs | paired | shuffled | destroy thermal correspondence |
| RshiftT | paired | shifted | spatial correspondence sensitivity |
| RstaleT | paired | stale same-sequence | temporal correspondence sensitivity |
| RdegT | paired | degraded | thermal-quality sensitivity |
| Rdeg-T | degraded | paired | RGB-quality sensitivity |

Null is not one universal zero image. Repeat critical conclusions using:

- dataset mean;
- heavy blur preserving coarse luminance;
- calibrated sensor-like noise;
- cross-sample image;
- same-sequence distant/stale frame.

### 10.3 Two-modality factorial effects

For a prediction-level utility `u`, define:

```text
u_00 = u(null RGB, null T)
u_R0 = u(real RGB, null T)
u_0T = u(null RGB, real T)
u_RT = u(real RGB, real T)

RGB main effect     = u_R0 - u_00
thermal main effect = u_0T - u_00
interaction         = u_RT - u_R0 - u_0T + u_00
```

Possible `u` values:

- matched-object confidence;
- IoU;
- negative center error;
- TP indicator at a fixed operating point.

MR is a nonlinear dataset-level ranked functional. Do not substitute MR into this
per-instance equation and call the result a Shapley causal attribution.

### 10.4 Corruption curves

- [x] Thermal translation: symmetric `dx,dy` grid in pixels.
- [x] Thermal blur: several kernel levels.
- [x] Thermal contrast compression and additive-noise severity curves.
- [x] RGB brightness, blur, and grayscale degradation.
- [x] Deterministic complete-modality nulls at evaluation. Random dropout
  probabilities are represented by Block D training, not stochastic evaluation.
- [ ] Optional stale-frame offsets when sequence data is available.

Plot performance against corruption severity with confidence bands. Report:

```text
clean score
area under corruption curve
relative degradation
failure threshold
recovery from robust training
```

### 10.5 Failure taxonomy

At a fixed FPPI operating point classify every eligible object/frame:

- fusion rescue;
- RGB-only rescue;
- thermal-only rescue;
- fusion regression;
- false positive introduced by fusion;
- false positive removed by fusion;
- localization improved;
- localization worsened;
- all methods fail;
- all methods succeed.

- [ ] Define categories programmatically before viewing examples.
- [ ] Report category prevalence and sequence-level confidence intervals.
- [ ] Sample qualitative examples from these tables, not manually from the image
  directory.

### Phase 8 exit gate

- [ ] At least two distinct null constructions support each modality-use conclusion.
- [ ] Prediction matching is deterministic and audited.
- [ ] Causal claims come from interventions with uncertainty, not attention maps.
- [ ] Clean and corrupted behavior are reported separately.

## 11. Phase 9: Interpretability and Internal Diagnostics

Interpretability has three levels here:

1. **Behavioral:** what changes under input intervention?
2. **Representational:** what information and geometry exist in each branch?
3. **Attributional:** which spatial features influence a specific output?

Behavioral evidence is strongest for causal modality-use claims. Attribution maps are
supporting evidence.

### 11.1 Targeted detection attribution

For LayerCAM, Grad-CAM++, EigenCAM, or related methods:

- [x] Select an exact post-NMS detection or GT-matched pre-NMS candidate.
- [x] Define the scalar target: objectness, class score, IoU-aware score, or a box
  coordinate.
- [x] Preserve the candidate identity across interventions.
- [ ] Extract maps at RGB backbone levels, thermal backbone levels, fusion
  inputs/outputs, neck levels, and head.
- [x] Resize maps with a declared bilinear method and store RGB and thermal maps
  separately.
- [x] Never backpropagate an unspecified sum across every detection and interpret it
  as object-specific evidence.

### 11.2 Quantitative attribution evaluation

- [x] Inside-ground-truth-box attribution energy.
- [x] Pointing-game accuracy.
- [x] Deletion curve using most-attributed pixels/patches first.
- [x] Insertion curve.
- [x] Fixed-candidate confidence change.
- [x] Fixed-candidate IoU and center-error change across deletion/insertion.
- [x] Sanity check after randomizing the learned head and full model.
- [x] Sensitivity to explanation method through Grad-CAM/Integrated-Gradients
  agreement on the same target. Alternative target-score sensitivity remains
  deferred.

A method that produces similar maps after weight randomization is not evidence about
the trained detector.

### 11.3 Feature statistics

At dark3/dark4/dark5 and fusion outputs, log:

```text
activation mean/std
L1/L2 norm
sparsity
fraction near activation saturation
channel covariance
effective rank
top singular values
condition number
RGB-T cosine similarity
RGB-T CKA
fused-to-RGB CKA
fused-to-thermal CKA
```

Compute these on a fixed, stratified probe set containing Day/Night, small/large,
occluded/unoccluded, high/low RGB contrast, and high/low thermal contrast examples.

### 11.4 Weight and gate analysis

For concat projection `W[R;T]`, split:

```text
W = [W_R, W_T]
```

Track:

- `||W_R||` and `||W_T||`;
- ratio of modality block norms;
- singular spectra and effective rank;
- per-output-channel modality ratio;
- change from initialization;
- relation to input intervention effects.

For gates/attention:

- distribution, entropy, saturation, and spatial smoothness;
- Day/Night and contrast-stratified means;
- correlation with measured modality quality;
- response under shuffled/shifted inputs.

Do not infer importance from weight norm alone. Input magnitude and downstream
nonlinearity also matter.

### 11.5 Gradient analysis

Per step or fixed diagnostic minibatch:

```text
||grad RGB branch||
||grad thermal branch||
||grad fusion block||
cos(grad_RGB_on_shared, grad_T_on_shared)
gradient variance
update-to-weight ratio
```

- [x] Track top-level branch weight, gradient, and gradient-to-weight norms every
  20 steps, not only at epoch 40.
- [ ] Use the same diagnostic minibatch across models/seeds.
- [ ] Relate early gradient conflict to later branch contribution.

### 11.6 Frozen-feature probes

At initialization and selected epochs:

- [x] Freeze RGB, thermal, and fused features.
- [x] Train identical closed-form ridge probes for person/background information.
- [x] Probe sub-cell localization/center information.
- [x] Use the same train-cv/val-cv samples, ridge capacity, and fixed
  regularization without validation tuning.
- [ ] Compare probe scores with full detector and intervention results.

A fused feature with worse probe performance than both branches is evidence of
information destruction. A good probe with poor detector performance instead points
toward neck/head optimization or decoding.

### 11.7 Cross-method consistency table

For each conclusion, compare:

```text
input intervention
corruption curve
feature statistics
weight/gate analysis
gradient analysis
probe result
saliency attribution
```

Mark agreement, contradiction, or missing evidence. Do not suppress contradictions;
they often reveal that an explanation method is measuring sensitivity rather than
actual reliance.

### Phase 9 exit gate

- [ ] Interpretability analysis covers the same predeclared population as behavioral
  analysis.
- [x] Heatmaps have quantitative localization, faithfulness, completeness, and
  parameter-randomization sanity checks.
- [x] Any causal statement is supported primarily by intervention results.
- [ ] Internal diagnostics distinguish domination, collapse, conflict, and
  information destruction where possible.

## 12. Phase 10: Statistics and Reporting

### 12.1 Primary summaries

For every core arm report:

```text
five raw seed values
mean
sample standard deviation (ddof=1)
median
95% confidence interval
paired difference versus prespecified baseline
paired confidence interval
```

Do not report population standard deviation for a sample of training seeds.

### 12.2 Pairing

Use seed-paired differences when arms use the same seed/data-order policy:

```text
d_s = MR_fusion,s - MR_baseline,s
```

Lower MR is better, so negative `d_s` favors fusion. Report every `d_s`, its mean,
sample standard deviation, and confidence interval.

### 12.3 Dependence structure

- Seeds are independent training replicates.
- Frames within a video are not independent.
- Day/Night frames have different distributions.

Use sequence/video-level bootstrap for dataset uncertainty. Do not bootstrap
individual frames as though they were IID.

### 12.4 Multiple comparisons

- [ ] Label the primary baseline and primary metric before the matrix runs.
- [ ] Distinguish confirmatory comparisons from exploratory rankings.
- [ ] Report the number of operators/depths examined.
- [ ] Use multiplicity-aware intervals/tests for a large confirmatory family, or state
  explicitly that the comparison is exploratory.
- [ ] Avoid converting a noisy best-of-many result into a single unqualified claim.

### 12.5 Practical significance

The historical 1-2 point seed variability is not a universal noise constant.
Estimate uncertainty under the corrected protocol.

Predeclare:

- smallest MR improvement worth pursuing;
- maximum acceptable latency/memory increase;
- robustness degradation tolerance;
- whether Day or Night regression can be accepted for an All improvement.

### 12.6 Metric interpretation

- Lower log-average MR is better.
- All MR is computed from a globally ranked All curve.
- All MR is not a frame-count-weighted average of Day and Night MR.
- A confidence threshold used for display is not the benchmark operating point.
- Report complete curves or the nine points, not only one scalar.

### 12.7 Result table schema

```text
experiment_id
family
operator
levels
anchor
seed
params
FLOPs
latency
All MR
Day MR
Night MR
recall@0.1 FPPI
recall@1.0 FPPI
run status
artifact hash
```

### Phase 10 exit gate

- [ ] Raw numbers are available behind every aggregate.
- [ ] Paired and dataset-level uncertainty are reported correctly.
- [ ] Confirmatory and exploratory findings are labeled.
- [ ] Claims include effect size, uncertainty, and compute cost.

## 13. Phase 11: Literature Reproduction and Claim Matrix

For every paper, create a row containing:

```text
paper
original claim
original architecture/dataset/metric
PRISM implementation mapping
faithful elements
intentional deviations
reproduction acceptance criterion
comparison baseline
criterion for "beat"
artifact links
result
```

### 13.1 Required papers

- [ ] **Multispectral Deep Neural Networks for Pedestrian Detection, BMVC 2016.**
  Reproduce the early/halfway/late distinction with literal computational graphs.
  Test whether halfway remains best on YOLOX after capacity control.
  <https://bmva-archive.org.uk/bmvc/2016/papers/paper073/paper073.pdf>

- [ ] **MBNet, ECCV 2020.**
  Reproduce DMAF and examine the claim that modality imbalance hinders dual-stream
  optimization. Compare branch gradients, effective rank, probes, and interventions
  against concat.
  <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630766.pdf>

- [ ] **AR-CNN, ICCV 2019.**
  Reproduce controlled spatial shift degradation and an alignment-aware/reliability
  mechanism. Report accuracy against shift severity.
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.html>

- [ ] **GAFF, WACV 2021.**
  Implement the guided spatial/channel attention reference. Test whether learned
  masks correlate with causal modality interventions and quality strata.
  <https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf>

- [ ] **ProbEn, ECCV 2022.**
  Implement probabilistic score fusion and uncertainty-weighted box fusion rather
  than a generic NMS merge. Compare clean accuracy, missing-modality behavior,
  parameters, FLOPs, and end-to-end latency.
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1448_ECCV_2022_paper.php>

- [ ] **Causal Mode Multiplexer, CVPR 2024.**
  Test the claim that day/night-conditioned data create modality shortcuts. Reproduce
  its relevant strata or released evaluation where licensing and data permit.
  <https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Causal_Mode_Multiplexer_A_Novel_Framework_for_Unbiased_Multispectral_Pedestrian_CVPR_2024_paper.html>

- [ ] **A Closer Look at Multimodal Representation Collapse, ICML 2025.**
  Translate noisy-feature entanglement and rank-bottleneck predictions into
  convolutional RGB-T fusion. State which diagnostics transfer and which do not.
  <https://proceedings.mlr.press/v267/chaudhuri25a.html>

- [ ] **What Makes Training Multi-Modal Classification Networks Hard?, CVPR
  2020.** Test unequal overfitting and learning rates through raw branch
  trajectories and post-selection horizon saturation. Do not assume a
  classification remedy transfers without a detector experiment.
  <https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf>

- [ ] **Modality Competition, ICML 2022; Greedy Multimodal Learning, ICML
  2022; Unimodal Bias, ICML 2024.** Use branch trajectories, frozen probes, and
  modality-null interventions to distinguish failure to learn one predictive
  stream from ordinary overfitting or scale mismatch. Only then test auxiliary
  unimodal supervision or gradient modulation in a separately declared
  training-method block.
  <https://proceedings.mlr.press/v162/huang22e.html>
  <https://proceedings.mlr.press/v162/wu22d.html>
  <https://proceedings.mlr.press/v235/zhang24aa.html>

- [ ] **MS-DETR, 2023/2024.** Compare bounded cross-attention as a mechanism
  probe, then separately test training-only unimodal auxiliary supervision if
  the core winner exhibits imbalance. Do not call the local operator MS-DETR.
  <https://arxiv.org/abs/2302.00290>

- [ ] **Countering Multi-modal Representation Collapse through Rank-targeted
  Fusion, WACV 2026.** Use effective rank as a falsifiable diagnostic and later
  test rank-targeted regularization only if rank loss agrees with interventions.
  <https://openaccess.thecvf.com/content/WACV2026/html/Kim_Countering_Multi-modal_Representation_Collapse_through_Rank-targeted_Fusion_WACV_2026_paper.html>

- [ ] **Beyond Fusion, WACV 2024.** Quantify visible-branch regressions under
  darkness/blur before considering a hallucination branch.
  <https://openaccess.thecvf.com/content/WACV2024/papers/Xie_Beyond_Fusion_Modality_Hallucination-Based_Multispectral_Fusion_for_Pedestrian_Detection_WACV_2024_paper.pdf>

- [ ] **Learning a Robust RGB-Thermal Detector for Extreme Modality Imbalance,
  Pattern Recognition Letters 2025.**
  Reproduce pseudo-degradation and base/auxiliary consistency under controlled
  missing/degraded modalities.
  <https://arxiv.org/abs/2505.22154>

- [ ] Review **Does Thermal Really Always Matter for RGB-T Salient Object
  Detection?** for illumination-conditioned hypotheses, while clearly separating
  salient-object detection from pedestrian detection.
  <https://arxiv.org/abs/2210.04266>

### 13.2 Meaning of "beat"

Do not write "beats paper X" unless:

- the relevant claim is explicitly named;
- the protocol is comparable or deviations are disclosed;
- the same metric and data condition are used;
- uncertainty is reported;
- compute and robustness tradeoffs are included.

A higher clean-data score does not beat a robustness claim. A faster model does not
beat an accuracy claim. An attention visualization does not beat a causal analysis.

### Phase 11 exit gate

- [ ] Literature claims are converted into falsifiable experiments.
- [ ] Reproduction fidelity and deviations are documented.
- [ ] Every comparative claim has an explicit acceptance rule.

## 14. Phase 12: Population-Level Visual Evidence

The existing four-frame grid is a smoke visualization only:

- it selects the first two qualifying Day and Night frames;
- it uses a fixed confidence threshold of 0.30;
- title parentheses are detection counts, not ground-truth counts;
- fusion detections are displayed on thermal, hiding RGB evidence;
- it does not represent the dataset population or establish causality.

### Required visualization data

Build the per-frame/object analysis table first. Then select:

- deterministic random examples;
- common successes;
- fusion rescues;
- fusion regressions;
- modality-specific rescues;
- high-confidence false positives;
- localization improvements/regressions;
- small, occluded, Day, Night, misaligned, and low-contrast cases.

### Figure requirements

- [ ] Show RGB and thermal side by side.
- [ ] Show ground truth and predictions from RGB, thermal, and fusion.
- [ ] Include frame ID, sequence, GT count, TP/FP/FN, confidence, and operating point.
- [ ] Show the relevant intervention next to the paired-input prediction.
- [ ] Use a fixed-FPPI operating point derived from validation, not display threshold
  0.30.
- [ ] Use consistent colors and box conventions.
- [ ] State category prevalence so rare examples are not presented as typical.
- [ ] Include regressions and random examples, not only visually persuasive wins.

### Quantitative figures

- [ ] MR-FPPI curves with seed variability.
- [ ] Paired seed difference plots.
- [ ] Day/Night and scale/occlusion small multiples.
- [ ] Corruption severity curves.
- [ ] Failure taxonomy prevalence.
- [ ] Gate/gradient/rank trajectories.
- [ ] Attribution metrics, with example maps only as supporting panels.

### Phase 12 exit gate

- [ ] Every qualitative example is traceable to a population-level table.
- [ ] Figures use honest operating points and disclose selection.
- [ ] Visualizations support rather than substitute for quantitative evidence.

## 15. Phase 13: Reproducibility and Repository Cleanup

### 15.1 Required manifests

The corrected runner now implements this contract with durable lifecycle rows,
attempt-specific logs, atomic manifests/checkpoints, environment and input
hashes, and hashed per-frame validation outputs. The unchecked items below are
evidence gates: they become complete only after the remote runs have produced
and archived the corresponding artifacts.

Every reported run must store:

```text
experiment ID
command
Git SHA and dirty diff
config
model graph specification
dataset/split/annotation hashes
seed
environment
hardware
start/end timestamps
checkpoint hashes
evaluator SHA/config
raw metric files
per-image detections
failure status
```

### 15.2 Proposed artifact organization

```text
artifacts/
  manifests/
  splits/
  evaluator_parity/
  detections/
  metrics/
  diagnostics/
  interpretability/
  figures/
  tables/
```

Large binary files may live outside Git, but manifests and hashes must remain in the
repository.

### 15.3 Deferred documentation audit

After the architecture is fixed:

- [ ] Check training sample counts in README/docs.
- [x] Reconcile `data/README.md` with actual paths and annotations.
- [ ] Add every promised config.
- [x] Keep experiment tracking local and append-only; W&B is not a dependency.
- [ ] Report real hardware and timing protocol.
- [ ] Add parameter/FLOP/memory/latency methodology.
- [ ] Correct phase/paper-status claims.
- [ ] Remove references to results invalidated by test leakage or evaluator changes.

### Phase 13 exit gate

- [ ] A clean checkout plus documented data can regenerate each reported table.
- [ ] Every result cell maps to raw detections, metrics, run manifest, and checkpoint.
- [ ] Documentation describes the system that actually produced the results.

## 16. Official-Test Release Gate

Before the corrected official test is evaluated, freeze and timestamp:

- [ ] Research questions and hypotheses.
- [ ] Dataset and sequence split hashes.
- [ ] Evaluator commit and parity evidence.
- [ ] Training recipe.
- [ ] Architecture graph and operator.
- [ ] Seed list.
- [ ] Epoch/checkpoint rule.
- [ ] Primary and secondary metrics.
- [ ] Statistical analysis code.
- [ ] Causal intervention definitions.
- [ ] Figure/example selection rules.

Then:

1. Train the frozen finalists.
2. Confirm all run manifests and validation artifacts.
3. Export official-test detections once per frozen model/seed.
4. Score with the pinned canonical evaluator.
5. Run the predeclared statistical and interpretability analyses.
6. Report all results, including regressions, failed seeds, and contradictions.

Do not return to architecture selection after viewing test results. Any subsequent
model is a new study requiring a new held-out confirmation protocol.

## 17. Immediate Ordered Work Queue

### Gate A: recover and certify

1. [ ] Preserve all artifacts from the other machine.
2. [ ] Build the historical run ledger.
3. [ ] Clone and pin the MLPD evaluator bundle.
4. [ ] Reproduce all bundled reference detections.
5. [ ] Implement the PRISM reference-format detection exporter.
6. [ ] Build synthetic evaluator fixtures.
7. [ ] Differential-test released and PRISM detections.
8. [ ] Fix or retire the local evaluator.
9. [ ] Re-score historical detections with the canonical evaluator.

### Gate B: freeze the clean protocol

10. [ ] Audit dataset pairing and annotations.
11. [ ] Freeze sequence-disjoint train/validation lists.
12. [ ] Implement the official-test firewall.
13. [ ] Freeze the 40-epoch from-scratch recipe and five seeds.
14. [ ] Complete RGB and thermal sanity runs.

### Gate C: repair architecture definitions

15. [x] Write tensor-level graph specifications.
16. [x] Implement true one-shot dark3/dark4/dark5 fusion.
17. [x] Implement multi-scale fusion with explicit anchor-only fallback.
18. [x] Implement graph-level modality-dependence tests.
19. [x] Quarantine or rename the old `fuse_stages` experiments.

### Gate D: controlled empirical study

20. [ ] Run unimodal and duplicated-input capacity controls.
21. [ ] Run the concat-based fusion-family matrix.
22. [ ] Select one graph using validation.
23. [ ] Run the operator ladder on that graph.
24. [ ] Freeze the clean finalist.
25. [ ] Run corruption and robust-fusion experiments.

### Gate E: causal explanation and release

26. [ ] Run factorial modality interventions.
27. [ ] Build the failure taxonomy and population tables.
28. [ ] Run feature, weight, gate, gradient, probe, and attribution analyses.
29. [ ] Complete the literature claim matrix.
30. [ ] Freeze the official-test release package.
31. [ ] Evaluate the official test and produce the final report.

## 18. Definition of Done

The project is ready to make a defensible fusion claim only when:

- the benchmark evaluator is independently reproduced and pinned;
- test data played no role in model selection;
- every architecture name matches its actual graph;
- fusion is compared against unimodal, capacity, duplicated-input, early, late, and
  probabilistic controls;
- core conclusions reproduce across five seeds;
- effect sizes and uncertainty are reported from raw numbers;
- causal modality-use claims are supported by controlled interventions;
- interpretability methods pass quantitative sanity checks;
- robustness to corruption/misalignment is measured;
- literature comparisons reproduce the relevant claim rather than only citing it;
- every number and figure is traceable to immutable artifacts.
