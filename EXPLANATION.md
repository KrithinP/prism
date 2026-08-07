# PRISM: Architecture and Experimental Design

This document describes what the repository implements, why each experiment
exists, and the exact order in which a clean machine should run the study.
`docs/LITERATURE.md` contains the source review behind these decisions.

## 1. Research question

The controlled question is:

> Under one detector family, data protocol, training budget, and evaluator,
> which RGB-thermal interaction graph and fusion operator use complementary
> information without corrupting or collapsing either modality?

This is not a best-possible KAIST leaderboard experiment. Every discovery run:

- starts from random initialization;
- trains for exactly 40 epochs;
- uses the same five seeds: 0, 1, 2, 3, 4;
- uses a frozen train/validation split;
- keeps the official test set inaccessible during model selection.

The 40-epoch horizon is an experimental control, not a claim of convergence.
Initial experiments fix secondary choices such as epoch count so major
architectural decisions can be compared cleanly. Training duration and other
optimization choices can be saturated after the fusion design is fixed.

## 2. What the earlier results mean

The reported 16-epoch test-set results for RGB, thermal, and `F-cat-345` are
historical exploratory evidence only. They cannot support a final claim because:

- the test set was used during architecture work;
- the old evaluator had not yet been differentially certified;
- the old `fuse_stages` names did not describe literal one-shot fusion graphs;
- the training horizon differs from the now-frozen 40 epochs;
- ignore regions were previously dropped during training;
- the local YOLOX loss/assignment and BatchNorm details differed from upstream.

They are useful as a failure report that motivated this study. They are not
carried into the corrected result table.

## 3. What "provisional evaluator" meant

An evaluator is provisional when its output resembles the benchmark metric but
has not demonstrated semantic equivalence to the released implementation.
Tiny choices change KAIST MR:

- whether boxes outside the image are clipped;
- whether boundary-straddling GT is clipped or ignored;
- `>` versus `>=` at IoU 0.5;
- stable ordering of equal-score detections;
- which qualifying ignore region absorbs a detection;
- frame-ID base and exact FPPI reference points.

PRISM's canonical evaluator was repaired against the pinned MLPD evaluator. The
differential harness covers the five released detection files and the bundle's
published table. `scripts/parity_check.py` recreates a fail-closed certificate
and stores commands, stdout, stderr, hashes, and per-file deltas. The other
machine must still preserve that generated certificate with the run artifacts.

## 4. Data definition

### 4.1 Files

The study uses the official KAIST image archive:

```text
Google Drive ID: 1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc
MD5 file ID:     1KQ9lZXX9nfZuhbqBUEdlU2Yjx3nzeg5c
```

Training labels are the MSDS-RCNN sanitized release:

```text
Google Drive ID: 1mKIzND2046aXHknzr28LJHvFH_6Ssa_q
frames:          7,601
```

Final test labels are the improved annotations pinned with the MLPD evaluator:

```text
test-all-20:   2,252
test-day-20:   1,455
test-night-20:   797
```

### 4.2 Images

The released files decoded by this pipeline are 640x512. RGB is decoded BGR to
RGB. Thermal is an 8-bit grayscale JPEG decoded into three equal channels. Both
are divided by 255 before the network.

### 4.3 Labels

The detector has one positive class:

```text
person -> target class 0
```

Ambiguous/group/cyclist labels and explicit bbGt ignore flags become:

```text
people, cyclist, person?, person?a, ignore=1 -> target class -1
```

Class `-1` boxes never become positives. Anchors centered inside these boxes are
removed from negative objectness supervision. At evaluation, detections absorbed
by ignore regions are neither TP nor FP under the reference rules.

### 4.4 Split

The 7,601 training frames are split at complete `setXX/VYYY` sequence
granularity. No video can occur in both `train-cv` and `val-cv`. Random
frame-level splitting is prohibited because adjacent video frames share
backgrounds and pedestrian instances.

The official 2,252-frame test set is not part of this split and is not reachable
from `prism.train`.

## 5. Base detector

### 5.1 Accurate name

The base is:

> a YOLOX-S-derived graph under a PRISM-specific fixed training recipe.

It is not an exact reproduction of the official YOLOX-S training procedure.
The audit is pinned in `external/yolox_reference.json`.

### 5.2 Graph

For `base_ch=32` and `base_depth=1`, the unimodal graph matches the YOLOX-S
topology:

```text
input 3x512x640
  -> Focus stem
  -> CSPDarknet
       dark3: 128x64x80, stride 8
       dark4: 256x32x40, stride 16
       dark5: 512x16x20, stride 32
  -> YOLO PAFPN, top-down then bottom-up
  -> decoupled 128-channel head at all three scales
  -> [box 4, objectness 1, class 1] per cell
```

The head is anchor-free. Box centers and sizes decode as:

```text
xy = (raw_xy + grid) * stride
wh = exp(raw_wh) * stride
```

Objectness and class probabilities are multiplied at inference.

### 5.3 Upstream semantics retained

The audited base includes:

- Focus/CSPDarknet/PAFPN topology;
- YOLOX-style decoupled head;
- class and objectness prior probability 0.01;
- BatchNorm `eps=1e-3`, `momentum=0.03`;
- IoU loss `1 - IoU^2`;
- IoU-weighted classification targets;
- current center-only SimOTA candidate geometry with radius 1.5;
- dynamic-k matching;
- NMS IoU 0.65.

### 5.4 Intentional recipe deviations

PRISM uses:

- one class;
- native 640x512 inputs;
- 40 epochs, not 300;
- random initialization, not pretrained weights;
- paired horizontal flip and RGB-only brightness/contrast jitter;
- no Mosaic, MixUp, random resize, EMA, or final L1/no-augmentation phase;
- SGD with Nesterov, warmup then cosine;
- gradient clipping;
- KAIST ignore-region masking;
- confidence floor 0.001 for the MR-FPPI sweep.

These choices are identical across comparable arms. The project can determine
which graph optimizes best under this fixed budget. It cannot conclude that a
losing graph would also lose after separately tuned or fully saturated training.

## 6. Fusion formulation

Let:

```text
R_l = E_r^l(RGB)
T_l = E_t^l(Thermal)
F_l = fusion operator at level l
N   = shared PAFPN neck
H   = shared YOLOX head
A_l = single-modality adapter
```

The two early backbones are independent because RGB texture and thermal
intensity statistics differ.

### 6.1 Unimodal

```text
y_rgb = H(N(E_r(RGB)))
y_t   = H(N(E_t(Thermal)))
```

### 6.2 Early input fusion

```text
x = concat(RGB, Thermal)       # six channels
y = H(N(E_early(x)))
```

There is one stream from the first convolution onward. It is inexpensive, but
assumes strong pixel correspondence and forces one stem to serve both sensor
distributions.

### 6.3 One-shot middle fusion

For a meeting level `L`:

```text
Z_L     = F_L(R_L, T_L)
Z_{L+1} = S_{L+1}(Z_L)
...
Z_5     = S_5(Z_4)
y       = H(N(P3,P4,P5))
```

The private streams end at `L`; there is exactly one shared suffix. Neck skips
below `L` come only from the declared anchor:

```text
thermal anchor, L=4: [A_3(T_3), Z_4, Z_5]
RGB anchor, L=4:     [A_3(R_3), Z_4, Z_5]
```

This is materially different from independently fusing several backbone
outputs.

### 6.4 Multi-scale fusion

For selected level set `S`:

```text
Z_l = F_l(R_l,T_l), l in S
Z_l = A_l(anchor_l), l not in S
y   = H(N(Z_3,Z_4,Z_5))
```

Both complete backbones remain independent. Unselected levels never silently
sum the modalities.

### 6.5 Late fusion

```text
D_r = detector_rgb(RGB)
D_t = detector_t(Thermal)
D   = merge(D_r,D_t)
```

PRISM tests concatenated detections followed by NMS and ProbEn. Late fusion
costs two full detector passes but does not expose intermediate features to
cross-modal corruption.

## 7. Fusion operators

Block C compares operators only after Block B freezes a graph.

Every normal Block C operator returns `C` channels and passes through the same
YOLOX-style BatchNorm and SiLU. Projection-based operators no longer receive an
extra normalization advantage.

### C1: average

```text
Z = (R + T) / 2
```

Zero learned mixing parameters before common normalization.

### C2: concat projection

```text
Z = Conv1x1(concat(R,T))
```

The primary learned linear-mixing baseline.

### C3: global scalar

```text
alpha = sigmoid(a)
Z = alpha*R + (1-alpha)*T
```

`alpha` is shared by every image and starts at 0.5.

### C4: channel mixture

```text
alpha_c = sigmoid(MLP(GAP(concat(R,T))))
Z = alpha_c*R + (1-alpha_c)*T
```

Trust can change by image and channel.

### C5: anchored gated residual

```text
g = sigmoid(G(concat(R,T)))
Z = anchor + g*complement
```

The graph's declared anchor controls the residual path.

### C6: GAFF-inspired attention

Channel attention and spatial attention reweight the concatenated feature before
a 1x1 projection. It is a family-level controlled operator, not a complete GAFF
reimplementation.

### C7: DMAF-inspired differential fusion

The absolute RGB/T feature difference is globally pooled and converted to
channel weights for the concatenated feature. It tests the differential-signal
hypothesis, not exact MBNet fidelity.

### C8: learned alignment

A small convolution predicts a 2-D offset field. The complement is bilinearly
warped relative to the declared anchor and then projected with the anchor.
Offsets receive gradient and are logged.

### C9: bounded cross-attention

Queries retain every spatial location. Keys and values are adaptively pooled to
at most 256 tokens per level. The declared anchor supplies queries and the
complement supplies keys/values.

### C10: sample-adaptive quality gate

Global mean and standard deviation from both features predict one trust
coefficient per image. This isolates sample-adaptive trust from the global
scalar and higher-capacity channel gates.

### C11/C12: expensive-operator capacity controls

C11 is a nonlinear concat-plus-1x1 network with exactly `4*C^2` operator
parameters, matching C9 cross-attention. C12 is a concat-plus-spatial-convolution
network with exactly `20*C^2 + 4*C + 2` parameters, matching C8 learned
alignment including its normalization and offset biases. Dynamic MACs and
latency are still reported because equal parameters do not imply equal compute.

### Separate nestedness control

Block A also includes:

```text
Z = anchor + alpha*complement
alpha initialized exactly to 0
```

It bypasses common post-normalization so that `alpha=0` is exactly the anchor
feature at initialization. The architecture's hypothesis class therefore
contains its anchor branch. This does **not** guarantee that joint SGD preserves
the independently trained unimodal detector or its metric. It is not ranked as
a Block C operator.

## 8. Experiment matrix

### Block A: baselines and controls

| ID | Model | Question |
|---|---|---|
| A1 | RGB YOLOX | Visible baseline |
| A2 | Thermal YOLOX | Thermal baseline |
| A3 | Wider RGB | Can capacity explain a fusion gain? |
| A4 | Wider thermal | Can capacity explain a fusion gain? |
| A5 | RGB+RGB dual branch | Does branch count alone help? |
| A6 | T+T dual branch | Does branch count alone help? |
| A7 | Paired nested residual, thermal anchor | Does a nested initialization protect thermal? |
| A8 | Paired nested residual, RGB anchor | Does a nested initialization protect RGB? |

### Block B: fusion graph

Concat is fixed while graph placement changes:

```text
B1      early
B2-B4   one-shot dark3/dark4/dark5
B5-B10  multi-scale {3},{4},{5},{3,4},{4,5},{3,4,5}
B11     late union-NMS
B12     late ProbEn
B13     selected learned graph with RGB anchor
```

The Block C template is the minimum five-seed mean validation Reasonable All MR
among feature-fusion graphs B2-B10. B1 is still a required early-fusion
comparator, but has no intermediate graph on which Block C or B13 can operate.

### Block C: operator

The selected Block B feature graph is fixed. C1-C10 test mechanisms; C11
matches C9's parameter count with a concat network; C12 matches C8's parameter
count with a spatial concat network. All twelve run for five seeds. The winner
is selected by the same predeclared validation criterion.

### Block D: robustness training

The selected Block C architecture is retrained:

```text
D1 modality dropout p=0.2
D2 random thermal translation up to 8 input pixels
D3 both
```

These answer a robustness question and cannot retroactively replace the clean
architecture winner.

## 9. Evaluation and statistics

### 9.1 Metric

Reasonable retains pedestrians at least 55 pixels high with no or partial
occlusion. Detections are ranked globally. Miss rate is sampled at nine
log-spaced FPPI references from 1e-2 to 1 and geometrically averaged.

Lower is better.

All MR is not a weighted average of Day and Night MR. Validation reports All
only because the frozen CV split does not pretend to have official test subset
membership. Official Day/Night values are produced only on the official lists.

### 9.2 Seed uncertainty

The raw five seed values are primary. `scripts/summarize_results.py` reports:

- mean and sample standard deviation;
- two-sided Student-t 95% interval;
- raw paired seed differences against RGB and thermal;
- whether both upper interval bounds are below zero.

Five seeds provide limited inferential power. Rankings remain exploratory unless
confirmed.

### 9.3 Dataset uncertainty

Frames in one video are dependent. `scripts/bootstrap_comparison.py` resamples
whole `setXX/VYYY` sequences, uses identical draws for candidate and baseline,
and stores every replicate. It does not bootstrap individual frames.

Training-seed and dataset-sequence uncertainty answer different questions and
are reported separately.

### 9.4 Compute

`scripts/profile_checkpoint.py` reports:

- total/trainable parameters;
- dynamic Conv2d, Linear, and cross-attention MACs;
- FLOPs using the explicit two-FLOP-per-MAC convention;
- every raw latency sample, then mean/std/median/p90/range;
- CUDA peak allocated memory.

Latency comparisons require the same hardware, software, precision, input, and
warmup/iteration settings.

## 10. Causal and interpretability analysis

### 10.1 Behavioral interventions

`scripts/evaluate_interventions.py` keeps weights fixed and evaluates:

- paired inputs;
- train-mean and zero replacement for each modality;
- both modalities at train mean or zero;
- RGB brightness at 0.75/0.50/0.25, blur at 5/9/15, and grayscale;
- thermal contrast at 0.75/0.50/0.25, blur at 5/9/15, and noise at
  0.05/0.10/0.20 of the 8-bit range;
- symmetric thermal shifts of 4/8/16 pixels in x and y.

It stores every per-frame detection and GT row in hashed NPZ artifacts before
reporting MR change versus paired input, along with raw frame luminance,
contrast, and target-size covariates for stratification. These are causal effects
of a defined input intervention on this fixed model's behavior. They do not
prove that a physical sensor is causally harmful in the world.

MR is a nonlinear ranked dataset functional. The script intentionally does not
substitute MR into an additive Shapley equation.

### 10.2 Representations and gates

`scripts/analyze_representations.py` writes:

- raw per-frame/level RGB, thermal, and fused mean/std/RMS/sparsity/kurtosis;
- RGB/T pooled-feature cosine;
- raw gate, mask, offset, or attention summaries;
- every pooled vector in NPZ;
- derived RGB-T/fused-to-branch linear CKA, singular spectra, effective and
  numerical rank, conditioning, and operator parameter-block norms.

The raw records precede the summaries. CKA, rank, and gate values are descriptive
and must agree with behavioral interventions before supporting a modality-use
story.

`scripts/probe_representations.py` freezes the detector, samples RGB, thermal,
and fused vectors at GT centers plus deterministic background cells, fits the
same closed-form ridge probes on train-cv, and evaluates on val-cv. It reports
person/background AUC and balanced accuracy plus sub-cell center RMSE. Raw
features, labels, IDs, scores, and center predictions are stored first. A strong
branch probe with a weak detector points toward downstream use/optimization; a
weak fused probe relative to both branches is consistent with information loss.

### 10.3 Targeted attribution

`scripts/analyze_attributions.py` uses the first predeclared validation frames
with a scored pedestrian and a clean-pair candidate at IoU >= 0.5. The target is
that fixed pre-NMS candidate's `sigmoid(objectness) * sigmoid(class)` score.

It writes:

- train-mean Integrated Gradients maps for RGB and thermal;
- branch Grad-CAM at the deepest level that actually participates in fusion;
- inside-GT attribution mass and pointing hits;
- fixed-candidate deletion and insertion curves for score, decoded-box IoU, and
  center error;
- Integrated-Gradients completeness error;
- cross-method map correlation;
- learned-head and full-model parameter-randomization checks;
- raw maps in NPZ before any aggregate.

This is quantitative supporting evidence. It does not turn saliency into a
causal claim, and it does not include a label-randomization retraining study.
Behavioral interventions remain primary.

## 11. Clean-machine execution

Run from repository root.

### 11.1 Environment

```sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
```

The configured PyTorch wheels target CUDA 12.8. Confirm the installed wheel
supports the target GPU before training.

### 11.2 Prepare and certify

```sh
make prepare-kaist
make certify-evaluator
make test
make preflight
make data-stats
```

`prepare-kaist` downloads, checks MD5, extracts, pins the evaluator, converts
labels, makes sequence-disjoint splits, audits every frame, and writes a data
manifest. `preflight` fails on missing pairs, wrong counts, stale hashes,
protocol drift, failed parity, or absent CUDA.

### 11.3 Run discovery

```sh
make study-discovery
```

This runs A1-A8, B1-B10, B11-B12, freezes the best eligible feature-fusion
graph from B2-B10, and runs B13.
The runner is append-only and reuses only checkpoints whose seed, epoch, full
recipe hash, protocol version, and executable-source fingerprint match. Each
run saves checkpoints and validation MR every 10 epochs and raw loss/LR/branch
gradient-to-weight diagnostics every 20 steps.

### 11.4 Run operators and robustness

```sh
make study-operators
make study-robustness
make summary
```

Block C consumes `outputs/selection/block_b_winner.json`. Block D consumes
`outputs/selection/block_c_winner.json`.

### 11.5 Run the predeclared diagnostics

```sh
make study-diagnostics
```

`scripts/run_posthoc_studies.py` resolves exact checkpoints from the ledger. It
profiles every learned graph at seed 0; runs five-seed interventions and
representation analysis on A7/A8, concat, the selected operator, and D1-D3;
runs five-seed frozen probes and attribution for concat and the selected
operator; and bootstraps the selected operator against A1 and A2 for each paired
seed. Existing outputs are reusable only when their embedded checkpoint hashes
match.

### 11.6 Freeze and evaluate official test

Only after validation selection and the analysis plan are frozen:

```sh
PYTHONPATH= .venv/bin/python scripts/freeze_release.py \
  --experiment final_fusion_seed0 \
  --checkpoint model="$WINNER_CKPT" \
  --selection-report outputs/selection/block_c_winner.json \
  --reason 'predeclared final fusion model' \
  --out outputs/releases/final_fusion_seed0.json

PYTHONPATH= .venv/bin/python -m prism.evaluate_model \
  --ckpt "$WINNER_CKPT" \
  --root data/raw/kaist-cvpr15 \
  --split test --official-test \
  --manifest outputs/releases/final_fusion_seed0.json \
  --export outputs/official/final_fusion_seed0.txt \
  --json-out outputs/official/final_fusion_seed0.local.json

PYTHONPATH= .venv/bin/python scripts/parity_check.py \
  --det-txts outputs/official/final_fusion_seed0.txt \
  --out-dir outputs/official/parity_final_fusion_seed0
```

The last command records the authoritative pinned-evaluator score for the
exported detections. Viewing test results must not trigger another architecture
selection cycle.

## 12. Interpretation of possible outcomes

### Fusion loses and A7/A8 remain near their anchor

The flexible mixing operator or optimization is the likely problem, not an
inevitable information penalty. Check gate saturation, feature scale, gradient
conflict, and shift sensitivity.

### Fusion loses and nested controls also lose

The dual-branch training dynamics, fixed optimization horizon, or data pipeline
may prevent preservation of the unimodal solution. Compare trajectories and
later run a predeclared longer-horizon confirmation.

### Fusion beats thermal but not RGB, or vice versa

This is not a general fusion win. Report the failing baseline and relevant
illumination/quality strata.

### Clean fusion wins but collapses under modality nulls

The model exploits complementarity but is not robust to sensor failure. D1/D3
address that different objective; report any clean-performance tradeoff.

### Gates favor one modality and nulling it has little effect

The gate is not a valid importance explanation. Prefer behavioral evidence.

### Rank or CKA changes without prediction changes

The representation metric is descriptive but not task-causal in that case.

## 13. Remaining boundaries

After the architecture study, separate projects may add:

- saturation of epochs and augmentation;
- faithful MBNet, GAFF, AR-CNN, MS-DETR, or CMM reproduction;
- auxiliary unimodal heads, cross-modal distillation, or rank regularization;
- paired-annotation alignment experiments;
- temporal 120/60 Hz feature caching and age embeddings;
- different-FOV geometry;
- real Y16 thermal preprocessing.

None should be mixed retrospectively into the frozen 40-epoch matrix.
