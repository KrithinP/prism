# RGB-T Fusion Literature Review

Last audited: 2026-08-06.

This is the research basis for PRISM. It covers paired visible/thermal pedestrian
detection, general multimodal optimization, and interpretability methods that are
actually relevant to the question: why can a learned fusion model perform worse
than a model with less input?

Only primary papers, official proceedings pages, official project pages, and
released code are used for claims. A paper's absolute result is not treated as
comparable unless its train annotations, test annotations, image sampling,
reasonable-subset definition, detector, initialization, and evaluator agree.

## 1. Scope and terminology

KAIST calls its modalities `color` and `thermal`; this repository uses `RGB` and
`thermal`. KAIST thermal files are 8-bit grayscale JPEGs replicated to three
channels by the loader. They are not raw radiometric Y16 data.

The current study is deliberately narrower than the deployment scenario that
motivated it:

- paired KAIST frames only;
- pedestrian detection only;
- no 120/60 Hz temporal cache;
- no camera calibration or different-FOV rectification;
- no claim about airborne objects;
- models trained from scratch for a fixed 40 epochs.

The result is an architecture study, not a leaderboard reproduction or a fielded
asynchronous sensor system.

## 2. Dataset and protocol literature

### 2.1 Original KAIST benchmark

Hwang et al. introduced a vehicle-mounted color/thermal benchmark captured with
a beam splitter. The official page reports about 95k pairs, 20 Hz, 103,128
annotations, and day/night traffic scenes.

Primary sources:

- Hwang et al., *Multispectral Pedestrian Detection: Benchmark Dataset and
  Baselines*, CVPR 2015:
  <https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf>
- Official dataset and download page:
  <https://soonminhwang.github.io/rgbt-ped-detection/>

The official page describes 640x480 images. The released JPEGs and improved
annotation canvas used here are 640x512. PRISM records the decoded dimensions and
fails if a frame differs.

### 2.2 Annotation protocols are not interchangeable

At least three training conventions occur in RGB-T papers:

1. Original KAIST labels sampled at a fixed frame interval.
2. The 7,601-frame sanitized training set released with MSDS-RCNN.
3. The KAIST-Paired annotations released with AR-CNN, with separate boxes and
   pair relations across modalities.

MSDS-RCNN manually corrected localization, classification, missing labels, and
misaligned regions, then released the 7,601 valid sanitized training frames.
Its paper reports a large effect from annotation cleanup, which means annotation
choice can dominate an architecture ablation.

Primary source:

- Li et al., *Multispectral Pedestrian Detection via Simultaneous Detection and
  Segmentation*, BMVC 2018: <https://arxiv.org/abs/1808.04818>
- Official project: <https://li-chengyang.github.io/home/MSDS-RCNN/>

AR-CNN instead supplies modality-specific paired boxes to study weak alignment.
That protocol is valuable for alignment research, but it is not the same training
set as the MSDS sanitized 7,601-frame protocol.

Primary source:

- Zhang et al., *Weakly Aligned Cross-Modal Learning for Multispectral
  Pedestrian Detection*, ICCV 2019:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.html>

PRISM freezes:

- training: MSDS sanitized annotations, exactly 7,601 pairs;
- selection: a sequence-disjoint split of those 7,601 frames;
- final test: improved test annotations on exactly 2,252 `test-all-20` pairs;
- headline metric: Reasonable All log-average miss rate over FPPI 1e-2 to 1.

Consequences:

- PRISM may compare its own arms causally because they share one protocol.
- It may discuss whether a paper's mechanism transfers.
- It must not claim a numeric win over a paper trained with KAIST-Paired or a
  different frame list without a protocol-matched reproduction.

## 3. First-principles formulation

Let `R` be RGB, `T` thermal, and `Y` the pedestrian targets. For loss `L`, an
unrestricted predictor with both modalities cannot have worse optimal risk than
an unrestricted RGB-only predictor:

```text
inf_f E[L(f(R,T),Y)] <= inf_g E[L(g(R),Y)]
```

because the multimodal class can choose `f(R,T)=g(R)` and ignore `T`.

Observed degradation therefore does not prove that information itself is
harmful. It proves that at least one practical assumption failed:

- the fusion hypothesis class does not contain the unimodal solution;
- finite data lets the larger model overfit;
- one optimization schedule is inappropriate for branches that learn at
  different rates;
- gradients or shared neurons entangle predictive and noisy features;
- misregistration makes same-coordinate features describe different objects;
- normalization or projection destroys the stronger branch;
- the dataset rewards a day/night or warm-object shortcut;
- the fixed 40-epoch budget leaves the larger model under-optimized;
- data, decoding, NMS, or evaluator semantics differ.

PRISM's nested residual control tests the first item:

```text
Z = anchor + alpha * complement, alpha initialized to 0
```

At `alpha=0`, the fusion block is exactly the anchor feature function. This arm
is intentionally separate from the capacity-matched operator ranking.

## 4. Evolution of RGB-T fusion

### 4.1 Fusion placement

Liu et al. compared multiple two-stream Faster R-CNN fusion placements and found
middle or "halfway" fusion strongest in their setting. The durable contribution
is not a universal guarantee that one named layer is optimal; it is the
experimental separation of early, middle, and late interaction.

- Liu et al., *Multispectral Deep Neural Networks for Pedestrian Detection*,
  BMVC 2016:
  <https://www.bmva-archive.org.uk/bmvc/2016/papers/paper073/index.html>

PRISM translates this question into detector-independent graph definitions:

- early: input concatenation before the first convolution;
- one-shot: streams meet once at dark3, dark4, or dark5 and never split again;
- multi-scale: independently computed features are fused at a declared subset
  of dark3/dark4/dark5;
- late: independent detections are merged after both heads.

The local models are not faithful Faster R-CNN reproductions. They test the same
placement hypothesis on a YOLOX-S-derived detector.

### 4.2 Weak alignment and learned alignment

AR-CNN showed that nominally aligned pairs retain position shifts. It proposed
region feature alignment, feature reliability reweighting, RoI jitter, and
paired modality-specific annotations.

Implications for PRISM:

- include literal single-level and multi-level graphs;
- inject symmetric thermal shifts at training and evaluation;
- stratify degradation by shift direction and magnitude;
- compare high-resolution and coarse fusion;
- compare concat, learned offset warping, attention, and late fusion.

PRISM's alignment operator predicts a dense 2-D offset and warps one feature map
with `grid_sample`. It is AR-CNN-motivated, not an AR-CNN reproduction: it has no
region proposals or paired per-modality box regression.

### 4.3 Modality imbalance and quality-aware fusion

MBNet names two relevant problems: unequal modality contribution and
misalignment. It proposes Differential Modality Aware Fusion (DMAF) plus an
illumination-aware feature alignment component.

- Zhou et al., *Improving Multispectral Pedestrian Detection by Addressing
  Modality Imbalance Problems*, ECCV 2020:
  <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3128_ECCV_2020_paper.php>

PRISM includes:

- a simplified differential, channel-reweighting operator;
- a sample-adaptive scalar quality gate based on RGB/T feature statistics;
- per-frame gate dumps;
- RGB illumination and thermal contrast/noise interventions.

The local `dmaf` is explicitly "DMAF-inspired." It does not reproduce MBNet's
SSD detector, exact differential equations, or illumination-aware alignment.

### 4.4 Guided channel and spatial attention

GAFF learns inter- and intra-modality attention to dynamically reweight
multispectral features.

- Zhang et al., *Guided Attentive Feature Fusion for Multispectral Pedestrian
  Detection*, WACV 2021:
  <https://openaccess.thecvf.com/content/WACV2021/html/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.html>

PRISM's GAFF-inspired operator applies channel and spatial attention to
concatenated features. It is a controlled attention-family arm, not the complete
paper network. Its masks are logged as diagnostics, never treated as causal
importance without interventions.

### 4.5 Probabilistic decision fusion

ProbEn derives score fusion from Bayes' rule under conditional independence and
handles detections missing from one modality through marginalization.

- Chen et al., *Multimodal Object Detection via Probabilistic Ensembling*,
  ECCV 2022:
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1448_ECCV_2022_paper.php>
- Released code:
  <https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling>

Late fusion is essential because it asks how much benefit can be obtained
without mixing intermediate features. PRISM implements both merged NMS and the
released uniform-prior probability product. RGB and thermal checkpoints are
paired by seed.

### 4.6 Sparse cross-modal attention and auxiliary supervision

MS-DETR uses separate modality backbones/encoders, sparsely samples each
modality in a loosely coupled decoder, and retains modality branches for
instance-aware modality-balanced optimization.

- Xing et al., *MS-DETR: Multispectral Pedestrian Detection Transformer with
  Loosely Coupled Fusion and Modality-Balanced Optimization*:
  <https://arxiv.org/abs/2302.00290>

PRISM includes a bounded cross-attention ceiling: full-resolution queries but at
most 256 key/value tokens per level. This is not MS-DETR: it lacks deformable
sampling, object queries, decoder-slot matching, and auxiliary decoder losses.
The paper motivates two later experiments:

- training-only unimodal auxiliary detection supervision;
- per-instance, rather than only per-image, modality quality analysis.

These are not part of the first core matrix because they change the training
objective as well as the fusion operator.

### 4.7 Bias beyond ordinary day/night performance

Causal Mode Multiplexer argues that common multispectral datasets concentrate
on RGB-object/thermal-object day examples and RGB-no-object/thermal-object night
examples. Models can learn these correlations and generalize poorly to
counterfactual modality modes such as visible evidence without a thermal
signature.

- Kim et al., *Causal Mode Multiplexer*, CVPR 2024:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Causal_Mode_Multiplexer_A_Novel_Framework_for_Unbiased_Multispectral_Pedestrian_CVPR_2024_paper.html>

PRISM cannot reproduce the full CMM claim on ordinary KAIST validation alone.
It can:

- measure RGB/T null and degradation effects;
- cross illumination and thermal-contrast strata;
- report unusual disagreement cases;
- later evaluate released ROTX-MP under a separate protocol.

The distinction matters: a KAIST improvement does not by itself establish
unbiased cross-mode generalization.

### 4.8 Noisy visible features and hallucination

Beyond Fusion explicitly notes that a visible branch under poor visibility can
inject noise and proposes a hallucination-based alternative.

- Xie et al., *Beyond Fusion: Modality Hallucination-Based Multispectral Fusion
  for Pedestrian Detection*, WACV 2024:
  <https://openaccess.thecvf.com/content/WACV2024/papers/Xie_Beyond_Fusion_Modality_Hallucination-Based_Multispectral_Fusion_for_Pedestrian_Detection_WACV_2024_paper.pdf>

PRISM does not implement hallucination in the core matrix. The paper strengthens
the need to report cases where paired fusion regresses relative to thermal, and
to test RGB blur/darkness rather than assuming every paired input is beneficial.

## 5. General multimodal optimization literature

The detector literature describes the symptoms; general multimodal work helps
separate mechanisms.

### 5.1 Unequal overfitting and learning dynamics

Wang et al. show in multimodal classification that larger joint models can
overfit and modalities can generalize at different rates, making one shared
optimization strategy suboptimal. Their Gradient Blending is not directly
implemented here, but the diagnosis transfers.

- Wang et al., *What Makes Training Multi-Modal Classification Networks Hard?*,
  CVPR 2020:
  <https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf>

Required measurements:

- raw train and validation trajectories for every seed;
- branch-specific gradient norms and gradient-to-weight ratios;
- validation slope near epoch 40;
- longer-horizon confirmation only after architecture selection.

### 5.2 Greedy learning, competition, and unimodal bias

Several papers formalize a related but distinct failure: joint optimization can
learn the easiest modality first and then fail to learn predictive features from
the other stream. Huang et al. prove such modality competition for a simplified
late-fusion setting; Wu et al. characterize conditional learning speed; Zhang et
al. analyze how architecture and data statistics create unimodal bias in deep
linear networks.

- Huang et al., *Modality Competition: What Makes Joint Training of Multi-modal
  Network Fail in Deep Learning? (Provably)*, ICML 2022:
  <https://proceedings.mlr.press/v162/huang22e.html>
- Wu et al., *Characterizing and Overcoming the Greedy Nature of Learning in
  Multi-modal Deep Neural Networks*, ICML 2022:
  <https://proceedings.mlr.press/v162/wu22d.html>
- Zhang et al., *Understanding Unimodal Bias in Multimodal Deep Linear
  Networks*, ICML 2024:
  <https://proceedings.mlr.press/v235/zhang24aa.html>

These results do not prove that a convolutional YOLOX detector has the same
failure mechanism. They make the mechanism falsifiable here:

- one branch's gradients and frozen-feature probe improve early while the other
  remains near initialization;
- nulling the apparently ignored modality has little effect even though its
  unimodal detector is predictive;
- this asymmetry is consistent across seeds and is not explained by input scale
  or a saturated gate.

The fixed 40-epoch architecture matrix measures the symptom without changing
the optimizer by arm. Branch-specific optimization, auxiliary unimodal heads,
or gradient modulation are follow-up interventions only if these diagnostics
support competition; adding them to the discovery matrix now would confound
architecture with training method.

### 5.3 Representation and modality collapse

Chaudhuri et al. analyze modality collapse as noisy features from one modality
sharing fusion neurons with predictive features from another, masking useful
contributions and creating a rank bottleneck. They motivate cross-modal
distillation and basis reallocation.

- Chaudhuri et al., *A Closer Look at Multimodal Representation Collapse*,
  ICML 2025: <https://proceedings.mlr.press/v267/chaudhuri25a.html>

Kim et al. connect feature and modality collapse to effective-rank diagnostics
and rank-targeted fusion in a different application domain.

- Kim et al., *Countering Multi-modal Representation Collapse through
  Rank-targeted Fusion*, WACV 2026:
  <https://openaccess.thecvf.com/content/WACV2026/html/Kim_Countering_Multi-modal_Representation_Collapse_through_Rank-targeted_Fusion_WACV_2026_paper.html>

Transferable diagnostics:

- effective rank and singular spectra of RGB, thermal, and fused features;
- CKA between branches;
- fused-to-branch similarity;
- modality null effects;
- frozen feature probes.

These metrics are descriptive. A lower rank is not automatically harmful, and a
high RGB/T CKA is not automatically desirable. They become evidence only when
linked to prediction behavior.

## 6. Interpretability literature

### 6.1 Attribution methods

Grad-CAM uses target gradients flowing into convolutional feature maps to produce
a coarse target-specific localization map.

- Selvaraju et al., ICCV 2017:
  <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>

Integrated Gradients defines an attribution path from a reference input and is
motivated by sensitivity and implementation-invariance axioms.

- Sundararajan et al., ICML 2017:
  <https://proceedings.mlr.press/v70/sundararajan17a.html>

For a detector, the scalar target must be named: a particular pre-NMS candidate's
objectness, class score, box coordinate, or a fixed GT-matched score. Summing all
detections and drawing one map is not object-specific evidence.

### 6.2 Why attractive maps are insufficient

Adebayo et al. show that some saliency methods can be insensitive to model
parameters or the data-generating process; weight and label randomization are
required sanity checks.

- *Sanity Checks for Saliency Maps*, NeurIPS 2018:
  <https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html>

Jain and Wallace show that attention weights need not track other importance
measures and substantially different attention distributions can yield similar
predictions.

- *Attention is not Explanation*, NAACL 2019:
  <https://aclanthology.org/N19-1357/>

Therefore:

- gates and masks are internal states, not causal explanations;
- heatmaps require randomization, deletion/insertion, and target sensitivity;
- modality reliance is established primarily by controlled input interventions.

### 6.3 Representation similarity

Centered Kernel Alignment compares representation geometry across samples and is
more suitable than transformation-invariant CCA when feature dimension exceeds
the number of examples.

- Kornblith et al., *Similarity of Neural Network Representations Revisited*,
  ICML 2019: <https://proceedings.mlr.press/v97/kornblith19a>

`scripts/analyze_representations.py` stores every pooled feature vector used for
CKA in NPZ form and per-frame norms/gates in JSON before any summary.

`scripts/analyze_attributions.py` targets one GT-matched pre-NMS candidate,
compares train-mean Integrated Gradients with branch Grad-CAM, quantifies
inside-box/pointing/deletion/insertion behavior, and performs learned-head and
full-model parameter-randomization checks. These diagnostics corroborate, but
do not replace, fixed-weight modality interventions.

`scripts/probe_representations.py` fits identical closed-form probes on frozen
train-cv branch/fused features and evaluates them on val-cv. It tests whether
pedestrian and center information remains linearly accessible when the detector
metric degrades.

## 7. Paper-to-experiment traceability

| Literature claim | PRISM test | Acceptance boundary |
|---|---|---|
| Middle placement can beat early/late | B1-B12 | Five seeds; same validation protocol; compute disclosed |
| Extra branches alone can look beneficial | A3-A6 | Paired RGB-T must exceed width and duplicate-input controls |
| Fusion should be able to ignore a bad modality | A7-A8 | Nested coefficient starts at exactly zero; paired result compared to its anchor |
| Modality quality varies by image | C3 vs C10 | Sample-adaptive gate must beat global scalar under clean and degraded conditions |
| Differential fusion addresses imbalance | C7 | DMAF-inspired arm plus branch/rank/intervention diagnostics |
| Attention improves selective fusion | C6/C9 | Accuracy, latency, masks, and intervention agreement; masks alone do not pass |
| A larger fusion operator can masquerade as a better mechanism | C9 vs C11; C8 vs C12 | Exact added-parameter matches plus measured MACs, memory, and latency |
| Misalignment hurts local fusion | C8, D2-D3, shift sweep | Symmetric severity curve and comparison to coarse/late fusion |
| Missing modalities require robustness training | D1/D3 | Clean MR and null-modality MR both reported |
| Late probabilistic fusion is a strong baseline | B11-B12 | Paired seed checkpoints and released ProbEn score equation |
| Fusion can learn modality shortcuts | intervention/strata suite | Behavioral changes on counterfactual conditions; no claim from ordinary Day/Night alone |
| Joint training can neglect one predictive modality | trajectories + probes + null interventions | Persistent cross-seed asymmetry; ignored branch is useful unimodally; input-scale and gate-saturation alternatives rejected |
| Noisy-feature entanglement causes collapse | representation suite | Rank/CKA/probes agree with behavior; otherwise inconclusive |

## 8. What is implemented faithfully

Faithful or literal:

- early, one-shot, multi-scale, and late graph definitions;
- released uniform-prior ProbEn score product;
- current pinned YOLOX-S graph/head/loss/SimOTA semantics for `base_ch=32`;
- evaluator semantics differentially matched to the pinned MLPD release.

Controlled simplifications:

- `gaff`: GAFF/CBAM-family channel and spatial attention;
- `dmaf`: differential-statistic channel reweighting;
- `alignment`: dense feature translation, not region alignment;
- `cross-attention`: bounded convolutional feature attention, not MS-DETR;
- `quality-gated`: sample-adaptive quality proxy, not MBNet illumination alignment.

Not yet implemented and not part of core claims:

- exact paper reproduction of GAFF, DMAF/MBNet, AR-CNN, or MS-DETR;
- CMM mode-specific training or ROTX-MP evaluation;
- unimodal auxiliary detection heads or cross-modal distillation;
- label-randomization retraining and exact optimizer-update diagnostics;
- temporal/asynchronous fusion.

## 9. Claim rules

PRISM may claim an architectural effect only when:

1. The five raw seed values and failure states are available.
2. The candidate beats both unimodal baselines or is explicitly called a null or
   negative result.
3. Capacity and duplicated-input controls do not explain the effect.
4. Selection occurred only on the frozen sequence-disjoint validation set.
5. Dataset/annotation/evaluator hashes are fixed.
6. Compute and robustness tradeoffs are reported.

PRISM may claim model-level causal reliance only when:

1. A named input is intervened on while model weights and other inputs are fixed.
2. At least two null constructions agree, such as train-mean and zero replacement.
3. Prediction identity is controlled for object-level analyses.
4. Uncertainty is reported at the sequence and training-seed levels.
5. Gates, attention, saliency, CKA, and rank are used as corroborating diagnostics,
   not substitutes for the intervention.

It may not generalize those model-behavior effects into a claim that a physical
sensor is intrinsically harmful or unnecessary.
