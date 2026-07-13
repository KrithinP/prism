# RGB-Thermal Pedestrian Detection with Middle Fusion — Literature Review

Scope: everything you need to know before building a two-stream YOLOX with halfway fusion for the
KAIST benchmark, deployable on a Jetson Orin NX. Every citation below was verified against the
paper, its arXiv page, its official repo, or its publisher page (July 2026). Where a number could
not be re-verified from a primary source it is marked as such.

---

## 1. The Benchmark: KAIST Multispectral Pedestrian Detection

**Paper:** Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, In So Kweon, *"Multispectral
Pedestrian Detection: Benchmark Dataset and Baseline"*, **CVPR 2015**.
Site: https://soonminhwang.github.io/rgbt-ped-detection/ · Code/data:
https://github.com/SoonminHwang/rgbt-ped-detection

### 1.1 How it was captured — and why that matters

The rig is a color camera + FLIR thermal camera behind a **beam splitter**, mounted on a car.
Both cameras see the *same optical axis*, so image pairs are **geometrically registered by
hardware** (up to calibration residual and the resolution mismatch: RGB 640×480, thermal
320×256 upscaled to 640×480). Capture is hardware-synced at **20 Hz**, day and night, in
campus/road/downtown scenes.

This is the single most important fact about KAIST: **it is an aligned-modality dataset.** Any
fusion operator you validate on KAIST is being validated under near-zero parallax and near-zero
temporal skew. A real rig with two separate lenses does not give you this (see §4).

### 1.2 Size and splits

- 95,328 RGB-T pairs total; 103,128 dense annotations; 1,182 unique pedestrians. Classes:
  `person`, `people` (groups), `cyclist`; occlusion tagged as none / partial / heavy.
- Video sets `set00–05` = train (50,172 frames), `set06–11` = test (45,156 frames).
- Nobody trains/tests on every frame (20 Hz video is massively redundant). The de-facto standard:
  - **Test: every 20th frame of set06–11 → 2,252 images** (≈1,455 day / 797 night). This is the
    universal test set; the day/night breakdown gives the Day/Night columns in every paper.
  - **Train:** historically every 2nd frame (~25k images); the modern standard is the
    **sanitized train set of 7,601 images** (see below). ProbEn (ECCV 2022) states this
    7,601/2,252 protocol explicitly, and MLPD, TFDet, MS-DETR all follow it.

### 1.3 The annotation history — get this right or your numbers are meaningless

The original annotations are noisy: missing boxes, mislabeled boxes, boxes that fit neither
modality. Three cleanups exist, and the field settled on a specific combination:

1. **Improved *test* annotations — Liu et al. 2018.** Jingjing Liu, Shaoting Zhang, Shu Wang,
   Dimitris Metaxas (the Halfway Fusion authors) re-annotated the 2,252-image test set
   ("Improved annotations of test set of KAIST", 2018). Distributed from the Halfway-Fusion
   project (github.com/denny1108/multispectral-pedestrian-py-faster-rcnn; also mirrored in the
   resource list below). Switching from original to improved test annotations changes MR by
   **>15 points** for the same detector — this is the #1 source of incomparable numbers.
2. **Sanitized *train* annotations — Li et al., BMVC 2018 (MSDS-RCNN).** Chengyang Li, Dan Song,
   Ruofeng Tong, Min Tang, *"Multispectral Pedestrian Detection via Simultaneous Detection and
   Segmentation"*. They cleaned the training set (imprecise localization, misclassification,
   misaligned regions) → **7,601 train images**. MSDS-RCNN's own ablation: sanitized training
   alone moves them from 11.63 → 7.45 MR. Download from the MSDS-RCNN page
   (https://li-chengyang.github.io/home/MSDS-RCNN/).
3. **KAIST-Paired — Zhang et al., ICCV 2019 (AR-CNN).** Lu Zhang, Xiangyu Zhu, Xiangyu Chen,
   Xu Yang, Zhen Lei, Zhiyong Liu, *"Weakly Aligned Cross-Modal Learning for Multispectral
   Pedestrian Detection"*. They relabeled train boxes **per modality** (a box in RGB *and* a
   linked box in thermal), exposing that even beam-split KAIST has small residual RGB↔T position
   shifts. Get it from https://github.com/luzhang16/AR-CNN. Use it if you study misalignment;
   most leaderboard numbers do not require it.

**What the field uses today:** train on the **Li et al. sanitized train set (7,601 imgs)**, test
on the **Liu et al. improved test annotations (2,252 imgs)**, "Reasonable" setting. Every number
in the table in §2 uses this protocol unless flagged.

> Note: the two cleanups are frequently mis-attributed in later papers (Liu↔Li swapped). The
> correct attribution is as above — Liu = test, Li = train — confirmed against the ProbEn
> reference list and the MSDS-RCNN and CalayZhou resource pages.

### 1.4 Evaluation protocol

- **Metric:** log-average miss rate, written **MR⁻² (or LAMR)**: miss rate averaged over 9 FPPI
  points evenly log-spaced in **[10⁻², 10⁰]** false positives per image. Lower is better. A TP
  requires **IoU > 0.5**; detections matching "ignore" ground truth are discarded (Caltech
  rules, Dollár et al. toolbox conventions).
- **"Reasonable" subset:** pedestrians **≥ 55 px tall**, **not occluded or partially occluded**
  (heavy occlusion → ignore; `people`/`cyclist` → ignore). This is the headline number.
- **All / Day / Night:** same metric on the full test set and its day/night partitions. Some
  papers additionally report an "All" *scale/occlusion* condition (near+medium+far, all
  occlusion) — do not confuse "All (Day+Night, Reasonable)" with "All (all scales)". MS-DETR's
  table reports both, side by side.
- **Official code:** the original MATLAB devkit ships with the dataset repo
  (github.com/SoonminHwang/rgbt-ped-detection, based on Piotr Dollár's toolbox). Practical
  Python options used by recent papers: the **MLPD repo's `evaluation_script/evaluation_script.py`**
  (github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection) which ships
  `KAIST_annotation.json` (the improved test annotations in COCO-ish JSON) and plots
  MR-vs-FPPI curves against the SOTA; the **MS-DETR repo** carries the same JSON + eval script.
  There is also a Sejong-RCV EvalAI challenge ("Multispectral Pedestrian Detection Challenge",
  eval.ai challenge 2343) using the sanitized test protocol.

### 1.5 How reproductions silently become incomparable — checklist

Every one of these has burned someone in the literature:

1. **Original vs improved test annotations** (±15 MR points). If a paper's Halfway-Fusion
   baseline says ~37, it's original annotations; if ~26, improved. ProbEn's Table 2 mixes
   as-published numbers of both eras — its Halfway Fusion row (36.99) is the *original*-anno
   number while its MBNet row (8.13) is improved. Know which era each number is from.
2. **Original vs sanitized train set** (MSDS-RCNN: 11.63 → 7.45 from train cleanup alone).
3. **Reasonable vs All-condition** numbers confused (e.g. GAFF: 6.48 Reasonable vs ~27 All-condition).
4. **IoU threshold / ignore-region handling** — must use the Caltech-style matching with ignore
   boxes, not plain COCO matching.
5. **FPPI averaging** — 9 points in [1e-2, 1e0]; some ports average over [1e-4, 1e0] or use
   different curve interpolation.
6. **Test-set sampling** — must be the standard 2,252-frame subsample, not all 45k frames.
7. **Input resolution** — most papers evaluate at 640×512; changing it changes the 55 px
   "Reasonable" boundary in feature space and small-pedestrian recall.
8. **Day/night split source** — use the official set-level day/night tags.

Rule for your project: fix protocol = {sanitized train 7,601, improved test 2,252, Reasonable,
MLPD eval script}, run *every* baseline and ablation through that one script, and never quote a
published number next to yours without checking its annotation era.

---

## 2. The Canonical Paper Trail

Chronological. Every paper verified. "MR" below = MR⁻² Reasonable-All on the improved test
annotations unless stated.

**Hwang et al., CVPR 2015 — the benchmark + ACF+T+THOG.**
Extends Aggregated Channel Features with thermal channels (ACF+T+THOG). Hand-crafted,
pre-deep-learning. As re-plotted by Liu et al.: 54.74 MR (original annotations); on improved
test annotations the commonly quoted figure is ≈47.3. Significance: proved thermal channels help
and gave the field its dataset and metric.

**Liu et al., BMVC 2016 — "Multispectral Deep Neural Networks for Pedestrian Detection." THE
foundational middle-fusion paper.**
(Jingjing Liu, Shaoting Zhang, Shu Wang, Dimitris Metaxas; arXiv:1611.02644.)
Base detector: Faster R-CNN, VGG-16, two branches. Four fusion designs (verified from the BMVC
PDF):
- *Early Fusion*: concatenate feature maps **after conv1 (C1)**, then a **Network-in-Network
  (NIN) 1×1 conv** reduces the concatenation back to 128 channels so pretrained VGG filters can
  be reused.
- *Halfway Fusion*: same concat+NIN, but **after conv4 (C4)**.
- *Late Fusion*: concatenate the last FC features (F7); RPN uses both branches' C5.
- *Score Fusion*: cascade the two detectors and average confidences 50/50.
Results, original test annotations (All / Day / Night MR): ACF-C-T 54.74; **FasterRCNN-C 48.59 /
42.51 / 64.39; FasterRCNN-T 47.59 / 50.13 / 40.93**; Early 40.44; **Halfway 36.99 / 36.84 /
35.49**; Late 40.46; Score 40.06. On improved test annotations, Halfway Fusion = **25.75**
(24.88 Day / 26.59 Night — the consensus re-evaluation quoted by MS-DETR et al.).
Their stated reason halfway wins: **C4 features carry semantic content while still retaining
fine spatial detail — fusion needs both.** Early fusion drowns in low-level modality statistics
mismatch; late fusion has already committed each stream to a decision and can only recombine,
not interact. This one experiment is why the entire field builds two-stream backbones fused at
mid-level, and it is the empirical anchor for your project.

**König et al., CVPRW 2017 — Fusion RPN + BF.**
(Daniel König, Michael Adam, Christian Jarvers, Georg Layher, Heiko Neumann, Michael Teutsch,
*"Fully Convolutional Region Proposal Networks for Multispectral Person Detection"*.)
Multispectral RPN on VGG-16, fusing the two streams by concat+NIN **at conv5** (they ablated the
fusion stage and also found deep-but-not-final fusion best), then a Boosted Decision Forest
re-scores proposals. **29.83 MR** original annotations as published; **18.29 / 19.57 / 16.27**
on improved annotations (consensus tables). First to show RPN-level fusion + classical
re-scoring; also an early data point that night performance can *exceed* day once thermal is
properly fused.

**Li et al., BMVC 2018 — MSDS-RCNN** (same Li as the sanitized annotations; arXiv:1808.04818).
Two-stream Faster R-CNN + an auxiliary **segmentation supervision** branch ("simultaneous
detection and segmentation"). 11.34 / 10.53 / 12.94 on improved-test + original-train; **7.49 /
8.09 / 5.92** when trained on their sanitized train set (as tabulated by TFDet). Two lessons:
segmentation supervision is a cheap accuracy lever, and annotation cleanup is worth more than
most architecture ideas of that era.

**Li et al., Pattern Recognition 2019 — IAF R-CNN (Illumination-Aware Faster R-CNN).**
(Chengyang Li, Dan Song, Ruofeng Tong, Min Tang; Pattern Recognition 85:161–171;
arXiv:1803.05347.) A small **illumination estimation subnetwork** looks at the RGB image,
predicts a day/night scalar w∈[0,1], and the final detection is a **weighted late fusion** of
the color and thermal detectors' outputs via a gating function of w. **15.73 / 14.55 / 18.26**
(improved anno; 20.95 as published on original anno). The canonical "illumination-aware
weighting" paper.

**Guan et al., Information Fusion 2019 — IATDNN+IASS.**
(Dayan Guan, Yanpeng Cao, Jiangxin Yang, Yanlong Cao, Michael Ying Yang, *"Fusion of
multispectral data through illumination-aware deep neural networks for pedestrian detection"*,
Inf. Fusion 50:148–157; arXiv:1802.09972.) Same illumination-gating idea but applied to
**feature-level** two-stream fusion + illumination-aware semantic segmentation supervision.
**14.95 / 14.67 / 15.72** (improved anno; 26.37 as published on original).

**Zhang et al., Information Fusion 2019 — CIAN (Cross-modality Interactive Attention Network).**
(Lu Zhang, Zhiyong Liu, Shifeng Zhang, Xu Yang, Hong Qiao, Kaizhu Huang, Amir Hussain; Inf.
Fusion 50:20–29.) First to make the two streams *talk to each other* during fusion: a
cross-modality attention module reweights each stream using a jointly-computed context vector,
rather than gating on a global illumination scalar. **14.12 / 14.78 / 11.13.** Notable as the
first big night-side win (11.13) among feature-fusion methods of its era.

**Zhang et al., ICCV 2019 — AR-CNN (+ KAIST-Paired).**
(Lu Zhang, Xiangyu Zhu, Xiangyu Chen, Xu Yang, Zhen Lei, Zhiyong Liu; arXiv:1901.02645.)
The misalignment paper (see §4): **Region Feature Alignment (RFA)** predicts a per-region
RGB↔T offset and shifts region features before fusion; **RoI-jitter** augments training with
synthetic misalignment; **confidence-aware fusion** reweights region features by per-modality
reliability. Evaluated with KAIST-Paired annotations. **9.34 / 9.94 / 8.38.**

**Zhou et al., ECCV 2020 — MBNet (Modality Balance Network).**
(Kailai Zhou, Linsen Chen, Xun Cao, *"Improving Multispectral Pedestrian Detection by Addressing
Modality Imbalance Problems"*, ECCV 2020, Springer LNCS; arXiv:2008.03043; code
github.com/CalayZhou/MBNet.) Note: **ECCV**, not CVPR. Two ideas: **DMAF (Differential Modality
Aware Fusion)** — take the *difference* F_RGB − F_T, global-pool it, squash it, and use it to
re-excite the *other* modality's channels (a differential SE gate, cheap); and an
**illumination-aware feature alignment** module that weights modality contributions by predicted
illumination and aligns features. Single-stage (SSD-style, ResNet-50). **8.13 / 8.28 / 7.86.**
For your project MBNet is the most directly relevant "cheap, clever middle fusion" reference.

**Kim et al., IEEE RA-L (with IROS) 2021 — MLPD (Multi-Label Pedestrian Detection).**
(Jiwon Kim, Hyeongjun Kim, Taejoo Kim, Namil Kim, Yukyung Choi; RA-L 6(4):7846–7853; code
github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.) SSD-style two-stream halfway
architecture (VGG-16 or ResNet-50), **multi-label heads** (separate labels for
paired/unpaired input states) + **semi-unpaired augmentation** (apply *different* geometric
transforms to the two streams during training so the network never assumes perfect pairing).
No fancy fusion operator — fusion is plain halfway concat; the gains come from training
strategy. **7.58 / 7.95 / 6.95** (VGG-16). Ships the standard Python evaluator.

**Zhang et al., WACV 2021 — GAFF (Guided Attentive Feature Fusion).**
(Heng Zhang, Elisa Fromont, Sébastien Lefèvre, Bruno Avignon; WACV 2021; code
github.com/zhanghengdev/GAFF.) Fusion = **intra-modality attention** (SE-style channel
recalibration per stream, plus spatial attention *supervised by pedestrian masks*) +
**inter-modality attention** (each modality receives residual, location-aware corrections
guided by the other), then 1×1 projection. Explicitly designed for **low overhead**.
**6.48 / 8.35 / 3.46.** The 3.46 Night number stood as the night SOTA for years and is the
strongest evidence that *cheap guided gating*, not heavy attention, is sufficient on aligned
data. GAFF is arguably the closest published spirit-match to what you want to build on Jetson.

**Chen et al., ECCV 2022 (oral) — ProbEn (Probabilistic Ensembling). The late-fusion reality check.**
(Yi-Ting Chen, Jinghao Shi, Zelin Ye, Christoph Mertz, Deva Ramanan, Shu Kong;
arXiv:2104.02904; ECVA paper 136690139.) **Non-learned, detection-level Bayesian fusion** of
independently trained detectors: assuming conditional independence of modalities given the
class, fuse *logits* by summation (score fusion) and fuse boxes by variance-weighted averaging;
handles detections missing in one modality gracefully. Key numbers (their Table 1/2, verified
from the PDF): their own Faster R-CNN singles — RGB 18.67, thermal 18.99, early-fusion 19.36,
mid-fusion 14.48; plain **NMS-as-fusion of RGB+T detections already hits 10.78**, beating their
mid-fusion; ProbEn over RGB+T = 8.50; ProbEn3 (RGB+T+mid-fusion) = 7.66; **ProbEn3 over
off-the-shelf MLPD and GAFF = 5.14 / 6.04 / 3.59 — new SOTA at publication, beating every
learned mid-fusion method.** The uncomfortable, load-bearing lesson: *a strong late-fusion
ensemble of good single-modality detectors beats most published mid-fusion architectures.* Any
mid-fusion result must be compared against this, not just against other mid-fusion papers. Cost
caveat: ProbEn runs 2–3 full detectors — on a Jetson power/latency budget that is exactly what
you cannot afford, which is the honest engineering argument for mid-fusion.

**Fang, Han, Wang, 2021 — CFT (Cross-Modality Fusion Transformer).**
(Fang Qingyun, Han Dapeng, Wang Zhaokui; arXiv:2111.00273; code
github.com/DocF/multispectral-object-detection.) **YOLOv5** made two-stream, with transformer
self-attention blocks inserted at backbone stages that jointly attend over concatenated
RGB+T tokens (intra- and inter-modality attention in one op). Evaluated on FLIR-aligned, LLVIP
(mAP50 97.5, mAP 63.6), VEDAI — **not on KAIST MR**, so it does not appear in the MR table.
Relevance to you: it is the closest published template for "two-stream YOLO with mid-fusion",
and it defines the expensive end of the operator menu (dense self-attention at stride-8
resolution — quadratic in H×W, painful on Orin NX).

**Shen et al., Pattern Recognition 2024 (vol 145, 109913) — ICAFusion.**
(Jifeng Shen et al.; arXiv:2308.07504.) Dual **cross-attention** transformers for global
RGB↔T interaction, with an **iterative parameter-sharing** scheme (the same fusion block
re-applied, weights shared across iterations) to cut parameters/compute; also argues
cross-attention gives robustness to misalignment because interaction is global rather than
strictly per-pixel. Evaluated on KAIST, FLIR, VEDAI (KAIST MR reported in-paper; their headline
gains are mAP on FLIR/VEDAI and faster inference than CFT-style stacks).

**Xing et al., IEEE T-ITS 2024 — MS-DETR.**
(Yinghui Xing, Shuo Yang, Song Wang, Shizhou Zhang, Guoqiang Liang, Xiuwei Zhang, Yanning
Zhang; T-ITS, DOI 10.1109/TITS.2024.3450584; arXiv:2302.00290; code
github.com/YinghuiXing/MS-DETR.) DETR-style detector, two ResNet-50 streams, **loosely coupled
fusion**: instead of dense per-pixel fusion, sparsely sample keypoints per modality
(deformable-attention style) and fuse them with learned weights; plus modality-balanced
optimization via separate visible/thermal decoder branches aligned by an instance-wise dynamic
loss. **6.13 / 7.78 / 3.18**, and much stronger under the All-scales condition (20.64 vs GAFF's
27.30). Cost: **76.4M params, 258.8 GFLOPs @ 640×512** — a non-starter on Orin NX, but the
right "expensive ceiling" reference.

**Zhang et al., IEEE TNNLS 2025 (36(7):13276–13290) — TFDet. Current verified SOTA on KAIST.**
(Xue Zhang, Xiaohan Zhang, Jiangtao Wang, Jiacheng Ying, Zehua Sheng, Heng Yu, Chunguang Li,
Hui-Liang Shen; arXiv:2305.16580.) "Target-aware fusion": fuse (channel-shuffled group conv +
modulated deformable conv for adaptive receptive field + global channel recalibration), then
**refine with a segmentation branch** and a *correlation-maximum loss* that explicitly
sharpens foreground-background contrast in the fused feature map. **4.47 / 5.22 / 3.36**
(sanitized train, improved test). ~0.13 s/frame at 512×640 on an RTX 3060 as reported. The
lesson TFDet adds: making the *fused features discriminative* (supervision) matters as much as
the mixing operator itself — consistent with MSDS-RCNN and GAFF both profiting from
mask supervision.

**Kim et al., CVPR 2024 — Causal Mode Multiplexer (CMM) + ROTX-MP.** (arXiv:2403.01300; code
github.com/ssbin0914/Causal-Mode-Multiplexer.) Not an MR-leaderboard paper but essential
context: shows multispectral detectors learn a **thermal shortcut** ("modality bias") because
KAIST-style data only contains RO-TO (day: both visible) and RX-TO (night: thermal-only)
pedestrians; on RO-TX cases (visible in RGB, invisible in thermal — glass, insulated clothing)
detectors collapse. They release the **ROTX-MP** test set (1,000 images) and a counterfactual-
intervention training framework. Directly relevant to interceptor-style deployments where the
thermal channel can be defeated.

### 2.1 Results table

Protocol: KAIST test 2,252 imgs, improved test annotations (Liu et al. 2018), Reasonable
setting, MR⁻² % (lower better). Sources: MS-DETR Table (T-ITS 2024), ProbEn Tables 1–2 (ECCV
2022), TFDet (TNNLS 2025), MLPD repo. Params/GFLOPs are reported only where the paper states
them; most of this literature never reported compute.

| Year | Method | Base detector / backbone | Fusion type | MR All | Day | Night | Cost (as reported) |
|---|---|---|---|---|---|---|---|
| 2015 | ACF+T+THOG (Hwang) | ACF (hand-crafted) | early channel stack | ≈47.3 | 42.6 | 56.2 | CPU-era |
| 2016 | Halfway Fusion (Liu) | Faster R-CNN / 2×VGG16 | mid: concat+NIN @C4 | 25.75 | 24.88 | 26.59 | – |
| 2017 | Fusion RPN+BF (König) | RPN+BDT / 2×VGG16 | mid: concat+NIN @C5 | 18.29 | 19.57 | 16.27 | – |
| 2019 | IAF R-CNN (Li, PR) | Faster R-CNN / 2×VGG16 | late, illumination-gated | 15.73 | 14.55 | 18.26 | – |
| 2019 | IATDNN+IASS (Guan, InfFus) | Faster R-CNN / 2×VGG16 | mid, illumination-gated | 14.95 | 14.67 | 15.72 | – |
| 2019 | CIAN (Zhang, InfFus) | one-stage / 2×VGG16 | mid, cross-modal attention | 14.12 | 14.78 | 11.13 | – |
| 2018 | MSDS-RCNN (Li, BMVC) | Faster R-CNN / 2×VGG16 | mid + seg supervision | 11.34¹ | 10.53 | 12.94 | – |
| 2019 | AR-CNN (Zhang, ICCV) | Faster R-CNN / 2×VGG16 | mid + region alignment | 9.34 | 9.94 | 8.38 | – |
| 2020 | MBNet (Zhou, ECCV) | SSD-style / 2×ResNet50 | mid: differential SE + illum. align | 8.13 | 8.28 | 7.86 | ~real-time (GPU) |
| 2021 | MLPD (Kim, RA-L) | SSD-style / 2×VGG16 | mid: plain concat; training tricks | 7.58 | 7.95 | 6.95 | single-stage |
| 2021 | GAFF (Zhang, WACV) | single-stage / 2×ResNet | mid: guided intra+inter attention (cheap) | 6.48 | 8.35 | 3.46 | low overhead (design goal) |
| 2022 | ProbEn3 w/ GAFF (Chen, ECCV) | ensemble of 3 detectors | **late**: Bayesian score+box fusion | 5.14 | 6.04 | 3.59 | ≈3× detector cost |
| 2024 | MS-DETR (Xing, T-ITS) | DETR / 2×ResNet50 | mid: sparse keypoint cross-attn | 6.13 | 7.78 | 3.18 | 76.4M, 258.8 GFLOPs |
| 2025 | TFDet (Zhang, TNNLS) | Faster R-CNN & FCOS / 2×ResNet50 | mid: adaptive conv + channel recal + seg-refine | **4.47** | 5.22 | 3.36 | 0.13 s @512×640, RTX 3060 |

¹ 7.49/8.09/5.92 when retrained on sanitized train annotations (TFDet's table).
Reference singles (ProbEn's in-house Faster R-CNN): RGB-only 18.67 (14.56 D / 27.42 N),
thermal-only 18.99 (24.59 D / 7.76 N), early fusion 19.36, plain mid-fusion 14.48.

Reading the trend honestly: 2016→2021 progress came from (a) annotation cleanup, (b) better
training (augmentation, seg supervision), (c) cheap gating — not from expensive operators.
The two expensive entries (MS-DETR, TFDet) buy their last 1–2 points with heavy machinery, and
plain late-fusion ProbEn still sits between them. On aligned data the fusion operator is *not*
the main lever; discriminative fused features and clean training are.

---

## 3. The Fusion-Operator Taxonomy

Setting: two feature maps F_rgb, F_t ∈ ℝ^{C×H×W} at some backbone stage; produce F_fused
(usually C channels so the rest of the detector is unchanged). Costs given as extra
multiply-accumulates (MACs) beyond the two backbones, for one fusion point. For your scale
reference at stride 16, 640×512 input, C=256: H×W = 40×32 = 1,280 positions.

**(a) Element-wise sum / mean / max / product — free.**
Cost ≈ C·H·W adds (~0.3 MFLOPs at the scale above; noise). No parameters; assumes the two
feature spaces are already commensurable (they aren't, at init, if streams are trained
separately — fine if trained jointly end-to-end). Sum/mean is the default inside FPN-style
necks and works surprisingly well as a *floor* baseline; max acts as a per-cell OR-gate (some
robustness to one modality dropping out); product acts as an AND-gate (dangerous: kills
detections that only one modality sees — exactly the night-time RGB-dead case). Literature: used
as ablation baselines everywhere; nobody wins with these alone, but the gap to the winner is
often only 1–3 MR points.

**(b) Concatenation + 1×1 conv (NIN fusion) — the Liu et al. workhorse. Cheap.**
Concat to 2C, project to C: cost = 2C·C·H·W MACs (≈168 MMACs at the scale above — vs ~tens of
GFLOPs for a backbone; negligible). Parameters 2C². This is what Halfway Fusion (BMVC 2016),
Fusion RPN+BF (2017), MSDS-RCNN, and MLPD actually use. The 1×1 learns per-channel linear
mixing — a superset of weighted sum. **Default choice; everything else must beat this.**

**(c) Learned-scalar / weighted fusion — free.**
F = α·F_rgb + (1−α)·F_t with α either a learned global parameter or a per-stage parameter.
Between (a) and (b). Rarely reported alone; subsumed by (b). Its interesting variant is when α
comes from *data* → (f).

**(d) Gated fusion (learned spatial or channel gate) — cheap.**
A small conv on [F_rgb‖F_t] predicts gate G ∈ [0,1]^{1×H×W} (spatial) or [0,1]^{C} (channel);
F = G⊙F_rgb + (1−G)⊙F_t. Cost: one or two 1×1/3×3 convs (same order as (b)).
Literature: **GFD-SSD** (Zheng, Izzat, Ravi, arXiv:1903.06999, 2019) — Gated Fusion Units
between two SSD streams; they found sigmoid-gated mixing beats stacked fusion and note a ReLU
variant (unbounded gate) with product→sum swap; ran on KAIST at 2× the speed of Faster-R-CNN
fusions. Gates give the network an explicit "trust which modality, where" mechanism — the
minimal operator that addresses the actual physics (RGB dies at night, thermal dies on glass).

**(e) SE-style channel recalibration / CBAM channel+spatial — cheap, Jetson-friendly.**
SE (Hu, Shen, Sun, CVPR 2018): global-avg-pool → FC(C→C/r) → FC(C/r→C) → sigmoid → scale
channels. Cost ~2C²/r MACs — *thousands of times* cheaper than the backbone; the caveat on
embedded HW is that pool+FC+scale are memory-bound, low-arithmetic-intensity ops, so measure
latency, not FLOPs. CBAM (Woo, Park, Lee, Kweon, ECCV 2018) adds a 7×7-conv spatial map.
In fusion use: apply SE/CBAM to [F_rgb‖F_t] before the 1×1 projection, or cross-wise (one
modality's global vector gates the other's channels). Literature: **GAFF** is exactly this
family (SE per stream + mask-guided spatial attention + cross-modal residual guidance) and hit
6.48/3.46-night — the best cost/accuracy point in the whole literature. **MBNet's DMAF** is the
differential variant: SE on the *difference* F_rgb−F_t, applied cross-wise.

**(f) Illumination-aware weighting — day/night gate from a subnetwork. Cheap but blunt.**
A tiny CNN on the (downsampled) RGB image predicts illumination w; w gates modality
contributions globally (IAF R-CNN at decision level; IATDNN at feature level; MBNet's
illumination branch). Cost: negligible (runs once per image on a thumbnail). Two honest
caveats from the literature: (1) day/night is a *global, bimodal* proxy — headlights, tunnels,
shadows are local, which is why CIAN/GAFF-style *local* gating overtook global illumination
gates on the leaderboard; (2) it adds a failure mode (mispredicted illumination flips the
gate). Worth having as an ablation arm precisely because it's the interpretable version of (d).

**(g) Differential / complementary fusion (MBNet-style) — cheap.**
Feed the *difference signal* to the gate: D = F_rgb − F_t; each modality is enhanced by an SE
gate computed from D (global-pooled, tanh), so each stream absorbs what the other has and it
lacks. Cost: SE-level. Motivation: the sum path carries common-mode information, the difference
path carries complementary information — gate on the latter. MBNet demonstrated this + modality
balancing = 8.13 with a single-stage detector in 2020.

**(h) Cross-modal feature rectification, CMX-style — moderate, no full attention.**
CMX (Jiacheng Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, Rainer Stiefelhagen,
*"CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers"*, IEEE T-ITS 2023;
arXiv:2203.04838): before fusing, a **Cross-Modal Feature Rectification Module** uses channel-
and spatial-attention computed from *both* modalities to correct each modality's features
(each stream is rectified by the other), then a heavier Feature Fusion Module mixes them. The
CM-FRM half is pool+MLP+conv machinery — SE/CBAM-cost, no token attention — and is separable
from their transformer FFM. Takeaway for you: "rectify-then-fuse" (fix each stream with cheap
cross-statistics, then plain concat+1×1) is a Jetson-compatible pattern with strong published
evidence from the RGB-X segmentation side; TFDet's fuse-then-refine is the detection-side
sibling.

**(i) Cross-attention / transformer fusion — the expensive ceiling.**
Dense self-attention over concatenated modality tokens (CFT) costs O((2HW)²·C) per fusion
stage — at stride 8, 80×64 tokens per modality → 10,240 tokens → ~10⁸·C MACs *per layer*, plus
quadratic memory: not Orin-NX-viable at useful resolutions. Published mitigations: sparse/
deformable sampling (**MS-DETR** "loosely coupled fusion": attend at a few hundred sampled
keypoints, not all pixels), **ICAFusion**'s iterative shared-weight cross-attention blocks, and
doing attention only at C5/stride-32 where HW is small. What attention actually buys, per the
misalignment literature (§4): *global correspondence*, i.e. tolerance to spatial offsets —
which KAIST barely has. On aligned data its measured advantage over GAFF-class gating is
~1–2 MR points at 10–100× the fusion cost. Keep it as the ceiling reference in the ablation
table, not the deployment candidate.

### Where to fuse, and weight sharing

- **Which stage(s).** Liu et al.: single fusion at C4 beats C1 and F7 (that's the "halfway"
  result). König et al. independently landed on deep-conv-stage fusion (C5-level concat+NIN).
  Modern single-stage systems (MLPD, GAFF, MBNet, CFT, TFDet) fuse at **every FPN level** —
  i.e. fuse C3, C4, C5 (or P3–P5) with the *same cheap operator per level* and feed a single
  fused FPN to one head. For a YOLOX: run two CSPDarknet streams through backbone stages, fuse
  at the three feature levels that feed PAFPN (strides 8/16/32), keep one PAFPN + one head.
  The known-but-under-ablated middle ground: fuse *earlier* (stride 4) helps small pedestrians
  in principle but costs the most compute and is most sensitive to misalignment; nobody has a
  clean modern ablation of single-level vs multi-level fusion on KAIST — cheap paper for you
  (§7).
- **Why "halfway" is best — the accepted story.** Early layers encode modality-specific
  low-level statistics (RGB textures/edges vs thermal blobs/gradients); fusing there forces one
  set of filters to serve two distributions and amplifies registration error at high resolution.
  Late/decision-level fusion cannot create joint features at all — it can only arbitrate between
  two already-formed opinions (ProbEn shows how far pure arbitration goes: far, but it needs
  two full networks). Mid-level features are abstract enough to be commensurable across
  modalities yet spatial enough to keep localization detail. That is Liu et al.'s original
  argument and nothing since has overturned it — though note ProbEn's result means "halfway
  beats late" is only cleanly true at *matched compute*.
- **Shared weights?** The near-universal choice is **two independent streams** (no sharing):
  modality statistics differ too much, and the thermal stream is usually *initialized* from
  RGB-pretrained weights but then diverges. Parameter sharing appears only as a compute trick
  (ICAFusion shares fusion-block weights across iterations, not across modalities). If Orin
  memory is tight, published precedent for a *shared* backbone with modality-specific BN exists
  in the broader multimodal literature but not as a winning KAIST configuration — treat it as
  an ablation, not the default.

---

## 4. Misalignment: what breaks off the beam splitter

KAIST is the best-case geometry. A real interceptor rig has: different sensors with different
FOV and distortion, **parallax** from baseline separation (disparity grows as 1/depth — worst
exactly for close, fast targets), imperfect extrinsic calibration that drifts with vibration
and temperature, and **temporal skew** (different frame rates / unsynced triggers; a target at
10 m crossing at 5 m/s moves ~9 px per 10 ms of skew at 640-px/60° optics). The literature:

**Datasets along the alignment spectrum:**
- **KAIST** — beam-split, hardware-synced. Aligned (with small residual shifts; see AR-CNN).
- **CVC-14** (González et al., *"Pedestrian Detection at Day/Night Time with Visible and FIR
  Cameras: A Comparison"*, Sensors 16(6):820, 2016) — visible + FIR from **separate cameras,
  not registered and not perfectly synchronized**; 640×471 @10 Hz; day/night sequences
  (~3,695 day + 3,390 night train images, ~700+700 test); **separate per-modality bounding
  boxes** because no single box fits both. The only classic benchmark that is honestly
  unregistered — always report on it if your claim is misalignment robustness (MBNet, AR-CNN,
  MS-DETR all do).
- **FLIR ADAS** — RGB+thermal, **unaligned both spatially and temporally** (ProbEn Fig. 4
  documents this), annotations on thermal only. The community instead uses **FLIR-aligned**:
  Heng Zhang et al. (*"Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine
  Blocks"*, ICIP 2020; github.com/zhanghengdev/CFR) manually aligned pairs → 4,128 train /
  1,013 val. 3 classes (person/car/bicycle), mAP metric.
- **LLVIP** (Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, Wenli Zhou, ICCVW 2021;
  arXiv:2108.10831) — 16,836 pairs, surveillance viewpoint, mostly very dark scenes, **strictly
  aligned in time and space** (registered crops). Good night-time stress test; mAP metric.
- **M3FD** (released with **TarDAL**: Jinyuan Liu et al., *"Target-aware Dual Adversarial
  Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for
  Object Detection"*, CVPR 2022; github.com/JinyuanLiu-CV/TarDAL) — 4,200 aligned IR-visible
  pairs, multi-scenario (including degraded weather), 6 classes, YOLO-format labels. Comes from
  the image-fusion community; commonly used as a third detection benchmark.

**Techniques for tolerating spatial misregistration in the fusion neck (no deformable attention):**
1. **Predict-and-shift region/feature offsets — AR-CNN (ICCV 2019).** A small head predicts the
   RGB↔T shift per region proposal; region features are translated before fusion. This is a
   2-DoF (or 4-DoF) *local translation* model — much cheaper than deformable attention and
   sufficient when misalignment is a smooth, low-magnitude field. KAIST-Paired quantifies that
   even beam-split data has such shifts. Anchor-free analogue for YOLOX: predict a per-cell
   (dx,dy) for the thermal feature map at each fused level and bilinearly resample the thermal
   features before the fusion op — a "flow-lite" alignment layer, a few 3×3 convs.
2. **Train-time misalignment augmentation — AR-CNN's RoI-jitter, MLPD's semi-unpaired
   augmentation.** Deliberately decalibrate the two streams during training (independent
   shifts/flips/crops) so the fused features learn shift tolerance. Zero inference cost. MLPD's
   7.58 with *plain concat fusion* is largely attributable to this — the single highest-value,
   lowest-cost idea in the misalignment literature for your build.
3. **Coarse-only fusion:** fuse at stride 16/32 only, where a k-pixel image offset shrinks to
   k/16 feature cells; accept the small-object recall cost. Implicit in many designs; cheap and
   effective when misalignment is bounded.
4. **Detection-level (late) fusion — ProbEn.** Box-level fusion with an IoU-matching tolerance
   sidesteps feature misalignment entirely; ProbEn explicitly demonstrates this on unaligned
   FLIR, where it beats mid-fusion approaches. The principled fallback if your rig's parallax
   turns out unmanageable — at the cost of running two detectors.
5. **Global-interaction fusion** (cross-attention: ICAFusion's stated motivation, MS-DETR's
   sparse sampling, and 2024–25 offset-aware works like RSDet "removal-then-selection"
   (arXiv:2401.10731) and COMO offset-guided Mamba fusion (arXiv:2412.18076)) — works, but this
   is the expensive family you're excluding on Jetson.

**Temporal skew: mostly an open problem.** The literature acknowledges it (CVC-14's authors
note the streams are not synchronized; ProbEn notes FLIR is temporally unaligned) but almost
no method models time explicitly — the standard treatment is "nearest frame + treat residual
motion as spatial misalignment", which items 1–4 above then absorb. There is early 2026-era
work on spatiotemporal fusion (e.g. "Strip-Fusion: Spatiotemporal Fusion for Multispectral
Pedestrian Detection", arXiv:2601.18008 — surfaced in search, not yet peer-review-verified;
treat accordingly), but no established benchmark isolates temporal skew. For a real rig the
engineering answer is hardware triggering; the research answer is unwritten — a genuine gap
(§7).

---

## 5. Sanity Baselines: RGB-only and thermal-only

Run these first, on your exact YOLOX + protocol, before any fusion code exists.

**The established facts on KAIST (Reasonable, MR⁻², lower better):**

| Detector | All | Day | Night | Source |
|---|---|---|---|---|
| Faster R-CNN, RGB-only | 48.59 | 42.51 | **64.39** | Liu BMVC 2016 (orig. anno) |
| Faster R-CNN, thermal-only | 47.59 | 50.13 | **40.93** | Liu BMVC 2016 (orig. anno) |
| Faster R-CNN, RGB-only | 18.67 | 14.56 | **27.42** | ProbEn ECCV 2022 (modern protocol) |
| Faster R-CNN, thermal-only | 18.99 | **24.59** | **7.76** | ProbEn ECCV 2022 |
| DETR-style, visible-only | 18.18 | – | – | MS-DETR T-ITS 2024 |
| DETR-style, thermal-only | 15.61 | – | – | MS-DETR T-ITS 2024 |

Pattern, stable across a decade and across detector families: **thermal ≥ RGB overall (KAIST
is ~40% night frames), thermal crushes RGB at night (7.76 vs 27.42 in ProbEn's runs — a 3.5×
gap), RGB wins by a solid margin in daytime.** On more night-heavy data the gap widens: on
LLVIP, thermal-only 65.3 mAP vs visible-only 50.8 (MS-DETR's table). Expect your thermal-only
YOLOX to be the stronger single stream overall, and by a wide margin at night.

**Why "fusion that doesn't beat the best single modality" is the classic silent failure.**
Three mechanisms, all documented:
1. **Early fusion can *underperform* singles** — ProbEn measured early fusion at 19.36 vs 18.67
   RGB-only. Naive mixing lets the noisy modality corrupt the informative one.
2. **Modality imbalance** (MBNet's thesis): joint training lets the easier modality (thermal on
   KAIST) dominate the gradients; the RGB stream under-trains, and the fused model converges to
   "thermal detector with dead weight attached" — matching, not beating, thermal-only, while
   costing 2× compute. Because thermal-only is already strong, a lazy fused net *looks*
   respectable. Always compare against the *best single*, per split (Day AND Night), not just All.
3. **Modality bias / thermal shortcut** (CMM, CVPR 2024): the fused detector learns "pedestrian
   ⇔ warm blob" and fails on visible-but-cold cases — invisible on KAIST's test distribution,
   catastrophic off it.
The acceptance test for any fusion run: MR_fused < min(MR_rgb, MR_thermal) on **All, Day, and
Night simultaneously**, at matched training budget. Day is the one that usually fails silently
(check GAFF: 8.35 Day vs MLPD's 7.95 — even SOTA methods trade Day for Night).

---

## 6. YOLOX Specifics for a Two-Stream Build

**Paper:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun, *"YOLOX: Exceeding YOLO
Series in 2021"*, arXiv:2107.08430 (Megvii; tech report). Components that matter here:
**anchor-free** head (per-cell direct box regression), **decoupled head** (separate cls/reg
branches), **SimOTA** dynamic label assignment (top-k cost-based, k from IoU dynamics),
**strong augmentation** (Mosaic + MixUp, disabled for the last ~15 epochs), CSPDarknet + PAFPN.
YOLOX-S is 9.0M params / 26.8 GFLOPs @640 — two-stream backbone pushes that to roughly 1.7–1.8×
FLOPs (backbone doubles; neck+head shared), still comfortably inside Orin NX INT8 budget.

What breaks or must be adapted:

1. **Augmentation must be modality-synchronized.** Mosaic composes 4 images with random scale/
   translate; MixUp blends pairs; random affine + flip follow. In a two-stream pipeline every
   *geometric* parameter (mosaic layout, crop coords, scale, flip, affine) must be sampled
   **once** and applied to RGB and thermal identically, or you inject gross artificial
   misalignment and labels match neither stream. Implementation reality: stack RGB+T into one
   6-channel array before the augmentation pipeline, split at the model input. *Photometric*
   augs (HSV jitter, color jitter) must apply to RGB only — HSV math on thermal intensities is
   meaningless and breaks the thermal calibration; give thermal its own intensity-jitter/noise
   aug if any. Exception, deliberate: MLPD/AR-CNN-style *unpaired* geometric jitter as a
   misalignment-robustness ablation — but that must be a controlled, small perturbation you
   opted into, not a pipeline bug.
2. **MixUp across modalities:** blend RGB_a with RGB_b and T_a with T_b using the **same λ and
   the same partner image** — never mix modalities across different scenes independently.
3. **Thermal is single-channel.** KAIST thermal is 8-bit grayscale (from a 14-bit sensor).
   Standard practice: replicate to 3 channels and reuse RGB-pretrained conv1, or (equivalent,
   cheaper) sum conv1's input-channel weights into a 1-channel conv. Normalize thermal with
   *thermal* statistics — ProbEn found precise thermal mean subtraction (their train mean:
   135.438) actually matters. For a real rig later: decide AGC/16-bit→8-bit tone-mapping policy
   once and keep it fixed between train and deploy; this silently dominates thermal-stream
   behavior and is nowhere standardized in the literature.
4. **Second-stream initialization.** Both streams start from the same COCO/ImageNet-pretrained
   backbone; the thermal stream fine-tunes into thermal statistics. Everyone does this; nobody
   has shown a better thermal-specific pretraining on KAIST-scale data (too small to train from
   scratch: 7.6k images). Freeze-vs-finetune of the RGB stream is an easy ablation.
5. **SimOTA needs no structural change** — fusion happens before the head, so assignment sees
   one fused feature pyramid. Two caveats: (i) if you add per-modality auxiliary heads
   (MS-DETR-style modality balancing, or MLPD-style multi-label supervision), each auxiliary
   head needs its own assignment against *its* modality's visibility (this is where
   KAIST-Paired's per-modality boxes become useful); (ii) SimOTA's cost uses cls×IoU — early in
   training a dominant thermal stream will win the assignment dynamics, which is the
   modality-imbalance mechanism (§5) showing up inside the assigner. Worth logging per-modality
   gradient norms.
6. **Strong-aug schedule interacts with small data.** COCO-tuned Mosaic/MixUp strength on 7.6k
   KAIST images is unexplored territory; MLPD-class results suggest augmentation is *the* lever,
   but expect to retune (weaker MixUp, longer no-aug tail). Nobody has published a YOLOX-on-KAIST
   MR baseline to copy — you will be establishing it (see §7).
7. **Eval plumbing:** YOLOX outputs COCO-style; you need a converter to the KAIST eval format
   and the MLPD `evaluation_script.py` (§1.4). Score threshold low (0.01) so the FPPI sweep has
   support down to 10⁻².
8. **Deployment notes (Orin NX):** two-stream CSPDarknet is the cost center — everything in §3
   (a)–(h) is latency-noise next to it; TensorRT INT8 handles concat+1×1 and SE gates natively
   (sigmoid+mul fuse fine), whereas dense attention at stride 8 does not quantize or tile
   nearly as well. Calibrate INT8 with *both* modalities' statistics in the calibration set.

---

## 7. Open Questions — where an ablation study earns its keep

Genuinely unsettled, with the receipts:

1. **Mid vs late fusion at matched compute.** ProbEn (late, non-learned) beat every learned
   mid-fusion method as of 2022; TFDet (mid) later beat ProbEn, but with a heavy fusion module —
   while ProbEn pays ~3× detector cost. Nobody has published the honest iso-FLOPs comparison:
   one two-stream mid-fusion detector vs an ensemble of two half-size single-modality detectors.
   On an embedded budget this is *the* question, and it is open.
2. **Does the fusion operator matter at all, on aligned data?** MLPD (plain concat, good
   augmentation) 7.58; GAFF (cheap attention) 6.48; MBNet (differential SE) 8.13; MS-DETR
   (sparse cross-attention, 259 GFLOPs) 6.13. The spread across a 100× fusion-cost range is
   ~2 MR points, while training-protocol differences alone are worth more (MSDS-RCNN: 11.63→7.45
   from annotation cleanup). No paper holds detector, schedule, and augmentation fixed while
   swapping *only* the operator across the full taxonomy of §3. That controlled sweep on a
   modern single-stage detector does not exist. It is exactly your project.
3. **Statistical significance is absent from the field.** Typical gaps between adjacent
   leaderboard entries (0.3–1.0 MR) are quoted from single runs; no error bars, no seed
   variance, anywhere in the canonical trail. On 2,252 test images with ~1,500 reasonable
   pedestrians, run-to-run variance plausibly rivals the claimed deltas. Report mean±std over
   ≥3 seeds and you'll be more rigorous than the literature you cite.
4. **Where to fuse in a modern FPN detector.** Liu's C4 result is from 2016 VGG Faster R-CNN;
   modern methods fuse at all pyramid levels by default, untested against single-level
   alternatives. Single-level vs multi-level vs progressively-weighted fusion on YOLOX: open.
5. **Global illumination gate vs local gating.** IAF/IATDNN-style day/night gates were overtaken
   by local attention (CIAN, GAFF) on the leaderboard, but the comparison was never apples-to-
   apples (different detectors/eras). Whether a free global gate captures most of what a local
   gate does, on a modern detector, is unknown.
6. **Weight sharing between streams.** No winning KAIST method shares backbone weights; nobody
   has carefully measured the accuracy cost of sharing (with per-modality BN) against the ~45%
   backbone-memory saving. Embedded-relevant, unstudied.
7. **Modality bias / thermal shortcut.** CMM (CVPR 2024) showed detectors collapse on RO-TX
   (visible-only) pedestrians and KAIST cannot measure this. How much cheap-gated mid-fusion
   suffers vs late fusion, and whether unpaired augmentation mitigates it, is open. ROTX-MP
   exists as a probe set.
8. **Temporal skew.** Essentially unstudied (§4). Even a simulated-skew ablation (offset the
   thermal stream by k frames on KAIST video and sweep k) would be a novel, publishable curve —
   and directly relevant to any real dual-sensor rig.
9. **Thermal-stream pretraining.** RGB-pretrained init is universal faute de mieux; no
   established thermal foundation pretraining at this scale. Whether it matters is unmeasured.
10. **Benchmark saturation.** SOTA is at 4.47 MR on 2,252 images with known residual label noise
    — annotation-noise floor territory. Day performance (5.22) now lags night (3.36): the
    remaining KAIST errors are daytime hard cases and far-scale, occluded pedestrians outside
    the Reasonable subset (All-condition MRs are still ~20+). Claims of further Reasonable-set
    progress increasingly measure protocol fit, not capability — which strengthens the case
    that a *controlled operator study* (your project) is worth more than another 0.3-point SOTA.

---

## Primary sources (all verified July 2026)

- Hwang, Park, Kim, Choi, Kweon. *Multispectral Pedestrian Detection: Benchmark Dataset and Baseline.* CVPR 2015. https://soonminhwang.github.io/rgbt-ped-detection/ · https://openaccess.thecvf.com/content_cvpr_2015/html/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.html
- Liu, Zhang, Wang, Metaxas. *Multispectral Deep Neural Networks for Pedestrian Detection.* BMVC 2016. https://arxiv.org/abs/1611.02644 · https://bmva-archive.org.uk/bmvc/2016/papers/paper073/index.html — improved test annotations (2018) via https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn
- König, Adam, Jarvers, Layher, Neumann, Teutsch. *Fully Convolutional Region Proposal Networks for Multispectral Person Detection.* CVPRW 2017. https://openaccess.thecvf.com/content_cvpr_2017_workshops/w3/html/Konig_Fully_Convolutional_Region_CVPR_2017_paper.html
- Li, Song, Tong, Tang. *Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation.* BMVC 2018. https://arxiv.org/abs/1808.04818 · sanitized annotations: https://li-chengyang.github.io/home/MSDS-RCNN/
- Li, Song, Tong, Tang. *Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection.* Pattern Recognition 85:161–171, 2019. https://arxiv.org/abs/1803.05347
- Guan, Cao, Yang, Cao, Yang. *Fusion of Multispectral Data Through Illumination-aware Deep Neural Networks for Pedestrian Detection.* Information Fusion 50:148–157, 2019. https://arxiv.org/abs/1802.09972
- Zhang, Liu, Zhang, Yang, Qiao, Huang, Hussain. *Cross-modality Interactive Attention Network for Multispectral Pedestrian Detection.* Information Fusion 50:20–29, 2019.
- Zhang, Zhu, Chen, Yang, Lei, Liu. *Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection.* ICCV 2019. https://arxiv.org/abs/1901.02645 · KAIST-Paired: https://github.com/luzhang16/AR-CNN
- Zheng, Izzat, Ravi. *GFD-SSD: Gated Fusion Double SSD for Multispectral Pedestrian Detection.* arXiv:1903.06999, 2019.
- Zhou, Chen, Cao. *Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems (MBNet).* ECCV 2020. https://arxiv.org/abs/2008.03043 · https://github.com/CalayZhou/MBNet
- Zhang, Fromont, Lefèvre, Avignon. *Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks.* ICIP 2020 (FLIR-aligned). https://github.com/zhanghengdev/CFR
- Zhang, Fromont, Lefèvre, Avignon. *Guided Attentive Feature Fusion for Multispectral Pedestrian Detection (GAFF).* WACV 2021. https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf · https://github.com/zhanghengdev/GAFF
- Kim, Kim, Kim, Kim, Choi. *MLPD: Multi-Label Pedestrian Detector in Multispectral Domain.* IEEE RA-L 6(4):7846–7853, 2021. https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection
- Jia, Zhu, Li, Tang, Zhou. *LLVIP: A Visible-infrared Paired Dataset for Low-light Vision.* ICCVW 2021. https://arxiv.org/abs/2108.10831
- Fang, Han, Wang. *Cross-Modality Fusion Transformer for Multispectral Object Detection (CFT).* arXiv:2111.00273, 2021. https://github.com/DocF/multispectral-object-detection
- Ge, Liu, Wang, Li, Sun. *YOLOX: Exceeding YOLO Series in 2021.* arXiv:2107.08430.
- Liu, Fan, Huang, Wu, Liu, Zhong, Luo. *Target-aware Dual Adversarial Learning ... (TarDAL / M3FD).* CVPR 2022. https://github.com/JinyuanLiu-CV/TarDAL
- Chen, Shi, Ye, Mertz, Ramanan, Kong. *Multimodal Object Detection via Probabilistic Ensembling (ProbEn).* ECCV 2022 (oral). https://arxiv.org/abs/2104.02904 · https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf · https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling
- Zhang, Liu, Yang, Hu, Liu, Stiefelhagen. *CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers.* IEEE T-ITS 2023. https://arxiv.org/abs/2203.04838 · https://github.com/huaaaliu/RGBX_Semantic_Segmentation
- Shen et al. *ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection.* Pattern Recognition 145:109913, 2024. https://arxiv.org/abs/2308.07504
- Kim, Shin, Yu, Kim, Ro. *Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection.* CVPR 2024. https://arxiv.org/abs/2403.01300 · https://github.com/ssbin0914/Causal-Mode-Multiplexer
- Xing, Yang, Wang, Zhang, Liang, Zhang, Zhang. *MS-DETR: Multispectral Pedestrian Detection Transformer with Loosely Coupled Fusion and Modality-Balanced Optimization.* IEEE T-ITS 2024. https://arxiv.org/abs/2302.00290 · https://github.com/YinghuiXing/MS-DETR
- Zhang, Zhang, Wang, Ying, Sheng, Yu, Li, Shen. *TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection.* IEEE TNNLS 36(7):13276–13290, 2025. https://arxiv.org/abs/2305.16580
- González et al. *Pedestrian Detection at Day/Night Time with Visible and FIR Cameras: A Comparison (CVC-14).* Sensors 16(6):820, 2016. https://www.mdpi.com/1424-8220/16/6/820
- Community resource lists (annotations, eval tools, method index): https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource · https://github.com/yuanmaoxun/Awesome-RGBT-Fusion
