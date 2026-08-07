# PRISM — Curriculum

> **Superseded educational draft.** This curriculum predates the corrected
> validation-first protocol. It is retained as background reading only. It must
> not determine data, evaluator, architecture, training, or reporting choices;
> use `../EXPLANATION.md`, `EXPERIMENT_PLAN.md`, and `LITERATURE.md`.

**Goal:** go from "strong robotics engineer who has trained detectors" to "professional in multispectral detection research" — able to design a middle-fusion RGB+Thermal detector on YOLOX, run an ablation study that a reviewer would accept, train it properly on KAIST, and report metrics that mean something.

**Assumed starting point:** ROS2/PX4/Isaac Sim, PyTorch fluent, general CNNs, has trained detectors before. So this curriculum does **not** teach you backprop or what a convolution is. It teaches the things that will actually bite you.

**Rule of the house (R2, theory before build):** every build phase has a theory gate in front of it. The gate is listed in the 6-week schedule at the bottom.

---

## How to read this document

Every resource is tagged:

- **[MUST]** — do not skip. If you only do the MUST items, you can still do this project well. There are 14 of them.
- **[REF]** — reference. Read the named section when you hit the named problem. Do not read cover to cover.
- **[DEPTH]** — for becoming a professional rather than finishing the task. Post-project, or on weekends.

Every entry says **what it's for** and **which parts**. Nothing here is "read this 800-page book."

**Everything marked [MUST] is free and legal to access.** Where a paid book is the best reference, it's tagged [REF] or [DEPTH] and a free substitute is given.

---

## The 5 things that will decide whether this project succeeds

Read this section twice. Most of the curriculum below exists to serve these five points.

1. **KAIST's metric is not mAP.** It's log-average miss rate (MR, often written MR⁻²) on the *reasonable* subset — **lower is better**. If you report mAP and only mAP, your results are not comparable to any paper in this field. (Track 5.)

2. **KAIST's original annotations are broken, and this can dwarf your entire ablation.** Li et al. (MSDS-RCNN) showed that swapping the original training annotations for *sanitized* ones moves miss rate from **11.63% → 7.45% MR** — i.e. annotation noise was responsible for about a third of the inference error. Your fusion tricks will be worth 1–2 MR points. The annotation choice is worth ~4. **Fix the data before you ablate anything**, or you will be measuring your labels, not your architecture. (Tracks 1, 5.)

3. **You must know your seed noise floor before you believe any ablation result.** If run-to-run std is ±0.8 MR and your fancy gated fusion beats concat by 0.5 MR, you have discovered nothing. Measure variance first. (Track 5.)

4. **Fusion comparisons are unfair by default.** A cross-attention fusion block adds parameters and FLOPs. Comparing it against a plain-sum baseline that has neither is not an ablation, it's a bigger-model result. You need **parameter/FLOP-matched controls**. (Track 5.)

5. **KAIST is weakly aligned, and so is your real rig.** The RGB/thermal pairs are *not* pixel-perfect — there's a position shift. Your rig makes this worse: FOV mismatch, parallax from the physical baseline, and a 120fps RGB / 60fps thermal skew that offsets *moving* objects in time as well as space. Fusion operators that assume perfect pixel correspondence (element-wise sum at high resolution) degrade under misalignment; ones that pool or attend are more forgiving. **This is a first-class design axis for your ablation, not an afterthought.** (Tracks 3, 4.)

---

## TRACK 1 — Object detection fundamentals (so YOLOX isn't a black box)

**Why:** you're modifying YOLOX's backbone/neck junction. If you don't know what SimOTA does, you will silently break label assignment when you change the feature maps feeding the head, and you'll blame the fusion.

### [MUST] YOLOX paper
**"YOLOX: Exceeding YOLO Series in 2021"** — Ge, Liu, Wang, Li, Sun (Megvii).
https://arxiv.org/abs/2107.08430

**Read:** all of it — it's only 8 pages. Then re-read **§2.1 (implementation details)** and **§2.2** carefully. The parts that matter for you:
- **Decoupled head** — classification and regression get separate branches. This is why YOLOX converges faster than YOLOv5 and why it's a good fusion base.
- **Anchor-free** — one prediction per location, no anchor box priors. Simplifies your life enormously; there are no anchor scales to re-tune when you change feature statistics via fusion.
- **SimOTA** — the label assignment. Dynamic: the number of positive samples per ground-truth is decided per-image by the loss matrix.
- **Strong augmentation (Mosaic + MixUp) is turned OFF for the last 15 epochs.** Remember this. It matters for multimodal (see Track 3) because naive Mosaic will destroy your RGB/thermal pairing if you augment the two streams independently.

### [MUST] Read the YOLOX source. The code *is* the lesson.
https://github.com/Megvii-BaseDetection/YOLOX

Read these five files, in this order. This is a ~3 hour exercise and it is the highest-value 3 hours in Track 1:

| File | What to learn from it |
|---|---|
| `yolox/models/yolox.py` | The top-level assembly: backbone → head. This is where your fusion will slot in. |
| `yolox/models/darknet.py` | CSPDarknet. Note the stage outputs `dark3`, `dark4`, `dark5` — **these are your candidate fusion points.** |
| `yolox/models/yolo_pafpn.py` | The PAFPN neck. Note it *consumes* `dark3/4/5`. Whatever you fuse must produce tensors with these exact shapes, or you rewrite the neck. |
| `yolox/models/yolo_head.py` | The decoupled head **and `get_assignments()` — this is SimOTA.** Read `get_assignments` and `dynamic_k_matching` line by line. |
| `yolox/models/losses.py` | The IoU loss. Small file, read it all. |

**Concrete understanding check:** without looking, draw the tensor shapes flowing out of `dark3/dark4/dark5` for a 640×640 input on YOLOX-S, and mark exactly where a two-stream fusion module would attach. If you can't, re-read.

### [MUST] Label assignment — the concept behind SimOTA
**"OTA: Optimal Transport Assignment for Object Detection"** — Ge et al., CVPR 2021 (same authors as YOLOX).
https://arxiv.org/abs/2103.14259 · code: https://github.com/Megvii-BaseDetection/OTA

**Read:** §1, §3 (the OT formulation), and **§3.3**. SimOTA is OTA with the Sinkhorn iteration dropped in favour of a simple top-k ("dynamic k"). Understanding *why* they could drop it tells you how robust the assignment is.

**Why you care:** label assignment decides which feature-map locations are positives. Fusion changes the feature statistics at those locations. If your fusion module changes the loss landscape enough, SimOTA will assign differently — and you can get an accuracy change that has *nothing to do with better features* and everything to do with a different positive/negative split. Knowing this is the difference between an engineer and a researcher.

### [MUST] Anchor-free detection, the canonical treatment
**FCOS: "Fully Convolutional One-Stage Object Detection"** — Tian, Shen, Chen, He (ICCV 2019).
https://arxiv.org/abs/1904.01355

**Read:** §3.1 and §3.2 (centerness). ~4 pages. This is the clearest explanation of anchor-free dense prediction that exists, and YOLOX's head is a direct descendant.

### [REF] The necks
- **FPN** — "Feature Pyramid Networks for Object Detection", Lin et al. https://arxiv.org/abs/1612.03144 — **read §3 only** (the top-down pathway + lateral connections). 2 pages.
- **PANet** — "Path Aggregation Network for Instance Segmentation", Liu et al. https://arxiv.org/abs/1803.01534 — **read §3.1 only** (the bottom-up path augmentation). YOLOX's PAFPN is this. 1 page.

Together these two are 3 pages and fully explain YOLOX's neck. Do not read more of them.

### [REF] Losses and NMS
- **Focal Loss / RetinaNet** — https://arxiv.org/abs/1708.02002 — **read §3 (focal loss) and §4 (class imbalance)**. YOLOX doesn't use focal loss by default, but understanding the foreground/background imbalance problem is mandatory background for understanding *why* label assignment matters at all.
- **GIoU** — "Generalized Intersection over Union", Rezatofighi et al. (CVPR 2019). https://arxiv.org/abs/1902.09630 — **read §2–3.** Why plain IoU has zero gradient for non-overlapping boxes.
- **DIoU/CIoU** — "Distance-IoU Loss", Zheng et al. (AAAI 2020). https://arxiv.org/abs/1911.08287 — **read §3–4.** Adds center-distance and aspect-ratio terms. Worth a cheap ablation row in your study (swap YOLOX's IoU loss for CIoU — one-line change, might be free accuracy).

### [REF] The anchor-based vs anchor-free question, settled
**ATSS: "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection"** — Zhang et al., CVPR 2020.
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.pdf

**Read §3 (the "essential difference" experiments).** The punchline — that the anchor-based/anchor-free gap is *almost entirely* about label assignment, not about anchors — is one of the best pieces of empirical CV reasoning in the literature. **This paper is also a masterclass in ablation design; read it again for Track 5.**

### [REF] Reference implementations to read when stuck
- **mmdetection** — https://github.com/open-mmlab/mmdetection — the most modular detection codebase. When you want to see "how do people normally structure a two-stream backbone", read how mmdet composes backbone/neck/head configs.
- **Ultralytics** — https://github.com/ultralytics/ultralytics — read for augmentation pipeline patterns (mosaic/mixup implementations), not for architecture.

### [DEPTH] Lecture, if you want it spoken aloud
**Justin Johnson, UMich EECS 498-007 / 598-005, Lecture 16 — "Object Detection and Segmentation"**
https://www.youtube.com/watch?v=zHSjrqS0jAY

Covers the R-CNN → Fast → Faster → single-stage arc in 80 minutes. Watch at 1.5×. Skip if the papers above already landed — this is the same content, slower.

---

## TRACK 2 — Deep-learning-for-vision core (only what you're shaky on)

**Why this track is short:** you already know CNNs and PyTorch. Skipping the fundamentals is the correct call. What actually breaks multimodal training runs is **normalization, initialization, and augmentation** — so that's all this track covers.

### [MUST] Normalization and training dynamics — the *practical* version
**Karpathy, "Neural Networks: Zero to Hero" — Lecture 3: "Building makemore Part 3: Activations & Gradients, BatchNorm"**
https://karpathy.ai/zero-to-hero.html · playlist: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ · code: https://github.com/karpathy/nn-zero-to-hero

**Watch only Lecture 3.** Skip the rest of the playlist (it's an LLM course; not your problem). Lecture 3 is ~1h55m of exactly the thing that will bite you: what BatchNorm actually does to activation statistics, why dead/saturated units happen, how to diagnose a bad init by *looking at the histograms*.

**Why this specific lecture, for this specific project:** you are about to build a network where **two streams with wildly different input statistics** (8-bit RGB, roughly natural-image distributed; 14/16-bit thermal, unimodal, low-contrast, sensor-dependent offset) meet at a fusion point. BatchNorm statistics per-stream will *not* match. This is the #1 cause of "my fusion model trains worse than my RGB baseline." You need to be able to look at activation histograms at the fusion junction and say what's wrong.

**Concrete understanding check:** instrument your two-stream model, plot the mean/std of activations at the last backbone stage for each modality *before* fusion. If they differ by more than ~2×, you have found a real bug, and you now know how to fix it (per-stream norm, or normalize thermal input properly).

### [MUST] Augmentation for multimodal — there is no paper, so here's the rule
There is no canonical reference for "how to augment an RGB-T pair," which is exactly why people get it wrong. The rule:

> **Every geometric augmentation must be applied identically to both modalities. Every photometric augmentation must be applied to the RGB stream only.**

Flip, crop, scale, Mosaic, MixUp → same transform, same random parameters, both streams. HSV jitter, brightness, contrast, color augmentation → RGB only; applying HSV jitter to a thermal image is physically meaningless and destroys the radiometric signal.

Get this wrong and you break the pixel correspondence that middle fusion depends on. **Write a unit test that asserts a fixed pixel in RGB and its corresponding pixel in thermal land in the same place after your augmentation pipeline.** Read YOLOX's `yolox/data/data_augment.py` and its mosaic dataset wrapper to see where you'd have to intervene.

### [REF] The deep-learning textbook to keep open
**Simon J.D. Prince, "Understanding Deep Learning"** (MIT Press, 2023). **Free PDF**, CC BY-NC-ND.
https://github.com/udlbook/udlbook

The best modern DL book, and the diagrams are outstanding. **For this project read only:**
- **Ch. 7 — Gradients and initialization** (why your two-stream net's gradients explode/vanish)
- **Ch. 9 — Regularization**
- **Ch. 10 — Convolutional networks**
- **Ch. 11 — Residual networks and batch normalization** ← the important one
- **Ch. 20 — Why does deep learning work?** (read last, for taste)

Ignore the other 15 chapters for now.

### [REF] The other free DL book — use it as a code reference, not a read
**"Dive into Deep Learning"** — Zhang, Lipton, Li, Smola (Cambridge UP, 2023). Free, runnable.
https://d2l.ai/

Every chapter is executable notebooks. **Use it as a lookup:** Ch. 7–8 (modern CNNs), Ch. 8.6 (ResNet), Ch. 14 (computer vision: augmentation, bounding boxes, anchor boxes, multiscale detection). Best used when you want a *working minimal implementation* of a concept to compare your code against.

### [REF] The vision course, if you want structure
**Justin Johnson, UMich EECS 498-007 / 598-005 — "Deep Learning for Computer Vision"** (full playlist)
https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r

This is the modern, better-recorded successor to CS231n (Johnson co-taught CS231n). **For you, only two lectures are worth the time:**
- **Lecture 7 — Convolutional Networks** https://www.youtube.com/watch?v=gJ5UENsAQGY (skip if confident; the normalization section is the useful bit)
- **Lectures 10 & 11 — Training Neural Networks I & II** (in the playlist) — learning-rate schedules, weight decay, the "recipe". This is the applied version of Prince Ch. 7/11.

### [REF] Stanford CS231n course notes
https://cs231n.github.io/

The *notes* (not the videos) remain the best quick reference on the web for optimization, backprop intuition, and CNN architectures. Bookmark `neural-networks-3` (learning, babysitting the training process) and re-read it the first time a training run goes sideways. Free.

### [DEPTH] The reference textbooks
- **Bishop & Bishop, "Deep Learning: Foundations and Concepts"** (Springer, 2024) — the rigorous, probabilistic treatment. Buy it if you want depth on *why*. Not needed for this project.
- **Goodfellow, Bengio, Courville, "Deep Learning"** (MIT Press, 2016; free HTML at https://www.deeplearningbook.org/) — historically important, now dated on architectures. **If you read one chapter, read Ch. 8 (Optimization for Training Deep Models).** Do not read it as your primary source; Prince has superseded it.

---

## TRACK 3 — Multimodal / sensor fusion (the actual research contribution)

**Why:** this is where your ablation lives. Everything else is scaffolding. Read this track properly.

### [MUST] The original "halfway fusion" paper — the reason middle fusion is the default
**"Multispectral Deep Neural Networks for Pedestrian Detection"** — Liu, Zhang, Wang, Metaxas (BMVC 2016).
https://arxiv.org/abs/1611.02644

**Read all of it (8 pages).** This is *the* paper that established your project's premise. They compared four fusion stages on KAIST — Early Fusion, Halfway Fusion, Late Fusion, Score Fusion — and found **Halfway Fusion (fusing mid-level conv features) won**, beating the baseline by ~11% and cutting miss rate by 3.5 points.

**This is literally the ablation study you have been asked to run, in its 2016 form. Read it as a template, then ask: what would I do differently with 10 years of hindsight, an anchor-free base, and modern fusion operators?** That question is your project.

**Why halfway wins (the intuition you must be able to articulate):**
- **Early fusion** (concatenate at the input, 4-channel or 6-channel tensor) forces a single shared filter bank to serve two radically different input distributions. Low-level features (edges, textures) are *modality-specific*; a Gabor-like filter tuned for RGB texture is wrong for thermal. Early fusion also has zero robustness to misalignment — it hard-assumes pixel correspondence.
- **Late fusion** (two full detectors, merge boxes/scores) throws away all cross-modal *feature* interaction. The two streams never help each other see; they only vote at the end. You can't recover a pedestrian that neither stream detected alone but that fusion would have made visible.
- **Middle fusion** keeps low-level feature extraction modality-specific (good — the streams learn their own edge/texture detectors) but merges at a semantic level where both streams have abstracted to "person-ish blob at this location." Semantic-level features are also **more spatially tolerant** — they've been downsampled 8–32×, so a 4-pixel parallax shift at input becomes a sub-pixel shift at `dark5`. **This is why middle fusion is also the misalignment-robust choice, and it's the strongest argument for your architecture given your rig.**

### [MUST] Modality imbalance and illumination-aware gating
**MBNet: "Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems"** — Zhou, Chen, Cao (ECCV 2020).
https://arxiv.org/abs/2008.03043 · code: https://github.com/CalayZhou/MBNet

**Read §3 in full.** Two ideas you will very likely put in your ablation:
- **DMAF (Differential Modality Aware Fusion)** — use the *difference* between the two modalities' features as a signal to modulate each stream. Cheap, clever.
- **Illumination-aware feature alignment** — a small subnet predicts day/night from the RGB image and *gates* how much each modality contributes. This is the "gating" mechanism in your brief.

**Why this matters for your ablation:** the modalities are not equally useful at all times. Thermal dominates at night; RGB dominates in daylight and for small/distant objects. A fusion operator with a **static** weighting (plain sum, plain concat) cannot express this. A **gated/attention** operator can. **The day/night split in your results table is where this will show up — and if you only report the aggregate number, you will hide your own most interesting finding.** Report `All / Day / Night` as three columns. Always.

### [MUST] Misalignment — the one that maps onto your rig
**AR-CNN: "Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection"** — Zhang et al. (ICCV 2019).
https://arxiv.org/abs/1901.02645 · code: https://github.com/luzhang16/AR-CNN

**Read §1, §3.1, §3.2.** Key facts:
- KAIST pairs are **not strictly aligned** — the same pedestrian sits at different positions in RGB vs thermal. Everyone who reports on KAIST is quietly training through this.
- They **re-annotated KAIST with per-modality boxes** → the **KAIST-Paired Annotation**. Use it if you want to *measure* misalignment sensitivity.
- Their **Region Feature Alignment (RFA)** module predicts the shift and aligns region features adaptively.

**This is the single most transferable paper in the whole curriculum for your real hardware.** Your rig has FOV mismatch + parallax + a 60/120fps temporal skew. AR-CNN is the prior art for "what to do when your modalities don't line up." A **deliberate misalignment-robustness ablation** — synthetically shift the thermal image by 0/2/4/8 px and measure MR degradation across your fusion operators — would be a genuinely novel and *useful* contribution that most KAIST papers don't bother with, and it directly de-risks your deployment. **Strongly consider making this a headline row in your study.**

### [MUST] The fusion operator zoo — pick your ablation axes
There is no single paper for this; the design space is what you're being paid to explore. Here is the space, laid out. **Your ablation is a structured walk through these axes.**

**Axis A — WHERE to fuse (the "middle" in middle fusion):**
- after `dark3` (stride 8, high-res, semantically shallow, misalignment-sensitive)
- after `dark4` (stride 16)
- after `dark5` (stride 32, deep, semantically rich, misalignment-tolerant)
- multi-stage (fuse at 3, 4, *and* 5 — more params, more interaction)
- This axis alone is a clean 4-row table and directly reproduces/updates Liu et al. 2016 on a modern base.

**Axis B — HOW to fuse (the operator), roughly in increasing cost:**
| Operator | Params | Notes |
|---|---|---|
| Element-wise **sum** | 0 | The honest zero-param baseline. Assumes aligned, comparably-scaled features. |
| Element-wise **mean/max** | 0 | Max is a crude "trust whichever stream is more excited" |
| **Concat + 1×1 conv** | 2C² | The workhorse. Lets the net *learn* the mixing weights. Usually the one to beat. |
| **Learned scalar weights** (αF_rgb + βF_t) | 2 | Nearly free; tests whether a *global* modality weighting is all you need |
| **Channel attention (SE-style)** on concatenated features | ~2C²/r | Per-channel, input-dependent gating |
| **Illumination-gated** (MBNet-style) | small | Gate conditioned on a day/night subnet |
| **Cross-attention / transformer** (CFT-style) | large | Most expressive, most expensive, most likely to overfit on KAIST's small effective dataset |

**Axis C — backbone weight sharing:** two independent backbones (2× params) vs. fully shared weights (1× params, forces a modality-agnostic feature space — usually bad) vs. shared *late* stages only. **This axis is where parameter-matching gets subtle and where a careless study becomes invalid.**

**Axis D — what feeds the neck:** fused features only, or fused + per-modality features (a "keep the originals" residual path).

**Axis E — modality dropout regularization:** randomly zero one stream during training with probability p. Tests and *builds* robustness to sensor failure. Extremely relevant to a real drone/vehicle (thermal shutter NUC event = ~0.5s of garbage frames, see Track 4). Cheap to add, could be a headline result.

### [MUST] Late fusion done *right* — your strongest control
**ProbEn: "Multimodal Object Detection via Probabilistic Ensembling"** — Chen et al. (ECCV 2022).
https://arxiv.org/abs/2104.02904

**Read §3.** The uncomfortable finding: a **carefully done probabilistic late fusion is startlingly strong** — competitive with or beating many elaborate mid-fusion networks. **Read this paper specifically so that your late-fusion control is a *strong* control and not a strawman.** If you ablate middle fusion against a deliberately weak late-fusion baseline, your study is decorative. This is the difference between an honest result and a paper that gets shredded in review.

### [REF] Transformer-based fusion
**CFT: "Cross-Modality Fusion Transformer for Multispectral Object Detection"** — Qingyun et al.
https://arxiv.org/abs/2111.00273

**Read §3.** A YOLOv5-based cross-attention fusion — architecturally the closest published thing to what you're building (single-stage YOLO base + middle fusion). Good source for the "expensive end" of your operator axis. Note their param/FLOP cost honestly when you compare against it.

### [REF] The literature maps — use these to find prior art, don't read them linearly
- **Multispectral Pedestrian Detection Resource** — https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource — datasets, methods, **annotations**, and eval tools. Your first stop for getting the sanitized/improved annotation files.
- **Awesome-RGBT-Fusion** — https://github.com/yuanmaoxun/Awesome-RGBT-Fusion — broader RGB-T (detection, segmentation, tracking).

**Use these to build your related-work table and to check nobody has already published your exact ablation.** Do this in Week 1, not Week 6.

### [REF] Other RGB-T datasets (for a generalization row)
- **LLVIP** — https://arxiv.org/abs/2108.10831 · https://github.com/bupt-ai-cz/LLVIP — 15,488 **strictly aligned** pairs, low-light, pedestrians. Because it's well-aligned it's an excellent *contrast* to KAIST: **if your fusion method's advantage over baselines shrinks on LLVIP, that tells you your gain was really misalignment-robustness.** That's a genuinely insightful experiment and it costs you one extra training run.
- **FLIR ADAS** — https://oem.flir.com/solutions/automotive/adas-dataset-form/ — free with registration; use the "aligned" community version (~5.1k pairs, person/car/bicycle) if you want a multi-class generalization row.

---

## TRACK 4 — Camera geometry & thermal sensor physics

**Why this track is not optional:** your deployment rig has FOV mismatch, parallax, and a 120fps/60fps temporal skew. Every one of those is a *geometry* problem, and none of them are fixed by better fusion weights. Also: you are going to be asked "why does thermal fail here?" and the answer is physics, not machine learning.

### Part A — Camera geometry

#### [MUST] Camera calibration, properly
**Cyrill Stachniss (Uni Bonn) — Photogrammetry I & II** (free, YouTube)
Playlist: https://www.youtube.com/playlist?list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y
Course page + slides: https://www.ipb.uni-bonn.de/photogrammetry-i-ii/index.html

**Watch, from the playlist:**
- **"Camera Calibration using Zhang's Method"** — https://www.youtube.com/watch?v=-9He7Nu3u8s ← **watch this one first and twice.** This is exactly what you will do to your rig.
- The lectures on **pinhole camera model / intrinsics & extrinsics**, **direct linear transform (DLT)**, and **homography**.

**Do not watch the whole playlist** — it's a full semester of geodesy. Take the four topics above.

**Why Stachniss over everything else:** he is the clearest lecturer on this material alive, the slides are downloadable, and it's a robotics framing (which is your framing). This is the free replacement for Hartley & Zisserman.

#### [MUST] The specific geometry problems your rig has

You need to be able to state these precisely. Study until you can:

1. **FOV mismatch.** Your RGB and thermal have different focal lengths/sensor sizes → different fields of view. Resolving this is a **resampling/cropping** problem: compute the overlapping frustum, warp one to the other's grid. **Consequence for training: you must decide whether to train on the intersection (lose FOV) or pad (introduce invalid regions). Say which one you did, in the paper.**

2. **Parallax.** The two cameras are physically separated by a baseline `b`. The disparity of a point at depth `Z` is `d ≈ f·b/Z`. **This means a single homography can only align the two views for one plane at one depth.** Pedestrians at 5m and buildings at 50m *cannot both* be aligned by one warp. **This is the fundamental reason KAIST is "weakly aligned" and why AR-CNN exists.** It is *not* a calibration bug you can fix — it's geometry. You can only mitigate it (fuse deep, where features are spatially pooled) or model it (learn the shift, AR-CNN-style).
   → **Understanding check:** with your actual baseline and focal length, compute the pixel disparity for a pedestrian at 3m, 10m, and 30m. If the 3m number is >4px, element-wise fusion at stride-8 is already compromised and you have a *quantitative* justification for fusing deeper. Put this calculation in your report. It will impress whoever reads it.

3. **Temporal skew (120fps RGB / 60fps thermal).** Even with perfect spatial calibration, if the two exposures aren't simultaneous, a moving object is displaced by `v · Δt`. At 60Hz thermal and worst-case 8.3ms skew, a person walking at 1.5 m/s moves ~12mm — negligible. **But a vehicle at 15 m/s moves 12.5cm, and your drone's own ego-motion moves the whole scene.** Fixes, in order of preference: hardware sync/trigger → timestamp-nearest pairing → interpolation. **Log which frame-pairing rule you used.** This is the kind of detail that separates a real system from a demo.

#### [MUST] Homography and image warping
**Szeliski, "Computer Vision: Algorithms and Applications", 2nd ed.** — **free PDF** from the author.
https://szeliski.org/Book/ (download: https://szeliski.org/Book/download.php)

**Read only:**
- **§2.1 — Geometric primitives and transformations** (2D/3D transforms, homogeneous coords, the transformation hierarchy)
- **§2.1.4 / §2.2 — Photometric image formation and the camera model** (intrinsics, lens distortion) ← also serves Part B
- **§8.1 — Feature-based alignment / pose estimation** (how you'd actually estimate the RGB↔thermal warp)
- **§11 — Stereo correspondence** — **skim §11.1 (epipolar geometry) only**, for the parallax/disparity relation.

That's ~60 pages out of 1200. **This is your geometry reference book; keep it open, don't read it linearly.**

#### [REF] Multiple View Geometry — the bible, mostly paid
**Hartley & Zisserman, "Multiple View Geometry in Computer Vision," 2nd ed.** (Cambridge, 2004)
https://www.robots.ox.ac.uk/~vgg/hzbook/ — **free sample chapters** (Introduction, "Epipolar Geometry and the Fundamental Matrix", The Trifocal Tensor), plus MATLAB code and errata. The full book is paid.

**Verdict: you do not need this for PRISM.** It is the definitive reference and worth owning eventually, but for a two-camera rigidly-mounted rig, Szeliski §2 + Stachniss's calibration lecture cover you completely. Grab the **free epipolar geometry chapter** if you want the parallax story in full rigour. Tagged [DEPTH] for your career, [REF] for this project.

### Part B — Thermal imaging physics

**This is the part almost nobody on a CV team actually knows, and it's where you can become the person in the room who does.**

#### [MUST] The fundamentals of IR thermal imaging
**Vollmer & Möllmann, "Infrared Thermal Imaging: Fundamentals, Research and Applications," 2nd ed. (Wiley-VCH, 2018) — Chapter 1, "Fundamentals of Infrared Thermal Imaging."**
**Free legal sample chapter (full Ch. 1, ~60pp):** https://application.wiley-vch.de/books/sample/3527413510_c01.pdf
(Book page, if you ever want the rest: https://www.wiley.com/en-us/infrared-thermal-imaging-fundamentals-research-and-applications-2nd-edition-p-9783527413515)

**Read Chapter 1 in full — it's free and it's the whole physics foundation.** It covers, and you must come away able to explain:
- **Blackbody radiation, Planck's law, Wien's displacement law** — *why* room-temperature objects (~300K) peak around 10µm, which is *why* pedestrian detection uses **LWIR (8–14µm)** and not MWIR or SWIR. A human body is a ~310K emitter; that's the entire physical basis of your project.
- **Emissivity (ε).** A thermal camera does not measure temperature — it measures **radiance**, and infers apparent temperature *assuming* an emissivity. Human skin is ε≈0.98 (nearly a blackbody — which is why people show up so well). Polished metal is ε≈0.1 and is highly reflective in LWIR — **a metal surface in a thermal image mostly shows you a reflection of the surroundings, not its own temperature.** This is why cars look weird in thermal.
- **Atmospheric transmission windows.** The 8–14µm window exists because that's where the atmosphere doesn't absorb. Humidity, rain, and fog *do* attenuate LWIR, degrading range — relevant to your drone's operating envelope.
- **The external influences diagram (Fig 1.2 in the chapter):** what the camera sees = object emission + reflected ambient + atmospheric emission + optics emission. All four are real, and the last two are why thermal cameras need internal calibration.

**The single most important consequence for you: GLASS IS OPAQUE TO LWIR.** Ordinary glass transmits visible light and blocks 8–14µm almost completely. Therefore:
- You **cannot see a person through a car windshield or a building window** in thermal — you see the glass's own surface temperature.
- Thermal lenses cannot be made of glass — they're **germanium** (which is why thermal optics are expensive, and why they're opaque-looking and gold-ish).
- **In an RGB+T fusion system, this is a genuine, principled case where RGB carries information thermal *physically cannot*.** It is one of the best answers you can give to "why fuse at all, why not just use thermal at night?" **Put this in your report's motivation section.**

**Concrete understanding check:** be able to answer, from physics, in one paragraph each: (a) Why do pedestrians sometimes *disappear* in thermal on a hot day? [thermal crossover — when ambient ≈ skin temperature, contrast → 0. Directly explains why your Day column will be worse than your Night column.] (b) Why does a person behind a window vanish in thermal but not RGB? (c) Why is a car's windshield hot-looking but its polished hood ambiguous?

#### [MUST] NUC, FFC, and fixed-pattern noise — the *engineering* reality
**Teledyne FLIR Lepton Engineering Datasheet** (free)
https://cdn.sparkfun.com/assets/f/6/3/4/c/Lepton_Engineering_Datasheet_Rev200.pdf

**Read the sections on the VOx microbolometer array, non-uniformity correction (NUC), and flat-field correction (FFC).** Even if you're not using a Lepton, this datasheet is the clearest free explanation of the practical pipeline. (The FLIR Boson datasheet covers the same ground if you're on a Boson.)

**What you must take away:**
- An **uncooled microbolometer** works by each pixel *heating up* and changing resistance. Every pixel has a slightly different response curve → **fixed-pattern noise (FPN)**: a static, structured, non-random noise pattern overlaid on every frame.
- **NUC** applies per-pixel gain/offset correction terms to cancel FPN. **FFC** (the "shutter click") periodically closes an internal shutter to re-derive the offset term against a uniform thermal reference.
- **This has three hard consequences for your project:**
  1. **FFC blanks the sensor for ~0.5s at unpredictable intervals.** On a 60fps thermal stream that's ~30 dead/garbage frames. Your fusion model *will* see them at deploy time. **→ This is the deployment-side argument for the modality-dropout ablation (Axis E). Train for the failure you know is coming.**
  2. **Residual FPN drifts with sensor temperature.** A model trained on a cold sensor sees different noise than one deployed on a hot one. If you augment, consider adding synthetic FPN/row-noise as a thermal-specific augmentation — **and note that this is a thermal-only augmentation, another reason the "photometric aug on RGB only" rule needs a thermal-specific counterpart.**
  3. **Thermal is 14/16-bit raw radiometric, not 8-bit.** How you normalize it into the network (global fixed range? per-frame min/max? AGC?) is a **real preprocessing decision that will change your results** — per-frame min/max normalization makes the network's input statistics depend on whether a hot object is in frame, which is a subtle train/test distribution bug. **This deserves its own ablation row and almost nobody reports it.**

#### [REF] Radiometry, from a computer vision angle
**Shree Nayar (Columbia) — "First Principles of Computer Vision"** (free)
Site: https://fpcv.cs.columbia.edu/ · YouTube: https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw
**Radiometry and Reflectance monograph (free PDF):** https://cave.cs.columbia.edu/Statics/monographs/Radiometry%20and%20Reflectance%20FPCV-3-1.pdf

**Watch the "Radiometry and Reflectance" module** (start: https://www.youtube.com/watch?v=tflz0loWhIY). This is the CV-side complement to Vollmer's physics-side: radiance vs irradiance, BRDF, why surfaces look the way they do. **It gives you the vocabulary to say precisely what "emissivity" is doing to your images.** Nayar's lectures are exceptionally well-made. Also excellent on the image-formation/optics modules if you want them.

---

## TRACK 5 — Empirical research methodology (how to run an ablation that isn't decorative)

**Read this track before you run a single experiment.** This is the track that turns "I trained some models" into research. It is also the track most engineers skip, which is precisely why it's your edge.

### [MUST] What makes research dishonest (usually without meaning to)
**"Troubling Trends in Machine Learning Scholarship"** — Lipton & Steinhardt (ICML 2018 debates).
https://arxiv.org/abs/1807.03341

**Read all of it — 15 pages, and read it slowly.** The four failure modes:
1. Failure to distinguish **explanation from speculation**
2. Failure to **identify the sources of empirical gains** ← **this is the whole reason your project is an ablation study**
3. **Mathiness** — equations used to impress rather than clarify
4. **Misuse of language** — suggestive terms that overclaim

**§3.2 ("Failure to identify the sources of empirical gains") is your project's mission statement.** The paper's canonical example: a model gets +2 points, the authors credit their novel architecture, and someone later shows the gain came entirely from a longer training schedule. **You are being paid specifically to not let this happen.** Every row in your table must isolate exactly one variable.

### [MUST] Seeds, variance, and your noise floor
**"Torch.manual_seed(3407) is all you need: On the influence of random seeds..."** — David Picard.
https://arxiv.org/abs/2109.08203

**Read it (the title is a joke; the paper is not).** Picard scanned up to 10,000 seeds on CIFAR-10. The finding: even when variance is modest, **it is surprisingly easy to find an outlier seed that looks much better or much worse than average.** A single-seed comparison is a coin flip dressed as a result.

**"Accounting for Variance in Machine Learning Benchmarks"** — Bouthillier et al. (MLSys 2021).
https://arxiv.org/abs/2103.03098

**Read §2 and §3.** Sources of variance: data sampling, data augmentation, weight init, hyperparameter choice. They show how to budget your compute across seeds vs configs — genuinely useful when you have limited GPU hours (you do).

**The operational rule for PRISM — do this in Week 3, before any ablation:**
> Train your **baseline** configuration **3–5 times** with different seeds, changing nothing else. Compute the **standard deviation of your MR**. **That number is your noise floor.** Any ablation delta smaller than ~2σ is not a result — it's weather. Report **mean ± std** for every row, and state your seed count. If you cannot afford 3 seeds per row for all rows, run 3+ seeds on the *baseline* and on your *best* config, and be explicit that the middle rows are single-seed.

**If you do only one thing from this entire curriculum, do this.** It is the cheapest possible way to avoid publishing a fiction, and it's the first thing a good reviewer checks.

### [MUST] The gold-standard ablation to imitate
**ATSS** (already listed in Track 1) — https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.pdf

**Re-read §3 with methodology eyes.** Watch how they:
- Take two methods that differ in *many* ways
- Systematically **equalize every difference one at a time** ("we add the improvements of FCOS to RetinaNet one by one")
- Show the gap closes to ~0 → **therefore** the remaining difference (label assignment) is the *real* cause
- Only *then* propose their method

**That structure — equalize, isolate, attribute, then propose — is exactly the structure your PRISM report should have.** Copy it shamelessly.

### [MUST] The rules for *your* ablation. Print this and stick it on your monitor.

**1. Controls (non-negotiable — you need all four, and they must be strong):**
- **RGB-only YOLOX** (same backbone, same schedule)
- **Thermal-only YOLOX** (same backbone, same schedule)
- **Early fusion** (channel-concat at input)
- **Late fusion** (two detectors + principled box/score merge — **use ProbEn, not a strawman**)

Middle fusion must beat **all four** before you claim anything. If thermal-only beats your fusion model, you have a bug (or a genuinely interesting negative result — but check for the bug first).

**2. Matched budgets — the thing that invalidates most fusion papers:**
A two-stream backbone has **~2× the backbone parameters** of the RGB-only baseline. **Your fusion model beating the single-stream baseline is therefore not, by itself, evidence that fusion works — it might just be evidence that 2× params works.** You must control for this. Options:
- Compare against a **width-scaled single-stream** model with matched params/FLOPs (e.g. RGB-only YOLOX-M vs. fusion of two YOLOX-S backbones)
- Report **params and GFLOPs in a column of every table row**, always. No exceptions.
- When comparing *fusion operators* to each other, hold the backbones fixed so only the operator's params vary — and report that delta explicitly.

**3. Change one thing per row.** If a row changes both the fusion stage *and* the operator, it tells you nothing about either.

**4. Fix the training schedule across all rows.** Same epochs, same LR schedule, same augmentation, same input resolution, same warmup, same aug-off-for-last-15-epochs. **A "better" result from an accidentally longer schedule is Lipton & Steinhardt's §3.2 in the flesh.**

**5. Fix the data first, then never touch it.** Decide your annotation set (sanitized train / improved test), your train/test frame sampling, and your image size **in Week 3, before any ablation row runs.** Changing data mid-study invalidates every row you already ran.

**6. Report Day / Night / All separately.** Aggregate-only reporting hides the single most interesting effect in RGB-T fusion (see MBNet). This is where gating earns its keep — or fails to.

**7. State negative results.** "Cross-attention fusion cost 3× the FLOPs and did not beat concat+1×1" is a **genuinely valuable finding** and a sign of an honest study. Report it with the same prominence as a win. The field is full of papers that quietly dropped the rows that didn't work.

### [MUST] The metric — get this exactly right or nothing else matters
**KAIST benchmark** — Hwang et al., "Multispectral Pedestrian Detection: Benchmark Dataset and Baselines" (CVPR 2015)
Project: https://soonminhwang.github.io/rgbt-ped-detection/ · Code/eval: https://github.com/SoonminHwang/rgbt-ped-detection
Paper: https://openaccess.thecvf.com/content_cvpr_2015/html/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.html

**Read the paper's evaluation section, and then read the evaluation code.** The rules:
- **The metric is log-average miss rate (MR, "MR⁻²"): LOWER IS BETTER.** It is computed by averaging miss rate at 9 FPPI (false positives per image) points log-spaced in [10⁻², 10⁰]. It is **not** mAP. Reporting only mAP makes you incomparable to the entire literature.
- **Evaluate on the `reasonable` subset** (typically: height ≥ 55px, no/partial occlusion) as the headline number. The `all`, `scale`, and `occlusion` subsets are secondary.
- **Report All / Day / Night.** Three columns. Every time.
- **Dataset is 95k aligned color-thermal pairs (640×480, 20Hz)** captured from a vehicle, ~103k annotations, 1,182 unique pedestrians.

**Two traps that will silently ruin your numbers:**
- **Frame correlation.** KAIST is *video* sampled at 20Hz. Consecutive frames are nearly identical. If you sample train/test naively you get near-duplicate leakage and inflated performance. **The community convention is to sample every Nth frame** (commonly every 2nd for training, every 20th for testing). **Follow the convention used by MBNet/MSDS-RCNN so your numbers are comparable — and state your sampling rule explicitly in the report.**
- **1,182 unique pedestrians** across 95k frames means the *effective* dataset is far smaller than the frame count suggests. **This is why high-capacity fusion (cross-attention) may overfit, and it's a real hypothesis your ablation can test.**

### [MUST] Fix the annotations before you ablate
**MSDS-RCNN: "Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation"** — Li et al. (BMVC 2018).
https://arxiv.org/abs/1808.04818 · project: https://li-chengyang.github.io/home/MSDS-RCNN/

**Read the section on the sanitized training annotations.** The headline number, again, because it governs your whole project:

> **Training on sanitized annotations improved miss rate from 11.63% → 7.45% MR. Annotation noise accounted for roughly a third of the inference error.**

**Therefore, before running a single ablation row, you must lock down:**
- **Training annotations:** the **sanitized** set (Li et al.)
- **Test annotations:** the **improved/reasonable** test annotations (Liu et al.)
- Both are linked from https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource

**And state the choice loudly in your report.** Papers using original vs sanitized annotations are **not comparable**, and half the confusion in this subfield comes from people not saying which they used. Being explicit here is free credibility.

### [MUST] Experiment tracking — immutable local artifacts

The corrected study intentionally has no hosted-tracker dependency. The
append-only ledger, checkpoint metadata, raw JSONL histories, frozen manifests,
and hashed diagnostic artifacts are the authoritative record.

**What to log, from run #1:**
- The **full config** (fusion stage, operator, seed, LR, schedule, annotation set, sampling rule, image size)
- **The git commit hash.** Every run. Non-negotiable — otherwise Week-6 you cannot reproduce Week-3's result.
- **Params and GFLOPs**, as logged metrics, so they appear as table columns automatically
- **MR (reasonable/day/night)**, mAP, and the full MR-FPPI curve
- **Latency** on the target device (once you have Track 6 running)
- A **qualitative image panel** — predictions on the same 8 fixed validation images, every run. You will catch bugs by eye that no metric will show you.

Do not use a hyperparameter-sweep tool for architectural ablations. The rows are
deliberate, controlled interventions executed by `scripts/run_experiments.py`.

### [REF] Reading & writing a results table
There's no single paper for this. The rules:
- **Columns:** Method | Fusion stage | Fusion op | Params (M) | GFLOPs | **MR-All ↓** | **MR-Day ↓** | **MR-Night ↓** | mAP ↑ | Latency (ms)
- Mark the **arrow direction** on every metric (↓ for MR). Readers *will* misread MR as "higher is better" if you don't.
- **Bold the best, underline the second best.** Standard convention.
- Report **mean ± std** and **state the seed count in the caption.**
- Group rows into blocks with rules between them: *controls* | *fusion stage sweep* | *fusion operator sweep* | *final model*. A table that isn't grouped is unreadable.
- **Every claim in your prose must point at a specific row.** If a sentence in your abstract has no row behind it, delete the sentence.

---

## TRACK 6 — Deployment & efficiency

**Why:** the point of a YOLOX base (rather than a two-stage detector) is real-time deployment. A fusion module that doubles latency has a cost, and that cost belongs in the results table. **Do this track last — but do not skip it, because "which fusion operator is worth its latency" is a genuinely important research question and it's one your ablation is uniquely positioned to answer.**

### [MUST] FLOPs ≠ latency. Internalize this.
There is no paper to read; there's a measurement to make.

**The rule:** FLOPs measure arithmetic. Latency is often dominated by **memory bandwidth** and **kernel launch overhead**, not arithmetic. Consequences you *will* hit:
- **Concatenation is 0 FLOPs but is not free** — it allocates and copies a large tensor. Fusing at stride-8 (high resolution) means copying a big feature map.
- **Element-wise sum is 0 params and ~0 FLOPs and is genuinely nearly free.** This is a real advantage and it belongs in the discussion.
- **Attention over a stride-8 feature map is brutally expensive** (quadratic in spatial size, and memory-bound). Over stride-32 it's cheap. **This interacts directly with your Axis A (where to fuse) — the same operator has a completely different cost at different stages.** That interaction is a genuine finding and a great figure: **plot MR vs latency for every (stage, operator) pair and draw the Pareto front.** That single figure could be the best thing in your report.
- **A two-stream backbone roughly doubles backbone latency**, and that's usually the dominant cost — far more than any fusion operator. **Measure it, and say it.** Many fusion papers agonize over a cheap operator while ignoring that they doubled the backbone.

**Always report latency measured on the target device, batch size 1, with warmup, averaged over 100+ runs.** Never report latency from a desktop GPU as if it means anything for a Jetson.

### [MUST] TensorRT + INT8
**NVIDIA TensorRT documentation** — https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
**Quantization schemes** — https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html

**Read:** the **quantization / working-with-quantized-types** section, and the **`trtexec`** tooling docs.

**The workflow:** PyTorch → ONNX export → TensorRT engine → benchmark with `trtexec`.

**What you must understand about INT8 PTQ (post-training quantization):**
- It needs a **calibration dataset** — a representative sample of real inputs — from which TensorRT estimates the activation range (scale) per tensor.
- **The multimodal wrinkle nobody warns you about: your calibration set must contain BOTH modalities, in realistic proportion, including night frames.** Calibrating only on daytime RGB-dominant data will pick activation ranges that clip your thermal stream's dynamic range at night. **This is a genuinely subtle bug and a nice thing to mention in your report.**
- **Calibrate on the device you deploy on.** Results vary across devices.
- Note YOLOX ships ONNX/TensorRT/OpenVINO deployment examples in its repo — **start from those, don't write the export from scratch.**

**Benchmark command pattern (from the NVIDIA docs/forums):**
`/usr/src/tensorrt/bin/trtexec --loadEngine=model.engine --batch=1 --avgRuns=100 --warmUp=500`

### [MUST] The efficiency table row
Every ablation row gets: **Params (M) · GFLOPs · FP16 latency (ms) · INT8 latency (ms) · MR after INT8**.

**That last column is the one that matters and the one nobody reports.** A fusion operator that wins in FP32 but collapses under INT8 quantization is a *bad operator for a drone*. Attention-based fusion with large activation dynamic range is the prime suspect here. **If you find that concat+1×1 survives INT8 better than cross-attention, that is a real, useful, publishable, deployment-relevant finding** — and it's the kind of result that only comes from someone who does both the research and the deployment. **That's the professional you're trying to become. This row is your differentiator.**

---

## THE 6-WEEK SCHEDULE

**Assumed budget: ~15 h/week** (~2h/weekday + ~5h weekend). Split roughly **40% study / 60% build**, front-loaded toward study. This is a real budget for a student with a startup job — it is deliberately not 40h/week.

**Theory gates are hard gates (R2).** Do not start a phase's build until its gate is passed.

---

### WEEK 1 — Fundamentals + data reality
**Theme: understand the base detector and discover that your dataset is a minefield.**

**Study (~9h)**
- YOLOX paper, all of it, twice **[MUST]** (2h)
- Read the 5 YOLOX source files **[MUST]** (3h) — `yolox.py`, `darknet.py`, `yolo_pafpn.py`, `yolo_head.py`, `losses.py`
- FCOS §3.1–3.2 + FPN §3 + PANet §3.1 **[MUST/REF]** (1.5h)
- OTA paper §1, §3 **[MUST]** (1.5h)
- Skim the two literature-map repos; note every paper that has already done a fusion-stage ablation **[REF]** (1h)

**Build (~6h)**
- Environment: YOLOX running, COCO-pretrained weights, reproduce the reported YOLOX-S COCO number on a small val subset. **You must be able to trust your baseline before you modify it.**
- Download KAIST. Write a dataloader that returns **aligned (RGB, thermal, boxes)** triplets.
- **Visualize 50 random pairs with boxes overlaid on both modalities.** Look at them. Actually look.

**✅ Gate / understanding check (all three must pass):**
1. Draw, from memory, the YOLOX-S tensor shapes at `dark3/4/5` for 640×640 input, and mark where fusion attaches.
2. Explain in writing (½ page) what `dynamic_k_matching` does and what happens to it if a fusion module shifts the feature scale at the head input.
3. **Show me 5 KAIST pairs where the RGB and thermal boxes visibly do not line up.** If you can't find any, you're not looking hard enough — and that's the point of the exercise.

---

### WEEK 2 — Fusion theory + geometry/physics
**Theme: know why middle fusion wins, and know what your sensors physically are.**

**Study (~10h)**
- Liu et al. **Halfway Fusion** (BMVC 2016) — all of it **[MUST]** (1.5h)
- **MBNet** §3 **[MUST]** (1.5h)
- **AR-CNN** §1, §3.1–3.2 **[MUST]** (1.5h)
- **ProbEn** §3 **[MUST]** (1h) — so your late-fusion control is strong
- **Vollmer & Möllmann Ch. 1** (free PDF) **[MUST]** (2.5h) — the whole chapter
- Stachniss: **Zhang's calibration method** + pinhole/intrinsics/homography lectures **[MUST]** (2h)

**Build (~5h)**
- **Lock the data.** Get the **sanitized training** annotations and **improved
  test** annotations. The corrected protocol and exact sources are frozen in
  `../data/README.md`; do not substitute another annotation/frame convention.
- Implement and verify the **augmentation correspondence unit test**: assert that a geometric aug maps a known RGB pixel and its thermal counterpart to the same location; assert photometric augs touch RGB only.
- Compute the **parallax table** for your real rig: pixel disparity at 3m / 10m / 30m given your baseline and focal length.

**✅ Gate / understanding check:**
1. **One page, written:** "Why middle fusion?" — argue it from (a) modality-specific low-level features, (b) spatial tolerance to misalignment at low resolution, (c) preserved cross-modal feature interaction that late fusion discards. Cite Liu et al. **This page becomes your report's motivation section — write it properly.**
2. Answer the three thermal-physics questions from Track 4B (thermal crossover / glass opacity / metal emissivity) from physics, not hand-waving.
3. Show the parallax table. **State the fusion stage your own numbers justify.**

---

### WEEK 3 — Methodology + baselines + the noise floor
**Theme: build the controls and find out how much of your future result is going to be noise. This is the most important week.**

**Study (~6h)**
- **Lipton & Steinhardt**, all of it, slowly **[MUST]** (2h)
- **Picard (seeds)** + **Bouthillier §2–3** **[MUST]** (1.5h)
- **ATSS §3**, re-read with methodology eyes **[MUST]** (1h)
- **KAIST eval protocol** — read the evaluation *code*, not just the paper **[MUST]** (1.5h)
- Read the ledger and manifest contracts **[MUST]** (0.5h)

**Build (~9h)**
- Implement the **KAIST MR⁻² evaluator** and **verify it reproduces a published number** for a known method. **If your evaluator is wrong, every week after this is wasted. Do not skip the verification.**
- Train the **four controls**: RGB-only, Thermal-only, Early fusion, Late fusion (ProbEn-style merge).
- **Train the baseline 3–5× with different seeds. Compute σ of MR.**
- Verify local config/source hashes, params/MACs, MR-all/day/night, and fixed
  diagnostic artifacts are recorded.

**✅ Gate / understanding check:**
1. **State your noise floor: "baseline MR = X ± σ over N seeds."** Write σ on a sticky note and put it on your monitor. **Every future claim must clear 2σ.**
2. Your evaluator reproduces a published KAIST MR to within a fraction of a point. Show the comparison.
3. Sanity check that must pass: **thermal-only beats RGB-only at night; RGB-only beats or matches thermal-only in day.** If this doesn't hold, you have a data bug (very likely a normalization bug on the thermal stream). Fix it before proceeding.

---

### WEEK 4 — The fusion-stage ablation (Axis A)
**Theme: the modern re-run of Liu et al. 2016 — where should we fuse?**

**Study (~4h)**
- Re-read Prince **Ch. 11** (BatchNorm/residual) **[REF]** (1.5h) — you're about to hit two-stream normalization problems and you want the theory fresh
- Karpathy **Lecture 3** (activations, gradients, BatchNorm) **[MUST]** (2h) — watch at 1.25×, do the histogram exercise

**Build (~11h)**
- Implement the two-stream backbone with a **swappable fusion point**: after `dark3` / `dark4` / `dark5` / multi-stage. Use the simplest operator (concat+1×1) and hold it fixed — **one variable per row.**
- Train all 4 stage variants. **Same schedule, same aug, same everything.**
- **Instrument activation statistics at the fusion junction for both streams. Plot the histograms.**

**✅ Gate / understanding check:**
1. **The 4-row stage table**, with Params/GFLOPs columns, MR-all/day/night, and your σ marked. **Which rows actually clear 2σ?** Say so explicitly — including "none of them do," if that's the truth.
2. The activation histograms at the fusion point for RGB vs thermal streams. **Do they match in scale?** If they don't, explain what you did about it.
3. Compare your result to Liu et al. 2016. **Does halfway still win on an anchor-free base in 2026?** Whatever the answer, that's a real finding — and if it *doesn't* replicate, that is the more interesting result and you should chase it.

---

### WEEK 5 — The fusion-operator ablation (Axes B–E) + misalignment
**Theme: what to fuse with, and does it survive reality.**

**Study (~4h)**
- **CFT** §3 **[REF]** (1h) — the expensive end of the operator axis
- Re-read **MBNet** §3 on gating **[MUST]** (1h) — you're implementing it now
- Szeliski **§2.1** + **§8.1** **[MUST]** (2h) — you're about to warp images deliberately

**Build (~11h)**
- Fix the fusion stage to Week 4's winner. Now sweep the **operator**: sum · concat+1×1 · learned-scalar-weight · SE/channel-attention · illumination-gated · cross-attention.
- **Parameter-matched control:** a width-scaled single-stream model with params/FLOPs matched to your fusion model. **This is the row that makes your study valid.**
- **The misalignment experiment (your differentiator):** synthetically shift the thermal image by **0 / 2 / 4 / 8 px** at test time and measure MR degradation **for each fusion operator.** Which operators are robust? **This is the result that transfers to your real parallax-afflicted rig.**
- **Modality dropout (Axis E):** train with random single-stream dropout, evaluate under simulated thermal FFC blackout.

**✅ Gate / understanding check:**
1. **The operator table**, param/FLOP-matched, mean±std, day/night split, with the width-scaled single-stream control included. **Does middle fusion beat a parameter-matched single-stream model? Answer honestly.** If it doesn't, that's the finding, and it's a good one.
2. **The misalignment degradation curve** — MR vs shift, one line per operator. Explain the ranking *from geometry* (which operators assume pixel correspondence? which pool it away?).
3. **One paragraph of negative results.** What didn't work, and — importantly — is it "didn't work" or "didn't clear 2σ"? **Those are different claims. Know which one you're making.**

---

### WEEK 6 — Deployment, Pareto front, and the write-up
**Theme: turn experiments into a result someone can act on.**

**Study (~4h)**
- **TensorRT** quantization docs + `trtexec` **[MUST]** (2h)
- Re-read **Lipton & Steinhardt §3.2** **[MUST]** (0.5h) — read it again *while writing*, as a checklist against your own draft
- Results-table conventions (Track 5) **[REF]** (0.5h)
- Skim **LLVIP** paper **[REF]** (1h) — for the generalization row

**Build (~11h)**
- Export best model + baselines: PyTorch → ONNX → TensorRT. **FP16 and INT8.** Calibration set **must include night frames and both modalities in realistic proportion.**
- Benchmark on the **Jetson**, batch 1, warmup, ≥100 runs. Fill the **latency + MR-after-INT8** columns for every operator.
- **The Pareto figure: MR vs latency for every (stage, operator) pair, Pareto front drawn.** This is your headline figure.
- *(Stretch, if the week allows)* One **LLVIP** run — does the fusion advantage shrink on a *strictly aligned* dataset? If yes, **your gain was misalignment-robustness, and that's a sharper, more honest claim than "our fusion is better."**
- **Write the report.** Structure it like ATSS: equalize → isolate → attribute → propose.

**✅ Final gate — you are done when:**
1. **The full results table** exists: grouped blocks (controls / stage sweep / operator sweep / final), Params · GFLOPs · MR-all/day/night ± std · mAP · FP16 ms · INT8 ms · MR-after-INT8, arrows marked, best bolded, seed count in the caption.
2. **The Pareto figure** exists and you can defend every point on the front.
3. **Every claim in your abstract points at a specific table row.** Delete any claim that can't.
4. **You can state, in one sentence each:** what you changed, what it bought you, **what it cost in latency**, and **how confident you are relative to seed noise.**
5. **Your negative results are in the paper**, with the same prominence as the positive ones.

---

## The MUST-DO list, in one place (14 items)

If everything goes wrong and you have a fraction of the time, do exactly these.

| # | Item | Track | Why |
|---|---|---|---|
| 1 | **YOLOX paper** — https://arxiv.org/abs/2107.08430 | 1 | Your base detector |
| 2 | **YOLOX source** (5 files) — https://github.com/Megvii-BaseDetection/YOLOX | 1 | The code is the lesson |
| 3 | **OTA / SimOTA** — https://arxiv.org/abs/2103.14259 | 1 | Label assignment will bite you |
| 4 | **FCOS §3** — https://arxiv.org/abs/1904.01355 | 1 | Anchor-free, clearly explained |
| 5 | **Karpathy Lec 3** (BatchNorm/activations) — https://karpathy.ai/zero-to-hero.html | 2 | Two-stream normalization is *the* silent killer |
| 6 | **Halfway Fusion** — https://arxiv.org/abs/1611.02644 | 3 | Your project's premise, in its 2016 form |
| 7 | **MBNet** — https://arxiv.org/abs/2008.03043 | 3 | Gating + modality imbalance |
| 8 | **AR-CNN** — https://arxiv.org/abs/1901.02645 | 3 | Misalignment — maps straight onto your rig |
| 9 | **ProbEn** — https://arxiv.org/abs/2104.02904 | 3 | So your late-fusion control isn't a strawman |
| 10 | **Vollmer & Möllmann Ch. 1** (free) — https://application.wiley-vch.de/books/sample/3527413510_c01.pdf | 4 | Thermal physics: emissivity, LWIR, **glass** |
| 11 | **Stachniss — Zhang's calibration** — https://www.youtube.com/watch?v=-9He7Nu3u8s | 4 | You will calibrate this rig |
| 12 | **Lipton & Steinhardt** — https://arxiv.org/abs/1807.03341 | 5 | "Failure to identify sources of empirical gains" = your job |
| 13 | **Picard, seeds** — https://arxiv.org/abs/2109.08203 | 5 | **Measure your noise floor before believing anything** |
| 14 | **MSDS-RCNN sanitized annotations** — https://arxiv.org/abs/1808.04818 | 5 | Bad labels ≫ your fusion gains. Fix data first. |

---

## What "professional in this space" actually means

Finishing the task means you produce a table. Being a professional means you can be handed a *new* multimodal detection problem and know, immediately:

- **What the metric should be, and why the obvious one is wrong.**
- That the **dataset's labels are probably broken**, and how to check.
- **What the sensors physically can and cannot see** — and that the answer comes from radiometry, not from a validation curve.
- That your architecture must be **matched against a parameter-equalized control** or your result means nothing.
- That a **result inside seed noise is not a result.**
- That **latency on the target device** is part of the result, not an afterthought.
- **When to report that your idea didn't work** — and that doing so makes you more credible, not less.

That list is what this curriculum is actually teaching. The KAIST table is just the exercise.

---

*All resources above were verified to exist and be accessible as described. Free/open-access items are marked with their source; the two paid books (Hartley & Zisserman, Vollmer & Möllmann full text) are tagged [REF]/[DEPTH] only, and free substitutes are given for both — Szeliski + Stachniss for geometry, and Vollmer's free Chapter 1 + the FLIR datasheet for thermal.*
