# Concepts to learn — the model-building phase

In build order. Each concept is needed for the code you type right after it.
**[core]** = you must understand it. **[skim]** = know what it is, move on.

Search terms are given verbatim — they're what actually returns good results.

---

## Group 1 — The blocks (`blocks.py`, typing now)

1. **[core] Convolution: kernel, stride, padding** — how a conv changes the spatial size.
   → search: `convolution output size formula stride padding`
   *Why:* stride is the single most important number in this project. Stride 2 halves the
   image; everything else preserves it. That's how the backbone controls scale.

2. **[core] Batch normalization** — why every conv is followed by one.
   → search: `batch normalization explained neural networks`
   *Why:* our two streams (RGB, thermal) have very different pixel statistics. BatchNorm
   behaviour at the fusion point is the #1 cause of "fusion is worse than single-stream."

3. **[skim] SiLU / Swish activation**
   → search: `SiLU swish activation function`

4. **[core] Residual / skip connections**
   → search: `residual connection ResNet skip connection explained`
   *Why:* the `Bottleneck` block uses one. Without it, deep stacks don't train.

5. **[skim] Bottleneck block (1x1 squeeze -> 3x3 -> expand)**
   → search: `bottleneck block 1x1 convolution channel reduction`

6. **[core] CSP — Cross Stage Partial**
   → search: `CSPNet cross stage partial network explained`
   *Why:* this is `CSPLayer`, the block CSPDarknet is mostly made of. Split channels, process
   half, concatenate. Same accuracy, ~half the compute.

7. **[skim] SPP — Spatial Pyramid Pooling**
   → search: `spatial pyramid pooling SPP object detection`

---

## Group 2 — The backbone (`darknet.py`, next)

8. **[core] Feature maps, stride, receptive field**
   → search: `feature map stride receptive field CNN explained`
   *Why:* this is the stride table in textbook Ch.4, and the entire misalignment argument.

9. **[core] Why a backbone outputs THREE scales**
   → search: `multi scale feature extraction object detection backbone`
   *Why:* small pedestrians are found at stride 8, large ones at stride 32.

10. **[skim] YOLOX Focus / stem layer**
    → search: `YOLOX focus layer stem slice`

---

## Group 3 — The neck

11. **[core] FPN — Feature Pyramid Network**
    → search: `Feature Pyramid Network FPN explained object detection`
    *Why:* top-down path — semantic meaning flows down to high-resolution maps.

12. **[core] PAN / PAFPN — Path Aggregation**
    → search: `PANet path aggregation network bottom-up path`
    *Why:* YOLOX's neck is FPN + a bottom-up pass. Location flows back up.

13. **[skim] Lateral connections, upsampling in CNNs**
    → search: `lateral connection upsample feature pyramid`

---

## Group 4 — The head

14. **[core] Anchor-based vs anchor-free**
    → search: `anchor free object detection explained FCOS`
    *Why:* YOLOX is anchor-free — predicts boxes directly. Big simplification for us: no
    anchor tuning when fusion changes feature statistics.

15. **[core] Decoupled head**
    → search: `YOLOX decoupled head classification regression branch`
    *Why:* separate branches for class / objectness / box. Why YOLOX converges fast.

16. **[skim] Objectness score**
    → search: `objectness score object detection`

17. **[core] NMS — Non-Maximum Suppression**
    → search: `non maximum suppression NMS explained`
    *Why:* removes duplicate boxes. You already met it in the evaluator's one-to-one matching.

---

## Group 5 — Training (we port this, but understand it)

18. **[core] Label assignment — the whole idea**
    → search: `label assignment object detection positive negative samples`
    *Why:* deciding which of the 6720 grid cells is "responsible" for each pedestrian. Fusion
    changes features, which can change assignment, which changes accuracy for reasons that have
    nothing to do with better features. Know this trap.

19. **[skim] SimOTA / OTA**
    → search: `SimOTA YOLOX label assignment` then `OTA optimal transport assignment detection`

20. **[core] IoU loss / GIoU**
    → search: `GIoU loss explained generalized intersection over union`
    *Why:* plain IoU has zero gradient when boxes don't overlap. GIoU fixes it.

21. **[skim] BCE loss for classification/objectness**
    → search: `binary cross entropy loss explained`

---

## Group 6 — Fusion (the phase after this — preview only)

22. **[core] Early / middle / late fusion**
    → search: `early middle late fusion multimodal deep learning`

23. **[core] Feature fusion operators**
    → search: `feature fusion concatenation element-wise addition attention gating`
    *Why:* this is Axis B of the ablation — the thing the whole project measures.

---

## The honest note

There is **no video** covering YOLOX implementation specifically (it's from 2021; the good
courses predate it). Videos will teach you concepts 1–17. For 18–21, read the official YOLOX
source: `yolox/models/yolo_head.py`, functions `get_assignments` and `dynamic_k_matching`.

Best single video for the *shape* of a from-scratch build: Aladdin Persson, "YOLOv1 from
Scratch". Different architecture, same build sequence as ours.

Textbook Ch.4 covers concepts 8–17 and 22–23 in the exact form this project needs them.
