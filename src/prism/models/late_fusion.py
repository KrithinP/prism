"""Late / decision fusion: merge the *detections* of two independent unimodal
detectors. This is family B11/B12 of the experiment matrix (TODO.md 6.6) -- the
"does fusion earn its place over a simple late-fusion ensemble" control.

M operates on post-NMS per-image detection tensors (K, 6) = x1,y1,x2,y2,score,cls.
union_nms merges by concatenation + one NMS pass; proben_merge associates
boxes into NMS clusters and fuses scores with the Bayes product rule (unmatched
boxes are kept, never silently dropped). This control applies the released-code
uniform-prior product rule to YOLOX confidence scores. Those scores are not
guaranteed calibrated posteriors merely because the training recipes match, so
the result is an empirical late-fusion control, not a claim of Bayesian
calibration.
"""

from __future__ import annotations

import torch
from torchvision.ops import nms


def _xyxy(dets):
    return dets[:, :4]


def _box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between every box in a (N,4) and b (M,4) -> (N, M)."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(2)
    union = _box_area(a)[:, None] + _box_area(b)[None, :] - inter
    return inter / union.clamp(min=1e-9)


def union_nms(dets_rgb: torch.Tensor, dets_thermal: torch.Tensor, iou_thr=0.5) -> torch.Tensor:
    """Union the two detection sets and run one NMS pass over the merged set.
    The merge is order-insensitive to camera: an object found by both cameras
    survives once with the higher confidence."""
    if dets_rgb.numel() == 0:
        return dets_thermal
    if dets_thermal.numel() == 0:
        return dets_rgb
    merged = torch.cat([dets_rgb, dets_thermal], dim=0)
    keep = nms(_xyxy(merged), merged[:, 4], iou_thr)
    return merged[keep]


def proben_merge(
    dets_rgb: torch.Tensor, dets_thermal: torch.Tensor, iou_thr=0.5, box_fusion="avg"
) -> torch.Tensor:
    """Probabilistic ensembling (ProbEn) score + box fusion, per the published
    method: Chen et al., "Multimodal Object Detection via Probabilistic
    Ensembling", ECCV 2022 (arXiv:2104.02904, code: Jamie725/RGBT-detection).

    - Association is an NMS-cluster merge over the concatenated detection set
      (score-descending): each box joins the cluster of the highest-scoring box
      it overlaps by >= iou_thr (the same merge decision torchvision NMS makes,
      so union_nms and proben_merge cannot disagree at the boundary); clusters
      may hold 2+ boxes from either camera. Boxes with no overlap stay singleton
      clusters (missing-modality behavior: a detector must not die because one
      camera failed).
    - Score fusion is the released-code UNIFORM-PRIOR product rule:
          p_fused = prod(p_i) / (prod(p_i) + prod(1 - p_i))
      for the cluster's scores. This is NOT the noisy-OR rule 1 - prod(1 - p_i):
      ProbEn treats disagreeing evidence (0.7 vs 0.3) as ~0.5, noisy-OR as 0.79.
      Scores are clamped to [eps, 1-eps] first (log-space safety); the reference
      relies on sigmoid scores never hitting 0/1 exactly.
      The general Bayes derivation contains a class-prior correction. We follow
      the released implementation and state this deviation explicitly because
      YOLOX's objectness*class score is not a calibrated posterior.
    - Box fusion: 'avg' = unweighted mean of member boxes (the reference's
      avg_bbox_fusion); 'score' = score-weighted mean. The paper's 'var' variant
      needs learned per-box variance heads, which this study does not have.
    - A final NMS pass removes residual chain-case duplicates (a box overlapping
      two clusters' kept boxes can otherwise survive twice).
    """
    if dets_rgb.numel() == 0:
        return dets_thermal
    if dets_thermal.numel() == 0:
        return dets_rgb

    merged = torch.cat([dets_rgb, dets_thermal], dim=0)
    order = torch.argsort(merged[:, 4], descending=True)
    n = merged.shape[0]
    ious = _pairwise_iou(_xyxy(merged), _xyxy(merged))

    pos_of = torch.empty_like(order)  # detection index -> sort position
    pos_of[order] = torch.arange(n, device=order.device)

    cluster_of = torch.full((n,), -1, dtype=torch.long)
    clusters: list[list[int]] = []
    for i in order.tolist():
        if cluster_of[i] != -1:
            continue
        cid = len(clusters)
        clusters.append([])
        # scan every detection at-or-below this one in score order: the
        # representative itself (IoU 1) plus all unclaimed lower boxes it
        # overlaps. Must start at the SORT POSITION, not the index i.
        for q in range(pos_of[i].item(), n):
            jj = order[q].item()
            if cluster_of[jj] == -1 and ious[i, jj] >= iou_thr:
                cluster_of[jj] = cid
                clusters[cid].append(jj)

    eps = 1e-6
    out = []
    for members in clusters:
        m = merged[members]
        # Score fusion in float64: float32's round trip 1 - (1 - eps) is NOT
        # eps (relative error ~1e-2 for eps=1e-6), which breaks the symmetry
        # of the Bayes rule exactly at the clamp boundary. The reference's
        # log-space formulation exists for the same reason.
        p = m[:, 4].double().clamp(eps, 1 - eps)
        fused = p.prod() / (p.prod() + (1 - p).prod())
        if box_fusion == "score":
            w = (m[:, 4] + eps).unsqueeze(1)
            box = (m[:, :4] * w).sum(0) / w.sum()
        elif box_fusion == "avg":
            box = m[:, :4].mean(0)
        else:
            raise ValueError(f"box_fusion must be 'avg' or 'score', got {box_fusion!r}")
        out.append(torch.cat([box, fused[None].float(), m[0, 5:6]]))
    res = torch.stack(out)
    keep = nms(_xyxy(res), res[:, 4], iou_thr)
    return res[keep]
