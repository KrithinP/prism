"""Stage 1 of the MR^-2 evaluator: IoU, and per-frame detection matching.

Given one frame's detections and ground-truth boxes, decide which detections are
true positives, which are false positives, and which are absorbed by ignore
regions (and therefore don't count at all). See textbook Ch.3.
"""

from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in `a` (M,4) and every box in `b` (N,4) -> (M,N).
    Boxes are (x1, y1, x2, y2)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    # intersection corners: broadcast a as (M,1,4) against b as (1,N,4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return (inter / np.clip(union, 1e-9, None)).astype(np.float32)


def match_frame(det_boxes, det_scores, gt_boxes, gt_ignore, iou_thr=0.5):
    """Match one frame's detections to its ground truth.

    Returns:
      labels  : (K,) of 1 (true positive) or 0 (false positive), for the K
                detections that were NOT absorbed by an ignore region, ordered
                high-score first.
      scores  : (K,) the matching detection scores (for the threshold sweep later).
      n_pos   : number of scored (non-ignore) ground-truth boxes in this frame
                -- the denominator for miss rate.
    """
    det_boxes = np.asarray(det_boxes, np.float32).reshape(-1, 4)
    det_scores = np.asarray(det_scores, np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    gt_ignore = np.asarray(gt_ignore, bool).reshape(-1)

    order = np.argsort(-det_scores)  # highest confidence first
    ious = box_iou(det_boxes, gt_boxes)  # (M, N)
    gt_taken = np.zeros(len(gt_boxes), bool)  # each GT can be claimed once

    labels, scores = [], []
    for i in order:
        # best available ground-truth for this detection
        j, best = -1, iou_thr
        for k in range(len(gt_boxes)):
            if gt_taken[k] or ious[i, k] < best:
                continue
            j, best = k, ious[i, k]
        if j == -1:  # matched nothing -> false positive
            labels.append(0)
            scores.append(det_scores[i])
        elif gt_ignore[j]:  # matched an ignore region -> discard entirely
            gt_taken[j] = True  # (still consume it so nothing else matches it)
        else:  # matched a real pedestrian -> true positive
            gt_taken[j] = True
            labels.append(1)
            scores.append(det_scores[i])

    n_pos = int((~gt_ignore).sum())
    return np.asarray(labels, np.int64), np.asarray(scores, np.float32), n_pos
