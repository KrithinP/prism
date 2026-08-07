"""Canonical KAIST evaluator: the reference semantics that PRISM reports with.

TODO.md Phase 1: the local evaluator must be certified against the pinned MLPD
evaluation bundle, or retired. This module implements the *canonical* semantics
exactly as the released evaluation defines them (TODO.md 3.4):

    IoU threshold: 0.5 (matches >= 0.5, exact ties count)
    max detections per frame: 1000
    FPPI reference points: the pinned bundle's exact array (0.0100 ... 1.0000)
    Reasonable height: >= 55 px, occlusion none/partial, class person
    image boundary range: [5, 5, 635, 507]
        - ground truth outside it is IGNORED (absorb-only), never clipped;
        - detections are evaluated RAW (the reference iou() uses raw coords).
    Day = first 1455 sorted test IDs, Night = the rest (reference imgIds[:1455])

Matching rules that change scores near ties (NOT trivia):
  * regular ground truth is matched by intersection-over-union at threshold >=,
    at most once; among equal-IoU candidates the LATER gt wins (the reference
    scans gt in order and adopts any iou >= the best so far);
  * ignored/crowd ground truth uses intersection divided by detection area and
    can absorb multiple detections; the FIRST qualifying region in input order
    absorbs (the reference stops its scan there);
  * non-ignored ground truth is considered BEFORE ignored ground truth;
  * detections are sorted stably by descending confidence.

`legacy_matching` differs: it uses IoU for ignore regions too (prism.eval.matching)
and does not clip to the boundary -- kept only for re-scoring exploratory runs.
"""

from __future__ import annotations

import numpy as np

from prism.data.kaist_constants import REASONABLE_MAX_OCCLUSION, REASONABLE_MIN_HEIGHT
from prism.eval.mr import log_average_miss_rate

IOU_THR = 0.5
IGNORE_AREA_RATIO = 0.5  # inter(det, ignore) / area(det) above which a det is absorbed
MAX_DETS_PER_FRAME = 1000
IMAGE_BOUNDARY = (5, 5, 635, 507)  # x1, y1, x2, y2; GT outside it is ignored


def _inter_area(a, b):
    """Intersection area between every box of a (M,4) and b (N,4) -> (M,N)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def _area(boxes):
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def _box_iou(a, b):
    inter = _inter_area(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def match_frame_reference(
    det_boxes, det_scores, gt_boxes, gt_classes, gt_occlusion, boundary=IMAGE_BOUNDARY
):
    """Match one frame under canonical KAIST semantics.

    Returns (labels, scores, n_pos, n_absorbed):
      labels    (K,) 1=TP / 0=FP for detections NOT absorbed by an ignore region,
                ordered high-score first (stable sort).
      scores    (K,) their confidences.
      n_pos     scored (regular) ground-truth boxes -- the miss-rate denominator.
      n_absorbed  detections discarded by ignore regions (counted, not scored).
    """
    det_boxes = np.asarray(det_boxes, np.float32).reshape(-1, 4)
    det_scores = np.asarray(det_scores, np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes, np.int64).reshape(-1)
    gt_occlusion = np.asarray(gt_occlusion, np.int64).reshape(-1)

    # detections are evaluated RAW: the reference iou() uses the raw coords and
    # never clips (a detection outside the image boundary is just an FP).

    # cap detections at the reference maximum (keep top-1000 by score)
    if len(det_scores) > MAX_DETS_PER_FRAME:
        keep = np.argsort(-det_scores, kind="stable")[:MAX_DETS_PER_FRAME]
        det_boxes, det_scores = det_boxes[keep], det_scores[keep]

    # ground truth too short, heavy-occluded, non-person, or outside the benchmark
    # image boundary is IGNORED (absorb-only), never clipped -- the reference ORs
    # each rule into the annotation's ignore flag before matching.
    heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    inside = (
        (gt_boxes[:, 0] >= boundary[0])
        & (gt_boxes[:, 1] >= boundary[1])
        & (gt_boxes[:, 2] <= boundary[2])
        & (gt_boxes[:, 3] <= boundary[3])
    )
    regular = (
        (gt_classes == 1)
        & (heights >= REASONABLE_MIN_HEIGHT)
        & (gt_occlusion <= REASONABLE_MAX_OCCLUSION)
        & inside
    )
    ignore = ~regular

    order = np.argsort(-det_scores, kind="stable")  # stable: ties keep input order
    ious = _box_iou(det_boxes, gt_boxes)
    det_area = np.clip(_area(det_boxes), 1e-9, None)[:, None]
    inter_ratios = _inter_area(det_boxes, gt_boxes) / det_area

    gt_taken = np.zeros(len(gt_boxes), bool)
    regular_idx = np.nonzero(regular)[0]
    ignore_idx = np.nonzero(ignore)[0]
    labels, scores, n_absorbed = [], [], 0
    for i in order:
        # 1) regular (non-ignored) ground truth first: IoU >= 0.5, at most once.
        #    The scan adopts any iou >= the best so far, so an equal-IoU later gt
        #    wins -- exactly the reference's `if ious < bstOa: continue`.
        best_j, best_iou = -1, IOU_THR
        for j in regular_idx:
            if not gt_taken[j] and ious[i, j] >= best_iou:
                best_j, best_iou = j, ious[i, j]
        if best_j >= 0:
            gt_taken[best_j] = True
            labels.append(1)
            scores.append(det_scores[i])
            continue
        # 2) then ignore regions: inter/det_area, absorbs any number; the FIRST
        #    qualifying region in input order absorbs (the reference scans gt in
        #    order and stops at the first ignore it matches).
        for j in ignore_idx:
            if inter_ratios[i, j] >= IGNORE_AREA_RATIO:
                n_absorbed += 1
                break
        else:
            labels.append(0)
            scores.append(det_scores[i])

    n_pos = int(regular.sum())
    return (np.asarray(labels, np.int64), np.asarray(scores, np.float32), n_pos, n_absorbed)


def check_fppi_reachable(labels, scores, n_images):
    """Fail loudly if the FPPI=1.0 reference point cannot be reached.

    MR^-2 samples miss rates at FPPI points up to 1.0. If the export confidence
    floor cut too many detections, the curve never gets there and the reported
    number is silently flattering -- so this raises instead (TODO.md 3.6).
    """
    if n_images <= 0 or len(scores) == 0:
        raise ValueError("no detections to evaluate -- cannot reach FPPI=1.0")
    max_fppi = int((np.asarray(labels) == 0).sum()) / n_images
    if max_fppi < 1.0:
        raise ValueError(
            f"cannot reach FPPI=1.0 (max FPPI={max_fppi:.3f} with ALL detections kept) "
            "-- export confidence floor is too high"
        )


def mr_for_frames(frames, key=None, require_reachable=True):
    """MR^-2 over a set of frames using the canonical matching. `key` selects a
    subset of frames ('all'/'day'/'night'); frames carry an 'is_day' flag.

    require_reachable=True: if even the lowest-confidence detection cannot reach
    FPPI=1.0 (i.e. the export confidence floor was set too high), raise instead of
    silently reporting a flattering miss rate.
    """
    labels, scores, n_pos = [], [], 0
    for f in frames:
        if key == "day" and not f.get("is_day"):
            continue
        if key == "night" and f.get("is_day"):
            continue
        lab, sc, np_, _ = match_frame_reference(
            f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
        )
        labels.append(lab)
        scores.append(sc)
        n_pos += np_
    labels = np.concatenate(labels) if labels else np.array([], np.int64)
    scores = np.concatenate(scores) if scores else np.array([], np.float32)
    n_imgs = sum(
        1
        for f in frames
        if key in (None, "all")
        or (key == "day" and f.get("is_day"))
        or (key == "night" and not f.get("is_day"))
    )
    if require_reachable and n_imgs and len(scores):
        check_fppi_reachable(labels, scores, n_imgs)
    return log_average_miss_rate(labels, scores, n_pos, n_imgs)


def evaluate_reference(frames, require_reachable=True):
    """The standard report: {all, day, night} MR^-2 over the Reasonable subset,
    computed with canonical matching. All/Day/Night are separate curves, each with
    its own FPPI axis (TODO.md 12.6: All is NOT a frame-count-weighted average)."""
    result = {"all": mr_for_frames(frames, "all", require_reachable)}
    if frames and all("is_day" in frame for frame in frames):
        result["day"] = (
            mr_for_frames(frames, "day", require_reachable)
            if any(frame["is_day"] for frame in frames)
            else None
        )
        result["night"] = (
            mr_for_frames(frames, "night", require_reachable)
            if any(not frame["is_day"] for frame in frames)
            else None
        )
    else:
        # A sequence-disjoint validation split does not inherit the official
        # test set's 1455/797 Day/Night membership. Returning None prevents an
        # all-night-by-default artifact from being mistaken for a real subset.
        result["day"] = None
        result["night"] = None
    return result
