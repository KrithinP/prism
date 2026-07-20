
"""Stage 4a: the Reasonable-subset filter, and dataset-level evaluation.

Ties matching (Stage 1) and the log-average (Stages 2-3) into one number.
See textbook Ch.3 Stage 4.
"""
from __future__ import annotations

import numpy as np

from prism.data.kaist_constants import (
    REASONABLE_MIN_HEIGHT, REASONABLE_MAX_OCCLUSION,
)
from prism.eval.matching import match_frame
from prism.eval.mr import log_average_miss_rate


def reasonable_ignore(gt_boxes, gt_classes, gt_occlusion):
    """True = don't score this ground-truth box in the Reasonable subset.

    Three reasons a box is ignored:
      1. it isn't a 'person' (people / cyclist / person?)  -- class 0
      2. it's shorter than 55 px                            -- too small to be fair
      3. it's heavily occluded (occlusion == 2)             -- mostly hidden
    Ignored boxes still ABSORB detections (they just don't count either way).
    """
    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    heights = gt_boxes[:, 3] - gt_boxes[:, 1]          # xyxy -> height
    ignore = np.asarray(gt_classes, np.int64) == 0
    ignore |= heights < REASONABLE_MIN_HEIGHT
    ignore |= np.asarray(gt_occlusion, np.int64) > REASONABLE_MAX_OCCLUSION
    return ignore


def evaluate(frames):
    """frames: list of dicts, each with det_boxes, det_scores, gt_boxes,
    gt_classes, gt_occlusion. Returns MR^-2 over the Reasonable subset."""
    all_labels, all_scores, n_pos = [], [], 0
    for f in frames:
        ignore = reasonable_ignore(f["gt_boxes"], f["gt_classes"], f["gt_occlusion"])
        labels, scores, npos = match_frame(
            f["det_boxes"], f["det_scores"], f["gt_boxes"], ignore
        )
        all_labels.append(labels)
        all_scores.append(scores)
        n_pos += npos
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    return log_average_miss_rate(labels, scores, n_pos, len(frames))
