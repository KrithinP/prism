"""Stage 4a: the Reasonable-subset filter, and dataset-level evaluation.

Ties matching (Stage 1) and the log-average (Stages 2-3) into one number.
See textbook Ch.3 Stage 4.
"""

from __future__ import annotations

import numpy as np

from prism.data.kaist_constants import (
    OCCLUSION_HEAVY,
    OCCLUSION_NONE,
    OCCLUSION_PARTIAL,
    REASONABLE_MAX_OCCLUSION,
    REASONABLE_MIN_HEIGHT,
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
    heights = gt_boxes[:, 3] - gt_boxes[:, 1]  # xyxy -> height
    ignore = np.asarray(gt_classes, np.int64) == 0
    ignore |= heights < REASONABLE_MIN_HEIGHT
    ignore |= np.asarray(gt_occlusion, np.int64) > REASONABLE_MAX_OCCLUSION
    return ignore


def subset_ignore(gt_boxes, gt_classes, gt_occlusion, min_height, max_height, allowed_occlusion):
    """True = don't score this box in THIS subset.

    Generalises `reasonable_ignore` to any (height range, occlusion set) condition.
    A box outside the subset is *ignored*, not deleted -- it still absorbs detections
    penalty-free. That is the Caltech/KAIST convention and it matters: deleting a
    far-away pedestrian would charge you a false positive for correctly finding it.
    """
    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    occ = np.asarray(gt_occlusion, np.int64)

    ignore = np.asarray(gt_classes, np.int64) == 0  # not a scored 'person'
    ignore |= heights < min_height
    ignore |= heights >= max_height
    ignore |= ~np.isin(occ, list(allowed_occlusion))
    return ignore


# The standard KAIST evaluation conditions (Hwang et al., CVPR 2015).
# Scale thresholds 45 px / 115 px come from the paper -- chosen from driving speed
# and braking distance, not from anything about the images.
#   name: (min_height, max_height, allowed occlusion tags)
NON_HEAVY = (OCCLUSION_NONE, OCCLUSION_PARTIAL)
SUBSETS = {
    "reasonable": (REASONABLE_MIN_HEIGHT, np.inf, NON_HEAVY),  # the headline
    # scale conditions (occlusion policy held at the Reasonable one, documented choice)
    "near": (115.0, np.inf, NON_HEAVY),
    "medium": (45.0, 115.0, NON_HEAVY),
    "far": (1.0, 45.0, NON_HEAVY),
    # occlusion conditions (height held at the Reasonable threshold)
    "occ-none": (REASONABLE_MIN_HEIGHT, np.inf, (OCCLUSION_NONE,)),
    "occ-partial": (REASONABLE_MIN_HEIGHT, np.inf, (OCCLUSION_PARTIAL,)),
    "occ-heavy": (REASONABLE_MIN_HEIGHT, np.inf, (OCCLUSION_HEAVY,)),
}


def _evaluate_with(frames, ignore_fn):
    all_labels, all_scores, n_pos = [], [], 0
    for f in frames:
        ignore = ignore_fn(f)
        labels, scores, npos = match_frame(f["det_boxes"], f["det_scores"], f["gt_boxes"], ignore)
        all_labels.append(labels)
        all_scores.append(scores)
        n_pos += npos
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    return log_average_miss_rate(labels, scores, n_pos, len(frames)), n_pos


def evaluate(frames):
    """frames: list of dicts, each with det_boxes, det_scores, gt_boxes,
    gt_classes, gt_occlusion. Returns MR^-2 over the Reasonable subset."""
    mr, _ = _evaluate_with(
        frames,
        lambda f: reasonable_ignore(f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]),
    )
    return mr


def evaluate_subsets(frames, names=None):
    """MR^-2 for each standard KAIST condition. Returns {name: (mr, n_pos)}.

    n_pos is reported alongside because a subset with very few pedestrians (Far and
    Heavy usually are) produces a noisy, near-meaningless MR -- read the count before
    reading the number.
    """
    out = {}
    for name in names or SUBSETS:
        min_h, max_h, occ = SUBSETS[name]
        out[name] = _evaluate_with(
            frames,
            lambda f, mn=min_h, mx=max_h, oc=occ: subset_ignore(
                f["gt_boxes"], f["gt_classes"], f["gt_occlusion"], mn, mx, oc
            ),
        )
    return out
