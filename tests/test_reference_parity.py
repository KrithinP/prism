"""Synthetic evaluator unit cases (TODO.md 3.6): deterministic fixtures that pin
the canonical matching semantics -- the intermediate TP/FP/ignore sequence, not
just the final MR. These run WITHOUT the pinned MLPD bundle (they encode its
documented rules); scripts/parity_check.py adds the differential check against the
actual pinned implementation.

Frame helper: (x1,y1,x2,y2) boxes, gt_classes 1=person 0=ignore, gt_occlusion 0..2.
"""

import numpy as np
import pytest

from prism.eval.canonical import (
    check_fppi_reachable,
    evaluate_reference,
    match_frame_reference,
    mr_for_frames,
)
from prism.eval.mr import REF_FPPI


def make_frame(dets=None, gts=None, is_day=True):
    """dets: list of (x1,y1,x2,y2,score); gts: list of (x1,y1,x2,y2,cls,occ)."""
    dets = dets or []
    gts = gts or []
    return {
        "det_boxes": np.asarray([d[:4] for d in dets], np.float32).reshape(-1, 4),
        "det_scores": np.asarray([d[4] for d in dets], np.float32),
        "gt_boxes": np.asarray([g[:4] for g in gts], np.float32).reshape(-1, 4),
        "gt_classes": np.asarray([g[4] for g in gts], np.int64),
        "gt_occlusion": np.asarray([g[5] for g in gts], np.int64),
        "is_day": is_day,
    }


def person(x1, y1, x2, y2, occ=0):
    return (x1, y1, x2, y2, 1, occ)


def ignore(x1, y1, x2, y2):
    return (x1, y1, x2, y2, 0, 0)


# ---------------------------------------------------------------- matching


def test_one_exact_detection_is_one_tp():
    f = make_frame(dets=[(100, 100, 200, 200, 0.9)], gts=[person(100, 100, 200, 200)])
    labels, _scores, n_pos, absorbed = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [1] and n_pos == 1 and absorbed == 0


def test_below_iou_05_is_fp_and_fn():
    f = make_frame(
        dets=[(100, 100, 150, 150, 0.9)],  # IoU 0.25 with the GT
        gts=[person(100, 100, 200, 200)],
    )
    labels, _scores, n_pos, _ = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [0] and n_pos == 1  # 1 FP, GT stays unmatched -> FN


def test_two_dets_one_gt_first_tp_second_fp():
    f = make_frame(
        dets=[(100, 100, 200, 200, 0.9), (100, 100, 200, 200, 0.8)],
        gts=[person(100, 100, 200, 200)],
    )
    labels, _scores, _, _ = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [1, 0]  # GT claimed once


def test_dets_inside_ignore_region_are_absorbed():
    f = make_frame(
        dets=[(110, 110, 190, 190, 0.9), (112, 112, 192, 192, 0.8)],
        gts=[ignore(100, 100, 200, 200)],
    )
    labels, _scores, n_pos, absorbed = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [] and absorbed == 2 and n_pos == 0


def test_regular_gt_wins_over_ignore_region():
    """Detection overlaps both a regular box and an ignore region: the regular
    match takes priority (non-ignored GT is considered first)."""
    f = make_frame(
        dets=[(150, 100, 250, 200, 0.9)],
        gts=[person(140, 100, 240, 200), ignore(100, 100, 300, 300)],
    )
    labels, _, _, absorbed = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [1] and absorbed == 0


def test_ignore_uses_inter_over_det_area_not_plain_iou():
    """A detection fully inside a big ignore region has LOW ordinary IoU (0.25)
    but inter/det_area = 1.0 -- under canonical semantics it is absorbed. Plain
    IoU-based matching (the old local evaluator) would call it an FP."""
    f = make_frame(dets=[(100, 100, 200, 200, 0.9)], gts=[ignore(100, 100, 300, 300)])
    labels, _, _, absorbed = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert absorbed == 1 and len(labels) == 0
    # ...but a detection overlapping the ignore region for < 50% of ITS OWN area
    # is a false positive, even though it overlaps the ignore box substantially.
    f2 = make_frame(dets=[(100, 100, 500, 200, 0.9)], gts=[ignore(100, 100, 300, 150)])
    labels2, _, _, absorbed2 = match_frame_reference(
        f2["det_boxes"], f2["det_scores"], f2["gt_boxes"], f2["gt_classes"], f2["gt_occlusion"]
    )
    assert absorbed2 == 0 and labels2.tolist() == [0]


def test_reasonable_height_boundary_is_exact():
    """55 px tall is scored, 54 px is ignored -- exactly, no epsilon."""
    f_55 = make_frame(dets=[(100, 100, 200, 155, 0.9)], gts=[person(100, 100, 200, 155)])
    _, _, n_pos, _ = match_frame_reference(
        f_55["det_boxes"],
        f_55["det_scores"],
        f_55["gt_boxes"],
        f_55["gt_classes"],
        f_55["gt_occlusion"],
    )
    assert n_pos == 1
    f_54 = make_frame(dets=[(100, 100, 200, 154, 0.9)], gts=[person(100, 100, 200, 154)])
    _labels, _, n_pos, absorbed = match_frame_reference(
        f_54["det_boxes"],
        f_54["det_scores"],
        f_54["gt_boxes"],
        f_54["gt_classes"],
        f_54["gt_occlusion"],
    )
    assert n_pos == 0 and absorbed == 1


def test_heavy_occlusion_ignored():
    f = make_frame(dets=[(100, 100, 200, 200, 0.9)], gts=[person(100, 100, 200, 200, occ=2)])
    _, _, n_pos, absorbed = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert n_pos == 0 and absorbed == 1


def test_exact_05_iou_counts_as_match():
    """The reference matches with `ious < bstOa: continue` -- an IoU of exactly
    0.5 is a TP, not a miss (only a strict < 0.5 fails)."""
    # 100x50 detection against a 100x100 GT: inter = 5000, union = 10000 -> 0.5
    f = make_frame(dets=[(100, 100, 200, 150, 0.9)], gts=[person(100, 100, 200, 200)])
    labels, _, n_pos, _ = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [1] and n_pos == 1


def test_image_boundary_semantics():
    """Reference boundary rule: GT outside [5,5,635,507] is IGNORED (absorb-only);
    detections are evaluated RAW and are never clipped."""
    f = make_frame(
        dets=[(100, 100, 200, 200, 0.9), (700, 100, 900, 200, 0.8)],
        gts=[person(100, 100, 200, 200)],
    )
    labels, _scores, _, _ = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [1, 0]  # outside detection = raw FP, not dropped
    # a GT straddling the boundary is ignored (absorb-only), even if >= 55px tall
    f2 = make_frame(dets=[(640, 100, 660, 200, 0.9)], gts=[person(640, 100, 660, 200)])
    labels2, _, n_pos, absorbed = match_frame_reference(
        f2["det_boxes"], f2["det_scores"], f2["gt_boxes"], f2["gt_classes"], f2["gt_occlusion"]
    )
    assert n_pos == 0 and absorbed == 1 and len(labels2) == 0


def test_tied_scores_stable_order():
    """Identical scores: stable ordering (input order preserved) decides TP/FP."""
    f = make_frame(
        dets=[(100, 100, 200, 200, 0.5), (100, 100, 200, 200, 0.5)],
        gts=[person(100, 100, 200, 200)],
    )
    labels, _, _, _ = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert labels.tolist() == [1, 0]


# ---------------------------------------------------------------- edge cases


def test_no_detections_no_gt():
    f = make_frame()
    labels, _scores, n_pos, _ = match_frame_reference(
        f["det_boxes"], f["det_scores"], f["gt_boxes"], f["gt_classes"], f["gt_occlusion"]
    )
    assert len(labels) == 0 and n_pos == 0
    assert np.isnan(mr_for_frames([f]))  # no positives -> MR undefined


def test_gt_with_no_detections():
    f = make_frame(gts=[person(100, 100, 200, 200)])
    assert mr_for_frames([f]) == 1.0  # 100% miss


def test_confidence_floor_too_high_fails_loudly():
    """With 1 image and 0 false positives at any threshold, FPPI=1.0 is
    unreachable -> the canonical evaluator must raise, not flatter."""
    f = make_frame(dets=[(100, 100, 200, 200, 0.9)], gts=[person(100, 100, 200, 200)])
    with pytest.raises(ValueError, match="FPPI"):
        check_fppi_reachable(np.array([1]), np.array([0.9]), n_images=1)
    with pytest.raises(ValueError):
        mr_for_frames([f], require_reachable=True)


def test_reference_fppi_points_exact():
    """The exact array from the pinned bundle (KAISTParams.fppiThrs), verbatim."""
    expected = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])
    assert np.array_equal(REF_FPPI, expected)


# ---------------------------------------------------------------- day/night


def test_day_night_curves_use_their_own_fppi_axis():
    """Day and Night MR are separate curves with separate FPPI denominators
    (TODO.md 12.6); All is a third curve, not a weighted average. Same content
    in each subset, but different denominators -> different curves."""
    # night: 1 image, 1 FP (2 TPs before it) -> steep curve
    night = make_frame(
        dets=[(100, 100, 200, 200, 0.9), (100, 200, 200, 300, 0.85), (300, 300, 400, 400, 0.2)],
        gts=[person(100, 100, 200, 200), person(100, 200, 200, 300)],
        is_day=False,
    )
    # day: 1 image, FP BEFORE any TP -> all-but-one sampled points miss everything
    day = make_frame(
        dets=[(300, 300, 400, 400, 0.95), (100, 100, 200, 200, 0.9), (500, 500, 600, 600, 0.2)],
        gts=[person(100, 100, 200, 200)],
        is_day=True,
    )
    res = evaluate_reference([night, day], require_reachable=False)
    # all: 2 images -> the FP lands at fppi 0.5, miss 2/3 on the sampled points
    assert res["night"] < res["all"] < res["day"]


def test_require_reachable_flag_offs():
    """Require-reachability is the default; subsets that cannot reach FPPI=1.0
    (e.g. a tiny synthetic set) opt out explicitly."""
    clean = make_frame(
        dets=[(100, 100, 200, 200, 0.9)], gts=[person(100, 100, 200, 200)], is_day=True
    )
    res = evaluate_reference([clean], require_reachable=False)
    assert res["day"] <= 1e-9  # perfect detection -> ~1e-10 clip floor
