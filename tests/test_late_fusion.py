"""Late-fusion semantics tests (TODO.md 6.6, family B11/B12): deterministic
fixtures for union-NMS and ProbEn merging of two unimodal detection sets.

Tensor format (K, 6) = x1, y1, x2, y2, score, cls. Boxes are arbitrary pixel
coordinates; no boundary or score-calibration semantics live here (both
unimodal detectors share one recipe, so scores are comparable by construction).
"""

import pytest
import torch

from prism.models.late_fusion import proben_merge, union_nms


def d(boxes_scores):
    """boxes_scores: list of (x1, y1, x2, y2, score) -> (K, 6) tensor."""
    return torch.as_tensor([(*box, 0) for box in boxes_scores], dtype=torch.float32)


def test_union_nms_keeps_single_object_once():
    """The same object detected by both cameras with different confidences:
    union + NMS keeps it once, with the higher score (order-insensitive)."""
    rgb = d([(10, 10, 50, 90, 0.7)])
    thm = d([(10, 10, 50, 90, 0.9)])
    out = union_nms(rgb, thm)
    assert out.shape[0] == 1
    assert out[0, 4].item() == pytest.approx(0.9)


def test_union_nms_keeps_distinct_objects():
    """Two different objects, each seen by one camera only: both survive."""
    rgb = d([(10, 10, 50, 90, 0.8)])
    thm = d([(200, 10, 240, 90, 0.8)])
    out = union_nms(rgb, thm)
    assert out.shape[0] == 2
    assert {round(float(x), 1) for x in out[:, 4]} == {0.8}


def test_union_nms_suppression_follows_iou_thr():
    """Overlapping boxes from the two cameras are NOT merged when their IoU is
    below the threshold (they are two different objects to the ensemble)."""
    rgb = d([(10, 10, 50, 90, 0.9)])
    thm = d([(10, 10, 100, 90, 0.6)])  # IoU < 0.5 with the rgb box
    out = union_nms(rgb, thm, iou_thr=0.5)
    assert out.shape[0] == 2


def test_union_nms_empty_sides():
    empty = torch.zeros((0, 6))
    single = d([(10, 10, 50, 90, 0.8)])
    assert union_nms(empty, single).shape[0] == 1
    assert union_nms(single, empty).shape[0] == 1
    assert union_nms(empty, empty).shape[0] == 0


# ----------------------------------------------------------------- ProbEn


def test_proben_combined_score_is_bayes_product_rule():
    """Matched pair: the published ProbEn score is
    prod(p)/(prod(p)+prod(1-p)) = 0.63/(0.63+0.03) = 0.9545... -- NOT noisy-OR
    (which would be 1 - 0.3*0.1 = 0.97)."""
    rgb = d([(10, 10, 50, 90, 0.7)])
    thm = d([(10, 10, 50, 90, 0.9)])
    out = proben_merge(rgb, thm)
    assert out.shape[0] == 1
    assert out[0, 4].item() == pytest.approx(0.7 * 0.9 / (0.7 * 0.9 + 0.3 * 0.1))


def test_proben_disagreeing_evidence_is_neutral():
    """The signature difference from noisy-OR: p=0.7 vs p=0.3 is conflicting
    evidence, so ProbEn reports 0.5; noisy-OR would report 0.79."""
    rgb = d([(10, 10, 50, 90, 0.7)])
    thm = d([(10, 10, 50, 90, 0.3)])
    out = proben_merge(rgb, thm)
    assert out[0, 4].item() == pytest.approx(0.5)
    assert out[0, 4].item() != pytest.approx(1 - 0.3 * 0.7)


def test_proben_box_is_unweighted_average():
    """Default box fusion is the reference's avg_bbox_fusion (plain mean)."""
    rgb = d([(10, 10, 50, 90, 0.2)])
    thm = d([(10, 10, 90, 90, 0.8)])
    out = proben_merge(rgb, thm)
    x2 = out[0, 2].item()
    assert x2 == pytest.approx((50 + 90) / 2.0)


def test_proben_box_fusion_score_option():
    rgb = d([(10, 10, 50, 90, 0.2)])
    thm = d([(10, 10, 90, 90, 0.8)])
    out = proben_merge(rgb, thm, box_fusion="score")
    x2 = out[0, 2].item()
    assert x2 == pytest.approx((50 * 0.2 + 90 * 0.8) / 1.0)


def test_proben_cluster_can_hold_three_boxes():
    """A cluster may contain 2+ boxes from either camera: 0.7 (rgb) and
    0.9, 0.8 (thermal, both overlapping) -> product rule over all three."""
    rgb = d([(10, 10, 50, 90, 0.7)])
    thm = d([(10, 10, 50, 90, 0.9), (12, 12, 48, 88, 0.8)])
    out = proben_merge(rgb, thm)
    assert out.shape[0] == 1
    p = torch.tensor([0.7, 0.9, 0.8])
    expected = p.prod() / (p.prod() + (1 - p).prod())
    assert out[0, 4].item() == pytest.approx(expected.item())


def test_proben_separated_pairs_stay_separate():
    """Two objects, each seen by both cameras: two clusters, both fused."""
    rgb = d([(10, 10, 50, 90, 0.7), (200, 200, 240, 280, 0.7)])
    thm = d([(10, 10, 50, 90, 0.9), (200, 200, 240, 280, 0.9)])
    out = proben_merge(rgb, thm)
    assert out.shape[0] == 2
    fused = 0.7 * 0.9 / (0.7 * 0.9 + 0.3 * 0.1)
    scores = sorted(out[:, 4].tolist())
    assert scores == pytest.approx([fused] * 2)


def test_proben_keeps_unmatched_boxes_with_own_score():
    """A box only one camera sees survives unchanged (missing-modality
    behavior): a singleton cluster's Bayes score is p/(p + (1-p)) = p."""
    rgb = d([(10, 10, 50, 90, 0.7)])
    thm = d([(300, 300, 340, 380, 0.5)])
    out = proben_merge(rgb, thm)
    assert out.shape[0] == 2
    assert {round(float(x), 4) for x in out[:, 4]} == {0.7, 0.5}


def test_proben_empty_sides():
    empty = torch.zeros((0, 6))
    single = d([(10, 10, 50, 90, 0.8)])
    assert proben_merge(empty, single).shape[0] == 1
    assert proben_merge(single, empty).shape[0] == 1
    assert proben_merge(empty, empty).shape[0] == 0


def test_proben_scores_clamped_to_unit_interval():
    """Scores outside [0,1] are clamped before combination: p1 -> 1-eps and
    p2 -> eps give a fused score of exactly 0.5 (perfectly conflicting)."""
    rgb = d([(10, 10, 50, 90, 1.7)])
    thm = d([(10, 10, 50, 90, -0.3)])
    out = proben_merge(rgb, thm)
    assert out[0, 4].item() == pytest.approx(0.5)


def test_proben_cluster_scan_starts_at_sort_position():
    """Regression: the top-scoring box lives at tensor index 2 (thermal second
    entry) while the 0.8 box sits at index 0 -- the cluster scan must start at
    the box's SORT position, not its numeric index, or the 0.8 box is missed
    and later NMS-suppressed. Correct fused score = Bayes(0.95, 0.8)."""
    rgb = d([(10, 10, 50, 90, 0.8)])
    thm = d([(200, 200, 240, 280, 0.6), (10, 10, 50, 90, 0.95)])
    out = proben_merge(rgb, thm)
    assert out.shape[0] == 2
    scores = sorted(out[:, 4].tolist())
    fused = 0.95 * 0.8 / (0.95 * 0.8 + 0.05 * 0.2)
    assert scores == pytest.approx([0.6, fused])


def test_proben_rejects_unknown_box_fusion():
    rgb = d([(10, 10, 50, 90, 0.7)])
    thm = d([(10, 10, 50, 90, 0.9)])
    with pytest.raises(ValueError, match="box_fusion"):
        proben_merge(rgb, thm, box_fusion="var")
