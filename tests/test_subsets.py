"""The standard KAIST conditions must select the right pedestrians.

Scale thresholds are 45 px / 115 px (Hwang et al., CVPR 2015). A box outside a
condition must be IGNORED, not deleted -- it still absorbs detections penalty-free.
"""
import numpy as np

from prism.eval.evaluate import SUBSETS, subset_ignore, evaluate_subsets


# heights: 200 (near), 80 (medium), 30 (far), 60 (reasonable, partial occ), 60 (heavy occ)
GT = np.array([
    [0, 0, 20, 200],    # h=200  near,   occ none
    [0, 0, 20, 80],     # h=80   medium, occ none
    [0, 0, 20, 30],     # h=30   far,    occ none
    [0, 0, 20, 60],     # h=60   medium, occ partial
    [0, 0, 20, 60],     # h=60   medium, occ heavy
], dtype=np.float32)
CLASSES = np.array([1, 1, 1, 1, 1])          # all real persons
OCC = np.array([0, 0, 0, 1, 2])


def _scored(name):
    mn, mx, occ = SUBSETS[name]
    return ~subset_ignore(GT, CLASSES, OCC, mn, mx, occ)


def test_scale_conditions_use_45_and_115_px():
    assert list(_scored("near")) == [True, False, False, False, False]      # only h=200
    # medium = 45..115: h=80 and h=60(partial); h=60(heavy) excluded by occlusion
    assert list(_scored("medium")) == [False, True, False, True, False]
    assert list(_scored("far")) == [False, False, True, False, False]       # only h=30


def test_occlusion_conditions_split_by_tag():
    assert list(_scored("occ-none")) == [True, True, False, False, False]   # h>=55, occ 0
    assert list(_scored("occ-partial")) == [False, False, False, True, False]
    assert list(_scored("occ-heavy")) == [False, False, False, False, True]


def test_scale_conditions_partition_the_non_heavy_boxes():
    """Near/medium/far must be disjoint and together cover every non-heavy person."""
    near, med, far = _scored("near"), _scored("medium"), _scored("far")
    assert not (near & med).any() and not (med & far).any() and not (near & far).any()
    covered = near | med | far
    non_heavy = (CLASSES == 1) & (OCC != 2)
    assert list(covered) == list(non_heavy)


def test_evaluate_subsets_returns_mr_and_count():
    frames = [{
        "det_boxes": np.array([[0, 0, 20, 200]], np.float32),
        "det_scores": np.array([0.9], np.float32),
        "gt_boxes": GT, "gt_classes": CLASSES, "gt_occlusion": OCC,
    }]
    res = evaluate_subsets(frames)
    assert set(res) == set(SUBSETS)
    mr_near, n_near = res["near"]
    assert n_near == 1                      # one near pedestrian is scored
    assert 0.0 <= mr_near <= 1.0
