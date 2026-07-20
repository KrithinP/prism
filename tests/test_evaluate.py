
"""Reasonable-subset filter: small/occluded pedestrians become ignore regions."""
import numpy as np

from prism.eval.evaluate import evaluate, reasonable_ignore


def test_reasonable_ignores_small_and_occluded():
    boxes = np.array([[0, 0, 20, 60],     # 60 px tall, person, not occluded -> SCORED
                      [0, 0, 20, 30],     # 30 px tall -> too small -> ignored
                      [0, 0, 20, 60],     # tall but occlusion=2 -> ignored
                      [0, 0, 20, 60]], np.float32)  # tall, but class 0 -> ignored
    classes = np.array([1, 1, 1, 0])
    occl = np.array([0, 0, 2, 0])
    ign = reasonable_ignore(boxes, classes, occl)
    assert ign.tolist() == [False, True, True, True]


def test_detection_on_small_pedestrian_is_not_penalised():
    # Frame 1: a scored pedestrian, detected correctly -> TP.
    # Frame 2: only a too-small pedestrian; we detect it -> absorbed, NOT a false positive.
    frames = [
        {"det_boxes": np.array([[0, 0, 20, 60]], np.float32),
         "det_scores": np.array([0.9], np.float32),
         "gt_boxes": np.array([[0, 0, 20, 60]], np.float32),
         "gt_classes": np.array([1]), "gt_occlusion": np.array([0])},
        {"det_boxes": np.array([[0, 0, 20, 30]], np.float32),
         "det_scores": np.array([0.8], np.float32),
         "gt_boxes": np.array([[0, 0, 20, 30]], np.float32),
         "gt_classes": np.array([1]), "gt_occlusion": np.array([0])},
    ]
    mr = evaluate(frames)
    # One scored pedestrian, found, and zero false positives -> perfect.
    assert mr < 1e-6
