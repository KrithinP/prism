"""Stage-1 evaluator test with a hand-computable answer."""
import numpy as np

from prism.eval.matching import box_iou, match_frame


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], np.float32)     # area 100
    b = np.array([[0, 0, 10, 10],                  # identical -> IoU 1.0
                  [5, 0, 15, 10],                  # half overlap -> 50/150 = 1/3
                  [20, 20, 30, 30]], np.float32)    # disjoint -> 0
    iou = box_iou(a, b)[0]
    assert np.isclose(iou[0], 1.0)
    assert np.isclose(iou[1], 1 / 3, atol=1e-4)
    assert np.isclose(iou[2], 0.0)


def test_match_frame_tp_fp_and_ignore():
    # Ground truth: one real pedestrian, one ignore region.
    gt_boxes = np.array([[0, 0, 10, 10],       # person  (scored)
                         [50, 50, 60, 60]], np.float32)  # ignore region
    gt_ignore = np.array([False, True])

    # Detections: hits the person (TP), hits nothing (FP), hits the ignore box (discarded).
    det_boxes = np.array([[0, 0, 10, 10],      # -> TP
                          [100, 100, 110, 110],# -> FP
                          [50, 50, 60, 60]], np.float32)  # -> discarded
    det_scores = np.array([0.9, 0.8, 0.7], np.float32)

    labels, scores, n_pos = match_frame(det_boxes, det_scores, gt_boxes, gt_ignore)

    assert n_pos == 1                          # only the person is scored
    assert labels.tolist() == [1, 0]           # TP then FP; the ignore-hit vanished
    assert len(labels) == 2                    # the discarded detection is gone
