"""MR^-2 on hand-computable cases."""

import numpy as np

from prism.eval.mr import log_average_miss_rate


def test_perfect_detector_is_zero():
    # 10 images, 10 pedestrians, all found, no false alarms -> miss rate 0 everywhere.
    labels = np.ones(10)  # 10 true positives
    scores = np.linspace(0.9, 0.5, 10)
    mr = log_average_miss_rate(labels, scores, n_pos=10, n_images=10)
    assert mr < 1e-6


def test_misses_half_is_half():
    # Finds 5 of 10, no false alarms -> miss rate is 0.5 at every operating point.
    labels = np.ones(5)  # 5 TP, 0 FP
    scores = np.linspace(0.9, 0.5, 5)
    mr = log_average_miss_rate(labels, scores, n_pos=10, n_images=10)
    assert np.isclose(mr, 0.5, atol=1e-6)


def test_no_detections_is_one():
    mr = log_average_miss_rate(np.array([]), np.array([]), n_pos=10, n_images=10)
    assert mr == 1.0
