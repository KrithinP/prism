"""Frozen-feature probe sampling and closed-form metrics are auditable."""

import numpy as np
import torch

from scripts.probe_representations import (
    balanced_accuracy,
    binary_auc,
    fit_ridge,
    predict_ridge,
    sample_locations,
)


def test_probe_locations_include_gt_center_and_avoid_boxes() -> None:
    boxes = torch.tensor([[20.0, 10.0, 40.0, 30.0]])
    rows = sample_locations(
        "set00/V000/I00000",
        boxes,
        torch.tensor([1]),
        feature_height=8,
        feature_width=10,
        image_height=80,
        image_width=100,
    )
    positive = next(row for row in rows if row["label"] == 1)
    background = next(row for row in rows if row["label"] == 0)
    assert (positive["x"], positive["y"]) == (3, 2)
    background_x = (background["x"] + 0.5) * 10
    background_y = (background["y"] + 0.5) * 10
    assert not (20 <= background_x < 40 and 10 <= background_y < 30)


def test_ridge_probe_recovers_simple_linear_signal() -> None:
    features = np.asarray([[-2.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    targets = np.asarray([-1.0, -1.0, 1.0, 1.0])
    model = fit_ridge(features, targets, regularization=1e-6)
    scores = predict_ridge(model, features)
    labels = (targets > 0).astype(np.int64)
    assert binary_auc(labels, scores) == 1.0
    assert balanced_accuracy(labels, scores) == 1.0


def test_auc_handles_tied_scores_by_average_rank() -> None:
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert binary_auc(labels, scores) == 0.5
