"""Representation summaries have known values on simple matrices."""

import torch

from scripts.analyze_representations import effective_rank, linear_cka


def test_linear_cka_is_one_for_identical_nonconstant_features():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    assert abs(linear_cka(features, features) - 1.0) < 1e-12


def test_effective_rank_detects_rank_one_representation():
    features = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    result = effective_rank(features)
    assert abs(result["effective_rank"] - 1.0) < 1e-12
