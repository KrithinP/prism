"""Attribution targets and quantitative checks are deterministic and explicit."""

import torch
from torch import nn

from scripts.analyze_attributions import (
    attribution_map,
    candidate_output,
    candidate_scores,
    faithfulness_curves,
    grad_cam,
    integrated_gradients,
    localization_metrics,
    select_candidate,
)


class _TinyFusion(nn.Module):
    def forward(self, rgb, thermal):
        batch = rgb.shape[0]
        regression = torch.zeros(batch, 4, 1, 1, device=rgb.device)
        objectness = (rgb.mean(dim=(1, 2, 3)) + thermal.mean(dim=(1, 2, 3))).view(batch, 1, 1, 1)
        classification = torch.zeros(batch, 1, 1, 1, device=rgb.device)
        return [torch.cat([regression, objectness, classification], dim=1)]


class _Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.dark3 = nn.Conv2d(3, 2, 1, bias=False)
        nn.init.ones_(self.dark3.weight)


class _TinyMiddleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_subset = [3]
        self.backbone_rgb = _Branch()
        self.backbone_thermal = _Branch()

    def forward(self, rgb, thermal):
        rgb_feature = self.backbone_rgb.dark3(rgb)
        thermal_feature = self.backbone_thermal.dark3(thermal)
        batch = rgb.shape[0]
        regression = torch.zeros(batch, 4, 1, 1, device=rgb.device)
        objectness = (rgb_feature + thermal_feature).mean(dim=(1, 2, 3)).view(batch, 1, 1, 1)
        classification = torch.zeros(batch, 1, 1, 1, device=rgb.device)
        return [torch.cat([regression, objectness, classification], dim=1)]


def test_candidate_selection_uses_highest_score_among_iou_matches() -> None:
    decoded = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 9.0, 0.0, 0.0],
            [20.0, 20.0, 30.0, 30.0, 0.0, 0.0],
        ]
    )
    scores = torch.tensor([0.4, 0.9, 0.99])
    selected = select_candidate(decoded, scores, torch.tensor([[0.0, 0.0, 10.0, 10.0]]), 0.5)
    assert selected["candidate_index"] == 1
    assert selected["gt_index"] == 0


def test_localization_metrics_have_exact_inside_mass_and_pointing() -> None:
    saliency = torch.zeros(4, 5)
    saliency[2, 3] = 2
    saliency[0, 0] = 1
    metrics = localization_metrics(saliency, torch.tensor([2.0, 1.0, 5.0, 4.0]))
    assert metrics["inside_gt_fraction"] == 2 / 3
    assert metrics["pointing_hit"] is True
    assert metrics["peak_xy"] == [3, 2]


def test_integrated_gradients_approximately_satisfies_completeness() -> None:
    model = _TinyFusion()
    rgb = torch.ones(1, 3, 2, 2)
    thermal = torch.full((1, 3, 2, 2), 2.0)
    baseline = torch.zeros_like(rgb)
    rgb_attr, thermal_attr, input_score, baseline_score = integrated_gradients(
        model, rgb, thermal, baseline, baseline, candidate_index=0, steps=128
    )
    assert candidate_scores(model(rgb, thermal)).shape == (1, 1)
    attributed = float(rgb_attr.sum() + thermal_attr.sum())
    assert abs(attributed - (input_score - baseline_score)) < 1e-5
    assert attribution_map(rgb_attr).shape == (2, 2)


def test_grad_cam_targets_both_branches_at_actual_fusion_level() -> None:
    model = _TinyMiddleFusion()
    rgb = torch.ones(1, 3, 4, 5)
    thermal = torch.full((1, 3, 4, 5), 0.5)
    rgb_cam, thermal_cam, level = grad_cam(model, "multi-scale", rgb, thermal, candidate_index=0)
    assert level == 3
    assert rgb_cam.shape == (4, 5) and thermal_cam.shape == (4, 5)
    assert rgb_cam.sum() > 0 and thermal_cam.sum() > 0


def test_faithfulness_curves_preserve_score_and_localization_outputs() -> None:
    model = _TinyFusion()
    rgb = torch.ones(1, 3, 2, 2)
    thermal = torch.full((1, 3, 2, 2), 2.0)
    baseline = torch.zeros_like(rgb)
    gt_box = torch.tensor([-4.0, -4.0, 4.0, 4.0])

    clean = candidate_output(model, rgb, thermal, 0, gt_box)
    assert clean["box_iou"] == 1.0
    assert clean["center_error_px"] == 0.0

    curves = faithfulness_curves(
        model,
        rgb,
        thermal,
        baseline,
        baseline,
        torch.ones(2, 2),
        torch.ones(2, 2),
        0,
        gt_box,
        [0.0, 1.0],
    )
    for modality in ("rgb", "thermal"):
        assert curves[modality]["fractions"] == [0.0, 1.0]
        assert len(curves[modality]["deletion"]) == 2
        assert len(curves[modality]["insertion"]) == 2
        assert set(curves[modality]["deletion"][0]) == {"score", "box_iou", "center_error_px"}
        assert "deletion_score_auc" in curves[modality]
        assert "insertion_score_auc" in curves[modality]
