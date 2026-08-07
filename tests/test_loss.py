"""The loss must be finite, flow gradients, and let the model overfit one image."""

import torch

from prism.models.loss import (
    YOLOXLoss,
    centers_inside_boxes,
    yolox_center_prior,
    yolox_iou_loss,
)
from prism.models.yolox import YOLOX


def _one_sample():
    img = torch.randn(1, 3, 512, 640)
    # one pedestrian box [cls, x1, y1, x2, y2]
    targets = [torch.tensor([[0.0, 300, 200, 340, 320]])]
    return img, targets


def test_loss_is_finite_and_has_gradients():
    model = YOLOX(num_classes=1)
    lossfn = YOLOXLoss(num_classes=1)
    img, targets = _one_sample()
    loss, parts = lossfn(model(img), targets)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert parts["num_fg"] > 0  # SimOTA assigned at least one positive


def test_background_loss_is_sane_at_init():
    """From-scratch, an all-background batch must not produce a giant loss.
    Prior-bias init (head.py) keeps it small; without it, SGD diverges to NaN."""
    torch.manual_seed(0)
    model = YOLOX(num_classes=1)
    lossfn = YOLOXLoss(num_classes=1)
    imgs = torch.randn(4, 3, 512, 640)
    empty = [torch.zeros(0, 5) for _ in range(4)]
    _loss, parts = lossfn(model(imgs), empty)
    # Healthy floor is ~270: 4 imgs x 6720 cells x -log(0.99) with prior pi=0.01.
    # The broken (no prior-bias) value was ~18000, so 500 cleanly separates them.
    assert parts["obj"] < 500.0, f"background loss too large at init: {parts['obj']:.1f}"


def test_iou_loss_matches_official_yolox_squared_form():
    iou = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(yolox_iou_loss(iou), torch.tensor([1.0, 0.75, 0.0]))


def test_ignore_region_removes_negative_objectness_supervision():
    lossfn = YOLOXLoss(num_classes=1, strides=(8,))
    output = torch.zeros(1, 6, 2, 2, requires_grad=True)
    # Covers all four anchor centres (4,4), (12,4), (4,12), (12,12).
    targets = [torch.tensor([[-1.0, 0.0, 0.0, 16.0, 16.0]])]
    loss, parts = lossfn([output], targets)
    assert loss.item() == 0.0
    assert parts["obj"] == 0.0


def test_ignore_region_uses_inclusive_anchor_centres():
    centers = torch.tensor([[4.0, 4.0], [12.0, 12.0], [20.0, 20.0]])
    boxes = torch.tensor([[4.0, 4.0, 12.0, 12.0]])
    assert centers_inside_boxes(centers, boxes).tolist() == [True, True, False]


def test_simota_center_prior_has_pinned_1_5_stride_boundary():
    centers = torch.tensor([[100.0, 100.0], [112.0, 100.0], [112.01, 100.0]])
    strides = torch.tensor([8.0, 8.0, 8.0])
    gt = torch.tensor([[100.0, 100.0, 20.0, 40.0]])
    assert yolox_center_prior(centers, strides, gt)[:, 0].tolist() == [True, True, False]


def test_model_can_overfit_one_image():
    """The cheapest, most reliable DL smoke test: memorise a single image."""
    torch.manual_seed(0)
    model = YOLOX(num_classes=1).train()
    lossfn = YOLOXLoss(num_classes=1)
    img, targets = _one_sample()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 150 steps, not 60: with prior-bias init (head.py) the model starts confidently
    # predicting background, so it must climb out of the -4.595 bias before it can fit
    # the box. Correct behaviour, just a slower start than the pre-fix model had.
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss, _ = lossfn(model(img), targets)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.5 * first, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
