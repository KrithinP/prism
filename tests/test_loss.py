"""The loss must be finite, flow gradients, and let the model overfit one image."""
import torch

from prism.models.yolox import YOLOX
from prism.models.loss import YOLOXLoss


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
    assert parts["num_fg"] > 0            # SimOTA assigned at least one positive


def test_model_can_overfit_one_image():
    """The cheapest, most reliable DL smoke test: memorise a single image."""
    torch.manual_seed(0)
    model = YOLOX(num_classes=1).train()
    lossfn = YOLOXLoss(num_classes=1)
    img, targets = _one_sample()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = None
    for _ in range(60):
        opt.zero_grad()
        loss, _ = lossfn(model(img), targets)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.5 * first, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
