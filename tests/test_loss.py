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


def test_background_loss_is_sane_at_init():
    """From-scratch, an all-background batch must not produce a giant loss.
    Prior-bias init (head.py) keeps it small; without it, SGD diverges to NaN."""
    torch.manual_seed(0)
    model = YOLOX(num_classes=1)
    lossfn = YOLOXLoss(num_classes=1)
    imgs = torch.randn(4, 3, 512, 640)
    empty = [torch.zeros(0, 5) for _ in range(4)]
    loss, parts = lossfn(model(imgs), empty)
    # Healthy floor is ~270: 4 imgs x 6720 cells x -log(0.99) with prior pi=0.01.
    # The broken (no prior-bias) value was ~18000, so 500 cleanly separates them.
    assert parts["obj"] < 500.0, f"background loss too large at init: {parts['obj']:.1f}"


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
