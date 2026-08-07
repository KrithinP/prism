"""The assembled detector runs end to end and has a sane size."""

import torch

from prism.models.yolox import YOLOX


def test_yolox_forward():
    model = YOLOX(num_classes=1).eval()
    outs = model(torch.randn(1, 3, 512, 640))
    assert [tuple(o.shape) for o in outs] == [(1, 6, 64, 80), (1, 6, 32, 40), (1, 6, 16, 20)]


def test_yolox_param_count_is_reasonable():
    # YOLOX-S is ~9M params. Ours is the same family; sanity-check the ballpark.
    model = YOLOX(num_classes=1)
    n = sum(p.numel() for p in model.parameters())
    assert 5e6 < n < 15e6, f"unexpected param count: {n / 1e6:.1f}M"
