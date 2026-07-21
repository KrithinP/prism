
"""Blocks must preserve or halve spatial size exactly as expected."""
import torch

from prism.models.blocks import ConvBNAct, Bottleneck, CSPLayer, SPPBottleneck


def test_conv_keeps_size_and_halves_on_stride2():
    x = torch.randn(2, 16, 32, 40)
    assert ConvBNAct(16, 32, k=3)(x).shape == (2, 32, 32, 40)
    assert ConvBNAct(16, 32, k=3, stride=2)(x).shape == (2, 32, 16, 20)


def test_bottleneck_and_csp_keep_size():
    x = torch.randn(2, 32, 16, 20)
    assert Bottleneck(32, 32)(x).shape == (2, 32, 16, 20)
    assert CSPLayer(32, 64, n=2)(x).shape == (2, 64, 16, 20)


def test_spp_keeps_size():
    x = torch.randn(2, 64, 10, 12)
    assert SPPBottleneck(64, 128)(x).shape == (2, 128, 10, 12)
