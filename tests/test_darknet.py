"""The backbone must emit exactly the three shapes from the Ch.4 table."""

import torch

from prism.models.darknet import CSPDarknet


def test_cspdarknet_emits_three_scales():
    m = CSPDarknet().eval()  # eval() so BatchNorm is deterministic with batch 1
    x = torch.randn(1, 3, 512, 640)  # one image: 3 channels, 512 tall, 640 wide
    c3, c4, c5 = m(x)
    assert c3.shape == (1, 128, 64, 80)  # dark3: stride 8   (512/8=64, 640/8=80)
    assert c4.shape == (1, 256, 32, 40)  # dark4: stride 16
    assert c5.shape == (1, 512, 16, 20)  # dark5: stride 32
