"""The neck must return three maps at the SAME shapes as the backbone gave it."""

import torch

from prism.models.darknet import CSPDarknet
from prism.models.pafpn import PAFPN


def test_pafpn_preserves_the_three_shapes():
    backbone = CSPDarknet().eval()
    neck = PAFPN().eval()
    x = torch.randn(1, 3, 512, 640)
    feats = backbone(x)  # dark3, dark4, dark5
    p3, p4, p5 = neck(feats)
    assert p3.shape == (1, 128, 64, 80)  # same as dark3
    assert p4.shape == (1, 256, 32, 40)  # same as dark4
    assert p5.shape == (1, 512, 16, 20)  # same as dark5


def test_pafpn_actually_mixed_information():
    """A neck output should differ from the raw backbone feature (it was enriched)."""
    backbone = CSPDarknet().eval()
    neck = PAFPN().eval()
    feats = backbone(torch.randn(1, 3, 512, 640))
    p3, _, _ = neck(feats)
    assert not torch.allclose(p3, feats[0])  # enriched, not passed through unchanged
