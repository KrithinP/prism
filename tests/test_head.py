
"""The head must emit 4+1+num_classes channels at each of the three scales."""
import torch

from prism.models.darknet import CSPDarknet
from prism.models.pafpn import PAFPN
from prism.models.head import DecoupledHead


def test_head_output_shapes_single_class():
    backbone, neck, head = CSPDarknet().eval(), PAFPN().eval(), DecoupledHead(num_classes=1).eval()
    feats = neck(backbone(torch.randn(1, 3, 512, 640)))
    outs = head(feats)
    # 4 (box) + 1 (obj) + 1 (class) = 6 channels per cell
    assert outs[0].shape == (1, 6, 64, 80)
    assert outs[1].shape == (1, 6, 32, 40)
    assert outs[2].shape == (1, 6, 16, 20)


def test_head_total_predictions():
    backbone, neck, head = CSPDarknet().eval(), PAFPN().eval(), DecoupledHead(num_classes=1).eval()
    outs = head(neck(backbone(torch.randn(1, 3, 512, 640))))
    total = sum(o.shape[2] * o.shape[3] for o in outs)
    assert total == 64 * 80 + 32 * 40 + 16 * 20 == 6720
