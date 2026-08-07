"""The two-stream fusion model must assemble, batch both cameras, and learn."""

import torch

from prism.data.detection import fusion_collate
from prism.models.fusion import ConcatFuse, SumFuse
from prism.models.loss import YOLOXLoss
from prism.models.yolox import YOLOX
from prism.models.yolox_fusion import TwoStreamYOLOX


def test_concat_fuse_preserves_shape():
    f = ConcatFuse(64)
    a, b = torch.randn(2, 64, 32, 40), torch.randn(2, 64, 32, 40)
    assert f(a, b).shape == (2, 64, 32, 40)  # 2C in, C out


def test_sum_fuse_is_elementwise():
    f = SumFuse()
    a, b = torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4)
    assert torch.allclose(f(a, b), a + b)


def test_two_stream_output_matches_single_stream_shapes():
    m = TwoStreamYOLOX(num_classes=1)
    rgb, thm = torch.randn(2, 3, 512, 640), torch.randn(2, 3, 512, 640)
    outs = m(rgb, thm)
    assert [o.shape for o in outs] == [
        (2, 6, 64, 80),
        (2, 6, 32, 40),
        (2, 6, 16, 20),
    ]  # 4+1+1 = 6 per scale


def test_two_stream_has_more_params_than_single():
    single = sum(p.numel() for p in YOLOX(num_classes=1).parameters())
    fused = sum(p.numel() for p in TwoStreamYOLOX(num_classes=1).parameters())
    assert fused > single  # two backbones + fusion convs
    assert 1.4 < fused / single < 2.0  # shared neck/head -> not a full 2x


def test_single_stage_config_still_feeds_all_scales():
    """Phase-2 config: fusing at dark4 only must still produce all three head scales."""
    m = TwoStreamYOLOX(fuse_stages=(4,))
    outs = m(torch.randn(1, 3, 512, 640), torch.randn(1, 3, 512, 640))
    assert len(outs) == 3


def test_fusion_collate_stacks_both_streams():
    batch = [
        (torch.zeros(3, 512, 640), torch.ones(3, 512, 640), torch.zeros(0, 5)),
        (torch.zeros(3, 512, 640), torch.ones(3, 512, 640), torch.tensor([[0.0, 1, 2, 3, 4]])),
    ]
    rgb, thm, targets = fusion_collate(batch)
    assert rgb.shape == (2, 3, 512, 640) and thm.shape == (2, 3, 512, 640)
    assert isinstance(targets, list) and len(targets) == 2


def test_fusion_model_can_overfit_one_batch():
    """The end-to-end proof the two-stream path learns: memorise one image."""
    torch.manual_seed(0)
    model = TwoStreamYOLOX(num_classes=1).train()
    lossfn = YOLOXLoss(num_classes=1)
    rgb, thm = torch.randn(1, 3, 512, 640), torch.randn(1, 3, 512, 640)
    targets = [torch.tensor([[0.0, 300, 200, 340, 320]])]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss, _ = lossfn(model(rgb, thm), targets)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.5 * first, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
