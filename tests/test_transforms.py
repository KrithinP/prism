"""Correspondence guardrail: geometric augs must move both streams identically."""
import torch

# pyrefly: ignore [missing-import]
from prism.data.transforms import PairedHFlip


def test_hflip_keeps_streams_aligned_and_moves_boxes():
    H, W = 32, 64
    rgb = torch.zeros(3, H, W)
    thermal = torch.zeros(3, H, W)
    # identical bright marker at the SAME pixel in both modalities
    r, c = 10, 12
    rgb[:, r, c] = 255
    thermal[:, r, c] = 255
    boxes = torch.tensor([[c - 2.0, r - 2.0, c + 2.0, r + 2.0]])  # box around the marker

    # p=1.0 forces the flip so the test is deterministic
    rgb_f, thermal_f, boxes_f = PairedHFlip(p=1.0)(rgb, thermal, boxes)

    # 1) both streams flipped to the SAME new location
    rgb_bright = (rgb_f[0] == 255).nonzero()
    thm_bright = (thermal_f[0] == 255).nonzero()
    assert torch.equal(rgb_bright, thm_bright), "streams diverged — pairing broken"
    new_c = W - 1 - c
    assert rgb_bright[0].tolist() == [r, new_c], "flip landed in the wrong place"

    # 2) the flipped box still contains the flipped marker
    x1, y1, x2, y2 = boxes_f[0].tolist()
    assert x1 <= new_c <= x2 and y1 <= r <= y2, "box did not follow the pixels"


def test_no_flip_when_p_zero():
    rgb = torch.rand(3, 8, 8)
    thermal = torch.rand(3, 8, 8)
    boxes = torch.tensor([[1.0, 1.0, 4.0, 4.0]])
    out_rgb, _, out_boxes = PairedHFlip(p=0.0)(rgb, thermal, boxes)
    assert torch.equal(out_rgb, rgb) and torch.equal(out_boxes, boxes)
