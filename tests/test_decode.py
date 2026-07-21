
"""Decoding must produce valid pixel boxes from the model's raw outputs."""
import torch

from prism.models.yolox import YOLOX
from prism.models.decode import decode_outputs, postprocess


def test_decode_shapes_and_ranges():
    model = YOLOX(num_classes=1).eval()
    outs = model(torch.randn(1, 3, 512, 640))
    dec = decode_outputs(outs)
    assert dec.shape == (1, 6720, 6)               # 4 box + 1 obj + 1 class
    # obj and class scores are probabilities after sigmoid
    assert (dec[..., 4:] >= 0).all() and (dec[..., 4:] <= 1).all()


def test_postprocess_returns_boxes():
    model = YOLOX(num_classes=1).eval()
    outs = model(torch.randn(1, 3, 512, 640))
    dets = postprocess(decode_outputs(outs), conf_thr=0.0)   # keep everything, then NMS
    assert isinstance(dets, list) and dets[0].shape[1] == 6  # x1,y1,x2,y2,score,class
