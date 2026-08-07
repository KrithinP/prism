"""Decoding must produce valid pixel boxes from the model's raw outputs."""

import pytest
import torch

from prism.models.decode import decode_outputs, postprocess
from prism.models.yolox import YOLOX


def test_decode_shapes_and_ranges():
    model = YOLOX(num_classes=1).eval()
    outs = model(torch.randn(1, 3, 512, 640))
    dec = decode_outputs(outs)
    assert dec.shape == (1, 6720, 6)  # 4 box + 1 obj + 1 class
    # obj and class scores are probabilities after sigmoid
    assert (dec[..., 4:] >= 0).all() and (dec[..., 4:] <= 1).all()


def test_postprocess_returns_boxes():
    model = YOLOX(num_classes=1).eval()
    outs = model(torch.randn(1, 3, 512, 640))
    dets = postprocess(decode_outputs(outs), conf_thr=0.0)  # keep everything, then NMS
    assert isinstance(dets, list) and dets[0].shape[1] == 6  # x1,y1,x2,y2,score,class


def test_decode_accepts_a_reduced_head_but_rejects_missing_strides():
    output = torch.zeros(1, 6, 1, 1)
    assert decode_outputs([output]).shape == (1, 1, 6)
    with pytest.raises(ValueError, match="output levels"):
        decode_outputs([output, output], strides=(8,))


def test_empty_postprocess_result_preserves_input_device_and_dtype():
    decoded = torch.zeros(1, 1, 6, dtype=torch.float64)
    result = postprocess(decoded, conf_thr=1.0)[0]
    assert result.shape == (0, 6)
    assert result.device == decoded.device
    assert result.dtype == decoded.dtype
