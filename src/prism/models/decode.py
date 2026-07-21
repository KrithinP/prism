
"""Turn the head's raw per-cell outputs into boxes in pixel coordinates.

Each cell predicts its box RELATIVE to itself:
  raw[0:2] = center offset from the cell's own grid position
  raw[2:4] = log-size (exponentiate, then scale by stride)
  raw[4]   = objectness logit
  raw[5:]  = class logits
See textbook Ch.4.
"""
from __future__ import annotations

import torch
from torchvision.ops import nms


def decode_outputs(outputs, strides=(8, 16, 32)):
    """outputs: list of (B, 4+1+C, H, W). Returns (B, total_cells, 4+1+C) with the
    box already in (x1, y1, x2, y2) pixel coordinates and scores sigmoid'd."""
    decoded = []
    for out, stride in zip(outputs, strides):
        B, ch, H, W = out.shape
        # (B, ch, H, W) -> (B, H*W, ch)
        out = out.flatten(2).permute(0, 2, 1)

        # build the grid of cell positions for this scale
        yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((xv, yv), dim=2).reshape(1, H * W, 2).to(out.device).float()

        # center = (offset + cell index) * stride ; size = exp(raw) * stride
        xy = (out[..., 0:2] + grid) * stride
        wh = torch.exp(out[..., 2:4]) * stride
        obj = out[..., 4:5].sigmoid()
        cls = out[..., 5:].sigmoid()

        x1y1 = xy - wh / 2                          # center -> corners
        x2y2 = xy + wh / 2
        decoded.append(torch.cat([x1y1, x2y2, obj, cls], dim=2))
    return torch.cat(decoded, dim=1)                # (B, total_cells, 4+1+C)


def postprocess(decoded, conf_thr=0.3, iou_thr=0.5):
    """Keep confident boxes and run NMS, per image. Returns a list (one per image)
    of (K, 6) tensors: x1,y1,x2,y2,score,class."""
    results = []
    for pred in decoded:                            # pred: (total_cells, 4+1+C)
        boxes = pred[:, :4]
        scores = pred[:, 4] * pred[:, 5:].max(dim=1).values   # obj * best class
        cls = pred[:, 5:].argmax(dim=1)
        keep = scores > conf_thr
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        if boxes.numel() == 0:
            results.append(torch.zeros((0, 6)))
            continue
        k = nms(boxes, scores, iou_thr)
        results.append(torch.cat([boxes[k], scores[k, None], cls[k, None].float()], dim=1))
    return results
