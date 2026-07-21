"""Run the UNTRAINED detector on a real KAIST frame and draw its boxes.

    python scripts/demo_untrained.py

The point: prove the whole path turns (image -> boxes) BEFORE training. The boxes
will be nonsense -- scattered, tiny, random -- because the weights are random. That
nonsense is the 'before' picture. After training, these snap onto the pedestrian.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from prism.data.kaist import parse_annotation
from prism.models.yolox import YOLOX
from prism.models.decode import decode_outputs, postprocess

FRAME = "set06/V000/I01259"
ROOT = Path("data/raw/kaist-cvpr15")
OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    torch.manual_seed(0)
    setn, vid, fr = FRAME.split("/")
    base = ROOT / "images" / setn / vid
    rgb = cv2.cvtColor(cv2.imread(str(base / "visible" / f"{fr}.jpg")), cv2.COLOR_BGR2RGB)
    ann = parse_annotation(ROOT / "annotations-xml-new-sanitized" / setn / vid / f"{fr}.xml")

    model = YOLOX(num_classes=1).eval()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        dets = postprocess(decode_outputs(model(x)), conf_thr=0.20, iou_thr=0.5)[0]

    # keep the 60 highest-scoring boxes so the picture is readable
    if len(dets):
        dets = dets[dets[:, 4].argsort(descending=True)][:60]

    canvas = rgb.copy()
    # predictions in RED (what the untrained net guessed)
    for x1, y1, x2, y2, score, _ in dets.tolist():
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    # ground truth in GREEN (where the pedestrian actually is)
    for (bx1, by1, bx2, by2), c in zip(ann["boxes"].tolist(), ann["classes"].tolist()):
        if c == 1:
            cv2.rectangle(canvas, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 220, 0), 2)

    out = OUT / "demo_untrained.png"
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"drew {len(dets)} predicted boxes (RED) vs ground truth (GREEN) -> {out}")


if __name__ == "__main__":
    main()
