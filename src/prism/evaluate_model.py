"""Turn a trained checkpoint into the standard KAIST number: MR^-2 All/Day/Night.

This is the bridge between training and the metric. It runs the model over a protocol
test split, collects detections, and feeds them to the Reasonable-subset evaluator.

Two details that decide whether the number is honest:

  1. **Collect detections at a very low confidence threshold.** MR^-2 sweeps the
     threshold itself -- walking detections high-score to low-score IS the sweep, and
     the FPPI axis runs out to 1 false positive per image. If we pre-filtered at, say,
     0.3 we would truncate the curve's right-hand side and report a flatteringly low
     miss rate. So conf_thr is 0.001 here, NOT the 0.3 used for visualisation.

  2. **Ground truth comes from the raw loader, not DetectionDataset.** The training
     wrapper drops ignore regions and remaps the class index; the evaluator needs the
     original person/ignore classes and occlusion flags so `reasonable_ignore` can do
     its job (people/cyclist/person? and small/occluded boxes absorb detections
     penalty-free rather than counting as misses).

Run: PYTHONPATH= .venv/bin/python -m prism.evaluate_model --ckpt checkpoints/thermal_seed0.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from prism.data.kaist import kaist_split
from prism.models.yolox import YOLOX
from prism.models.decode import decode_outputs, postprocess
from prism.eval.evaluate import evaluate

# Low on purpose -- see note (1) in the module docstring.
EVAL_CONF_THR = 0.001
EVAL_NMS_IOU = 0.5


@torch.no_grad()
def collect_frames(model, base, modality, device, batch_size=8):
    """Run the model over every frame of `base`; return evaluator-shaped dicts."""
    frames = []
    model.eval()
    for start in range(0, len(base), batch_size):
        samples = [base[i] for i in range(start, min(start + batch_size, len(base)))]
        imgs = torch.stack([s[modality].float() / 255.0 for s in samples]).to(device)
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            outputs = model(imgs)
        dets = postprocess(decode_outputs([o.float() for o in outputs]),
                           conf_thr=EVAL_CONF_THR, iou_thr=EVAL_NMS_IOU)
        for s, d in zip(samples, dets):
            d = d.cpu()
            frames.append({
                "det_boxes": d[:, :4].numpy(),
                "det_scores": d[:, 4].numpy(),
                "gt_boxes": s["boxes"].numpy(),
                "gt_classes": s["classes"].numpy(),      # 1 = person, 0 = ignore region
                "gt_occlusion": s["occlusion"].numpy(),
            })
    return frames


def evaluate_checkpoint(ckpt_path, root, modality=None, splits=("test", "test-day", "test-night"),
                        batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location="cpu")
    modality = modality or ck.get("modality", "rgb")

    model = YOLOX(num_classes=1)
    model.load_state_dict(ck["model"])
    model.to(device)

    results = {}
    for split in splits:
        base = kaist_split(root, split)
        frames = collect_frames(model, base, modality, device, batch_size)
        n_det = sum(len(f["det_scores"]) for f in frames)
        results[split] = {"mr": evaluate(frames), "frames": len(frames), "dets": n_det}
    return modality, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument("--modality", default=None, choices=[None, "rgb", "thermal"])
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    modality, res = evaluate_checkpoint(a.ckpt, a.root, a.modality, batch_size=a.batch_size)

    label = {"test": "All", "test-day": "Day", "test-night": "Night"}
    print(f"\nMR^-2 (Reasonable, lower is better) -- {a.ckpt}  [{modality}]")
    print(f"{'subset':<8}{'MR^-2':>10}{'frames':>9}{'dets':>10}")
    for split, r in res.items():
        pct = r["mr"] * 100
        print(f"{label[split]:<8}{pct:>9.2f}%{r['frames']:>9}{r['dets']:>10}")
    print("\nNOTE: comparable to published numbers only once the sanitized-train /"
          "\n      improved-test annotations are installed (see decisions log 2026-07-24).")


if __name__ == "__main__":
    main()
