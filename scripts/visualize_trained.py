"""Before/after visual: what the trained detector sees, plus the loss curve.

Loads checkpoints/thermal_seed0.pt (the overfit-smoke checkpoint), runs it on one
real KAIST thermal frame, and draws predictions (lime) against ground truth (red).
A second panel plots the training loss curve. Saves to docs/figures/.

Run: PYTHONPATH= .venv/bin/python scripts/visualize_trained.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from prism.data.kaist import PairedKAIST
from prism.data.detection import DetectionDataset
from prism.models.yolox import YOLOX
from prism.models.decode import decode_outputs, postprocess
from prism.train import positive_indices

ROOT = "data/raw/kaist-cvpr15"
OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)

# --- the loss curve numbers from Krithin's smoke run (thermal, 300 frames, 20 ep) ---
STEP = list(range(20, 760, 20))
LOSS = [42.185,20.350,10.491,8.569,7.572,6.304,5.752,4.783,4.420,3.310,3.397,2.998,
        3.771,2.938,2.510,2.524,2.066,2.458,2.561,1.821,1.996,1.847,1.392,2.161,1.886,
        1.474,1.743,1.573,1.810,1.213,1.305,1.257,1.214,1.394,1.127,1.104,1.248]
OBJ  = [37.354,15.689,6.327,4.827,4.273,3.403,2.542,2.270,1.514,1.132,1.200,0.755,1.251,
        0.984,0.526,0.631,0.444,0.549,0.609,0.381,0.487,0.284,0.271,0.495,0.434,0.387,
        0.268,0.282,0.383,0.247,0.255,0.223,0.238,0.243,0.238,0.163,0.286]
IOU  = [0.696,0.621,0.616,0.572,0.499,0.438,0.483,0.368,0.441,0.308,0.314,0.321,0.371,
        0.274,0.277,0.263,0.218,0.266,0.271,0.191,0.201,0.209,0.141,0.224,0.192,0.137,
        0.196,0.168,0.189,0.120,0.132,0.130,0.121,0.148,0.109,0.116,0.119]


def run(model, img):
    model.eval()
    with torch.no_grad():
        return postprocess(decode_outputs(model(img[None])), conf_thr=0.3, iou_thr=0.5)[0]


def draw(ax, disp, gt, pred, title):
    ax.imshow(disp, cmap="gray"); ax.set_title(title, fontsize=12); ax.axis("off")
    for b in gt:                                  # ground truth = red
        x1, y1, x2, y2 = b[1:5]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, edgecolor="red", lw=2))
    for p in pred:                                # predictions = lime, with score
        x1, y1, x2, y2, s, _ = p
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, edgecolor="lime", lw=1.5))
        ax.text(float(x1), float(y1) - 3, f"{s:.2f}", color="lime", fontsize=8)


def main():
    base = PairedKAIST(ROOT)
    ds = DetectionDataset(base, modality="thermal", transforms=None)

    # pick a training frame with a handful of pedestrians (clear to read)
    idx = positive_indices(base)[:300]
    choice = next((i for i in idx if 2 <= ds[i][1].shape[0] <= 5), idx[0])
    img, target = ds[choice]                      # img in [0,1], (3,512,640)
    disp = (img[0].numpy() * 255).astype("uint8")  # one channel for grayscale display

    trained = YOLOX(num_classes=1)
    trained.load_state_dict(torch.load("checkpoints/thermal_seed0.pt",
                                       map_location="cpu")["model"])
    untrained = YOLOX(num_classes=1)              # random init (with prior bias)

    pred_tr = run(trained, img)
    # untrained at a low threshold, to reveal its random scatter (prior bias silences 0.3)
    with torch.no_grad():
        pred_un = postprocess(decode_outputs(untrained(img[None])), conf_thr=0.05)[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    draw(axes[0], disp, target, pred_un,
         f"BEFORE — untrained: {len(pred_un)} random boxes (thr 0.05)")
    draw(axes[1], disp, target, pred_tr,
         f"AFTER — trained: {len(pred_tr)} boxes on people (thr 0.30)")
    fig.suptitle("Red = ground-truth pedestrians   |   Lime = model predictions",
                 fontsize=13)
    fig.tight_layout()
    f1 = OUT / "trained_before_after.png"
    fig.savefig(f1, dpi=110, bbox_inches="tight"); plt.close(fig)

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(STEP, LOSS, "-o", ms=3, label="total loss", color="#1f77b4")
    ax.plot(STEP, OBJ, "-s", ms=3, label="objectness (background suppression)", color="#d62728")
    ax.plot(STEP, IOU, "-^", ms=3, label="box IoU loss (fit)", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("loss (log scale)")
    ax.set_title("The loop descends — thermal, 300 frames, 20 epochs")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    f2 = OUT / "loss_curve.png"
    fig.savefig(f2, dpi=110, bbox_inches="tight"); plt.close(fig)

    print(f"frame: {base.ids[choice]}  gt={target.shape[0]}  "
          f"trained_pred={len(pred_tr)}  untrained_pred={len(pred_un)}")
    print(f"saved {f1}")
    print(f"saved {f2}")


if __name__ == "__main__":
    main()
