"""Eyeball check: fusion vs single-camera detections on real day/night test frames.

For a handful of test frames (some day, some night) draws a 3-column row:
  [thermal-only on thermal img]  [rgb-only on rgb img]  [fusion on thermal img]
Red = ground truth, lime = model prediction (+score). Lets you SEE whether fusion
finds pedestrians the single cameras miss -- especially in daytime, where the 12-point
gain lives. Runs on CPU so it doesn't disturb GPU training.

Run: PYTHONPATH= .venv/bin/python scripts/visualize_fusion.py
"""
from __future__ import annotations
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from prism.data.kaist import kaist_split
from prism.models.yolox import YOLOX
from prism.models.yolox_fusion import TwoStreamYOLOX
from prism.models.decode import decode_outputs, postprocess

ROOT = "data/raw/kaist-cvpr15"
OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.30          # clean threshold for the eye (eval uses 0.001)
DEV = "cpu"


def load(kind, path):
    m = (TwoStreamYOLOX if kind == "fusion" else YOLOX)(num_classes=1)
    m.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return m.eval().to(DEV)


@torch.no_grad()
def detect(model, *imgs):
    outs = model(*[i[None].to(DEV) for i in imgs])
    return postprocess(decode_outputs([o.float() for o in outs]), conf_thr=CONF)[0].cpu()


def draw(ax, img_chw, gt, preds, title, color=False):
    if color:                                            # RGB: show true colour (CHW->HWC)
        disp = img_chw.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(disp)
    else:                                                # thermal: single-channel grayscale
        ax.imshow((img_chw[0] * 255).numpy().astype("uint8"), cmap="gray")
    ax.set_title(title, fontsize=9); ax.axis("off")
    for b in gt:
        x1, y1, x2, y2 = b
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, edgecolor="red", lw=1.8))
    for p in preds:
        x1, y1, x2, y2, s, _ = p
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, edgecolor="lime", lw=1.4))
        ax.text(float(x1), float(y1) - 2, f"{s:.2f}", color="lime", fontsize=6)


def pick(split, n):
    """Indices of frames with 2-5 scored pedestrians, for readability."""
    base = kaist_split(ROOT, split)
    out = []
    for i in range(len(base)):
        s = base[i]
        npos = int((s["classes"] == 1).sum())
        if 2 <= npos <= 5:
            out.append((base, i));
        if len(out) >= n:
            break
    return out


def main():
    thermal = load("thermal", "outputs/baselines/thermal_seed0/thermal_seed0.pt")
    rgb = load("rgb", "outputs/baselines/rgb_seed0/rgb_seed0.pt")
    fusion = load("fusion", "outputs/fusion/seed0/fusion_seed0.pt")

    frames = [("DAY", b, i) for b, i in pick("test-day", 2)] + \
             [("NIGHT", b, i) for b, i in pick("test-night", 2)]

    fig, axes = plt.subplots(len(frames), 3, figsize=(13, 3.4 * len(frames)))
    for row, (tag, base, idx) in enumerate(frames):
        s = base[idx]
        rgb_img = s["rgb"].float() / 255.0
        thm_img = s["thermal"].float() / 255.0
        gt = s["boxes"][s["classes"] == 1]

        dt = detect(thermal, thm_img)
        dr = detect(rgb, rgb_img)
        df = detect(fusion, rgb_img, thm_img)

        draw(axes[row, 0], thm_img, gt, dt, f"{tag}  thermal-only ({len(dt)})")
        draw(axes[row, 1], rgb_img, gt, dr, f"{tag}  rgb-only ({len(dr)})", color=True)
        draw(axes[row, 2], thm_img, gt, df, f"{tag}  FUSION ({len(df)})")

    fig.suptitle("Red = ground truth   |   Lime = prediction (+score)   |   "
                 "GT count vs each model's detections", fontsize=11)
    fig.tight_layout()
    p = OUT / "fusion_vs_baselines_grid.png"
    fig.savefig(p, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")
    for tag, base, idx in frames:
        print(f"  {tag} frame: {base.ids[idx]}")


if __name__ == "__main__":
    main()
