"""Follow ONE real KAIST image through the pipeline, and draw what happens at each step.

    python scripts/trace_image.py

Writes outputs/figures/trace_<id>.png -- a single figure with three rows:
  Row 1  load        : raw RGB and thermal, with the ground-truth box (Phase 0 loader)
  Row 2  augment     : the same pair after a paired horizontal flip (Ch.2)
  Row 3  backbone    : the RGB stream's dark3/dark4/dark5 feature maps (Ch.4)

The backbone is UNTRAINED here (random weights), so the feature maps highlight edges and
texture, not pedestrians -- the point of row 3 is to *see the spatial shrink*: stride 8 -> 16 ->
32 turns a 512x640 image into a 64x80, then 32x40, then 16x20 summary.
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.data.kaist import parse_annotation
from prism.data.transforms import PairedHFlip
from prism.models.darknet import CSPDarknet

FRAME = "set06/V000/I01259"
ROOT = Path("data/raw/kaist-cvpr15")
OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
GREEN = (0, 200, 0)


def load_frame(fid):
    setn, vid, fr = fid.split("/")
    base = ROOT / "images" / setn / vid
    rgb = cv2.cvtColor(cv2.imread(str(base / "visible" / f"{fr}.jpg")), cv2.COLOR_BGR2RGB)
    thm = cv2.cvtColor(cv2.imread(str(base / "lwir" / f"{fr}.jpg")), cv2.COLOR_BGR2RGB)
    ann = parse_annotation(ROOT / "annotations-xml-new-sanitized" / setn / vid / f"{fr}.xml")
    return rgb, thm, ann


def draw(img, boxes, classes):
    img = img.copy()
    for (x1, y1, x2, y2), c in zip(boxes.tolist(), classes.tolist()):
        if c == 1:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
    return img


def feat_heat(fmap):
    """(1,C,H,W) -> a (H,W) heatmap = mean activation magnitude over channels, 0..1."""
    m = fmap[0].abs().mean(0).detach().numpy()
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    return m


def main():
    torch.manual_seed(0)                 # reproducible random-init feature maps
    rgb, thm, ann = load_frame(FRAME)
    boxes, classes = ann["boxes"], ann["classes"]

    # Row 2: paired flip (deterministic)
    to_chw = lambda a: torch.from_numpy(a).permute(2, 0, 1).float()
    rgb_t, thm_t, boxes_t = PairedHFlip(p=1.0)(
        to_chw(rgb), to_chw(thm), torch.from_numpy(boxes).float()
    )
    rgb_f = rgb_t.permute(1, 2, 0).byte().numpy()
    thm_f = thm_t.permute(1, 2, 0).byte().numpy()

    # Row 3: RGB through the (untrained) backbone
    backbone = CSPDarknet().eval()
    with torch.no_grad():
        c3, c4, c5 = backbone((to_chw(rgb) / 255.0).unsqueeze(0))

    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    for a in ax.ravel():
        a.axis("off")

    # Row 1 -- load
    ax[0, 0].imshow(draw(rgb, boxes, classes)); ax[0, 0].set_title("1a. RGB (visible), loaded + box")
    ax[0, 1].imshow(draw(thm, boxes, classes)); ax[0, 1].set_title("1b. Thermal (LWIR), same box")
    ax[0, 2].text(0.0, 0.5,
                  "LOAD (Phase 0)\n\n"
                  "- one paired sample\n"
                  "- box: x,y,w,h -> x1,y1,x2,y2\n"
                  "- 'person' scored; others ignored\n"
                  f"- image is {rgb.shape[1]}x{rgb.shape[0]} (WxH)",
                  fontsize=11, va="center")

    # Row 2 -- augment
    ax[1, 0].imshow(draw(rgb_f, boxes_t, classes)); ax[1, 0].set_title("2a. RGB after paired flip")
    ax[1, 1].imshow(draw(thm_f, boxes_t, classes)); ax[1, 1].set_title("2b. Thermal: SAME flip")
    ax[1, 2].text(0.0, 0.5,
                  "AUGMENT (Ch.2)\n\n"
                  "- geometric op hits BOTH\n  streams identically\n"
                  "- box moves with the pixels\n"
                  "- streams stay pixel-aligned\n  (the guardrail test proves it)",
                  fontsize=11, va="center")

    # Row 3 -- backbone feature maps
    for a, (name, f, stride) in zip(
        ax[2], [("dark3", c3, 8), ("dark4", c4, 16), ("dark5", c5, 32)]
    ):
        h = feat_heat(f)
        a.imshow(h, cmap="inferno")
        a.set_title(f"3. {name}: {f.shape[2]}x{f.shape[3]}  (stride {stride}, {f.shape[1]} ch)")
    fig.text(0.5, 0.02,
             "BACKBONE (Ch.4): the image is squeezed to three summaries. dark3 keeps detail "
             "(stride 8), dark5 keeps meaning (stride 32). Untrained here -> edges, not people; "
             "the point is the spatial shrink. These three maps are where fusion will attach.",
             ha="center", fontsize=10, wrap=True)

    fig.suptitle(f"One KAIST image through PRISM so far  --  {FRAME}", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    out = OUT / f"trace_{FRAME.replace('/', '_')}.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    # keep the committed doc copies in sync (markdown doc + textbook zip)
    import shutil
    for dst in ("docs/figures/pipeline_trace_phase1.png",
                "docs/textbook/figures/pipeline_trace_phase1.png"):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, dst)
    print("wrote", out, "(+ synced docs/figures and docs/textbook/figures)")
    print(f"shapes: dark3 {tuple(c3.shape)}  dark4 {tuple(c4.shape)}  dark5 {tuple(c5.shape)}")


if __name__ == "__main__":
    main()
