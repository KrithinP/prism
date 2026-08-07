"""Draw KAIST boxes on RGB and thermal, save side-by-side panels.

    python scripts/visualize_pairs.py

Writes outputs/vis/*.jpg. Open them and LOOK. Chapter 1 exit check: find 5 pairs
where the same box fits the person in one modality but sits slightly off in the
other -- that gap is the residual misalignment the whole project is about.
"""

import random
from pathlib import Path

import cv2
import numpy as np

from prism.data.kaist import PairedKAIST

OUT = Path("outputs/vis")
OUT.mkdir(parents=True, exist_ok=True)  # make the output folder if it doesn't exist

GREEN = (0, 255, 0)  # person  (a scored positive)
GRAY = (160, 160, 160)  # ignore  (people / cyclist / person?)


def chw_to_hwc(t):
    # (3, H, W) RGB tensor  ->  (H, W, 3) uint8 RGB numpy array, for OpenCV
    return t.permute(1, 2, 0).numpy().astype(np.uint8)


def draw(img, boxes, classes):
    img = img.copy()  # don't scribble on the original
    for (x1, y1, x2, y2), c in zip(boxes.tolist(), classes.tolist(), strict=True):
        color = GREEN if c == 1 else GRAY
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return img


def main(n=12, seed=0):
    ds = PairedKAIST("data/raw/kaist-cvpr15-preview")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)  # random but reproducible (fixed seed)

    written = 0
    for i in order:
        s = ds[i]
        if int((s["classes"] == 1).sum()) == 0:  # skip frames with no person
            continue
        rgb = draw(chw_to_hwc(s["rgb"]), s["boxes"], s["classes"])
        thm = draw(chw_to_hwc(s["thermal"]), s["boxes"], s["classes"])
        panel = np.hstack([rgb, thm])  # RGB | thermal
        fname = s["id"].replace("/", "_") + ".jpg"
        cv2.imwrite(str(OUT / fname), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        written += 1
        if written >= n:
            break
    print(f"wrote {written} panels to {OUT}/  -- open them and look.")


if __name__ == "__main__":  # only run main() when executed directly
    main()
