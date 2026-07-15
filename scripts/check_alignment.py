"""Amplify and inspect RGB<->thermal registration on KAIST.

The naked eye cannot see a 3-5 px offset on a dark night frame. So we amplify:

  1. Brighten the night RGB (percentile stretch) so it has visible edges at all.
  2. False-colour overlay: RED = thermal, CYAN = RGB. Where the two AGREE, edges
     are white/grey. Where they are OFFSET, you get a red ghost beside a cyan ghost
     -- misregistration becomes a colour fringe you can actually see.
  3. Per-pedestrian 5x zoom crops, RGB | thermal, so you can inspect one body.

    python scripts/check_alignment.py
"""
from pathlib import Path

import cv2
import numpy as np

from prism.data.kaist import PairedKAIST

OUT = Path("outputs/align")
OUT.mkdir(parents=True, exist_ok=True)


def gray(chw):
    hwc = chw.permute(1, 2, 0).numpy().astype(np.uint8)
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)


def stretch(g):
    """Percentile contrast stretch -- pulls a dark night frame up to full range."""
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi <= lo:
        return g
    return np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def main(n_frames=6, zoom=5, seed=0):
    ds = PairedKAIST("data/raw/kaist-cvpr15-preview")
    order = list(range(len(ds)))
    import random
    random.Random(seed).shuffle(order)

    made = 0
    for i in order:
        s = ds[i]
        # need a reasonably tall person to inspect at zoom
        tall = [(b, c) for b, c in zip(s["boxes"].tolist(), s["classes"].tolist())
                if c == 1 and (b[3] - b[1]) >= 60]
        if not tall:
            continue

        r = stretch(gray(s["rgb"]))       # brightened RGB luminance
        t = stretch(gray(s["thermal"]))   # normalised thermal

        # false-colour overlay: R=thermal, G=B=RGB  -> offsets show as red/cyan fringe
        overlay = np.dstack([t, r, r])
        cv2.imwrite(str(OUT / f"{s['id'].replace('/', '_')}_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # zoomed crops of the first tall person
        (x1, y1, x2, y2), _ = tall[0]
        pad = 8
        x1, y1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        x2, y2 = min(int(x2) + pad, r.shape[1]), min(int(y2) + pad, r.shape[0])
        rc = cv2.resize(r[y1:y2, x1:x2], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        tc = cv2.resize(t[y1:y2, x1:x2], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        h = max(rc.shape[0], tc.shape[0])
        pane = np.zeros((h, rc.shape[1] + tc.shape[1] + 10), np.uint8)
        pane[:rc.shape[0], :rc.shape[1]] = rc
        pane[:tc.shape[0], rc.shape[1] + 10:] = tc
        cv2.imwrite(str(OUT / f"{s['id'].replace('/', '_')}_zoom.jpg"), pane)

        made += 1
        if made >= n_frames:
            break
    print(f"wrote {made} overlay+zoom pairs to {OUT}/")


if __name__ == "__main__":
    main()
