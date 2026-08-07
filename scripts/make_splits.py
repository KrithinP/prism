"""Freeze the sequence-disjoint train/validation split (TODO.md Gate B step 11).

Reads the official sanitized train list, splits at set/video granularity with a
predeclared seed, writes train.txt/val.txt + split_manifest.json to splits/, and
prints the Day/Night/scale/occlusion strata breakdown (frames and boxes).

Run:  PYTHONPATH= .venv/bin/python scripts/make_splits.py --root data/raw/kaist-cvpr15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prism.data.kaist import PairedKAIST, parse_annotation
from prism.data.splits import split_by_video, write_split_files

SPLIT_DIR = Path("splits")
SPLIT_SEED = 20260730  # predeclared; freezing a split is a reviewed act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--val-fraction", type=float, default=0.15, help="target fraction of FRAMES in validation"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="predeclared split seed; changing it invalidates old runs",
    )
    ap.add_argument("--out-dir", default=str(SPLIT_DIR))
    a = ap.parse_args()

    base = PairedKAIST(
        a.root,
        ann_dirname="annotations-sanitized-train",
        split_file=str(Path(a.root) / "imageSets" / "train-sanitized.txt"),
    )
    ids = base.ids
    print(f"{len(ids)} train frames, {len({i.rsplit('/', 1)[0] for i in ids})} videos")

    train_ids, val_ids = split_by_video(ids, a.val_fraction, a.seed)
    out_dir = Path(a.out_dir)
    manifest = write_split_files(
        train_ids, val_ids, out_dir, source_ids=ids, seed=a.seed, val_fraction=a.val_fraction
    )
    print(f"train {len(train_ids)} frames / {manifest['train']['videos']} videos")
    print(f"val   {len(val_ids)} frames / {manifest['val']['videos']} videos")
    print(f"manifest: {out_dir / 'split_manifest.json'}")

    # strata report: frames + boxes by Day/Night proxy (frame index parity is NOT
    # day/night; this is frame counts only), and by scale/occlusion per annotation.
    for split, ids_ in (("train", train_ids), ("val", val_ids)):
        frames, boxes = 0, 0
        small = occluded = 0
        for fid in ids_:
            frames += 1
            _, _, xml = base._paths(fid)
            ann = parse_annotation(xml)
            pos = ann["classes"] == 1
            boxes += int(pos.sum())
            if pos.any():
                small += int(((ann["boxes"][pos, 3] - ann["boxes"][pos, 1]) < 55).sum())
                occluded += int((ann["occlusion"][pos] == 2).sum())
        print(
            f"{split}: {frames} frames, {boxes} positive boxes "
            f"({small} <55px, {occluded} heavily occluded)"
        )


if __name__ == "__main__":
    main()
