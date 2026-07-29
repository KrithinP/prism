"""Paired KAIST loader: one frame index -> (rgb, thermal, boxes, classes, occlusion)."""
from __future__ import annotations
import xml.etree.ElementTree as ET   # ET = ElementTree, Python's built-in XML parser
from pathlib import Path

import cv2                            # OpenCV: image decode
import numpy as np
import torch
from torch.utils.data import Dataset

from prism.data.kaist_constants import (
    POSITIVE_CLASSES, VISIBLE_DIRNAME, LWIR_DIRNAME,
)


def parse_annotation(xml_path: Path) -> dict:
    """Read one KAIST XML into arrays. Boxes come out as (x1,y1,x2,y2) pixels."""
    boxes, classes, occ = [], [], []
    if xml_path.exists():                         # frames with no people have no <object>
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            b = obj.find("bndbox")
            x = float(b.findtext("x")); y = float(b.findtext("y"))
            w = float(b.findtext("w")); h = float(b.findtext("h"))
            # THE conversion: KAIST stores top-left + width/height. We want two corners.
            boxes.append([x, y, x + w, y + h])
            # 1 = a scored positive ('person'); 0 = an ignore region (people/cyclist/person?).
            classes.append(1 if name in POSITIVE_CLASSES else 0)
            occ.append(int(obj.findtext("occlusion", default="0")))
    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "classes": np.asarray(classes, dtype=np.int64),
        "occlusion": np.asarray(occ, dtype=np.int64),
    }


class PairedKAIST(Dataset):
    """RGB+thermal pairs from a KAIST-style tree.

    root/
      images/setXX/VYYY/{visible,lwir}/IZZZZZ.jpg
      annotations-xml-new-sanitized/setXX/VYYY/IZZZZZ.xml
    """

    def __init__(self, root: str, ann_dirname: str = "annotations-xml-new-sanitized",
                 split_file: str | None = None):
        self.root = Path(root)
        self.ann_root = self.root / ann_dirname
        if split_file is not None:
            # Obey the official protocol list (set-split + frame-sampling baked in).
            ids = [ln.strip() for ln in Path(split_file).read_text().splitlines() if ln.strip()]
            self.ids = [fid for fid in ids if self._pair_exists(fid)]
        else:
            # Fallback: discover every frame that has BOTH modalities on disk.
            vis = sorted((self.root / "images").rglob(f"{VISIBLE_DIRNAME}/*.jpg"))
            self.ids = [self._id_from_visible(p) for p in vis
                        if p.with_name(p.name).parent.parent.joinpath(LWIR_DIRNAME, p.name).exists()]

    def _pair_exists(self, fid: str) -> bool:
        vis_p, lwir_p, _ = self._paths(fid)
        return vis_p.exists() and lwir_p.exists()

    @staticmethod
    def _id_from_visible(p: Path) -> str:
        # images/set04/V001/visible/I00001.jpg  ->  "set04/V001/I00001"
        return f"{p.parents[2].name}/{p.parents[1].name}/{p.stem}"

    def _paths(self, fid: str):
        setname, vid, frame = fid.split("/")
        base = self.root / "images" / setname / vid
        return (base / VISIBLE_DIRNAME / f"{frame}.jpg",
                base / LWIR_DIRNAME / f"{frame}.jpg",
                self.ann_root / setname / vid / f"{frame}.xml")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        fid = self.ids[i]
        vis_p, lwir_p, xml_p = self._paths(fid)
        # cv2 decodes as BGR; flip to RGB so channel order matches every pretrained backbone.
        rgb = cv2.cvtColor(cv2.imread(str(vis_p)), cv2.COLOR_BGR2RGB)
        thm = cv2.cvtColor(cv2.imread(str(lwir_p)), cv2.COLOR_BGR2RGB)
        ann = parse_annotation(xml_p)
        # HWC uint8 -> CHW float tensor. (H=height, W=width, C=channel.)
        to_chw = lambda a: torch.from_numpy(a).permute(2, 0, 1).contiguous()
        return {
            "id": fid,
            "rgb": to_chw(rgb),
            "thermal": to_chw(thm),
            "boxes": torch.from_numpy(ann["boxes"]),
            "classes": torch.from_numpy(ann["classes"]),
            "occlusion": torch.from_numpy(ann["occlusion"]),
        }


# Official KAIST protocol splits: (list file, annotation source).
#
# These point at the REAL annotation sets, converted from bbGt to XML on 2026-07-29
# (scripts/convert_kaist_bbgt.py). Note the training list: the field's standard
# training set is the 7601 sanitized frames (Li et al./MSDS-RCNN), NOT every-2nd-frame
# -- `train-sanitized.txt` is that list. Test evaluates on the improved annotations
# (Liu et al. 2016), which cover exactly the 2252 frames of test-all-20.
# (The old annotations-xml-new{,-sanitized} dirs were a byte-identical duplicate; they
# are left in place but no longer referenced. See decisions log 2026-07-24 / 07-29.)
_SPLITS = {
    "train":      ("train-sanitized.txt", "annotations-sanitized-train"),
    "test":       ("test-all-20.txt",     "annotations-improved-test"),
    "test-day":   ("test-day-20.txt",     "annotations-improved-test"),
    "test-night": ("test-night-20.txt",   "annotations-improved-test"),
}


def kaist_split(root: str, split: str) -> "PairedKAIST":
    """Build a PairedKAIST for a named protocol split ('train'/'test'/'test-day'/'test-night')."""
    fname, ann = _SPLITS[split]
    return PairedKAIST(root, ann_dirname=ann, split_file=str(Path(root) / "imageSets" / fname))
