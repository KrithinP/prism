"""Convert the REAL KAIST annotations (bbGt .txt) into our XML schema.

The genuine annotation sets ship in Piotr Dollar's bbGt v3 text format, not the nested
XML tree our loader reads:

    % bbGt version=3
    <label> <x> <y> <w> <h> <occ> <xv> <yv> <wv> <hv> <ign> <ang>

  - sanitized TRAIN (Li et al. / MSDS-RCNN): 7601 files, sets 00-05. THIS is the standard
    training set the literature uses -- smaller and different from our earlier guess
    (train-all-02, 25086). It defines both the labels AND which frames we train on.
  - improved TEST (Liu et al. 2016): 2252 files, exactly matching test-all-20.

This writes fresh XML into two clean dirs (leaving the old duplicate dirs untouched):
    annotations-sanitized-train/setXX/VYYY/IZZZZZ.xml
    annotations-improved-test/setXX/VYYY/IZZZZZ.xml
and emits imageSets/train-sanitized.txt (the 7601-frame training list).

Field mapping (verified): label -> <name>; x,y,w,h -> <bndbox>; field6 -> <occlusion>
(0/1/2 = none/partial/heavy, same as ours). All `ign` flags are 0 in this data. Labels
seen: person (scored) + people/cyclist/person?/person?a (all ignore, since only exact
'person' is positive).

Run:
  PYTHONPATH= .venv/bin/python scripts/convert_kaist_bbgt.py \
      --train-src ~/Downloads/sanitized_annotations \
      --test-src  ~/Downloads/annotations_KAIST_test_set
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape


IMG_W, IMG_H = 640, 512


def clip_object(o, stats):
    """Clip a box to the image, dropping it if no area survives.

    Real KAIST ships pedestrians truncated by the frame edge (x+w>640, x<0) and a
    small number of degenerate (w<=0/h<=0) boxes. The standard, evaluator-consistent
    handling is clip-then-drop: keep the visible part of an edge-truncated box, and
    discard a box that has no area left. Returns a new object dict with clipped
    x/y/w/h, or ``None`` to drop it. ``stats`` counts what happened.
    """
    x1 = max(0, min(o["x"], IMG_W))
    y1 = max(0, min(o["y"], IMG_H))
    x2 = max(0, min(o["x"] + o["w"], IMG_W))
    y2 = max(0, min(o["y"] + o["h"], IMG_H))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        stats["dropped"] += 1
        return None
    if (x1, y1, x2 - x1, y2 - y1) != (o["x"], o["y"], o["w"], o["h"]):
        stats["clipped"] += 1
    return dict(o, x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def parse_bbgt(path: str):
    """Return objects from one bbGt file, including its explicit ignore flag."""
    objs = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("%"):
                continue
            p = ln.split()
            # a valid object line starts with a label token and has >= 6 fields
            if len(p) < 6 or not p[0][0].isalpha():
                continue
            try:
                x, y, w, h, occ = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])
            except ValueError:
                continue
            try:
                ignore = bool(int(float(p[10]))) if len(p) > 10 else False
            except ValueError:
                ignore = False
            objs.append(
                {
                    "name": p[0],
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "occ": occ,
                    "ignore": ignore,
                }
            )
    return objs


def write_xml(dest: Path, fid: str, objs):
    """Write objs to dest/<fid>.xml in the schema parse_annotation() reads."""
    out = dest / f"{fid}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "<annotation>",
        f"  <filename>{fid}</filename>",
        "  <size><width>640</width><height>512</height><depth>3</depth></size>",
    ]
    for o in objs:
        lines += [
            "  <object>",
            f"    <name>{escape(o['name'])}</name>",
            f"    <bndbox><x>{o['x']}</x><y>{o['y']}</y><w>{o['w']}</w><h>{o['h']}</h></bndbox>",
            "    <pose>unknown</pose><truncated>0</truncated><difficult>0</difficult>",
            f"    <occlusion>{o['occ']}</occlusion>",
            "  </object>",
        ]
    lines.append("</annotation>")
    out.write_text("\n".join(lines))


def fid_from_name(txt_path: str) -> str:
    """set02_V001_I00145.txt -> set02/V001/I00145"""
    base = os.path.basename(txt_path)[:-4]
    return base.replace("_", "/")


def convert_tree(src: str, dest: Path, label_stats, occ_stats, clip_stats):
    ids = []
    for f in sorted(glob.glob(f"{src}/**/set*.txt", recursive=True)):
        fid = fid_from_name(f)
        if fid.count("/") != 2:
            continue
        objs = parse_bbgt(f)
        converted = []
        for o in objs:
            scored = o["name"] == "person" and not o.get("ignore", False)
            mapped = clip_object(dict(o, name=("person" if scored else "people")), clip_stats)
            if mapped is None:
                continue
            converted.append(mapped)
            label_stats[mapped["name"]] += 1
            occ_stats[o["occ"]] += 1
        write_xml(dest, fid, converted)
        ids.append(fid)
    return sorted(set(ids))


def fid_from_json_filename(filename: str) -> str:
    parts = Path(filename).parts
    for index, part in enumerate(parts):
        if part.startswith("set") and len(parts) >= index + 3:
            video = parts[index + 1]
            frame = Path(parts[-1]).stem
            return f"{part}/{video}/{frame}"
    raise ValueError(f"cannot derive KAIST frame id from {filename!r}")


def convert_test_json(src: Path, dest: Path, label_stats, occ_stats, clip_stats) -> list[str]:
    """Convert the pinned MLPD improved-test COCO JSON to loader XML.

    The local evaluator and the pinned reference now consume the same source GT.
    Ignored annotations are encoded as ``people`` so parse_annotation maps them
    to class 0 (absorb-only), while scored annotations remain ``person``.
    """
    data = json.loads(src.read_text())
    images = {
        int(image["id"]): fid_from_json_filename(image.get("file_name") or image["im_name"])
        for image in data["images"]
    }
    by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for annotation in data["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    ids = []
    for image_id, fid in sorted(images.items()):
        objects = []
        for annotation in by_image.get(image_id, []):
            x, y, w, h = annotation["bbox"]
            scored = annotation.get("category_id") == 1 and not annotation.get("ignore", 0)
            name = "person" if scored else "people"
            occ = int(annotation.get("occlusion", 0))
            clipped = clip_object({"name": name, "x": x, "y": y, "w": w, "h": h, "occ": occ}, clip_stats)
            if clipped is None:
                continue
            objects.append(clipped)
            label_stats[name] += 1
            occ_stats[occ] += 1
        write_xml(dest, fid, objects)
        ids.append(fid)
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-src", required=True, help="unpacked sanitized_annotations dir")
    test = ap.add_mutually_exclusive_group(required=True)
    test.add_argument("--test-src", help="unpacked improved test-set bbGt dir")
    test.add_argument("--test-json", help="pinned MLPD KAIST_annotation.json (recommended)")
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    a = ap.parse_args()

    root = Path(a.root)
    train_dest = root / "annotations-sanitized-train"
    test_dest = root / "annotations-improved-test"
    lstats, ostats = collections.Counter(), collections.Counter()
    train_clip, test_clip = collections.Counter(), collections.Counter()

    print("converting sanitized TRAIN ...")
    train_ids = convert_tree(os.path.expanduser(a.train_src), train_dest, lstats, ostats, train_clip)
    print(f"  wrote {len(train_ids)} XMLs -> {train_dest}")
    print(f"  boxes clipped to image bounds: {train_clip['clipped']}, degenerate dropped: {train_clip['dropped']}")

    print("converting improved TEST ...")
    test_ids = (
        convert_test_json(Path(a.test_json), test_dest, lstats, ostats, test_clip)
        if a.test_json
        else convert_tree(os.path.expanduser(a.test_src), test_dest, lstats, ostats, test_clip)
    )
    print(f"  wrote {len(test_ids)} XMLs -> {test_dest}")
    print(f"  boxes clipped to image bounds: {test_clip['clipped']}, degenerate dropped: {test_clip['dropped']}")

    # the training frame list (this REPLACES train-all-02 for training)
    list_path = root / "imageSets" / "train-sanitized.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(train_ids) + "\n")
    print(f"  wrote training list ({len(train_ids)}) -> {list_path}")

    # cross-check test coverage against the protocol list
    te20 = {
        ln.strip()
        for ln in (root / "imageSets" / "test-all-20.txt").read_text().splitlines()
        if ln.strip()
    }
    cover = len(set(test_ids) & te20)
    print(f"\n  test coverage vs test-all-20: {cover}/{len(te20)}")
    print(f"  labels: {dict(lstats)}")
    print(f"  occlusion: {dict(ostats)}  (0=none 1=partial 2=heavy)")

    # round-trip verification: read one converted XML back through our own parser
    from prism.data.kaist import parse_annotation

    sample = next(p for p in train_dest.rglob("*.xml"))
    ann = parse_annotation(sample)
    if len(train_ids) != 7601:
        raise SystemExit(f"sanitized training annotation count is {len(train_ids)}, expected 7601")
    if set(test_ids) != te20:
        missing = sorted(te20 - set(test_ids))
        extra = sorted(set(test_ids) - te20)
        raise SystemExit(
            f"improved test IDs do not equal test-all-20: "
            f"missing={len(missing)} {missing[:3]}, extra={len(extra)} {extra[:3]}"
        )
    print(f"\n  round-trip check on {sample.relative_to(root)}:")
    print(
        f"    boxes={len(ann['boxes'])} classes={list(ann['classes'][:6])} "
        f"occ={list(ann['occlusion'][:6])}"
    )
    print("\nDONE. Next: point _SPLITS at the new dirs + train-sanitized.txt, run make test.")


if __name__ == "__main__":
    main()
