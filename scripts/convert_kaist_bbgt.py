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
import os
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path("data/raw/kaist-cvpr15")


def parse_bbgt(path: str):
    """Return list of dicts {name,x,y,w,h,occ} from one bbGt file."""
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
            objs.append({"name": p[0], "x": x, "y": y, "w": w, "h": h, "occ": occ})
    return objs


def write_xml(dest: Path, fid: str, objs):
    """Write objs to dest/<fid>.xml in the schema parse_annotation() reads."""
    out = dest / f"{fid}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["<annotation>", f"  <filename>{fid}</filename>",
             "  <size><width>640</width><height>512</height><depth>3</depth></size>"]
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


def convert_tree(src: str, dest: Path, label_stats, occ_stats):
    ids = []
    for f in sorted(glob.glob(f"{src}/**/set*.txt", recursive=True)):
        fid = fid_from_name(f)
        if fid.count("/") != 2:
            continue
        objs = parse_bbgt(f)
        for o in objs:
            label_stats[o["name"]] += 1
            occ_stats[o["occ"]] += 1
        write_xml(dest, fid, objs)
        ids.append(fid)
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-src", required=True, help="unpacked sanitized_annotations dir")
    ap.add_argument("--test-src", required=True, help="unpacked improved test-set dir")
    a = ap.parse_args()

    train_dest = ROOT / "annotations-sanitized-train"
    test_dest = ROOT / "annotations-improved-test"
    lstats, ostats = collections.Counter(), collections.Counter()

    print("converting sanitized TRAIN ...")
    train_ids = convert_tree(os.path.expanduser(a.train_src), train_dest, lstats, ostats)
    print(f"  wrote {len(train_ids)} XMLs -> {train_dest}")

    print("converting improved TEST ...")
    test_ids = convert_tree(os.path.expanduser(a.test_src), test_dest, lstats, ostats)
    print(f"  wrote {len(test_ids)} XMLs -> {test_dest}")

    # the training frame list (this REPLACES train-all-02 for training)
    list_path = ROOT / "imageSets" / "train-sanitized.txt"
    list_path.write_text("\n".join(train_ids) + "\n")
    print(f"  wrote training list ({len(train_ids)}) -> {list_path}")

    # cross-check test coverage against the protocol list
    te20 = {ln.strip() for ln in (ROOT / "imageSets" / "test-all-20.txt").read_text().splitlines() if ln.strip()}
    cover = len(set(test_ids) & te20)
    print(f"\n  test coverage vs test-all-20: {cover}/{len(te20)}")
    print(f"  labels: {dict(lstats)}")
    print(f"  occlusion: {dict(ostats)}  (0=none 1=partial 2=heavy)")

    # round-trip verification: read one converted XML back through our own parser
    from prism.data.kaist import parse_annotation
    sample = next(p for p in train_dest.rglob("*.xml"))
    ann = parse_annotation(sample)
    print(f"\n  round-trip check on {sample.relative_to(ROOT)}:")
    print(f"    boxes={len(ann['boxes'])} classes={list(ann['classes'][:6])} "
          f"occ={list(ann['occlusion'][:6])}")
    print("\nDONE. Next: point _SPLITS at the new dirs + train-sanitized.txt, run make test.")


if __name__ == "__main__":
    main()
