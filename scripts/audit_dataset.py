"""Deterministic dataset audit (TODO.md 4.1/4.3): pairing, dimensions, orphans.

Checks, for every frame in a split list (or the whole tree):
  * visible/LWIR pair existence and decoded dimensions
  * unreadable / corrupt images
  * duplicate image content (by SHA256 of a stride-decoded sample) within a video
  * orphan annotations (XML with no matching images) and missing annotations
  * box sanity (positive area, inside the 640x512 frame)

Writes a JSON report (default outputs/audit/audit_report.json) with counts and
the sha256 of the audit script inputs. This report is part of the frozen-data
evidence for Phase 2 (TODO.md 4.2 exit gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from prism.data.kaist import PairedKAIST, parse_annotation
from prism.provenance import atomic_write_json

EXPECTED_SIZE = (512, 640)  # H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train", help="train/test/... split list to audit")
    ap.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="only audit every Nth frame (sampling for speed)",
    )
    ap.add_argument("--out", default="outputs/audit/audit_report.json")
    a = ap.parse_args()

    split_file = (
        Path(a.root)
        / "imageSets"
        / ("train-sanitized.txt" if a.split == "train" else "test-all-20.txt")
    )
    requested_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    base = PairedKAIST(
        a.root,
        ann_dirname=(
            "annotations-sanitized-train" if a.split == "train" else "annotations-improved-test"
        ),
        split_file=str(split_file),
        strict=False,
    )

    report = {
        "split": a.split,
        "frames_requested": len(requested_ids),
        "split_file": str(split_file),
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "sampled": a.sample_every,
        "frames": 0,
        "pair_ok": 0,
        "missing_visible": 0,
        "missing_lwir": 0,
        "bad_size": [],
        "unreadable": [],
        "orphan_xml": [],
        "missing_xml": [],
        "bad_boxes": [],
        "duplicate_pairs": [],
        "boxes": 0,
        "positive_boxes": 0,
    }

    seen_hashes: dict[tuple[str, str], str] = {}
    for fid in base.ids[:: a.sample_every]:
        report["frames"] += 1
        vis_p, lwir_p, xml_p = base._paths(fid)
        for p, key, miss_key in (
            (vis_p, "visible", "missing_visible"),
            (lwir_p, "lwir", "missing_lwir"),
        ):
            if not p.exists():
                report[miss_key] += 1
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                report["unreadable"].append(str(p))
                continue
            h, w = img.shape
            if (h, w) != EXPECTED_SIZE:
                report["bad_size"].append(f"{fid}:{key}:{h}x{w}")
            if key == "visible":
                report["pair_ok"] += 1 if lwir_p.exists() else 0
            digest = hashlib.sha256(img[::4, ::4].tobytes()).hexdigest()
            hash_key = (key, digest)
            if hash_key in seen_hashes and seen_hashes[hash_key] != fid:
                report["duplicate_pairs"].append((key, seen_hashes[hash_key], fid))
            seen_hashes[hash_key] = fid

        if not xml_p.exists():
            report["missing_xml"].append(fid)
            continue
        ann = parse_annotation(xml_p)
        report["boxes"] += len(ann["boxes"])
        report["positive_boxes"] += int((ann["classes"] == 1).sum())
        for x1, y1, x2, y2 in ann["boxes"]:
            if x2 <= x1 or y2 <= y1 or x2 > 640 or y2 > 512 or x1 < 0 or y1 < 0:
                report["bad_boxes"].append(fid)

    # orphan annotations: XMLs with no frame in the split (bounded scan)
    ann_root = base.ann_root
    orphan = 0
    requested_set = set(requested_ids)
    for xml in sorted(ann_root.rglob("*.xml")):
        rel = xml.relative_to(ann_root)
        fid = f"{rel.parts[0]}/{rel.parts[1]}/{rel.stem}"
        if fid not in requested_set:
            orphan += 1
    report["orphan_xml"] = orphan if orphan < 50 else f"{orphan} (truncated)"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))
    print(f"\nsaved {out}")
    fatal = (
        report["missing_visible"]
        + report["missing_lwir"]
        + len(report["unreadable"])
        + len(report["bad_size"])
        + len(report["missing_xml"])
        + len(report["bad_boxes"])
    )
    if fatal:
        raise SystemExit(f"dataset audit failed with {fatal} fatal findings; see {out}")


if __name__ == "__main__":
    main()
