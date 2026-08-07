"""PRISM -> reference-format detection exporter (TODO.md 3.5).

The pinned KAIST evaluator consumes one detection per CSV line:

    image_id,x,y,width,height,score

with semantics that are easy to get wrong:
  * image_id is ONE-BASED in the TXT; the reference loader subtracts one;
  * coordinates are top-left x,y,width,height (PRISM internally uses x1y1x2y2);
  * the image-ID mapping must come from KAIST_annotation.json, never from the
    dataloader order (dataloader order != benchmark order);
  * no rounding, no undocumented confidence threshold, no class column.

The exporter asserts: one-to-one coverage of all benchmark test frames (frames
with zero detections are still tracked), finite coordinates/scores, positive
width/height, and that every exported id lies in the annotation set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prism.provenance import atomic_write_bytes, sha256_file


def load_frame_key_map(
    annotation_json: Path, split_ids: list[str], *, allow_order_fallback: bool = False
) -> dict[str, int]:
    """frame_key (e.g. 'set06/V000/I00019') -> benchmark image_id (1-based).

    The authoritative map comes from the pinned bundle's KAIST_annotation.json
    ({'images': [{id, file_name}]}). The pinned bundle stores 0-based ids
    (0..2251 in JSON order); the TXT contract is ONE-BASED (the reference
    loader subtracts one, txt2json's `-1`), so the exported id is json_id + 1.
    If the bundle is absent the exporter falls back to the order of the split
    list (also 1-based) with a WARNING -- acceptable for local smoke runs only,
    never for a reported score.
    """
    if annotation_json.exists():
        ann = json.loads(annotation_json.read_text())
        key_map = {}
        for img in ann["images"]:
            name = img.get("file_name") or img.get("im_name")
            if not name:
                raise ValueError("annotation image has neither file_name nor im_name")
            fid = Path(name)
            # file_name looks like .../set06/V000/visible/I00019.jpg: strip the
            # modality component and keep the trailing setXX/VYYY/I####ZZZ tail,
            # robust to absolute vs relative paths in the bundle.
            parts = [p for p in fid.parts if p not in ("visible", "lwir", "/", "")]
            key = "/".join(parts[-3:])
            key = str(Path(key).with_suffix("")) if Path(key).suffix else key
            key_map[key] = int(img["id"]) + 1
        missing = [f for f in split_ids if f not in key_map]
        if missing:
            raise ValueError(
                f"{len(missing)} split frames missing from the annotation map, e.g. {missing[:3]}"
            )
        return key_map
    if not allow_order_fallback:
        raise FileNotFoundError(
            f"{annotation_json} is required for benchmark-safe image ID mapping; "
            "run scripts/pin_mlpd_eval.py --clone"
        )
    print(
        "WARNING: KAIST_annotation.json not found -- using split-list order as "
        "image ids (explicit smoke-only fallback)"
    )
    return {fid: i + 1 for i, fid in enumerate(sorted(split_ids))}


def export_detections(
    frames: dict[str, np.ndarray],
    id_map: dict[str, int],
    split_ids: list[str],
    out_txt: Path,
    annotation_ids: set[int] | None = None,
) -> dict:
    """Write one detection per line. `frames`: frame_key -> (boxes xyxy, scores).

    `annotation_ids` is the set of benchmark ids declared by KAIST_annotation.json;
    every exported id must lie in it (this is the anti-corruption check -- the id
    map itself is not trusted for the final assertion). When absent (smoke runs)
    the map's own values are used.

    Returns a coverage report dict. Raises on: duplicate/missing frames, non-finite
    coordinates, non-positive width/height, or an id not present in the annotation
    set.
    """
    ann_ids = annotation_ids if annotation_ids is not None else set(id_map.values())
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("split_ids contains duplicate frame keys")
    expected = set(split_ids)
    actual = set(frames)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "inference/export frame coverage mismatch: "
            f"missing={len(missing)} {missing[:3]}, extra={len(extra)} {extra[:3]}"
        )

    seen = set()
    lines = []
    for fid in split_ids:  # benchmark order, including 0-det frames
        if fid in seen:
            raise ValueError(f"duplicate frame key in export list: {fid}")
        seen.add(fid)
        if fid not in id_map:
            raise ValueError(f"frame {fid} has no benchmark image id")
        img_id = id_map[fid]
        boxes, scores = frames[fid]
        boxes = np.asarray(boxes).reshape(-1, 4)
        scores = np.asarray(scores).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError(f"box/score count mismatch in {fid}: {len(boxes)} vs {len(scores)}")
        for (x1, y1, x2, y2), sc in zip(boxes, scores, strict=True):
            x1, y1, x2, y2, sc = map(float, (x1, y1, x2, y2, sc))
            w, h = x2 - x1, y2 - y1
            if not all(np.isfinite([x1, y1, w, h, sc])):
                raise ValueError(f"non-finite detection in {fid}: {x1},{y1},{w},{h},{sc}")
            if w <= 0 or h <= 0:
                raise ValueError(f"non-positive box in {fid}: {w}x{h}")
            # .17g round-trips a Python float. The reference accepts arbitrary
            # decimal precision, so quantizing boxes/scores here only creates an
            # avoidable difference between local scoring and exported scoring.
            values = ",".join(format(v, ".17g") for v in (x1, y1, w, h, sc))
            lines.append(f"{img_id},{values}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_txt, ("\n".join(lines) + "\n").encode())

    exported_ids = {int(ln.split(",")[0]) for ln in lines}
    if not exported_ids <= ann_ids:
        raise ValueError("exported image ids outside the annotation set")
    report = {
        "frames_in_split": len(split_ids),
        "frames_with_detections": sum(
            1 for fid in split_ids if fid in frames and len(frames[fid][1]) > 0
        ),
        "total_detections": len(lines),
        "distinct_image_ids": len(exported_ids),
        "output": str(out_txt),
        "sha256": sha256_file(out_txt),
    }
    return report
