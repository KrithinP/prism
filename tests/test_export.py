"""Exporter contract tests (TODO.md 3.5)."""

import json

import numpy as np
import pytest

from prism.eval.export import export_detections, load_frame_key_map


@pytest.fixture
def ann_json(tmp_path):
    # the pinned bundle stores 0-based ids in JSON order; the TXT is 1-based
    ann = {
        "images": [
            {"id": 0, "file_name": "/d/set06/V000/visible/I00001.jpg"},
            {"id": 1, "file_name": "/d/set06/V000/lwir/I00002.jpg"},
            {"id": 2, "file_name": "/d/set06/V000/visible/I00003.jpg"},
        ]
    }
    p = tmp_path / "KAIST_annotation.json"
    p.write_text(json.dumps(ann))
    return p


def test_map_from_annotation_json(ann_json):
    split = ["set06/V000/I00001", "set06/V000/I00002", "set06/V000/I00003"]
    m = load_frame_key_map(ann_json, split)
    assert m == {"set06/V000/I00001": 1, "set06/V000/I00002": 2, "set06/V000/I00003": 3}


def test_map_accepts_pinned_bundle_im_name_schema(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "im_name": "set06/V000/I00019"}],
            }
        )
    )
    assert load_frame_key_map(path, ["set06/V000/I00019"]) == {
        "set06/V000/I00019": 1,
    }


def test_map_missing_frame_fails_loudly(ann_json):
    split = ["set06/V000/I00001", "set06/V000/I00999"]
    with pytest.raises(ValueError, match="missing from the annotation map"):
        load_frame_key_map(ann_json, split)


def test_export_one_line_per_detection_with_coverage(ann_json, tmp_path):
    split = ["set06/V000/I00001", "set06/V000/I00002", "set06/V000/I00003"]
    id_map = load_frame_key_map(ann_json, split)
    frames = {
        "set06/V000/I00001": (np.array([[10, 20, 110, 220]]), np.array([0.9])),
        "set06/V000/I00002": (np.empty((0, 4)), np.empty(0)),  # zero detections
        "set06/V000/I00003": (
            np.array([[5, 5, 55, 105], [300, 300, 400, 400]]),
            np.array([0.7, 0.2]),
        ),
    }
    out = tmp_path / "dets.txt"
    report = export_detections(frames, id_map, split, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0] == "1,10,20,100,200,0.90000000000000002"
    # benchmark order: frame1's det, then frame3's two dets (frame2 has none)
    assert lines[1] == "3,5,5,50,100,0.69999999999999996"
    assert lines[2] == "3,300,300,100,100,0.20000000000000001"
    assert report["frames_in_split"] == 3
    assert report["frames_with_detections"] == 2
    assert report["total_detections"] == 3
    assert report["distinct_image_ids"] == 2
    assert len(report["sha256"]) == 64


def test_export_rejects_nonfinite_and_zero_area(ann_json, tmp_path):
    split = ["set06/V000/I00001", "set06/V000/I00002", "set06/V000/I00003"]
    id_map = load_frame_key_map(ann_json, split)
    empty = (np.empty((0, 4)), np.empty(0))
    bad = {
        "set06/V000/I00001": (np.array([[10, 20, 10, 220]]), np.array([0.9])),
        "set06/V000/I00002": empty,
        "set06/V000/I00003": empty,
    }
    with pytest.raises(ValueError, match="non-positive"):
        export_detections(bad, id_map, split, tmp_path / "d.txt")
    bad2 = {
        "set06/V000/I00001": (np.array([[np.nan, 20, 110, 220]]), np.array([0.9])),
        "set06/V000/I00002": empty,
        "set06/V000/I00003": empty,
    }
    with pytest.raises(ValueError, match="non-finite"):
        export_detections(bad2, id_map, split, tmp_path / "d2.txt")


def test_export_rejects_id_outside_annotation_set(ann_json, tmp_path):
    split = ["set06/V000/I00001", "set06/V000/I00002", "set06/V000/I00003"]
    id_map = load_frame_key_map(ann_json, split)
    frames = {"set06/V000/I00001": (np.array([[1, 1, 10, 10]]), np.array([0.5]))}
    # corrupt the map so a split frame maps to an id NOT in the annotation set
    bad_map = dict(id_map)
    bad_map["set06/V000/I00001"] = 999
    with pytest.raises(ValueError, match="outside the annotation set"):
        complete = {fid: frames.get(fid, (np.empty((0, 4)), np.empty(0))) for fid in split}
        export_detections(complete, bad_map, split, tmp_path / "d.txt", annotation_ids={1, 2, 3})


def test_export_requires_exact_frame_coverage(ann_json, tmp_path):
    split = ["set06/V000/I00001", "set06/V000/I00002", "set06/V000/I00003"]
    id_map = load_frame_key_map(ann_json, split)
    incomplete = {
        "set06/V000/I00001": (np.empty((0, 4)), np.empty(0)),
    }
    with pytest.raises(ValueError, match="coverage mismatch"):
        export_detections(incomplete, id_map, split, tmp_path / "missing.txt")


def test_missing_annotation_map_fails_without_explicit_smoke_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark-safe"):
        load_frame_key_map(tmp_path / "missing.json", ["set06/V000/I00001"])
