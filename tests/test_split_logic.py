"""Sequence-disjoint split tests (TODO.md 4.2)."""

import json

from prism.data.splits import group_by_video, split_by_video, write_split_files


def _fake_ids(n_videos=5, frames_per=10):
    return [f"set06/V{i:03d}/I{j:05d}" for i in range(n_videos) for j in range(frames_per)]


def test_group_by_video():
    ids = _fake_ids(3, 4)
    groups = group_by_video(ids)
    assert len(groups) == 3
    assert all(len(v) == 4 for v in groups.values())


def test_split_is_video_disjoint():
    ids = _fake_ids(10, 10)
    train, val = split_by_video(ids, 0.2, seed=0)
    assert len(set(train) & set(val)) == 0
    train_vids = {i.rsplit("/", 1)[0] for i in train}
    val_vids = {i.rsplit("/", 1)[0] for i in val}
    assert train_vids.isdisjoint(val_vids)  # THE property


def test_split_is_deterministic():
    ids = _fake_ids(10, 10)
    a1, b1 = split_by_video(ids, 0.2, seed=7)
    a2, b2 = split_by_video(ids, 0.2, seed=7)
    assert a1 == a2 and b1 == b2


def test_split_fraction_approximately_respected():
    ids = _fake_ids(20, 8)  # 160 frames
    _train, val = split_by_video(ids, 0.15, seed=3)
    frac = len(val) / len(ids)
    assert 0.10 <= frac <= 0.20  # nearest-fit, videos whole


def test_split_covers_all_frames():
    ids = _fake_ids(6, 5)
    train, val = split_by_video(ids, 0.3, seed=1)
    assert sorted(train + val) == sorted(ids)


def test_write_split_files_manifest(tmp_path):
    ids = _fake_ids(4, 5)
    train, val = split_by_video(ids, 0.25, seed=0)
    manifest = write_split_files(train, val, tmp_path, source_ids=ids, seed=0, val_fraction=0.25)
    assert (tmp_path / "train.txt").exists() and (tmp_path / "val.txt").exists()
    assert manifest["train"]["frames"] == len(train)
    assert len(manifest["train"]["sha256"]) == 64
    # manifest re-readable and self-consistent
    again = json.loads((tmp_path / "split_manifest.json").read_text())
    assert again["split_policy"].startswith("sequence-disjoint")
    assert again["train"]["sha256"] == manifest["train"]["sha256"]
    assert again["seed"] == 0
    assert again["source"]["frames"] == len(ids)
