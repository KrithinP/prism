"""Sequence-disjoint train/validation split tooling (TODO.md 4.2).

Nearby frames from the same video are highly correlated; a random frame split
would leak backgrounds, pedestrians, and temporal context into validation. The
split happens at `setXX/VYYY` (video) granularity: complete videos land in
exactly one of train or validation, never both.

Outputs (committed to `splits/`):
  train.txt / val.txt       -- sorted frame lists, one frame per line
  split_manifest.json       -- sha256 of each list + the split parameters
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from prism.provenance import atomic_write_bytes, atomic_write_json, sha256_file


def group_by_video(ids: list[str]) -> dict[str, list[str]]:
    """ids: ['set06/V000/I00019', ...] -> {video: [frames]} preserving list order."""
    videos: dict[str, list[str]] = {}
    for fid in ids:
        videos.setdefault(fid.rsplit("/", 1)[0], []).append(fid)
    return videos


def split_by_video(ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split frame lists at video granularity into (train_ids, val_ids).

    Greedy nearest-fit: videos are shuffled (seeded), then validation is filled
    with the video whose frame count brings the validation total closest to the
    target without exceeding it; if the remaining budget cannot be filled exactly
    (the smallest remaining video overshoots), that smallest video is taken.
    Videos are never split between partitions.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    videos = group_by_video(ids)
    items = sorted(videos.items())  # deterministic order
    rng = random.Random(seed)
    rng.shuffle(items)  # seeded, reproducible
    n_val = round(len(ids) * val_fraction)

    remaining = list(items)
    val_frames, val = 0, []
    while remaining and val_frames < n_val:
        fits = [(i, v, f) for i, (v, f) in enumerate(remaining) if val_frames + len(f) <= n_val]
        if fits:
            i, v, f = min(fits, key=lambda t: abs(val_frames + len(t[2]) - n_val))
        else:
            i = min(range(len(remaining)), key=lambda i: len(remaining[i][1]))
            v, f = remaining[i]  # nearest overshoot
        remaining.pop(i)
        val.append((v, f))
        val_frames += len(f)
    train = remaining
    return ([f for _, fs in sorted(train) for f in fs], [f for _, fs in sorted(val) for f in fs])


def write_split_files(
    train_ids: list[str],
    val_ids: list[str],
    out_dir: str | Path,
    *,
    source_ids: list[str] | None = None,
    seed: int | None = None,
    val_fraction: float | None = None,
) -> dict:
    """Write train.txt/val.txt plus a sha256 manifest. Returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if set(train_ids) & set(val_ids):
        raise ValueError("train and validation frame lists overlap")
    if source_ids is not None and sorted(train_ids + val_ids) != sorted(source_ids):
        raise ValueError("train/validation lists do not exactly cover source_ids")
    manifest = {
        "schema_version": 1,
        "split_policy": "sequence-disjoint (set/video granularity)",
        "seed": seed,
        "target_val_fraction": val_fraction,
        "realized_val_fraction": len(val_ids) / max(1, len(train_ids) + len(val_ids)),
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
    }
    if source_ids is not None:
        source_text = "\n".join(source_ids) + "\n"
        manifest["source"] = {
            "frames": len(source_ids),
            "sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        }
    for name, ids in (("train", train_ids), ("val", val_ids)):
        p = out / f"{name}.txt"
        atomic_write_bytes(p, ("\n".join(sorted(ids)) + "\n").encode())
        manifest[name] = {
            "frames": len(ids),
            "videos": len({i.rsplit("/", 1)[0] for i in ids}),
            "sha256": sha256_file(p),
        }
    manifest_path = out / "split_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest
