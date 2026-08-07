"""Validation-only controlled interventions for a trained RGB-T fusion model.

These tests support causal statements about *this model's behavior under an
input intervention*. They do not by themselves identify the causal effect of a
sensor in the real world, and MR is nonlinear, so the script deliberately does
not report Shapley values or an additive interaction decomposition from MR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.data.kaist import kaist_split_cv
from prism.data.transforms import translate_with_zeros
from prism.eval.canonical import evaluate_reference
from prism.evaluate_model import collect_frames, load_checkpoint_model
from prism.provenance import atomic_save_npz, atomic_write_json, sha256_file

DEFAULT_CONDITIONS = (
    "paired",
    "rgb_mean",
    "thermal_mean",
    "both_mean",
    "rgb_zero",
    "thermal_zero",
    "both_zero",
    "rgb_brightness_0.75",
    "rgb_brightness_0.50",
    "rgb_brightness_0.25",
    "rgb_blur_5",
    "rgb_blur_9",
    "rgb_blur_15",
    "rgb_grayscale",
    "thermal_contrast_0.75",
    "thermal_contrast_0.50",
    "thermal_contrast_0.25",
    "thermal_blur_5",
    "thermal_blur_9",
    "thermal_blur_15",
    "thermal_noise_0.05",
    "thermal_noise_0.10",
    "thermal_noise_0.20",
    "thermal_shift_x_m16",
    "thermal_shift_x_m8",
    "thermal_shift_x_m4",
    "thermal_shift_x_p4",
    "thermal_shift_x_p8",
    "thermal_shift_x_p16",
    "thermal_shift_y_m16",
    "thermal_shift_y_m8",
    "thermal_shift_y_m4",
    "thermal_shift_y_p4",
    "thermal_shift_y_p8",
    "thermal_shift_y_p16",
)

SHIFTS = {
    "thermal_shift_x_m4": (-4, 0),
    "thermal_shift_x_m16": (-16, 0),
    "thermal_shift_x_m8": (-8, 0),
    "thermal_shift_x_p4": (4, 0),
    "thermal_shift_x_p8": (8, 0),
    "thermal_shift_x_p16": (16, 0),
    "thermal_shift_y_m4": (0, -4),
    "thermal_shift_y_m16": (0, -16),
    "thermal_shift_y_m8": (0, -8),
    "thermal_shift_y_p4": (0, 4),
    "thermal_shift_y_p8": (0, 8),
    "thermal_shift_y_p16": (0, 16),
}

RGB_BRIGHTNESS = {
    "rgb_brightness_0.75": 0.75,
    "rgb_brightness_0.50": 0.50,
    "rgb_brightness_0.25": 0.25,
}
RGB_BLUR = {"rgb_blur_5": 5, "rgb_blur_9": 9, "rgb_blur_15": 15}
THERMAL_CONTRAST = {
    "thermal_contrast_0.75": 0.75,
    "thermal_contrast_0.50": 0.50,
    "thermal_contrast_0.25": 0.25,
}
THERMAL_BLUR = {
    "thermal_blur_5": 5,
    "thermal_blur_9": 9,
    "thermal_blur_15": 15,
}
THERMAL_NOISE = {
    "thermal_noise_0.05": 0.05,
    "thermal_noise_0.10": 0.10,
    "thermal_noise_0.20": 0.20,
}


def _fill_like(image: torch.Tensor, channel_mean: list[float]) -> torch.Tensor:
    values = (
        torch.tensor(channel_mean, dtype=torch.float32, device=image.device).view(-1, 1, 1) * 255.0
    )
    return values.expand_as(image).clone()


def _seed(frame_id: str, condition: str) -> int:
    digest = hashlib.sha256(f"{frame_id}\0{condition}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def write_raw_frames(frames: list[dict], path: Path) -> dict:
    """Store every variable-length detection/GT row with frame offsets."""
    detection_offsets = [0]
    gt_offsets = [0]
    detection_boxes, detection_scores = [], []
    gt_boxes, gt_classes, gt_occlusion = [], [], []
    for frame in frames:
        detection_boxes.append(frame["det_boxes"])
        detection_scores.append(frame["det_scores"])
        gt_boxes.append(frame["gt_boxes"])
        gt_classes.append(frame["gt_classes"])
        gt_occlusion.append(frame["gt_occlusion"])
        detection_offsets.append(detection_offsets[-1] + len(frame["det_scores"]))
        gt_offsets.append(gt_offsets[-1] + len(frame["gt_boxes"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        path,
        frame_ids=np.asarray([frame["id"] for frame in frames]),
        detection_offsets=np.asarray(detection_offsets, dtype=np.int64),
        detection_boxes=np.concatenate(detection_boxes).astype(np.float32),
        detection_scores=np.concatenate(detection_scores).astype(np.float32),
        gt_offsets=np.asarray(gt_offsets, dtype=np.int64),
        gt_boxes=np.concatenate(gt_boxes).astype(np.float32),
        gt_classes=np.concatenate(gt_classes).astype(np.int64),
        gt_occlusion=np.concatenate(gt_occlusion).astype(np.int64),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "frames": len(frames),
        "detections": detection_offsets[-1],
        "ground_truth": gt_offsets[-1],
    }


class IntervenedDataset:
    """Read-only view that deterministically modifies one or both modalities."""

    def __init__(self, base, condition: str, statistics: dict, limit: int | None = None):
        if condition not in DEFAULT_CONDITIONS:
            raise ValueError(f"unknown intervention {condition!r}")
        self.base = base
        self.condition = condition
        self.statistics = statistics
        self.limit = min(limit, len(base)) if limit is not None else len(base)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        source = self.base[index]
        sample = dict(source)
        rgb = source["rgb"].clone()
        thermal = source["thermal"].clone()
        condition = self.condition
        rgb_mean = self.statistics["rgb"]["mean"]
        thermal_mean = self.statistics["thermal"]["mean"]

        if condition in ("rgb_mean", "both_mean"):
            rgb = _fill_like(rgb, rgb_mean)
        if condition in ("thermal_mean", "both_mean"):
            thermal = _fill_like(thermal, thermal_mean)
        if condition == "rgb_zero":
            rgb.zero_()
        if condition == "thermal_zero":
            thermal.zero_()
        if condition == "both_zero":
            rgb.zero_()
            thermal.zero_()
        if condition in RGB_BRIGHTNESS:
            rgb = rgb * RGB_BRIGHTNESS[condition]
        if condition in RGB_BLUR:
            kernel = RGB_BLUR[condition]
            rgb = F.avg_pool2d(rgb[None].float(), kernel, stride=1, padding=kernel // 2)[0]
        if condition == "rgb_grayscale":
            grayscale = rgb.float().mean(dim=0, keepdim=True)
            rgb = grayscale.expand_as(rgb)
        if condition in THERMAL_CONTRAST:
            per_image_mean = thermal.float().mean(dim=(1, 2), keepdim=True)
            thermal = per_image_mean + THERMAL_CONTRAST[condition] * (
                thermal.float() - per_image_mean
            )
        if condition in THERMAL_BLUR:
            kernel = THERMAL_BLUR[condition]
            thermal = F.avg_pool2d(thermal[None].float(), kernel, stride=1, padding=kernel // 2)[0]
        if condition in THERMAL_NOISE:
            generator = torch.Generator().manual_seed(_seed(source["id"], condition))
            noise = torch.randn(thermal.shape, generator=generator) * (
                THERMAL_NOISE[condition] * 255.0
            )
            thermal = thermal.float() + noise
        if condition in SHIFTS:
            thermal = translate_with_zeros(thermal, *SHIFTS[condition])

        sample["rgb"] = rgb.clamp(0, 255).to(source["rgb"].dtype)
        sample["thermal"] = thermal.clamp(0, 255).to(source["thermal"].dtype)
        return sample


def frame_covariates(base, limit: int | None) -> list[dict]:
    """Measured per-frame quality/target covariates for later stratification."""
    count = min(limit, len(base)) if limit is not None else len(base)
    rows = []
    for index in range(count):
        sample = base[index]
        rgb = sample["rgb"].float() / 255.0
        thermal = sample["thermal"].float() / 255.0
        scored_boxes = sample["boxes"][sample["classes"] == 1]
        heights = scored_boxes[:, 3] - scored_boxes[:, 1] if len(scored_boxes) else torch.zeros(0)
        rows.append(
            {
                "frame_id": sample["id"],
                "rgb_mean": float(rgb.mean()),
                "rgb_std": float(rgb.std(unbiased=False)),
                "thermal_mean": float(thermal.mean()),
                "thermal_std": float(thermal.std(unbiased=False)),
                "scored_gt_count": len(scored_boxes),
                "scored_gt_height_min": float(heights.min()) if len(heights) else None,
                "scored_gt_height_max": float(heights.max()) if len(heights) else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--stats", default="outputs/data/train_cv_stats.json")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="deterministic smoke subset")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    if "paired" not in args.conditions:
        raise SystemExit("conditions must include paired as the reference")

    stats_record = json.loads(Path(args.stats).read_text())
    if stats_record.get("split") != "train-cv":
        raise SystemExit("intervention statistics must come from train-cv")
    statistics = stats_record["statistics"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, modality, model_type = load_checkpoint_model(args.ckpt, device=device)
    if modality != "fusion":
        raise SystemExit("controlled modality interventions require a fusion checkpoint")

    base = kaist_split_cv(args.root, "val-cv")
    output = Path(args.out)
    raw_directory = output.parent / f"{output.stem}_raw"
    rows = []
    for condition in args.conditions:
        dataset = IntervenedDataset(base, condition, statistics, args.limit)
        frames = collect_frames(
            model,
            dataset,
            modality,
            device,
            batch_size=args.batch_size,
            input_policy="paired",
            day_ids=None,
        )
        metrics = evaluate_reference(frames)
        raw = write_raw_frames(frames, raw_directory / f"{condition}.npz")
        rows.append(
            {
                "condition": condition,
                "frames": len(frames),
                "detections": sum(len(frame["det_scores"]) for frame in frames),
                "mr_all": metrics["all"],
                "raw_frames": raw,
            }
        )
        print(f"{condition}: MR^-2 All={100 * metrics['all']:.3f}%")

    reference = next(row["mr_all"] for row in rows if row["condition"] == "paired")
    for row in rows:
        row["delta_mr_pp_vs_paired"] = 100.0 * (row["mr_all"] - reference)

    checkpoint = Path(args.ckpt)
    result = {
        "schema_version": 1,
        "scope": "model-behavior interventions on val-cv; not real-world sensor causality",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_type": model_type,
        "split": "val-cv",
        "stats_file": args.stats,
        "stats_sha256": sha256_file(args.stats),
        "raw_frame_covariates": frame_covariates(base, args.limit),
        "conditions": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
