"""Frozen RGB, thermal, and fused feature probes for a trained fusion model.

Features are sampled at scored GT centers plus deterministic background cells.
Identical closed-form ridge probes are fit on train-cv and evaluated on val-cv:

- person-vs-background score: ROC AUC and balanced accuracy;
- positive-cell sub-cell center offsets: two-dimensional RMSE.

The raw feature matrices, labels, IDs, targets, validation scores, and
predictions are saved before summaries. The detector remains frozen throughout.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from prism.data.kaist import kaist_split_cv
from prism.evaluate_model import load_checkpoint_model
from prism.provenance import atomic_save_npz, atomic_write_json, sha256_file


def _branch_maps(model, model_type: str, rgb: torch.Tensor, thermal: torch.Tensor):
    if model_type == "multi-scale":
        rgb_maps = dict(zip((3, 4, 5), model.backbone_rgb(rgb), strict=True))
        thermal_maps = dict(zip((3, 4, 5), model.backbone_thermal(thermal), strict=True))
        levels = list(model.fuse_subset)
        fused_maps = {
            level: model.fusers[str(level)](rgb_maps[level], thermal_maps[level])
            for level in levels
        }
    elif model_type == "one-shot":
        level = model.meet_level
        rgb_maps = model.backbone_rgb.forward_upto(rgb, level)
        thermal_maps = model.backbone_thermal.forward_upto(thermal, level)
        levels = [level]
        fused_maps = {level: model.fuser(rgb_maps[level], thermal_maps[level])}
    else:
        raise ValueError(f"frozen probes require a two-stream feature graph, got {model_type!r}")
    return rgb_maps, thermal_maps, fused_maps, levels


def _inside_any_box(x: float, y: float, boxes: torch.Tensor) -> bool:
    if boxes.numel() == 0:
        return False
    return bool(
        ((boxes[:, 0] <= x) & (x < boxes[:, 2]) & (boxes[:, 1] <= y) & (y < boxes[:, 3])).any()
    )


def sample_locations(
    frame_id: str,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
) -> list[dict]:
    """GT-center positives plus deterministic background cells."""
    rows = []
    positives = boxes[classes == 1]
    for index, box in enumerate(positives):
        center_x = float((box[0] + box[2]) / 2)
        center_y = float((box[1] + box[3]) / 2)
        feature_x = min(feature_width - 1, max(0, int(center_x / image_width * feature_width)))
        feature_y = min(feature_height - 1, max(0, int(center_y / image_height * feature_height)))
        rows.append(
            {
                "id": f"{frame_id}#person{index}",
                "label": 1,
                "x": feature_x,
                "y": feature_y,
                "offset": [
                    center_x / image_width * feature_width - feature_x,
                    center_y / image_height * feature_height - feature_y,
                ],
            }
        )

    backgrounds = max(1, len(positives))
    cells = feature_height * feature_width
    start = int.from_bytes(hashlib.sha256(frame_id.encode()).digest()[:8], "little") % cells
    added = 0
    for delta in range(cells):
        flat = (start + 104729 * delta) % cells
        feature_y, feature_x = divmod(flat, feature_width)
        pixel_x = (feature_x + 0.5) * image_width / feature_width
        pixel_y = (feature_y + 0.5) * image_height / feature_height
        if _inside_any_box(pixel_x, pixel_y, boxes):
            continue
        rows.append(
            {
                "id": f"{frame_id}#background{added}",
                "label": 0,
                "x": feature_x,
                "y": feature_y,
                "offset": [float("nan"), float("nan")],
            }
        )
        added += 1
        if added == backgrounds:
            break
    return rows


@torch.no_grad()
def collect_features(
    model,
    model_type: str,
    dataset,
    device: str,
    batch_size: int,
    limit: int,
) -> dict[str, dict]:
    records: dict[str, dict] = {}
    model.eval()
    limit = min(limit, len(dataset))
    for start in range(0, limit, batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, limit))]
        rgb = torch.stack([sample["rgb"].float() / 255.0 for sample in samples]).to(device)
        thermal = torch.stack([sample["thermal"].float() / 255.0 for sample in samples]).to(device)
        rgb_maps, thermal_maps, fused_maps, levels = _branch_maps(model, model_type, rgb, thermal)

        for level in levels:
            maps = {
                "rgb": rgb_maps[level],
                "thermal": thermal_maps[level],
                "fused": fused_maps[level],
            }
            _, _, feature_height, feature_width = maps["rgb"].shape
            for sample_index, sample in enumerate(samples):
                image_height, image_width = sample["rgb"].shape[-2:]
                locations = sample_locations(
                    sample["id"],
                    sample["boxes"],
                    sample["classes"],
                    feature_height,
                    feature_width,
                    image_height,
                    image_width,
                )
                for representation, feature_map in maps.items():
                    key = f"{representation}_l{level}"
                    record = records.setdefault(
                        key, {"features": [], "labels": [], "offsets": [], "ids": []}
                    )
                    for location in locations:
                        record["features"].append(
                            feature_map[sample_index, :, location["y"], location["x"]].float().cpu()
                        )
                        record["labels"].append(location["label"])
                        record["offsets"].append(location["offset"])
                        record["ids"].append(location["id"])

    packed = {}
    for key, record in records.items():
        packed[key] = {
            "features": torch.stack(record["features"]).numpy(),
            "labels": np.asarray(record["labels"], dtype=np.int64),
            "offsets": np.asarray(record["offsets"], dtype=np.float64),
            "ids": np.asarray(record["ids"]),
        }
    return packed


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    regularization: float,
) -> dict:
    """Standardized ridge with an unpenalized intercept."""
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (train_x - mean) / scale
    design = np.concatenate([standardized, np.ones((len(standardized), 1))], axis=1)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict_ridge(model: dict, features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.concatenate([standardized, np.ones((len(standardized), 1))], axis=1)
    return design @ model["weights"]


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        return None
    ranks = rankdata(scores)
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        return None
    predictions = scores >= 0
    return float(((predictions[positives] == 1).mean() + (predictions[negatives] == 0).mean()) / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", default=None)
    args = parser.parse_args()
    if args.ridge <= 0:
        raise SystemExit("--ridge must be positive and is frozen, not validation-tuned")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, modality, model_type = load_checkpoint_model(args.ckpt, device=device)
    if modality != "fusion":
        raise SystemExit("frozen RGB/T/fused probes require a fusion checkpoint")
    train = collect_features(
        model,
        model_type,
        kaist_split_cv(args.root, "train-cv"),
        device,
        args.batch_size,
        args.train_limit,
    )
    validation = collect_features(
        model,
        model_type,
        kaist_split_cv(args.root, "val-cv"),
        device,
        args.batch_size,
        args.val_limit,
    )
    if set(train) != set(validation):
        raise RuntimeError("train/validation representation keys differ")

    raw_arrays = {}
    summary = {}
    raw_validation = {}
    for key in sorted(train):
        train_record = train[key]
        val_record = validation[key]
        classifier = fit_ridge(
            train_record["features"],
            2.0 * train_record["labels"] - 1.0,
            args.ridge,
        )
        scores = predict_ridge(classifier, val_record["features"])
        positive_train = train_record["labels"] == 1
        positive_val = val_record["labels"] == 1
        localizer = fit_ridge(
            train_record["features"][positive_train],
            train_record["offsets"][positive_train],
            args.ridge,
        )
        predicted_offsets = predict_ridge(localizer, val_record["features"][positive_val])
        target_offsets = val_record["offsets"][positive_val]
        localization_rmse = float(np.sqrt(np.mean((predicted_offsets - target_offsets) ** 2)))
        summary[key] = {
            "train_samples": len(train_record["labels"]),
            "validation_samples": len(val_record["labels"]),
            "validation_positive_fraction": float(val_record["labels"].mean()),
            "person_background_auc": binary_auc(val_record["labels"], scores),
            "person_background_balanced_accuracy": balanced_accuracy(val_record["labels"], scores),
            "positive_subcell_center_rmse": localization_rmse,
        }
        raw_validation[key] = {
            "ids": val_record["ids"].tolist(),
            "labels": val_record["labels"].tolist(),
            "classification_scores": scores.tolist(),
            "positive_ids": val_record["ids"][positive_val].tolist(),
            "center_targets": target_offsets.tolist(),
            "center_predictions": predicted_offsets.tolist(),
        }
        for split_name, record in (("train", train_record), ("val", val_record)):
            for field in ("features", "labels", "offsets", "ids"):
                raw_arrays[f"{key}_{split_name}_{field}"] = record[field]

    checkpoint = Path(args.ckpt)
    output = Path(args.out)
    raw_output = Path(args.raw_out) if args.raw_out else output.with_suffix(".npz")
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_type": model_type,
        "train_split": "train-cv",
        "validation_split": "val-cv",
        "sampling": (
            "GT-center positives and deterministic non-box background cells "
            "from first frozen split frames"
        ),
        "probe": {
            "type": "closed-form standardized ridge with unpenalized intercept",
            "regularization": args.ridge,
            "selection": "fixed a priori; no validation tuning",
        },
        "derived_summary": summary,
        "raw_validation_predictions": raw_validation,
        "scope_warning": (
            "probe performance measures linearly accessible information under "
            "this sampling scheme; it is not detector performance or causality"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(raw_output, **raw_arrays)
    result["raw_npz"] = {
        "path": str(raw_output),
        "sha256": sha256_file(raw_output),
    }
    atomic_write_json(output, result)
    print(f"wrote {output} and {raw_output}")


if __name__ == "__main__":
    main()
