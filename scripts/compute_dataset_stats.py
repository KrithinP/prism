"""Compute train-only RGB and thermal channel statistics for interventions.

The output is used to replace a modality with its training-distribution mean.
It must be computed on ``train-cv`` only; computing normalization or ablation
constants from validation/test data would leak evaluation information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from prism.data.kaist import kaist_split_cv
from prism.provenance import atomic_write_json


def streaming_stats(dataset) -> dict:
    totals = {
        "rgb": {
            "sum": torch.zeros(3, dtype=torch.float64),
            "sumsq": torch.zeros(3, dtype=torch.float64),
            "pixels": 0,
        },
        "thermal": {
            "sum": torch.zeros(3, dtype=torch.float64),
            "sumsq": torch.zeros(3, dtype=torch.float64),
            "pixels": 0,
        },
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        for modality in ("rgb", "thermal"):
            image = sample[modality].double() / 255.0
            flat = image.flatten(1)
            totals[modality]["sum"] += flat.sum(dim=1)
            totals[modality]["sumsq"] += flat.square().sum(dim=1)
            totals[modality]["pixels"] += flat.shape[1]

    result = {}
    for modality, values in totals.items():
        count = values["pixels"]
        mean = values["sum"] / count
        variance = (values["sumsq"] / count - mean.square()).clamp(min=0)
        result[modality] = {
            "mean": mean.tolist(),
            "std": variance.sqrt().tolist(),
            "pixels_per_channel": count,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--out", default="outputs/data/train_cv_stats.json")
    args = parser.parse_args()

    split_path = Path("splits/train.txt")
    dataset = kaist_split_cv(args.root, "train-cv")
    result = {
        "schema_version": 1,
        "split": "train-cv",
        "frames": len(dataset),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "scale": "[0,1]",
        "statistics": streaming_stats(dataset),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
