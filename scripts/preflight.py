"""Fail-closed preflight for an unattended PRISM experiment machine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from prism.data.kaist import kaist_split, kaist_split_cv
from prism.protocol import ARCH_EPOCHS, PROTOCOL_VERSION, RECIPE, SEEDS
from prism.provenance import atomic_write_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"passed": bool(condition), "message": message})
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--parity-report", default="outputs/evaluator_parity/parity_report.json")
    parser.add_argument("--allow-no-cuda", action="store_true")
    parser.add_argument("--out", default="outputs/preflight.json")
    args = parser.parse_args()
    checks: list[dict] = []
    try:
        root = Path(args.root)
        expected_counts = {
            "train": 7601,
            "test": 2252,
            "test-day": 1455,
            "test-night": 797,
        }
        data_manifest = root / "prism_data_manifest.json"
        require(data_manifest.exists(), f"data manifest exists: {data_manifest}", checks)
        data = json.loads(data_manifest.read_text())
        require(data.get("audits_completed") is True, "full train/test audits completed", checks)
        require(
            data.get("protocol_counts") == {"train": 7601, "test": 2252},
            "manifest records exact train/test protocol counts",
            checks,
        )
        for split in ("train", "test"):
            audit_artifact = data.get("audit_reports", {}).get(split, {})
            audit_path = Path(audit_artifact.get("path", ""))
            require(audit_path.exists(), f"{split} audit report exists", checks)
            require(
                sha256(audit_path) == audit_artifact.get("sha256"),
                f"{split} audit report hash matches data manifest",
                checks,
            )
            audit = json.loads(audit_path.read_text())
            require(audit.get("sampled") == 1, f"{split} audit covered every frame", checks)
            require(
                audit.get("frames") == expected_counts[split],
                f"{split} audit frame count is exact",
                checks,
            )
            fatal_count = (
                int(audit.get("missing_visible", 0))
                + int(audit.get("missing_lwir", 0))
                + len(audit.get("unreadable", []))
                + len(audit.get("bad_size", []))
                + len(audit.get("missing_xml", []))
                + len(audit.get("bad_boxes", []))
            )
            require(fatal_count == 0, f"{split} audit has no fatal findings", checks)

        for split, expected in expected_counts.items():
            actual = len(kaist_split(args.root, split))
            require(actual == expected, f"{split} count {actual} == {expected}", checks)

        train_cv = kaist_split_cv(args.root, "train-cv")
        val_cv = kaist_split_cv(args.root, "val-cv")
        require(
            set(train_cv.ids).isdisjoint(val_cv.ids),
            "train-cv and val-cv frame IDs are disjoint",
            checks,
        )
        train_videos = {item.rsplit("/", 1)[0] for item in train_cv.ids}
        val_videos = {item.rsplit("/", 1)[0] for item in val_cv.ids}
        require(
            train_videos.isdisjoint(val_videos), "train-cv and val-cv videos are disjoint", checks
        )

        split_manifest_path = Path("splits/split_manifest.json")
        split_manifest = json.loads(split_manifest_path.read_text())
        for name in ("train", "val"):
            split_file = Path("splits") / f"{name}.txt"
            require(
                sha256(split_file) == split_manifest[name]["sha256"],
                f"{name} split hash matches manifest",
                checks,
            )

        parity_path = Path(args.parity_report)
        parity = json.loads(parity_path.read_text())
        require(parity.get("passed") is True, "evaluator parity certificate passes", checks)
        evaluator_manifest = Path("external/evaluator_manifest.json")
        require(
            parity.get("evaluator_manifest_sha256") == sha256(evaluator_manifest),
            "parity certificate matches current evaluator manifest",
            checks,
        )

        config = yaml.safe_load(Path("configs/architecture.yaml").read_text())
        require(
            config["optimization"]["epochs"] == ARCH_EPOCHS,
            f"config epochs == {ARCH_EPOCHS}",
            checks,
        )
        require(
            config["optimization"]["base_lr"] == RECIPE["lr"],
            f"config base LR == {RECIPE['lr']}",
            checks,
        )
        require(
            config["data"]["batch_size"] == RECIPE["batch_size"],
            f"config batch size == {RECIPE['batch_size']}",
            checks,
        )
        require(tuple(config["training"]["seeds"]) == SEEDS, f"config seeds == {SEEDS}", checks)
        require(
            config["training"]["checkpoint_every_epochs"] == RECIPE["checkpoint_every_epochs"],
            "config checkpoint cadence matches frozen recipe",
            checks,
        )
        require(
            config["training"]["validation_every_epochs"] == RECIPE["validation_every_epochs"],
            "config validation cadence matches frozen recipe",
            checks,
        )
        require(
            config["training"]["history_every_steps"] == RECIPE["history_every_steps"],
            "config history cadence matches frozen recipe",
            checks,
        )
        require(
            config["evaluation"]["nms_iou"] == RECIPE["nms_iou"],
            f"config NMS IoU == {RECIPE['nms_iou']}",
            checks,
        )
        require(
            config["evaluation"]["conf_floor"] == RECIPE["confidence_floor"],
            f"config confidence floor == {RECIPE['confidence_floor']}",
            checks,
        )
        require(
            torch.cuda.is_available() or args.allow_no_cuda,
            "CUDA available (or explicit --allow-no-cuda)",
            checks,
        )
        passed, error = True, None
    # Preflight is a fail-closed boundary: any unexpected validation failure
    # must still produce the machine-readable report consumed by automation.
    except Exception as exc:
        passed, error = False, f"{type(exc).__name__}: {exc}"

    report = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "passed": passed,
        "error": error,
        "checks": checks,
    }
    out = Path(args.out)
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("preflight FAILED")


if __name__ == "__main__":
    main()
