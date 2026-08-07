"""B11/B12 late-fusion runner (docs/EXPERIMENT_PLAN.md 7.2): merge the
DETECTIONS of two independent unimodal checkpoints (A1 rgb + A2 thermal) and
evaluate the merged set with the canonical evaluator. No training happens here;
the checkpoints are the frozen single-modality baselines.

  --merge union  -> B11: concatenate + one NMS pass (prism.models.late_fusion.union_nms)
  --merge proben -> B12: published ProbEn (Chen et al., ECCV 2022): NMS-cluster
                   association + Bayes product-rule score fusion + box averaging
                   (prism.models.late_fusion.proben_merge)

Firewall (same discipline as prism.evaluate_model): the official test splits
require --official-test AND a frozen manifest whose "checkpoints" list hashes
both checkpoint files. Every official-test invocation is appended to
outputs/test_invocations.jsonl. Validation splits need no manifest.

Ledger: every run appends a row to outputs/ledger.jsonl with the merge type,
both checkpoint paths, and the canonical MR All/Day/Night.

Usage:
  PYTHONPATH= .venv/bin/python scripts/late_fusion_runner.py \
      --rgb-ckpt outputs/experiments/A1_s0/rgb_seed0.pt \
      --thermal-ckpt outputs/experiments/A2_s0/thermal_seed0.pt \
      --merge proben --split val-cv --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from prism.data.kaist import day_frame_ids, kaist_split, kaist_split_cv
from prism.eval.canonical import evaluate_reference
from prism.eval.export import export_detections, load_frame_key_map
from prism.evaluate_model import (
    _TEST_INVOCATION_LOG,
    collect_frames,
    load_checkpoint_model,
    write_raw_frames,
)
from prism.models.late_fusion import proben_merge, union_nms
from prism.protocol import PROTOCOL_VERSION, TEST_SPLITS
from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    acquire_run_lock,
    append_jsonl,
    artifact_record,
    atomic_write_json,
    environment_snapshot,
    new_attempt_id,
    read_jsonl,
    release_run_lock,
    source_fingerprint,
    unfinished_attempts,
    utc_now,
)
from prism.release import load_and_validate_release_manifest

LEDGER = Path("outputs/ledger.jsonl")

_MERGES = {"union": union_nms, "proben": proben_merge}


def _check_manifest(manifest_path, rgb_ckpt, thm_ckpt) -> dict:
    """Validate a separately frozen release manifest for both checkpoints."""
    try:
        return load_and_validate_release_manifest(
            manifest_path, {"rgb": rgb_ckpt, "thermal": thm_ckpt}
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-ckpt", required=True)
    ap.add_argument("--thermal-ckpt", required=True)
    ap.add_argument("--merge", required=True, choices=list(_MERGES))
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument(
        "--split",
        default="val-cv",
        choices=["train-cv", "val-cv", "test", "test-day", "test-night"],
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--official-test", action="store_true")
    ap.add_argument("--manifest", default=None)
    ap.add_argument(
        "--arm-label", default=None, help="ledger arm id (default: B11 for union, B12 for proben)"
    )
    ap.add_argument("--seed", type=int, required=True, help="paired unimodal seed for the ledger")
    ap.add_argument(
        "--export", default=None, help="reference-format TXT export (full official test only)"
    )
    ap.add_argument(
        "--ann-json", default="external/MLPD-eval/evaluation_script/KAIST_annotation.json"
    )
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--raw-output-dir", default=None)
    a = ap.parse_args()

    if a.split in TEST_SPLITS:
        if not a.official_test:
            raise SystemExit(
                f"split={a.split} requires --official-test (test-set firewall, TODO.md 4.4)"
            )
        if not a.manifest:
            raise SystemExit("--official-test requires --manifest (frozen run manifest)")
        m = _check_manifest(a.manifest, a.rgb_ckpt, a.thermal_ckpt)
        experiment_id = m["experiment"]
    else:
        experiment_id = a.arm_label or ("B11" if a.merge == "union" else "B12")

    attempt_id = new_attempt_id(f"{experiment_id}_s{a.seed}")
    run_lock = acquire_run_lock(Path("outputs/.locks") / f"{experiment_id}_s{a.seed}.lock")
    for stale_id, stale in unfinished_attempts(
        read_jsonl(LEDGER), arm=experiment_id, seed=a.seed
    ).items():
        append_jsonl(
            LEDGER,
            {
                **stale,
                "event": "attempt_abandoned",
                "timestamp": utc_now(),
                "status": "abandoned",
                "error": "prior process ended without a terminal event",
                "superseded_by": attempt_id,
                "abandoned_attempt_id": stale_id,
            },
        )
    repo_root = Path(__file__).resolve().parents[1]
    provenance_inputs = [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("configs/architecture.yaml"),
        Path("splits/split_manifest.json"),
        Path("external/evaluator_manifest.json"),
        Path(a.root) / "prism_data_manifest.json",
    ]
    base_record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "arm": experiment_id,
        "label": f"late-fusion {a.merge}",
        "seed": a.seed,
        "merge": a.merge,
        "split": a.split,
        "protocol_version": PROTOCOL_VERSION,
        "command": [sys.executable, *sys.argv],
        "rgb_checkpoint": artifact_record(a.rgb_ckpt),
        "thermal_checkpoint": artifact_record(a.thermal_ckpt),
        "manifest": artifact_record(a.manifest) if a.manifest else None,
        "source_fingerprint": source_fingerprint(
            repo_root,
            ["src/prism", "scripts/late_fusion_runner.py"],
        ),
        "environment": environment_snapshot(
            repo_root,
            input_artifacts=[path for path in provenance_inputs if path.is_file()],
        ),
    }
    append_jsonl(
        LEDGER,
        {
            **base_record,
            "event": "attempt_started",
            "timestamp": utc_now(),
            "status": "started",
        },
    )
    if a.official_test:
        append_jsonl(
            _TEST_INVOCATION_LOG,
            {
                **base_record,
                "event": "official_test_started",
                "timestamp": utc_now(),
                "status": "started",
            },
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb_model, _, _ = load_checkpoint_model(a.rgb_ckpt, "rgb", device)
    thm_model, _, _ = load_checkpoint_model(a.thermal_ckpt, "thermal", device)
    base = (
        kaist_split_cv(a.root, a.split)
        if a.split in ("train-cv", "val-cv")
        else kaist_split(a.root, a.split)
    )
    day_ids = day_frame_ids(a.root) if a.split in TEST_SPLITS else None

    fr = collect_frames(
        rgb_model, base, "rgb", device, a.batch_size, input_policy="paired", day_ids=day_ids
    )
    ft = collect_frames(
        thm_model, base, "thermal", device, a.batch_size, input_policy="paired", day_ids=day_ids
    )
    if [frame["id"] for frame in fr] != [frame["id"] for frame in ft]:
        raise RuntimeError("RGB and thermal frame order differs")

    merge = _MERGES[a.merge]
    frames = []
    for r, t in zip(fr, ft, strict=True):
        dr = torch.zeros((len(r["det_scores"]), 6))
        if len(r["det_scores"]):
            dr[:, :4] = torch.from_numpy(r["det_boxes"])
            dr[:, 4] = torch.from_numpy(r["det_scores"])
        dt = torch.zeros((len(t["det_scores"]), 6))
        if len(t["det_scores"]):
            dt[:, :4] = torch.from_numpy(t["det_boxes"])
            dt[:, 4] = torch.from_numpy(t["det_scores"])
        merged = merge(dr, dt)
        frame = {
            "id": r["id"],
            "det_boxes": merged[:, :4].numpy(),
            "det_scores": merged[:, 4].numpy(),
            "gt_boxes": r["gt_boxes"],
            "gt_classes": r["gt_classes"],
            "gt_occlusion": r["gt_occlusion"],
        }
        if "is_day" in r:
            frame["is_day"] = r["is_day"]
        frames.append(frame)

    canon = evaluate_reference(frames)
    n_det = sum(len(f["det_scores"]) for f in frames)
    label = {
        "test": "All",
        "test-day": "Day",
        "test-night": "Night",
        "val-cv": "Val",
        "train-cv": "TrainCV",
    }
    print(f"\nLate fusion [{a.merge}] on {a.split} -- {a.rgb_ckpt} + {a.thermal_ckpt}")
    print(f"{'subset':<9}{'MR^-2':>10}{'day':>8}{'night':>9}{'frames':>9}{'dets':>10}")
    day = "n/a" if canon["day"] is None else f"{canon['day'] * 100:.2f}%"
    night = "n/a" if canon["night"] is None else f"{canon['night'] * 100:.2f}%"
    print(
        f"{label[a.split]:<9}{canon['all'] * 100:>9.2f}%"
        f"{day:>8}{night:>9}{len(frames):>9}{n_det:>10}"
    )

    completed_record = {
        **base_record,
        "event": "attempt_finished",
        "timestamp": utc_now(),
        "status": "completed",
        # Backward-compatible path fields remain available to existing summaries.
        "rgb_ckpt": a.rgb_ckpt,
        "thermal_ckpt": a.thermal_ckpt,
        "mr": canon["all"],
        "mr_day": canon["day"],
        "mr_night": canon["night"],
        "frames": len(frames),
        "dets": n_det,
    }
    if a.raw_output_dir:
        raw_path = Path(a.raw_output_dir) / f"{a.split}.frames.jsonl"
        completed_record["raw_frames"] = write_raw_frames(frames, raw_path)

    if a.export:
        if a.split != "test" or not a.official_test:
            raise SystemExit("--export is allowed only for a frozen full official test")
        ids = list(base.ids)
        id_map = load_frame_key_map(Path(a.ann_json), ids)
        by_id = {frame["id"]: (frame["det_boxes"], frame["det_scores"]) for frame in frames}
        annotation = json.loads(Path(a.ann_json).read_text())
        report = export_detections(
            by_id,
            id_map,
            ids,
            Path(a.export),
            annotation_ids={int(image["id"]) + 1 for image in annotation["images"]},
        )
        completed_record["export"] = report
        print(f"exported {report['total_detections']} detections -> {a.export}")

    if a.json_out:
        atomic_write_json(a.json_out, completed_record)
        completed_record["metrics_artifact"] = artifact_record(a.json_out)

    append_jsonl(LEDGER, completed_record)
    print(f"\nrow appended to {LEDGER}")

    if a.official_test:
        append_jsonl(
            _TEST_INVOCATION_LOG,
            {
                **completed_record,
                "event": "official_test_finished",
                "splits": [a.split],
                "results": {a.split: canon["all"]},
            },
        )
        print(f"logged official-test invocation -> {_TEST_INVOCATION_LOG}")
    release_run_lock(run_lock)


if __name__ == "__main__":
    main()
