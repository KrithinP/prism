"""Turn a trained checkpoint into the standard KAIST number: MR^-2 All/Day/Night.

This is the bridge between training and the metric. It runs the model over a
protocol test split, collects detections, and feeds them to the CANONICAL
evaluator (prism.eval.canonical -- the reference MLPD semantics, TODO.md Phase 1).

Two hard rules:

  1. Collect detections at a very low confidence threshold. MR^-2 sweeps the
     threshold itself -- walking detections high-score to low-score IS the sweep,
     and the FPPI axis runs out to 1 false positive per image. Pre-filtering at,
     say, 0.3 truncates the curve and reports a flatteringly low miss rate. So
     conf_thr is 0.001, NOT the 0.3 used for visualisation.

  2. THE TEST-SET FIREWALL (TODO.md 4.4). Evaluating the official test set
     requires --official-test AND a frozen run manifest (--manifest). The
     manifest must hash the exact checkpoint bytes (ckpt_sha256) plus the
     experiment id and args hash -- a manifest that merely exists is rejected.
     Without the flag, only validation splits are reachable. Every official-test
     invocation is appended to outputs/test_invocations.jsonl with a timestamp.

Checkpoint dispatch reads the saved model_type ('unimodal' | 'multi-scale' |
'one-shot' | 'legacy') so old quarantine-era checkpoints still load.

Run (validation, firewall-safe):
    PYTHONPATH= .venv/bin/python -m prism.evaluate_model --ckpt ck.pt --split val-cv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from prism.data.kaist import day_frame_ids, kaist_split, kaist_split_cv
from prism.eval.canonical import evaluate_reference
from prism.eval.evaluate import evaluate, evaluate_subsets
from prism.eval.export import export_detections, load_frame_key_map
from prism.models.decode import decode_outputs, postprocess
from prism.models.yolox import YOLOX
from prism.protocol import PROTOCOL_VERSION, TEST_SPLITS
from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    append_jsonl,
    artifact_record,
    atomic_write_json,
    atomic_write_jsonl,
    environment_snapshot,
    new_attempt_id,
    source_fingerprint,
    utc_now,
)
from prism.release import load_and_validate_release_manifest

EVAL_CONF_THR = 0.001
# Match YOLOX's detection evaluation default. The low confidence floor below is
# intentionally KAIST-specific because MR sweeps scores out to 1 FPPI.
EVAL_NMS_IOU = 0.65

_TEST_INVOCATION_LOG = Path("outputs/test_invocations.jsonl")


def _check_manifest(manifest_path, ckpt_path) -> dict:
    """Validate a separately frozen release manifest and exact checkpoint."""
    try:
        return load_and_validate_release_manifest(manifest_path, {"model": ckpt_path})
    except ValueError as error:
        raise SystemExit(str(error)) from error


def load_checkpoint_model(ckpt_path, modality=None, device="cpu"):
    """Build the model the checkpoint describes, using its saved config."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    modality = modality or ck.get("modality", "rgb")
    mtype = ck.get("model_type", "unimodal")
    if modality != "fusion":
        model = YOLOX(num_classes=1, base_ch=ck.get("base_ch", 32))
    elif mtype == "early":
        from prism.models.early_fusion import EarlyFusionYOLOX

        model = EarlyFusionYOLOX(num_classes=1, base_ch=ck.get("base_ch", 32))
    elif mtype == "one-shot":
        from prism.models.one_shot_fusion import OneShotFusionYOLOX

        f = ck.get("fusion", {})
        model = OneShotFusionYOLOX(
            num_classes=1,
            base_ch=ck.get("base_ch", 32),
            meet_level=f.get("meet_level", 4),
            anchor=f.get("anchor", "thermal"),
            operator=f.get("operator", "concat"),
        )
    elif mtype == "multi-scale":
        from prism.models.multi_scale_fusion import MultiScaleFusionYOLOX

        f = ck.get("fusion", {})
        model = MultiScaleFusionYOLOX(
            num_classes=1,
            base_ch=ck.get("base_ch", 32),
            fuse_subset=tuple(f.get("fuse_subset", (3, 4, 5))),
            anchor=f.get("anchor", "thermal"),
            operator=f.get("operator", "concat"),
        )
    else:  # legacy/quarantined checkpoints
        from prism.models.yolox_fusion import LegacyTwoStreamYOLOX

        model = LegacyTwoStreamYOLOX(
            num_classes=1, fuse_stages=tuple(ck.get("fuse_stages", (3, 4, 5)))
        )
    model.load_state_dict(ck["model"])
    model.to(device)
    return model, modality, mtype


@torch.no_grad()
def collect_frames(
    model, base, modality, device, batch_size=8, input_policy="paired", day_ids=None
):
    """Run the model over every frame of `base`; return evaluator-shaped dicts
    (with an is_day flag when day_ids is given)."""
    frames = []
    model.eval()
    is_fusion = modality == "fusion"
    for start in range(0, len(base), batch_size):
        samples = [base[i] for i in range(start, min(start + batch_size, len(base)))]
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            if is_fusion:
                rgb = torch.stack([s["rgb"].float() / 255.0 for s in samples]).to(device)
                thm = torch.stack([s["thermal"].float() / 255.0 for s in samples]).to(device)
                if input_policy == "rgb":
                    outputs = model(rgb, rgb)
                elif input_policy == "thermal":
                    outputs = model(thm, thm)
                else:
                    outputs = model(rgb, thm)
            else:
                imgs = torch.stack([s[modality].float() / 255.0 for s in samples]).to(device)
                outputs = model(imgs)
        dets = postprocess(
            decode_outputs([o.float() for o in outputs]),
            conf_thr=EVAL_CONF_THR,
            iou_thr=EVAL_NMS_IOU,
        )
        for s, d in zip(samples, dets, strict=True):
            d = d.cpu()
            frame = {
                "id": s["id"],
                "det_boxes": d[:, :4].numpy(),
                "det_scores": d[:, 4].numpy(),
                "gt_boxes": s["boxes"].numpy(),
                "gt_classes": s["classes"].numpy(),  # 1 = person, 0 = ignore region
                "gt_occlusion": s["occlusion"].numpy(),
            }
            if day_ids is not None:
                frame["is_day"] = s["id"] in day_ids
            frames.append(frame)
    return frames


def write_raw_frames(frames: list[dict], path: str | Path) -> dict:
    """Persist the evaluator inputs needed to reproduce an aggregate metric."""
    atomic_write_jsonl(
        path,
        (
            {
                "frame_id": frame["id"],
                "det_boxes_xyxy": frame["det_boxes"].tolist(),
                "det_scores": frame["det_scores"].tolist(),
                "gt_boxes_xyxy": frame["gt_boxes"].tolist(),
                "gt_classes": frame["gt_classes"].tolist(),
                "gt_occlusion": frame["gt_occlusion"].tolist(),
                "is_day": frame.get("is_day"),
            }
            for frame in frames
        ),
    )
    return artifact_record(path)


def evaluate_checkpoint(
    ckpt_path,
    root,
    modality=None,
    splits=("test", "test-day", "test-night"),
    batch_size=8,
    input_policy="paired",
    day_ids=None,
    raw_output_dir=None,
):
    """Evaluate a checkpoint. Returns (modality, results) where results maps
    split name -> {mr (canonical), mr_legacy (old local evaluator, cross-check),
    frames, dets, subsets}."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, modality, _ = load_checkpoint_model(ckpt_path, modality, device)
    results = {}
    for split in splits:
        base = (
            kaist_split_cv(root, split)
            if split in ("train-cv", "val-cv")
            else kaist_split(root, split)
        )
        frames = collect_frames(
            model, base, modality, device, batch_size, input_policy=input_policy, day_ids=day_ids
        )
        n_det = sum(len(f["det_scores"]) for f in frames)
        canonical = evaluate_reference(frames)
        result = {
            "mr": canonical["all"],
            "mr_day": canonical["day"],
            "mr_night": canonical["night"],
            "mr_legacy": evaluate(frames),
            "frames": len(frames),
            "dets": n_det,
            "subsets": evaluate_subsets(frames),
        }
        if raw_output_dir is not None:
            raw_path = Path(raw_output_dir) / f"{split}.frames.jsonl"
            result["raw_frames"] = write_raw_frames(frames, raw_path)
        results[split] = result
    return modality, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument("--modality", default=None, choices=[None, "rgb", "thermal"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--split",
        default="val-cv",
        choices=["train-cv", "val-cv", "test", "test-day", "test-night"],
    )
    ap.add_argument(
        "--official-test",
        action="store_true",
        help="evaluate the official test set (requires --manifest)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="frozen run manifest JSON (experiment id, model hash, config)",
    )
    ap.add_argument("--input-policy", default="paired", choices=["paired", "rgb", "thermal"])
    ap.add_argument(
        "--export", default=None, help="also export reference-format detections to this TXT"
    )
    ap.add_argument(
        "--ann-json",
        default="external/MLPD-eval/evaluation_script/KAIST_annotation.json",
        help="pinned bundle's annotation json for image-id mapping",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="write machine-readable metrics JSON (used by matrix runners)",
    )
    ap.add_argument(
        "--raw-output-dir",
        default=None,
        help="write hashed per-frame post-NMS detections and ground truth as JSONL",
    )
    ap.add_argument(
        "--parent-manifest",
        default=None,
        help="attempt manifest that already captures the execution environment",
    )
    a = ap.parse_args()

    # --- the firewall: official test requires the frozen-manifest gate ---
    if a.split in TEST_SPLITS and not a.official_test:
        raise SystemExit(
            f"split={a.split} requires --official-test (test-set firewall, TODO.md 4.4).\n"
            "  Validation evaluation:  --split val-cv\n"
            "  Official test:          --split test --official-test --manifest run.json"
        )
    if a.official_test and not a.manifest:
        raise SystemExit("--official-test requires --manifest (frozen run manifest)")
    if a.official_test and a.split not in TEST_SPLITS:
        raise SystemExit(f"--official-test is only for the official splits, got {a.split}")
    if a.official_test:
        _check_manifest(a.manifest, a.ckpt)  # the gate IS the hash check

    evaluation_id = new_attempt_id("official_test" if a.official_test else "validation")
    repo_root = Path(__file__).resolve().parents[2]
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
        "evaluation_id": evaluation_id,
        "timestamp": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "command": [sys.executable, *sys.argv],
        "checkpoint": artifact_record(a.ckpt),
        "manifest": artifact_record(a.manifest) if a.manifest else None,
        "root": str(Path(a.root).resolve()),
        "split": a.split,
        "input_policy": a.input_policy,
        "batch_size": a.batch_size,
        "source_fingerprint": source_fingerprint(
            repo_root,
            ["src/prism", "configs/architecture.yaml"],
        ),
        "parent_manifest": (artifact_record(a.parent_manifest) if a.parent_manifest else None),
        "environment": (
            None
            if a.parent_manifest
            else environment_snapshot(
                repo_root,
                input_artifacts=[path for path in provenance_inputs if path.is_file()],
            )
        ),
    }
    if a.official_test:
        append_jsonl(
            _TEST_INVOCATION_LOG,
            {
                **base_record,
                "event": "official_test_started",
                "status": "started",
            },
        )

    day_ids = day_frame_ids(a.root) if a.split in ("test", "test-day", "test-night") else None
    splits = (a.split,)
    modality, res = evaluate_checkpoint(
        a.ckpt,
        a.root,
        a.modality,
        splits=splits,
        batch_size=a.batch_size,
        input_policy=a.input_policy,
        day_ids=day_ids,
        raw_output_dir=a.raw_output_dir,
    )

    label = {
        "test": "All",
        "test-day": "Day",
        "test-night": "Night",
        "val-cv": "Val",
        "train-cv": "TrainCV",
    }
    print(f"\nMR^-2 (Reasonable, canonical evaluator, lower is better) -- {a.ckpt}  [{modality}]")
    print(f"{'subset':<9}{'MR^-2':>10}{'day':>10}{'night':>10}{'frames':>9}{'dets':>10}")
    for split, r in res.items():
        day = "n/a" if r["mr_day"] is None else f"{r['mr_day'] * 100:.2f}%"
        night = "n/a" if r["mr_night"] is None else f"{r['mr_night'] * 100:.2f}%"
        print(
            f"{label[split]:<9}{r['mr'] * 100:>9.2f}%"
            f"{day:>10}{night:>10}{r['frames']:>9}{r['dets']:>10}"
        )
    if any(
        r["mr_legacy"] is not None and abs(r["mr_legacy"] - r["mr"]) > 0.01 for r in res.values()
    ):
        print(
            "\nNOTE: canonical MR differs from the legacy local evaluator by >0.01 on some "
            "split -- the legacy numbers are quarantined exploratory evidence (TODO.md 3.7)."
        )

    export_report = None
    if a.export:
        if a.split != "test":
            raise SystemExit("--export is defined for the full test split")
        base = kaist_split(a.root, "test")
        ids = list(base.ids)
        id_map = load_frame_key_map(Path(a.ann_json), ids)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        export_model, export_modality, _ = load_checkpoint_model(a.ckpt, a.modality, device=device)
        frames = collect_frames(
            export_model,
            base,
            export_modality,
            device,
            batch_size=a.batch_size,
            input_policy=a.input_policy,
            day_ids=day_ids,
        )
        by_id = {f["id"]: (f["det_boxes"], f["det_scores"]) for f in frames}
        ann = json.loads(Path(a.ann_json).read_text()) if Path(a.ann_json).exists() else None
        # bundle ids are 0-based; the TXT contract is 1-based (see eval/export.py)
        export_report = export_detections(
            by_id,
            id_map,
            ids,
            Path(a.export),
            annotation_ids=({i["id"] + 1 for i in ann["images"]} if ann else None),
        )
        print(
            f"\nexported {export_report['total_detections']} detections -> {a.export} "
            f"(sha256 {export_report['sha256'][:12]}...)"
        )

    if a.json_out:
        payload = {
            **base_record,
            "modality": modality,
            "splits": {
                split: {
                    "mr_all": r["mr"],
                    "mr_day": r["mr_day"],
                    "mr_night": r["mr_night"],
                    "frames": r["frames"],
                    "detections": r["dets"],
                    "mr_legacy": r["mr_legacy"],
                    "subsets": r["subsets"],
                    "raw_frames": r.get("raw_frames"),
                }
                for split, r in res.items()
            },
            "evaluation": {
                "confidence_floor": EVAL_CONF_THR,
                "nms_iou": EVAL_NMS_IOU,
                "canonical_evaluator": "prism.eval.canonical",
            },
        }
        json_out = Path(a.json_out)
        atomic_write_json(json_out, payload)

    if a.official_test:
        append_jsonl(
            _TEST_INVOCATION_LOG,
            {
                **base_record,
                "event": "official_test_finished",
                "timestamp": utc_now(),
                "status": "completed",
                "splits": list(res),
                "results": {s: r["mr"] for s, r in res.items()},
                "json_output": artifact_record(a.json_out) if a.json_out else None,
                "export": export_report,
            },
        )
        print(f"\nlogged official-test invocation -> {_TEST_INVOCATION_LOG}")


if __name__ == "__main__":
    main()
