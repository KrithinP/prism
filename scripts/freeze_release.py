"""Freeze validation-selected checkpoint(s) for official-test evaluation.

This command never evaluates test data. It creates an immutable evidence bundle
that the official evaluators validate before the test set becomes reachable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from prism.protocol import PROTOCOL_VERSION, RECIPE, SEEDS
from prism.provenance import artifact_record, atomic_write_bytes, atomic_write_json
from prism.release import canonical_hash, sha256_file


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--checkpoint must be ROLE=PATH")
    role, path = value.split("=", 1)
    if not role or not path:
        raise argparse.ArgumentTypeError("--checkpoint must be ROLE=PATH")
    return role, Path(path)


def git_state() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit_sha": sha, "dirty": dirty}


def artifact(path: Path, *, require_passed: bool = False) -> dict:
    if not path.exists():
        raise SystemExit(f"required freeze artifact does not exist: {path}")
    if require_passed and json.loads(path.read_text()).get("passed") is not True:
        raise SystemExit(f"required certification is not passing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--split-manifest", default="splits/split_manifest.json")
    parser.add_argument("--parity-report", default="outputs/evaluator_parity/parity_report.json")
    parser.add_argument("--evaluator-manifest", default="external/evaluator_manifest.json")
    parser.add_argument("--ledger", default="outputs/ledger.jsonl")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {out}")
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise SystemExit("duplicate checkpoint role")
    if set(checkpoints) not in ({"model"}, {"rgb", "thermal"}):
        raise SystemExit("roles must be exactly {model} or {rgb, thermal}")

    code = git_state()
    if code["dirty"] and not args.allow_dirty:
        raise SystemExit(
            "worktree is dirty; commit the exact release source or pass --allow-dirty "
            "to record an explicit exception"
        )

    checkpoint_rows = []
    model_configs = {}
    for role, path in checkpoints.items():
        if not path.exists():
            raise SystemExit(f"checkpoint does not exist: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_configs[role] = {
            key: state.get(key)
            for key in (
                "modality",
                "model_type",
                "base_ch",
                "fusion",
                "fusion_inputs",
                "seed",
                "epoch",
                "protocol_version",
                "protocol",
            )
        }
        checkpoint_rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    evidence = {
        "experiment": args.experiment,
        "reason": args.reason,
        "model_configs": model_configs,
        "recipe": RECIPE,
        "seeds": SEEDS,
    }
    ledger = Path(args.ledger)
    if not ledger.is_file():
        raise SystemExit(f"experiment ledger does not exist: {ledger}")
    ledger_snapshot = out.with_name(f"{out.stem}.ledger.jsonl")
    if ledger_snapshot.exists():
        raise SystemExit(f"refusing to overwrite frozen ledger snapshot: {ledger_snapshot}")
    atomic_write_bytes(ledger_snapshot, ledger.read_bytes())

    manifest = {
        "schema_version": 2,
        "status": "frozen",
        "experiment": args.experiment,
        "reason": args.reason,
        "frozen_at": datetime.now(UTC).isoformat(),
        "args_hash": canonical_hash(evidence),
        "model_configs": model_configs,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "recipe": RECIPE,
            "seeds": list(SEEDS),
        },
        "code": code,
        "ledger": artifact_record(ledger_snapshot),
        "selection_report": artifact(Path(args.selection_report)),
        "split_manifest": artifact(Path(args.split_manifest)),
        "parity_report": artifact(Path(args.parity_report), require_passed=True),
        "evaluator_manifest": artifact(Path(args.evaluator_manifest)),
        "checkpoints": checkpoint_rows,
    }
    atomic_write_json(out, manifest)
    print(f"frozen release manifest: {out}")
    print(f"args hash: {manifest['args_hash']}")


if __name__ == "__main__":
    main()
