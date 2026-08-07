"""Frozen release-manifest validation.

Architecture discovery and official-test evaluation are deliberately separate.
The discovery runner cannot manufacture a release manifest. A reviewer freezes
one only after validation selection and evaluator certification are complete.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prism.provenance import sha256_file


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_and_validate_release_manifest(
    manifest_path: str | Path, requested_checkpoints: dict[str, str | Path]
) -> dict:
    """Validate a frozen manifest and exact checkpoint bytes, fail closed."""
    path = Path(manifest_path)
    if not path.exists():
        raise ValueError(f"release manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"release manifest is invalid JSON: {path}: {error}") from error

    required = (
        "schema_version",
        "status",
        "experiment",
        "args_hash",
        "frozen_at",
        "selection_report",
        "split_manifest",
        "parity_report",
        "evaluator_manifest",
        "ledger",
        "checkpoints",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"release manifest is missing fields: {missing}")
    if manifest["schema_version"] != 2 or manifest["status"] != "frozen":
        raise ValueError("release manifest must have schema_version=2 and status='frozen'")

    for artifact_key in (
        "selection_report",
        "split_manifest",
        "parity_report",
        "evaluator_manifest",
        "ledger",
    ):
        artifact = manifest[artifact_key]
        artifact_path = Path(artifact.get("path", ""))
        if not artifact_path.exists():
            raise ValueError(f"{artifact_key} no longer exists: {artifact_path}")
        actual = sha256_file(artifact_path)
        if actual != artifact.get("sha256"):
            raise ValueError(
                f"{artifact_key} hash mismatch: manifest={artifact.get('sha256')}, actual={actual}"
            )

    parity = json.loads(Path(manifest["parity_report"]["path"]).read_text())
    if parity.get("passed") is not True:
        raise ValueError("the frozen evaluator parity report is not passing")

    selection = json.loads(Path(manifest["selection_report"]["path"]).read_text())
    selection_ledger = selection.get("ledger")
    if not isinstance(selection_ledger, dict):
        raise ValueError("selection report is missing its immutable ledger snapshot")
    selection_ledger_path = Path(selection_ledger.get("path", ""))
    if not selection_ledger_path.is_file():
        raise ValueError(f"selection ledger snapshot no longer exists: {selection_ledger_path}")
    selection_ledger_actual = sha256_file(selection_ledger_path)
    if selection_ledger_actual != selection_ledger.get("sha256"):
        raise ValueError(
            "selection ledger snapshot hash mismatch: "
            f"manifest={selection_ledger.get('sha256')}, actual={selection_ledger_actual}"
        )

    declared = {item.get("role"): item for item in manifest["checkpoints"]}
    if set(requested_checkpoints) != set(declared):
        raise ValueError(
            "checkpoint roles differ from the frozen manifest: "
            f"requested={sorted(requested_checkpoints)}, declared={sorted(declared)}"
        )
    for role, requested in requested_checkpoints.items():
        item = declared[role]
        requested_path = Path(requested)
        declared_path = Path(item.get("path", ""))
        if requested_path.resolve() != declared_path.resolve():
            raise ValueError(
                f"{role} checkpoint path differs from frozen manifest: "
                f"{requested_path} != {declared_path}"
            )
        if not requested_path.exists():
            raise ValueError(f"{role} checkpoint does not exist: {requested_path}")
        actual = sha256_file(requested_path)
        if actual != item.get("sha256"):
            raise ValueError(
                f"{role} checkpoint hash mismatch: manifest={item.get('sha256')}, actual={actual}"
            )
    return manifest
