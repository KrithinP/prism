import json

import pytest

from prism.release import load_and_validate_release_manifest, sha256_file


def test_release_manifest_validates_every_frozen_artifact(tmp_path):
    artifacts = {}
    for name, contents in (
        ("selection", "{}"),
        ("split", "{}"),
        ("parity", '{"passed": true}'),
        ("evaluator", "{}"),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(contents)
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    selection_ledger = tmp_path / "selection.ledger.jsonl"
    selection_ledger.write_text('{"status":"completed"}\n')
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "ledger": {
                    "path": str(selection_ledger),
                    "sha256": sha256_file(selection_ledger),
                }
            }
        )
    )
    artifacts["selection"] = {
        "path": str(selection_path),
        "sha256": sha256_file(selection_path),
    }
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    manifest = {
        "schema_version": 2,
        "status": "frozen",
        "experiment": "x",
        "args_hash": "abc",
        "frozen_at": "now",
        "selection_report": artifacts["selection"],
        "split_manifest": artifacts["split"],
        "parity_report": artifacts["parity"],
        "evaluator_manifest": artifacts["evaluator"],
        "ledger": artifacts["selection"],
        "checkpoints": [
            {
                "role": "model",
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            }
        ],
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(manifest))
    loaded = load_and_validate_release_manifest(path, {"model": checkpoint})
    assert loaded["experiment"] == "x"

    selection_ledger.write_text('{"status":"modified"}\n')
    with pytest.raises(ValueError, match="selection ledger snapshot hash mismatch"):
        load_and_validate_release_manifest(path, {"model": checkpoint})
    selection_ledger.write_text('{"status":"completed"}\n')

    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_and_validate_release_manifest(path, {"model": checkpoint})
