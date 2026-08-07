"""Post-hoc orchestration reuses only exact checkpoint-derived artifacts."""

import json

import pytest

from scripts.run_posthoc_studies import (
    PROFILE_ARMS,
    load_checkpoints,
    output_matches,
    sha256,
)


def test_checkpoint_resolution_uses_latest_completed_attempt(tmp_path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "arm": "C2",
                        "seed": 0,
                        "status": "completed",
                        "checkpoint": str(first),
                    }
                ),
                json.dumps(
                    {
                        "arm": "C2",
                        "seed": 0,
                        "status": "failed",
                        "checkpoint": str(tmp_path / "failed.pt"),
                    }
                ),
                json.dumps(
                    {
                        "arm": "C2",
                        "seed": 0,
                        "status": "completed",
                        "checkpoint": str(second),
                    }
                ),
            ]
        )
    )
    assert load_checkpoints(ledger)[("C2", 0)] == second


def test_existing_output_must_match_exact_checkpoint_hash(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "profile.json"
    output.write_text(json.dumps({"checkpoint_sha256": sha256(checkpoint)}))
    assert output_matches(output, {"checkpoint": sha256(checkpoint)})
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        output_matches(output, {"checkpoint": "wrong"})


def test_existing_output_rejects_missing_raw_artifact(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "analysis.json"
    output.write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256(checkpoint),
                "raw_npz": {
                    "path": str(tmp_path / "missing.npz"),
                    "sha256": "missing",
                },
            }
        )
    )
    with pytest.raises(SystemExit, match="raw diagnostic artifact"):
        output_matches(output, {"checkpoint": sha256(checkpoint)})


def test_profiles_cover_learned_graphs_but_not_late_ensembles() -> None:
    assert "B1" in PROFILE_ARMS and "B13" in PROFILE_ARMS
    assert "B11" not in PROFILE_ARMS and "B12" not in PROFILE_ARMS
