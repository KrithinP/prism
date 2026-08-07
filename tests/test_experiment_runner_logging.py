"""The matrix runner must leave a complete attempt chain without running a model."""

import json
import sys
from pathlib import Path

import torch

from prism.protocol import PROTOCOL_VERSION
from prism.provenance import artifact_record, read_jsonl
from scripts import run_experiments
from scripts.audit_ledger import audit_ledger
from scripts.select_winner import validate_completed_evidence


def test_matrix_attempt_has_start_manifest_logs_artifacts_and_terminal_row(tmp_path, monkeypatch):
    output_root = tmp_path / "experiments"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(run_experiments, "OUT", output_root)
    monkeypatch.setattr(run_experiments, "LEDGER", ledger)
    monkeypatch.setattr(run_experiments, "source_fingerprint", lambda: "a" * 64)
    monkeypatch.setattr(
        run_experiments,
        "environment_snapshot",
        lambda *_args, **_kwargs: {"captured": True},
    )
    commands = []

    def fake_subprocess(command, stdout_path, stderr_path, **_kwargs):
        commands.append(command)
        stdout = Path(stdout_path)
        stderr = Path(stderr_path)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("stdout\n")
        stderr.write_text("")
        if command[1:3] == ["-m", "prism.train"]:
            checkpoint = output_root / "B10_s0" / "fusion_seed0.pt"
            torch.save(
                {
                    "run_args_hash": command[command.index("--run-args-hash") + 1],
                    "source_fingerprint": command[command.index("--source-fingerprint") + 1],
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": 0,
                    "epoch": 40,
                },
                checkpoint,
            )
        else:
            metrics = Path(command[command.index("--json-out") + 1])
            raw_dir = Path(command[command.index("--raw-output-dir") + 1])
            raw_dir.mkdir(parents=True)
            raw = raw_dir / "val-cv.frames.jsonl"
            raw.write_text('{"frame_id":"frame"}\n')
            metrics.write_text(
                json.dumps(
                    {
                        "splits": {
                            "val-cv": {
                                "mr_all": 0.25,
                                "mr_day": 0.2,
                                "mr_night": 0.3,
                                "frames": 1,
                                "detections": 1,
                                "raw_frames": artifact_record(raw),
                            }
                        }
                    }
                )
            )
        return {
            "command": command,
            "returncode": 0,
            "stdout": artifact_record(stdout),
            "stderr": artifact_record(stderr),
        }

    monkeypatch.setattr(run_experiments, "run_logged_subprocess", fake_subprocess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments.py",
            "--arm",
            "B10",
            "--seeds",
            "0",
            "--limit",
            "1",
            "--root",
            str(tmp_path / "data"),
        ],
    )

    run_experiments.main()

    rows = read_jsonl(ledger)
    assert [row["event"] for row in rows] == [
        "attempt_started",
        "attempt_finished",
    ]
    started, finished = rows
    assert started["attempt_id"] == finished["attempt_id"]
    assert finished["status"] == "completed"
    assert finished["checkpoint_artifact"]["sha256"]
    assert finished["validation_artifact"]["sha256"]
    assert finished["phases"]["training"]["stdout"]["sha256"]
    assert finished["phases"]["validation"]["stdout"]["sha256"]
    validate_completed_evidence(finished)

    manifest = json.loads(Path(started["attempt_manifest"]["path"]).read_text())
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["environment"] == {"captured": True}
    assert "--history-path" in manifest["planned_commands"]["train"]
    assert "--parent-manifest" in manifest["planned_commands"]["validation"]
    assert len(commands) == 2
    audit = audit_ledger(ledger)
    assert audit["passed"]
    assert audit["verified_artifact_references"] >= 8
