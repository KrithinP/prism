"""Experiment provenance must be durable, immutable, and secret-conscious."""

import json
import sys

import pytest
import torch

from prism.provenance import (
    acquire_run_lock,
    append_jsonl,
    artifact_record,
    atomic_save_npz,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    environment_snapshot,
    latest_terminal_rows,
    new_attempt_id,
    read_jsonl,
    release_run_lock,
    run_logged_subprocess,
    source_fingerprint,
    unfinished_attempts,
    verify_artifact_record,
)


def test_append_jsonl_preserves_all_rows_and_rejects_nan(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_jsonl(ledger, {"event": "started", "attempt": 1})
    append_jsonl(ledger, {"event": "completed", "attempt": 1})
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "completed"]

    with pytest.raises(ValueError, match="Out of range"):
        append_jsonl(ledger, {"bad": float("nan")})
    assert len(ledger.read_text().splitlines()) == 2


def test_jsonl_reader_reports_line_number_and_lifecycle_gaps(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_jsonl(
        ledger,
        {
            "event": "attempt_started",
            "attempt_id": "unfinished",
            "arm": "A1",
            "seed": 0,
        },
    )
    append_jsonl(
        ledger,
        {
            "event": "attempt_started",
            "attempt_id": "finished",
            "arm": "A1",
            "seed": 1,
        },
    )
    append_jsonl(
        ledger,
        {
            "event": "attempt_finished",
            "attempt_id": "finished",
            "arm": "A1",
            "seed": 1,
        },
    )
    assert set(unfinished_attempts(read_jsonl(ledger))) == {"unfinished"}

    ledger.write_text(ledger.read_text() + "{invalid\n")
    with pytest.raises(ValueError, match=r"ledger\.jsonl:4"):
        read_jsonl(ledger)


def test_latest_terminal_row_does_not_hide_a_later_failed_retry():
    rows = [
        {"arm": "A1", "seed": 0, "status": "completed"},
        {
            "arm": "A1",
            "seed": 0,
            "event": "attempt_finished",
            "status": "failed",
        },
    ]
    assert latest_terminal_rows(rows)[("A1", 0)]["status"] == "failed"


def test_experiment_lock_rejects_a_second_owner(tmp_path):
    lock = tmp_path / "A1_s0.lock"
    first = acquire_run_lock(lock)
    try:
        with pytest.raises(RuntimeError, match="another process"):
            acquire_run_lock(lock)
    finally:
        release_run_lock(first)
    second = acquire_run_lock(lock)
    release_run_lock(second)


def test_atomic_json_and_artifact_record(tmp_path):
    output = tmp_path / "record.json"
    atomic_write_json(output, {"status": "complete"})
    assert json.loads(output.read_text()) == {"status": "complete"}
    artifact = artifact_record(output)
    assert artifact["path"] == str(output.resolve())
    assert artifact["bytes"] == output.stat().st_size
    assert len(artifact["sha256"]) == 64
    assert verify_artifact_record(artifact) == output
    output.write_text("modified")
    with pytest.raises(ValueError, match="mismatch"):
        verify_artifact_record(artifact)


def test_atomic_bytes_preserves_exact_payload(tmp_path):
    output = tmp_path / "snapshot.jsonl"
    atomic_write_bytes(output, b'{"row":1}\n')
    assert output.read_bytes() == b'{"row":1}\n'


def test_source_fingerprint_changes_with_declared_source(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    module = source / "module.py"
    module.write_text("VALUE = 1\n")
    first = source_fingerprint(tmp_path, [source])
    module.write_text("VALUE = 2\n")
    second = source_fingerprint(tmp_path, [source])
    assert len(first) == 64
    assert first != second


def test_atomic_jsonl_is_complete_and_parseable(tmp_path):
    output = tmp_path / "frames.jsonl"
    atomic_write_jsonl(output, [{"frame": 1}, {"frame": 2}])
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"frame": 1},
        {"frame": 2},
    ]


def test_atomic_checkpoint_is_loadable_and_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "checkpoint.pt"
    atomic_torch_save(output, {"epoch": 10, "weights": torch.ones(2)})
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 10
    assert torch.equal(loaded["weights"], torch.ones(2))
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_npz_is_loadable_and_leaves_no_temporary_file(tmp_path):
    import numpy as np

    output = tmp_path / "raw.npz"
    atomic_save_npz(output, values=np.array([1, 2, 3]))
    with np.load(output) as loaded:
        assert loaded["values"].tolist() == [1, 2, 3]
    assert list(tmp_path.glob("*.tmp.npz")) == []


def test_logged_subprocess_keeps_exact_streams_and_refuses_overwrite(tmp_path):
    stdout = tmp_path / "attempt.stdout.log"
    stderr = tmp_path / "attempt.stderr.log"
    result = run_logged_subprocess(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        stdout,
        stderr,
    )
    assert result["returncode"] == 0
    assert stdout.read_text() == "out\n"
    assert stderr.read_text() == "err\n"
    assert result["stdout"]["sha256"] == artifact_record(stdout)["sha256"]

    with pytest.raises(FileExistsError):
        run_logged_subprocess([sys.executable, "-c", "pass"], stdout, stderr)


def test_logged_subprocess_records_launch_failure(tmp_path):
    result = run_logged_subprocess(
        [str(tmp_path / "does-not-exist")],
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
    )
    assert result["returncode"] is None
    assert result["launch_error"].startswith("FileNotFoundError:")
    assert result["stderr"]["bytes"] > 0


def test_environment_snapshot_does_not_dump_arbitrary_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_TEST_SECRET", "do-not-log")
    snapshot = environment_snapshot(tmp_path)
    encoded = json.dumps(snapshot)
    assert "PRISM_TEST_SECRET" not in encoded
    assert "do-not-log" not in encoded
    assert "python_version" in snapshot["process"]


def test_attempt_ids_are_unique_and_prefixed():
    first = new_attempt_id("A1_s0")
    second = new_attempt_id("A1_s0")
    assert first.startswith("A1_s0_")
    assert first != second
