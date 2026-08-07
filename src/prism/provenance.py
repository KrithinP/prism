"""Durable, secret-conscious provenance utilities for experiment processes.

JSONL records are append-only, process-locked, flushed, and fsynced. Large
artifacts are represented by absolute path, byte count, and streaming SHA256.
Environment capture is deliberately allowlisted so credentials and unrelated
shell state never enter an experiment manifest.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

PROVENANCE_SCHEMA_VERSION = 1
SAFE_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
)


def utc_now() -> str:
    """Return an unambiguous, timezone-aware timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_attempt_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}_{timestamp}_p{os.getpid()}_{uuid.uuid4().hex[:8]}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(
    repo_root: str | Path,
    paths: list[str | Path] | tuple[str | Path, ...],
) -> str:
    """Hash a declared executable source closure in stable relative-path order."""
    root = Path(repo_root).resolve()
    files = set()
    for value in paths:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            files.update(candidate for candidate in path.rglob("*.py") if candidate.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"source fingerprint input does not exist: {path}")
    digest = hashlib.sha256()
    for path in sorted(files):
        try:
            identity = str(path.relative_to(root))
        except ValueError:
            identity = str(path)
        digest.update(identity.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_record(path: str | Path) -> dict:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact does not exist: {artifact}")
    stat = artifact.stat()
    return {
        "path": str(artifact.resolve()),
        "bytes": stat.st_size,
        "sha256": sha256_file(artifact),
    }


def verify_artifact_record(record: dict, *, label: str = "artifact") -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    path = Path(record.get("path", ""))
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    if record.get("bytes") is not None and path.stat().st_size != record["bytes"]:
        raise ValueError(
            f"{label} size mismatch: expected {record['bytes']}, got {path.stat().st_size}"
        )
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise ValueError(f"{label} hash mismatch: expected {record.get('sha256')}, got {actual}")
    return path


def append_jsonl(path: str | Path, row: dict, *, durable: bool = True) -> None:
    """Append one complete JSON object without inter-process interleaving."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
    with destination.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(payload)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            lines = handle.read().splitlines()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {source}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {source}:{line_number}")
        rows.append(row)
    return rows


def unfinished_attempts(
    rows: list[dict],
    *,
    arm: str | None = None,
    seed: int | None = None,
) -> dict[str, dict]:
    """Return started experiment attempts with no terminal lifecycle event."""
    started = {}
    finished = set()
    for row in rows:
        attempt_id = row.get("attempt_id")
        if not attempt_id:
            continue
        if arm is not None and row.get("arm") != arm:
            continue
        if seed is not None and row.get("seed") != seed:
            continue
        if row.get("event") == "attempt_started":
            if attempt_id in started:
                raise ValueError(f"duplicate attempt_started event: {attempt_id}")
            started[attempt_id] = row
        elif row.get("event") in {"attempt_finished", "attempt_abandoned"}:
            if attempt_id in finished:
                raise ValueError(f"duplicate terminal attempt event: {attempt_id}")
            finished.add(attempt_id)
    return {attempt_id: row for attempt_id, row in started.items() if attempt_id not in finished}


def latest_terminal_rows(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Return the last terminal record for each arm/seed in append order."""
    latest = {}
    for row in rows:
        if row.get("arm") is None or row.get("seed") is None:
            continue
        event = row.get("event")
        is_terminal = event in {"attempt_finished", "attempt_abandoned"} or (
            event is None
            and (
                row.get("status") in {"completed", "failed", "abandoned"}
                or row.get("mr") is not None
            )
        )
        if is_terminal:
            latest[(str(row["arm"]), int(row["seed"]))] = row
    return latest


def acquire_run_lock(path: str | Path):
    """Acquire a non-blocking process lock; the caller must close the handle."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another process holds the experiment lock: {lock_path}") from None
    return handle


def release_run_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Write a JSON artifact atomically and durably."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write immutable binary/text evidence through a durable atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: str | Path, rows) -> None:
    """Write a complete JSONL artifact atomically and durably."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    """Write a checkpoint atomically so an interrupted save is never reusable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: str | Path, **arrays) -> None:
    """Write a compressed NumPy artifact atomically."""
    import numpy as np

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _git(repo_root: Path, *args: str) -> str | None:
    process = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def git_snapshot(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return {
        "commit_sha": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "diff_sha256": (hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None),
    }


def environment_snapshot(
    repo_root: str | Path,
    *,
    input_artifacts: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict:
    """Capture the execution environment without serializing arbitrary env vars."""
    artifacts = {}
    for value in input_artifacts:
        path = Path(value)
        if path.is_file():
            artifacts[str(path)] = artifact_record(path)

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _physical_memory_bytes(),
        },
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd().resolve()),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
        },
        "git": git_snapshot(repo_root),
        "dependencies": {
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "numpy": _package_version("numpy"),
            "opencv": _package_version("opencv-python-headless"),
        },
        "accelerator": {
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "devices": cuda_devices,
            "nvidia_smi": _nvidia_smi_snapshot(),
        },
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "environment": {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ},
        "input_artifacts": artifacts,
    }


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None


def _nvidia_smi_snapshot() -> dict:
    query = "index,uuid,name,driver_version,memory.total,pci.bus_id"
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "available": process.returncode == 0,
        "query": query,
        "rows": process.stdout.splitlines() if process.returncode == 0 else [],
        "error": process.stderr.strip() if process.returncode != 0 else None,
    }


def _package_version(distribution: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def run_logged_subprocess(
    command: list[str],
    stdout_path: str | Path,
    stderr_path: str | Path,
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> dict:
    """Run a command into unique log files and return its durable provenance."""
    stdout_file = Path(stdout_path)
    stderr_file = Path(stderr_path)
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()
    returncode = None
    launch_error = None

    with stdout_file.open("xb") as stdout, stderr_file.open("xb") as stderr:
        try:
            process = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                env=env,
                cwd=cwd,
                check=False,
            )
            returncode = process.returncode
        except Exception as error:
            launch_error = f"{type(error).__name__}: {error}"
            stderr.write(traceback.format_exc().encode())
        except BaseException:
            stderr.write(traceback.format_exc().encode())
            raise
        finally:
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())

    return {
        "command": command,
        "cwd": str(Path(cwd or Path.cwd()).resolve()),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "launch_error": launch_error,
        "stdout": artifact_record(stdout_file),
        "stderr": artifact_record(stderr_file),
    }
