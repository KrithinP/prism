"""Run the predeclared validation-only profiling and interpretability studies.

The runner resolves checkpoint paths from the append-only experiment ledger and
winner identity from the frozen Block C manifest. Existing outputs are reused
only when they hash the exact expected checkpoint(s); mismatches fail closed.
Each subprocess has separate stdout/stderr and an append-only attempt ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from prism.protocol import PROTOCOL_VERSION, SEEDS
from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    acquire_run_lock,
    append_jsonl,
    artifact_record,
    atomic_write_json,
    environment_snapshot,
    latest_terminal_rows,
    new_attempt_id,
    read_jsonl,
    release_run_lock,
    run_logged_subprocess,
    sha256_file,
    source_fingerprint,
    unfinished_attempts,
    utc_now,
)

PROFILE_ARMS = (
    [f"A{i}" for i in range(1, 9)]
    + [f"B{i}" for i in range(1, 11)]
    + ["B13"]
    + [f"C{i}" for i in range(1, 13)]
    + [f"D{i}" for i in range(1, 4)]
)


def sha256(path: str | Path) -> str:
    return sha256_file(path)


def load_checkpoints(ledger_path: Path) -> dict[tuple[str, int], Path]:
    latest = {}
    rows = read_jsonl(ledger_path)
    unresolved = unfinished_attempts(rows)
    if unresolved:
        raise SystemExit(
            "post-hoc analysis is blocked by unfinished experiment attempts: "
            + ", ".join(sorted(unresolved))
        )
    for row in latest_terminal_rows(rows).values():
        if (
            row.get("status") == "completed"
            and row.get("checkpoint")
            and row.get("seed") is not None
        ):
            latest[(row["arm"], int(row["seed"]))] = Path(row["checkpoint"])
    return latest


def require_checkpoint(checkpoints: dict[tuple[str, int], Path], arm: str, seed: int) -> Path:
    path = checkpoints.get((arm, seed))
    if path is None or not path.is_file():
        raise SystemExit(f"missing completed checkpoint for {arm} seed {seed}: {path}")
    return path


def output_matches(output: Path, expected: dict[str, str]) -> bool:
    if not output.exists():
        return False
    try:
        record = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"existing diagnostic output is unreadable: {output}: {error}") from error
    observed = {}
    if "checkpoint_sha256" in record:
        observed["checkpoint"] = record["checkpoint_sha256"]
    if isinstance(record.get("candidate"), dict):
        observed["candidate"] = record["candidate"].get("sha256")
    if isinstance(record.get("baseline"), dict):
        observed["baseline"] = record["baseline"].get("sha256")
    if observed != expected:
        raise SystemExit(
            f"refusing to overwrite stale diagnostic output {output}: "
            f"expected hashes {expected}, found {observed}"
        )
    artifacts = []
    for key in ("raw_npz", "maps_npz"):
        if isinstance(record.get(key), dict):
            artifacts.append(record[key])
    for condition in record.get("conditions", []):
        if isinstance(condition.get("raw_frames"), dict):
            artifacts.append(condition["raw_frames"])
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact.get("sha256"):
            raise SystemExit(f"raw diagnostic artifact is missing or hash-mismatched: {path}")
    return True


def run_logged(
    label: str,
    command: list[str],
    output: Path,
    expected_hashes: dict[str, str],
    log_dir: Path,
    attempt_ledger: Path,
    session_manifest: dict,
) -> bool:
    run_lock = acquire_run_lock(log_dir / ".locks" / f"{label}.lock")
    attempt_id = new_attempt_id(label)
    attempt_dir = log_dir / attempt_id
    started = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "event": "attempt_started",
        "timestamp": utc_now(),
        "attempt_id": attempt_id,
        "label": label,
        "command": command,
        "output": str(output),
        "expected_hashes": expected_hashes,
        "session_manifest": session_manifest,
        "subprocess_environment_overrides": {
            "PYTHONPATH": "",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
        },
        "status": "started",
    }
    append_jsonl(attempt_ledger, started)
    try:
        reused = output_matches(output, expected_hashes)
    except (Exception, SystemExit) as error:
        append_jsonl(
            attempt_ledger,
            {
                **started,
                "event": "attempt_finished",
                "timestamp": utc_now(),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        print(f"{label}: failed ({attempt_id})")
        release_run_lock(run_lock)
        return False
    if reused:
        append_jsonl(
            attempt_ledger,
            {
                **started,
                "event": "attempt_finished",
                "timestamp": utc_now(),
                "status": "reused",
                "output_artifact": artifact_record(output),
            },
        )
        print(f"{label}: reused ({attempt_id})")
        release_run_lock(run_lock)
        return True
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PYTHONPATH="",
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="0",
    )
    row = {
        **started,
        "event": "attempt_finished",
        "timestamp": None,
        "status": "failed",
    }
    try:
        process = run_logged_subprocess(
            command,
            attempt_dir / "stdout.log",
            attempt_dir / "stderr.log",
            env=env,
        )
        row["process"] = process
        if process["returncode"] != 0:
            row["error"] = f"subprocess exited with {process['returncode']}"
        elif not output_matches(output, expected_hashes):
            row["error"] = "subprocess exited successfully but produced no valid output"
        else:
            row["status"] = "completed"
            row["output_artifact"] = artifact_record(output)
    except (Exception, SystemExit) as error:
        row["error"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc()
    row["timestamp"] = utc_now()
    append_jsonl(attempt_ledger, row)
    print(f"{label}: {row['status']} ({attempt_id})")
    release_run_lock(run_lock)
    return row["status"] == "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["profiles", "mechanisms", "bootstrap", "all"], default="all"
    )
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--ledger", default="outputs/ledger.jsonl")
    parser.add_argument("--operator-selection", default="outputs/selection/block_c_winner.json")
    parser.add_argument("--stats", default="outputs/data/train_cv_stats.json")
    parser.add_argument("--out-dir", default="outputs/posthoc")
    parser.add_argument("--profile-warmup", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=200)
    parser.add_argument("--representation-limit", type=int, default=512)
    parser.add_argument("--attribution-limit", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    if not Path(args.ledger).is_file():
        raise SystemExit(f"missing experiment ledger: {args.ledger}")
    selection = json.loads(Path(args.operator_selection).read_text())
    if selection.get("status") != "selected" or not str(selection.get("winner", "")).startswith(
        "C"
    ):
        raise SystemExit(f"invalid Block C selection: {args.operator_selection}")
    winner = selection["winner"]
    checkpoints = load_checkpoints(Path(args.ledger))
    output_root = Path(args.out_dir)
    logs = output_root / "logs"
    attempts = output_root / "attempts.jsonl"
    session_id = new_attempt_id("posthoc_session")
    repo_root = Path(__file__).resolve().parents[1]
    source_paths = [
        "src/prism",
        "scripts/run_posthoc_studies.py",
        "scripts/profile_checkpoint.py",
        "scripts/evaluate_interventions.py",
        "scripts/analyze_representations.py",
        "scripts/probe_representations.py",
        "scripts/analyze_attributions.py",
        "scripts/bootstrap_comparison.py",
    ]
    provenance_paths = [
        Path(args.ledger),
        Path(args.operator_selection),
        Path(args.stats),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("configs/architecture.yaml"),
        Path(args.root) / "prism_data_manifest.json",
    ]
    session_path = output_root / "sessions" / f"{session_id}.json"
    atomic_write_json(
        session_path,
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "session_id": session_id,
            "started_at_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "command": [sys.executable, *sys.argv],
            "arguments": vars(args),
            "source_fingerprint": source_fingerprint(repo_root, source_paths),
            "environment": environment_snapshot(
                repo_root,
                input_artifacts=[path for path in provenance_paths if path.is_file()],
            ),
        },
    )
    session_artifact = artifact_record(session_path)
    append_jsonl(
        attempts,
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "event": "posthoc_session_started",
            "timestamp": utc_now(),
            "session_id": session_id,
            "session_manifest": session_artifact,
            "status": "started",
        },
    )
    failures = 0

    if args.suite in ("profiles", "all"):
        for arm in PROFILE_ARMS:
            checkpoint = require_checkpoint(checkpoints, arm, 0)
            output = output_root / "profiles" / f"{arm}_s0.json"
            command = [
                sys.executable,
                "scripts/profile_checkpoint.py",
                "--ckpt",
                str(checkpoint),
                "--warmup",
                str(args.profile_warmup),
                "--iterations",
                str(args.profile_iterations),
                "--out",
                str(output),
            ]
            failures += not run_logged(
                f"profile_{arm}_s0",
                command,
                output,
                {"checkpoint": sha256(checkpoint)},
                logs,
                attempts,
                session_artifact,
            )

    mechanism_arms = list(dict.fromkeys(["A7", "A8", "C2", winner, "D1", "D2", "D3"]))
    attribution_arms = list(dict.fromkeys(["C2", winner]))
    if args.suite in ("mechanisms", "all"):
        if not Path(args.stats).is_file():
            raise SystemExit(f"missing train-cv statistics: {args.stats}")
        for arm in mechanism_arms:
            for seed in SEEDS:
                checkpoint = require_checkpoint(checkpoints, arm, seed)
                expected = {"checkpoint": sha256(checkpoint)}

                intervention_output = output_root / "interventions" / f"{arm}_s{seed}.json"
                intervention_command = [
                    sys.executable,
                    "scripts/evaluate_interventions.py",
                    "--ckpt",
                    str(checkpoint),
                    "--root",
                    args.root,
                    "--stats",
                    args.stats,
                    "--out",
                    str(intervention_output),
                ]
                failures += not run_logged(
                    f"interventions_{arm}_s{seed}",
                    intervention_command,
                    intervention_output,
                    expected,
                    logs,
                    attempts,
                    session_artifact,
                )

                representation_output = output_root / "representations" / f"{arm}_s{seed}.json"
                representation_command = [
                    sys.executable,
                    "scripts/analyze_representations.py",
                    "--ckpt",
                    str(checkpoint),
                    "--root",
                    args.root,
                    "--limit",
                    str(args.representation_limit),
                    "--out",
                    str(representation_output),
                ]
                failures += not run_logged(
                    f"representations_{arm}_s{seed}",
                    representation_command,
                    representation_output,
                    expected,
                    logs,
                    attempts,
                    session_artifact,
                )

                if arm in attribution_arms:
                    probe_output = output_root / "probes" / f"{arm}_s{seed}.json"
                    probe_command = [
                        sys.executable,
                        "scripts/probe_representations.py",
                        "--ckpt",
                        str(checkpoint),
                        "--root",
                        args.root,
                        "--out",
                        str(probe_output),
                    ]
                    failures += not run_logged(
                        f"probes_{arm}_s{seed}",
                        probe_command,
                        probe_output,
                        expected,
                        logs,
                        attempts,
                        session_artifact,
                    )

                    attribution_output = output_root / "attributions" / f"{arm}_s{seed}.json"
                    attribution_command = [
                        sys.executable,
                        "scripts/analyze_attributions.py",
                        "--ckpt",
                        str(checkpoint),
                        "--root",
                        args.root,
                        "--stats",
                        args.stats,
                        "--limit",
                        str(args.attribution_limit),
                        "--out",
                        str(attribution_output),
                    ]
                    failures += not run_logged(
                        f"attributions_{arm}_s{seed}",
                        attribution_command,
                        attribution_output,
                        expected,
                        logs,
                        attempts,
                        session_artifact,
                    )

    if args.suite in ("bootstrap", "all"):
        for seed in SEEDS:
            candidate = require_checkpoint(checkpoints, winner, seed)
            for baseline_arm in ("A1", "A2"):
                baseline = require_checkpoint(checkpoints, baseline_arm, seed)
                output = output_root / "bootstrap" / f"{winner}_vs_{baseline_arm}_s{seed}.json"
                command = [
                    sys.executable,
                    "scripts/bootstrap_comparison.py",
                    "--candidate",
                    str(candidate),
                    "--baseline",
                    str(baseline),
                    "--root",
                    args.root,
                    "--repetitions",
                    str(args.bootstrap_repetitions),
                    "--seed",
                    str(20260806 + seed),
                    "--out",
                    str(output),
                ]
                failures += not run_logged(
                    f"bootstrap_{winner}_vs_{baseline_arm}_s{seed}",
                    command,
                    output,
                    {
                        "candidate": sha256(candidate),
                        "baseline": sha256(baseline),
                    },
                    logs,
                    attempts,
                    session_artifact,
                )

    append_jsonl(
        attempts,
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "event": "posthoc_session_finished",
            "timestamp": utc_now(),
            "session_id": session_id,
            "session_manifest": session_artifact,
            "status": "failed" if failures else "completed",
            "failed_subprocesses": failures,
        },
    )
    if failures:
        raise SystemExit(f"{failures} post-hoc study subprocess(es) failed")
    print(f"post-hoc suite complete: {output_root}")


if __name__ == "__main__":
    main()
