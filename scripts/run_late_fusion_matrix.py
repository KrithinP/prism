"""Evaluate B11/B12 on paired A1/A2 seeds using validation only."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from prism.protocol import SEEDS
from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    append_jsonl,
    artifact_record,
    new_attempt_id,
    run_logged_subprocess,
    utc_now,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--experiments", default="outputs/experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--merges", nargs="+", choices=["union", "proben"], default=["union", "proben"]
    )
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS:
        raise SystemExit(f"late-fusion core controls require paired seeds {SEEDS}")

    output = Path(args.experiments)
    log_dir = Path("outputs/late_fusion")
    log_dir.mkdir(parents=True, exist_ok=True)
    attempt_ledger = log_dir / "attempts.jsonl"
    failures = 0
    env = dict(os.environ, PYTHONPATH="", PYTHONUNBUFFERED="1")
    for merge in args.merges:
        arm = "B11" if merge == "union" else "B12"
        for seed in args.seeds:
            env["PYTHONHASHSEED"] = str(seed)
            rgb = output / f"A1_s{seed}" / f"rgb_seed{seed}.pt"
            thermal = output / f"A2_s{seed}" / f"thermal_seed{seed}.pt"
            if not rgb.exists() or not thermal.exists():
                raise SystemExit(
                    f"missing paired seed {seed}: rgb={rgb.exists()}, thermal={thermal.exists()}"
                )
            attempt_id = new_attempt_id(f"{arm}_s{seed}")
            attempt_dir = log_dir / attempt_id
            attempt_dir.mkdir(parents=True, exist_ok=False)
            metrics_path = attempt_dir / "metrics.json"
            command = [
                sys.executable,
                "scripts/late_fusion_runner.py",
                "--rgb-ckpt",
                str(rgb),
                "--thermal-ckpt",
                str(thermal),
                "--merge",
                merge,
                "--split",
                "val-cv",
                "--root",
                args.root,
                "--arm-label",
                arm,
                "--seed",
                str(seed),
                "--json-out",
                str(metrics_path),
                "--raw-output-dir",
                str(attempt_dir / "raw_frames"),
            ]
            base = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "arm": arm,
                "seed": seed,
                "merge": merge,
                "command": command,
                "subprocess_environment_overrides": {
                    "PYTHONPATH": "",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONHASHSEED": str(seed),
                },
                "inputs": {
                    "rgb_checkpoint": artifact_record(rgb),
                    "thermal_checkpoint": artifact_record(thermal),
                },
            }
            append_jsonl(
                attempt_ledger,
                {
                    **base,
                    "event": "attempt_started",
                    "timestamp": utc_now(),
                    "status": "started",
                },
            )
            terminal = {
                **base,
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
                terminal["process"] = process
                if process["returncode"] == 0 and metrics_path.is_file():
                    terminal["status"] = "completed"
                    terminal["metrics_artifact"] = artifact_record(metrics_path)
                elif process["returncode"] == 0:
                    terminal["error"] = "subprocess produced no metrics artifact"
                else:
                    terminal["error"] = f"subprocess exited with {process['returncode']}"
            except Exception as error:
                terminal["error"] = f"{type(error).__name__}: {error}"
                terminal["traceback"] = traceback.format_exc()
            terminal["timestamp"] = utc_now()
            append_jsonl(attempt_ledger, terminal)
            print(f"{arm} seed {seed}: {terminal['status']} ({attempt_id})")
            failures += int(terminal["status"] != "completed")
    if failures:
        raise SystemExit(f"{failures} late-fusion validation run(s) failed")


if __name__ == "__main__":
    main()
