"""Select and freeze the validation winner for a predeclared arm family."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from prism.protocol import PROTOCOL_VERSION, SEEDS
from prism.provenance import (
    artifact_record,
    atomic_write_bytes,
    atomic_write_json,
    latest_terminal_rows,
    read_jsonl,
    unfinished_attempts,
    verify_artifact_record,
)

DEFAULT_GRAPH_ARMS = [f"B{i}" for i in range(2, 11)]


def validate_completed_evidence(row: dict) -> None:
    identity = f"{row.get('arm')} seed {row.get('seed')}"
    if row.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"{identity} uses protocol {row.get('protocol_version')!r}, "
            f"expected {PROTOCOL_VERSION!r}"
        )
    for key in (
        "attempt_manifest",
        "result_manifest",
        "checkpoint_artifact",
        "validation_artifact",
    ):
        verify_artifact_record(row.get(key), label=f"{identity} {key}")
    for phase_name, phase in row.get("phases", {}).items():
        for stream in ("stdout", "stderr"):
            if stream in phase:
                verify_artifact_record(
                    phase[stream],
                    label=f"{identity} {phase_name} {stream}",
                )
    raw_frames = row.get("val", {}).get("raw_frames")
    verify_artifact_record(raw_frames, label=f"{identity} raw validation frames")

    result = json.loads(Path(row["result_manifest"]["path"]).read_text())
    if result.get("attempt_id") != row.get("attempt_id"):
        raise ValueError(f"{identity} result manifest attempt_id differs from ledger")
    if result.get("args_hash") != row.get("args_hash"):
        raise ValueError(f"{identity} result manifest args_hash differs from ledger")
    if result.get("val") != row.get("val"):
        raise ValueError(f"{identity} result manifest metrics differ from ledger")

    metrics = json.loads(Path(row["validation_artifact"]["path"]).read_text())
    if metrics.get("splits", {}).get("val-cv") != row.get("val"):
        raise ValueError(f"{identity} validation artifact metrics differ from ledger")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="outputs/ledger.jsonl")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_GRAPH_ARMS)
    parser.add_argument("--out", default="outputs/selection/block_b_winner.json")
    args = parser.parse_args()

    ledger = Path(args.ledger)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite a frozen selection: {out}")
    rows = read_jsonl(ledger)
    unresolved = {
        attempt_id: row
        for attempt_id, row in unfinished_attempts(rows).items()
        if row.get("arm") in args.arms
    }
    if unresolved:
        raise SystemExit(
            "selection is blocked by unfinished attempts: " + ", ".join(sorted(unresolved))
        )
    # A later failed retry must not be hidden by an older successful attempt.
    terminal = latest_terminal_rows(rows)
    eligible = {
        key: row
        for key, row in terminal.items()
        if key[0] in args.arms and row.get("status") == "completed"
    }

    summaries = []
    for arm in args.arms:
        arm_rows = [eligible.get((arm, seed)) for seed in SEEDS]
        missing = [seed for seed, row in zip(SEEDS, arm_rows, strict=True) if row is None]
        if missing:
            raise SystemExit(f"{arm} is missing completed validation seeds {missing}")
        try:
            for row in arm_rows:
                validate_completed_evidence(row)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        hashes = {row["args_hash"] for row in arm_rows}
        if len(hashes) != 1:
            raise SystemExit(f"{arm} has inconsistent protocol hashes across seeds")
        values = [row["val"]["mr_all"] for row in arm_rows]
        summaries.append(
            {
                "arm": arm,
                "label": arm_rows[0]["label"],
                "args": arm_rows[0]["args"],
                "args_hash": arm_rows[0]["args_hash"],
                "seeds": list(SEEDS),
                "values": values,
                "mean_mr_all": statistics.mean(values),
                "std_mr_all": statistics.stdev(values),
            }
        )
    summaries.sort(key=lambda row: (row["mean_mr_all"], row["arm"]))
    winner = summaries[0]
    ledger_snapshot = out.with_name(f"{out.stem}.ledger.jsonl")
    if ledger_snapshot.exists():
        raise SystemExit(f"refusing to overwrite a frozen ledger snapshot: {ledger_snapshot}")
    atomic_write_bytes(ledger_snapshot, ledger.read_bytes())
    report = {
        "schema_version": 1,
        "status": "selected",
        "selected_at": datetime.now(UTC).isoformat(),
        "criterion": "minimum five-seed mean val-cv Reasonable All MR^-2",
        "eligible_arms": args.arms,
        "winner": winner["arm"],
        "args": winner["args"],
        "args_hash": winner["args_hash"],
        "source_ledger": str(ledger.resolve()),
        "ledger": artifact_record(ledger_snapshot),
        "ranking": summaries,
    }
    atomic_write_json(out, report)
    print(f"selected {winner['arm']}: mean MR={winner['mean_mr_all']:.6f}")
    print(f"frozen selection: {out}")


if __name__ == "__main__":
    main()
