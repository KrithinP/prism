"""Fail-closed integrity audit for the append-only experiment ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    artifact_record,
    atomic_write_json,
    latest_terminal_rows,
    read_jsonl,
    unfinished_attempts,
    utc_now,
    verify_artifact_record,
)
from scripts.select_winner import validate_completed_evidence


def _artifact_records(value, prefix="row"):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield prefix, value
        for key, child in value.items():
            yield from _artifact_records(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _artifact_records(child, f"{prefix}[{index}]")


def audit_ledger(path: str | Path) -> dict:
    ledger = Path(path)
    rows = read_jsonl(ledger)
    errors = []
    try:
        unresolved = unfinished_attempts(rows)
    except ValueError as error:
        unresolved = {}
        errors.append(str(error))
    if unresolved:
        errors.append("unfinished attempts: " + ", ".join(sorted(unresolved)))

    artifact_count = 0
    verified = set()
    for index, row in enumerate(rows, start=1):
        for label, record in _artifact_records(row, f"row[{index}]"):
            identity = (record.get("path"), record.get("sha256"))
            if identity in verified:
                continue
            try:
                verify_artifact_record(record, label=label)
                verified.add(identity)
                artifact_count += 1
            except ValueError as error:
                errors.append(str(error))

    terminal = latest_terminal_rows(rows)
    for (arm, seed), row in terminal.items():
        if (
            row.get("status") == "completed"
            and isinstance(row.get("val"), dict)
            and row.get("checkpoint_artifact")
        ):
            try:
                validate_completed_evidence(row)
            except ValueError as error:
                errors.append(f"{arm} seed {seed}: {error}")

    statuses = Counter(
        str(row.get("status"))
        for row in rows
        if row.get("event") in {"attempt_finished", "attempt_abandoned"}
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "audited_at_utc": utc_now(),
        "passed": not errors,
        "ledger": artifact_record(ledger),
        "rows": len(rows),
        "latest_arm_seed_records": len(terminal),
        "verified_artifact_references": artifact_count,
        "terminal_status_counts": dict(sorted(statuses.items())),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="outputs/ledger.jsonl")
    parser.add_argument("--out", default="outputs/ledger_audit.json")
    args = parser.parse_args()

    report = audit_ledger(args.ledger)
    atomic_write_json(args.out, report)
    print(
        f"ledger audit: {'PASS' if report['passed'] else 'FAIL'}; "
        f"{report['rows']} rows, "
        f"{report['verified_artifact_references']} artifact references"
    )
    if not report["passed"]:
        for error in report["errors"]:
            print(f"  - {error}")
        raise SystemExit("ledger audit failed")


if __name__ == "__main__":
    main()
