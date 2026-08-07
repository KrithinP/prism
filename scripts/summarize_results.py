"""Summarize raw ledger rows with paired-seed uncertainty.

The primary inferential unit here is the training seed. This script reports a
two-sided 95% Student-t interval (df=4 for five seeds), not the earlier ad-hoc
``2 * SE`` rule. Dataset uncertainty is a separate sequence-bootstrap analysis
and must be computed from per-frame exports; it is not fabricated from seed
aggregates.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from pathlib import Path

from scipy.stats import t

from prism.protocol import SEEDS
from prism.provenance import (
    atomic_write_bytes,
    atomic_write_json,
    latest_terminal_rows,
    read_jsonl,
    unfinished_attempts,
)


def interval(values: list[float], confidence: float = 0.95) -> dict:
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else None
    half = t.ppf((1 + confidence) / 2, n - 1) * std / math.sqrt(n) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half if half is not None else None,
        "ci95_high": mean + half if half is not None else None,
    }


def metric(row: dict) -> float | None:
    if row.get("status") == "completed" and isinstance(row.get("val"), dict):
        return row["val"].get("mr_all")
    if row.get("split") == "val-cv" and row.get("mr") is not None:
        return row["mr"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="outputs/ledger.jsonl")
    parser.add_argument("--out-dir", default="outputs/summary")
    parser.add_argument("--baseline-rgb", default="A1")
    parser.add_argument("--baseline-thermal", default="A2")
    args = parser.parse_args()

    ledger = Path(args.ledger)
    rows = read_jsonl(ledger)
    unresolved = unfinished_attempts(rows)
    if unresolved:
        raise SystemExit(
            "summary is blocked by unfinished experiment attempts: " + ", ".join(sorted(unresolved))
        )
    latest = {}
    for row in latest_terminal_rows(rows).values():
        value = metric(row)
        if value is not None and row.get("seed") is not None:
            latest[(row["arm"], int(row["seed"]))] = (row, float(value))

    arms = sorted({arm for arm, _ in latest})
    summary = []
    for arm in arms:
        values_by_seed = {seed: latest[(arm, seed)][1] for seed in SEEDS if (arm, seed) in latest}
        stats = interval(list(values_by_seed.values()))
        item = {
            "arm": arm,
            "label": next(latest[(arm, seed)][0].get("label", arm) for seed in values_by_seed),
            "seeds": values_by_seed,
            **stats,
            "paired_vs_rgb": None,
            "paired_vs_thermal": None,
            "beats_both_95ci": False,
        }
        comparisons = {}
        for name, baseline in (
            ("paired_vs_rgb", args.baseline_rgb),
            ("paired_vs_thermal", args.baseline_thermal),
        ):
            common = [
                seed for seed in SEEDS if seed in values_by_seed and (baseline, seed) in latest
            ]
            if len(common) >= 2:
                deltas = [values_by_seed[seed] - latest[(baseline, seed)][1] for seed in common]
                comparisons[name] = {**interval(deltas), "values": deltas}
                item[name] = comparisons[name]
        if item["paired_vs_rgb"] and item["paired_vs_thermal"]:
            item["beats_both_95ci"] = (
                item["paired_vs_rgb"]["ci95_high"] < 0
                and item["paired_vs_thermal"]["ci95_high"] < 0
            )
        summary.append(item)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        out / "summary.json",
        {
            "metric": "val-cv Reasonable All MR^-2 (fraction; lower is better)",
            "seed_interval": "two-sided 95% Student-t",
            "dataset_interval": ("not computed; requires sequence bootstrap of per-frame outputs"),
            "arms": summary,
        },
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "arm",
            "label",
            "n",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "beats_both_95ci",
        ],
    )
    writer.writeheader()
    writer.writerows({key: item[key] for key in writer.fieldnames} for item in summary)
    atomic_write_bytes(out / "summary.csv", csv_buffer.getvalue().encode())
    print(f"wrote {out / 'summary.json'} and {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
