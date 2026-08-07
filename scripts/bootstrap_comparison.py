"""Paired sequence bootstrap for two validation checkpoints.

Frames are not independent in KAIST. This script resamples complete
``setXX/VYYY`` sequences with replacement and applies the identical draw to both
models. It preserves every bootstrap replicate in JSON before percentile
summaries. Seed-to-seed training uncertainty remains a separate analysis.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prism.data.kaist import kaist_split_cv
from prism.eval.canonical import evaluate_reference
from prism.evaluate_model import collect_frames, load_checkpoint_model
from prism.provenance import atomic_write_json, sha256_file


def group_frames(frames: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame["id"].rsplit("/", 1)[0]].append(frame)
    return dict(grouped)


def collect_checkpoint(path: str, base, device: str, batch_size: int) -> list[dict]:
    model, modality, _ = load_checkpoint_model(path, device=device)
    return collect_frames(
        model, base, modality, device, batch_size=batch_size, input_policy="paired", day_ids=None
    )


def bootstrap(
    candidate: list[dict], baseline: list[dict], repetitions: int, seed: int
) -> list[dict]:
    candidate_groups = group_frames(candidate)
    baseline_groups = group_frames(baseline)
    sequences = sorted(candidate_groups)
    if sequences != sorted(baseline_groups):
        raise ValueError("candidate and baseline sequence coverage differs")
    for sequence in sequences:
        candidate_ids = [frame["id"] for frame in candidate_groups[sequence]]
        baseline_ids = [frame["id"] for frame in baseline_groups[sequence]]
        if candidate_ids != baseline_ids:
            raise ValueError(f"frame order differs in sequence {sequence}")

    generator = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        sampled = generator.choice(sequences, size=len(sequences), replace=True).tolist()
        candidate_sample = [frame for sequence in sampled for frame in candidate_groups[sequence]]
        baseline_sample = [frame for sequence in sampled for frame in baseline_groups[sequence]]
        candidate_mr = evaluate_reference(candidate_sample)["all"]
        baseline_mr = evaluate_reference(baseline_sample)["all"]
        rows.append(
            {
                "replicate": repetition,
                "sampled_sequences": sampled,
                "candidate_mr": candidate_mr,
                "baseline_mr": baseline_mr,
                "delta_mr": candidate_mr - baseline_mr,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise SystemExit("use at least 100 bootstrap repetitions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = kaist_split_cv(args.root, "val-cv")
    candidate = collect_checkpoint(args.candidate, base, device, args.batch_size)
    baseline = collect_checkpoint(args.baseline, base, device, args.batch_size)
    rows = bootstrap(candidate, baseline, args.repetitions, args.seed)
    deltas = np.asarray([row["delta_mr"] for row in rows])
    observed_candidate = evaluate_reference(candidate)["all"]
    observed_baseline = evaluate_reference(baseline)["all"]

    def checkpoint_record(path: str):
        file = Path(path)
        return {"path": str(file), "sha256": sha256_file(file)}

    result = {
        "schema_version": 1,
        "split": "val-cv",
        "unit": "setXX/VYYY sequence",
        "sampling": "paired nonparametric sequence bootstrap with replacement",
        "seed": args.seed,
        "candidate": checkpoint_record(args.candidate),
        "baseline": checkpoint_record(args.baseline),
        "observed": {
            "candidate_mr": observed_candidate,
            "baseline_mr": observed_baseline,
            "delta_mr": observed_candidate - observed_baseline,
        },
        "raw_replicates": rows,
        "derived_summary": {
            "repetitions": len(rows),
            "delta_mean": float(deltas.mean()),
            "delta_std_population": float(deltas.std()),
            "delta_percentile_ci95": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
            "probability_delta_below_zero": float((deltas < 0).mean()),
        },
        "scope_warning": (
            "sequence bootstrap estimates validation-sequence sampling uncertainty "
            "for one checkpoint pair; seed-to-seed training uncertainty is separate"
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
