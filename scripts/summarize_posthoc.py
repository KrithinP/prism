"""Summarize post-hoc artifacts without discarding seed-level values."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from prism.provenance import atomic_write_json
from scripts.summarize_results import interval


def arm_seed(path: Path) -> tuple[str, int]:
    arm, seed = path.stem.rsplit("_s", 1)
    return arm, int(seed)


def add_value(groups: dict, key: tuple, seed: int, value) -> None:
    if value is not None and not isinstance(value, bool):
        groups[key][seed] = float(value)


def summarize_groups(groups: dict) -> list[dict]:
    rows = []
    for key in sorted(groups):
        values = groups[key]
        raw = {str(seed): values[seed] for seed in sorted(values)}
        rows.append(
            {
                "key": list(key),
                "raw_seed_values": raw,
                **interval(list(raw.values())),
            }
        )
    return rows


def flatten_numeric(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_numeric(child, (*prefix, key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def mean_or_none(values: list) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.mean(numeric) if numeric else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/posthoc")
    parser.add_argument("--out", default="outputs/posthoc/summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    result = {
        "schema_version": 1,
        "seed_interval": "two-sided 95% Student-t; raw seed values retained",
    }

    groups = defaultdict(dict)
    for path in sorted((root / "interventions").glob("*.json")):
        arm, seed = arm_seed(path)
        record = json.loads(path.read_text())
        for condition in record["conditions"]:
            add_value(groups, (arm, condition["condition"], "mr_all"), seed, condition["mr_all"])
            add_value(
                groups,
                (arm, condition["condition"], "delta_mr_pp_vs_paired"),
                seed,
                condition["delta_mr_pp_vs_paired"],
            )
    result["interventions"] = summarize_groups(groups)

    groups = defaultdict(dict)
    for path in sorted((root / "representations").glob("*.json")):
        arm, seed = arm_seed(path)
        record = json.loads(path.read_text())
        for metric_path, value in flatten_numeric(record["derived_summary"]):
            add_value(groups, (arm, *metric_path), seed, value)
    result["representations"] = summarize_groups(groups)

    groups = defaultdict(dict)
    for path in sorted((root / "probes").glob("*.json")):
        arm, seed = arm_seed(path)
        record = json.loads(path.read_text())
        for representation, metrics in record["derived_summary"].items():
            for metric, value in metrics.items():
                add_value(groups, (arm, representation, metric), seed, value)
    result["probes"] = summarize_groups(groups)

    groups = defaultdict(dict)
    for path in sorted((root / "attributions").glob("*.json")):
        arm, seed = arm_seed(path)
        record = json.loads(path.read_text())
        rows = record["raw_per_candidate"]
        derived = {
            "rgb_inside_gt_fraction": mean_or_none(
                [row["rgb"]["inside_gt_fraction"] for row in rows]
            ),
            "thermal_inside_gt_fraction": mean_or_none(
                [row["thermal"]["inside_gt_fraction"] for row in rows]
            ),
            "rgb_pointing_rate": mean_or_none([row["rgb"]["pointing_hit"] for row in rows]),
            "thermal_pointing_rate": mean_or_none([row["thermal"]["pointing_hit"] for row in rows]),
            "rgb_modality_mass_share": mean_or_none(
                [row["rgb"]["modality_abs_mass_share"] for row in rows]
            ),
            "thermal_modality_mass_share": mean_or_none(
                [row["thermal"]["modality_abs_mass_share"] for row in rows]
            ),
            "ig_completeness_absolute_error": mean_or_none(
                [row["ig_completeness"]["absolute_error"] for row in rows]
            ),
            "rgb_deletion_auc": mean_or_none(
                [row["faithfulness"]["rgb"]["deletion_score_auc"] for row in rows]
            ),
            "thermal_deletion_auc": mean_or_none(
                [row["faithfulness"]["thermal"]["deletion_score_auc"] for row in rows]
            ),
            "rgb_insertion_auc": mean_or_none(
                [row["faithfulness"]["rgb"]["insertion_score_auc"] for row in rows]
            ),
            "thermal_insertion_auc": mean_or_none(
                [row["faithfulness"]["thermal"]["insertion_score_auc"] for row in rows]
            ),
            "rgb_ig_gradcam_correlation": mean_or_none(
                [
                    row["grad_cam"]["agreement_with_integrated_gradients"]["rgb_map_correlation"]
                    for row in rows
                ]
            ),
            "thermal_ig_gradcam_correlation": mean_or_none(
                [
                    row["grad_cam"]["agreement_with_integrated_gradients"][
                        "thermal_map_correlation"
                    ]
                    for row in rows
                ]
            ),
        }
        for metric, value in derived.items():
            add_value(groups, (arm, metric), seed, value)
        for scope in ("head", "full_model"):
            sanity = [
                row for row in record["parameter_randomization"]["raw"] if row["scope"] == scope
            ]
            for modality in ("rgb", "thermal"):
                value = mean_or_none([row[f"{modality}_map_correlation"] for row in sanity])
                add_value(groups, (arm, f"{scope}_{modality}_map_correlation"), seed, value)
    result["attributions"] = summarize_groups(groups)

    result["profiles"] = [
        json.loads(path.read_text()) for path in sorted((root / "profiles").glob("*.json"))
    ]
    result["sequence_bootstrap"] = [
        json.loads(path.read_text()) for path in sorted((root / "bootstrap").glob("*.json"))
    ]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
