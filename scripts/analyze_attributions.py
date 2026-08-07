"""Targeted, quantitative RGB/T attribution for a fusion checkpoint.

The target is one fixed pre-NMS candidate whose decoded box overlaps a scored
ground-truth pedestrian by at least ``--match-iou`` under the clean paired
input. Integrated Gradients uses train-cv channel means as the reference.

Attribution is corroborating evidence only. The output includes raw maps,
inside-box energy, pointing accuracy, fixed-candidate deletion/insertion curves,
and parameter-randomization checks. Modality reliance must still be established
primarily by ``evaluate_interventions.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from prism.data.kaist import kaist_split_cv
from prism.evaluate_model import load_checkpoint_model
from prism.models.decode import decode_outputs
from prism.provenance import atomic_save_npz, atomic_write_json, sha256_file


def candidate_scores(outputs: list[torch.Tensor]) -> torch.Tensor:
    """Flatten head outputs in the same anchor order as ``decode_outputs``."""
    flattened = [output.flatten(2).permute(0, 2, 1) for output in outputs]
    raw = torch.cat(flattened, dim=1)
    return raw[..., 4].sigmoid() * raw[..., 5].sigmoid()


def select_candidate(
    decoded: torch.Tensor,
    scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    match_iou: float,
) -> dict | None:
    """Select the highest-score candidate matching any scored GT."""
    if gt_boxes.numel() == 0:
        return None
    overlaps = box_iou(decoded[:, :4], gt_boxes.to(decoded.device))
    best_iou, best_gt = overlaps.max(dim=1)
    eligible = torch.nonzero(best_iou >= match_iou, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return None
    candidate = eligible[scores[eligible].argmax()]
    gt_index = best_gt[candidate]
    return {
        "candidate_index": int(candidate),
        "gt_index": int(gt_index),
        "iou": float(best_iou[candidate]),
        "score": float(scores[candidate]),
        "box": decoded[candidate, :4].detach().cpu().tolist(),
    }


def _score(model, rgb: torch.Tensor, thermal: torch.Tensor, candidate_index: int) -> torch.Tensor:
    return candidate_scores(model(rgb, thermal))[0, candidate_index]


def candidate_output(
    model,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    candidate_index: int,
    gt_box: torch.Tensor,
) -> dict:
    outputs = model(rgb, thermal)
    score = candidate_scores(outputs)[0, candidate_index]
    box = decode_outputs(outputs)[0, candidate_index, :4]
    iou = box_iou(box[None], gt_box.to(box.device)[None])[0, 0]
    center = (box[:2] + box[2:]) / 2
    gt_center = (gt_box[:2].to(box.device) + gt_box[2:].to(box.device)) / 2
    return {
        "score": float(score),
        "box_iou": float(iou),
        "center_error_px": float((center - gt_center).norm()),
    }


def integrated_gradients(
    model,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    rgb_baseline: torch.Tensor,
    thermal_baseline: torch.Tensor,
    candidate_index: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Trapezoidal Integrated Gradients for one fixed candidate score."""
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two path intervals")
    rgb_delta = rgb - rgb_baseline
    thermal_delta = thermal - thermal_baseline
    rgb_gradient = torch.zeros_like(rgb)
    thermal_gradient = torch.zeros_like(thermal)
    baseline_score = None
    input_score = None

    for path_index in range(steps + 1):
        alpha = path_index / steps
        rgb_path = (rgb_baseline + alpha * rgb_delta).detach().requires_grad_(True)
        thermal_path = (thermal_baseline + alpha * thermal_delta).detach().requires_grad_(True)
        score = _score(model, rgb_path, thermal_path, candidate_index)
        rgb_grad, thermal_grad = torch.autograd.grad(
            score, (rgb_path, thermal_path), retain_graph=False
        )
        weight = 0.5 if path_index in (0, steps) else 1.0
        rgb_gradient += weight * rgb_grad
        thermal_gradient += weight * thermal_grad
        if path_index == 0:
            baseline_score = float(score.detach())
        if path_index == steps:
            input_score = float(score.detach())

    rgb_attr = rgb_delta * rgb_gradient / steps
    thermal_attr = thermal_delta * thermal_gradient / steps
    return rgb_attr, thermal_attr, input_score, baseline_score


def attribution_map(attribution: torch.Tensor) -> torch.Tensor:
    """Absolute channel-summed saliency for one BCHW attribution tensor."""
    return attribution[0].abs().sum(dim=0)


def fusion_point_modules(model, model_type: str):
    """Return RGB/T modules producing the deepest feature that is actually fused."""
    if model_type == "multi-scale":
        level = max(model.fuse_subset)
    elif model_type == "one-shot":
        level = model.meet_level
    else:
        raise ValueError(f"branch Grad-CAM requires a two-stream feature graph, got {model_type!r}")
    return (
        level,
        getattr(model.backbone_rgb, f"dark{level}"),
        getattr(model.backbone_thermal, f"dark{level}"),
    )


def grad_cam(
    model,
    model_type: str,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    candidate_index: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Grad-CAM for both private branches at their deepest fused level."""
    level, rgb_module, thermal_module = fusion_point_modules(model, model_type)
    activations = {}

    def capture(name):
        def hook(_module, _inputs, output):
            activations[name] = output

        return hook

    handles = [
        rgb_module.register_forward_hook(capture("rgb")),
        thermal_module.register_forward_hook(capture("thermal")),
    ]
    try:
        score = _score(model, rgb, thermal, candidate_index)
        rgb_gradient, thermal_gradient = torch.autograd.grad(
            score,
            (activations["rgb"], activations["thermal"]),
            retain_graph=False,
            allow_unused=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    maps = []
    for activation, gradient in (
        (activations["rgb"], rgb_gradient),
        (activations["thermal"], thermal_gradient),
    ):
        if gradient is None:
            cam = torch.zeros(
                activation.shape[0], 1, *activation.shape[2:], device=activation.device
            )
        else:
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        maps.append(
            F.interpolate(cam, size=rgb.shape[-2:], mode="bilinear", align_corners=False)[
                0, 0
            ].detach()
        )
    return maps[0], maps[1], level


def localization_metrics(saliency: torch.Tensor, box: torch.Tensor) -> dict:
    """Attribution mass inside a GT box and whether its maximum lies inside."""
    height, width = saliency.shape
    x1 = max(0, min(width, int(torch.floor(box[0]).item())))
    y1 = max(0, min(height, int(torch.floor(box[1]).item())))
    x2 = max(0, min(width, int(torch.ceil(box[2]).item())))
    y2 = max(0, min(height, int(torch.ceil(box[3]).item())))
    total = float(saliency.sum())
    inside = float(saliency[y1:y2, x1:x2].sum()) if x1 < x2 and y1 < y2 else 0.0
    peak = int(saliency.argmax())
    peak_y, peak_x = divmod(peak, width)
    return {
        "total_abs_attribution": total,
        "inside_gt_fraction": inside / max(total, 1e-20),
        "pointing_hit": bool(x1 <= peak_x < x2 and y1 <= peak_y < y2),
        "peak_xy": [peak_x, peak_y],
    }


def _replace_ranked(
    source: torch.Tensor,
    reference: torch.Tensor,
    ranking: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    result = source.clone()
    count = round(ranking.numel() * fraction)
    if count:
        flat_result = result.flatten(2)
        flat_reference = reference.flatten(2)
        indices = ranking[:count]
        flat_result[:, :, indices] = flat_reference[:, :, indices]
    return result


@torch.no_grad()
def faithfulness_curves(
    model,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    rgb_baseline: torch.Tensor,
    thermal_baseline: torch.Tensor,
    rgb_map: torch.Tensor,
    thermal_map: torch.Tensor,
    candidate_index: int,
    gt_box: torch.Tensor,
    fractions: list[float],
) -> dict:
    """Fixed-candidate deletion and insertion curves for each modality."""
    result = {}
    for modality, saliency in (("rgb", rgb_map), ("thermal", thermal_map)):
        ranking = torch.argsort(saliency.flatten(), descending=True)
        deletion, insertion = [], []
        for fraction in fractions:
            if modality == "rgb":
                deleted_rgb = _replace_ranked(rgb, rgb_baseline, ranking, fraction)
                inserted_rgb = _replace_ranked(rgb_baseline, rgb, ranking, fraction)
                deletion_output = candidate_output(
                    model, deleted_rgb, thermal, candidate_index, gt_box
                )
                insertion_output = candidate_output(
                    model, inserted_rgb, thermal, candidate_index, gt_box
                )
            else:
                deleted_thermal = _replace_ranked(thermal, thermal_baseline, ranking, fraction)
                inserted_thermal = _replace_ranked(thermal_baseline, thermal, ranking, fraction)
                deletion_output = candidate_output(
                    model, rgb, deleted_thermal, candidate_index, gt_box
                )
                insertion_output = candidate_output(
                    model, rgb, inserted_thermal, candidate_index, gt_box
                )
            deletion.append(deletion_output)
            insertion.append(insertion_output)
        deletion_scores = [row["score"] for row in deletion]
        insertion_scores = [row["score"] for row in insertion]
        result[modality] = {
            "fractions": fractions,
            "deletion": deletion,
            "insertion": insertion,
            "deletion_score_auc": float(np.trapezoid(deletion_scores, fractions)),
            "insertion_score_auc": float(np.trapezoid(insertion_scores, fractions)),
        }
    return result


def randomize_parameters(module: torch.nn.Module, seed: int) -> None:
    """Reset each leaf parameterized module under a deterministic RNG state."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for child in module.modules():
        if len(list(child.children())) == 0 and hasattr(child, "reset_parameters"):
            child.reset_parameters()


def centered_correlation(first: torch.Tensor, second: torch.Tensor) -> float | None:
    first = first.double().flatten()
    second = second.double().flatten()
    first -= first.mean()
    second -= second.mean()
    denominator = first.norm() * second.norm()
    return float(first.dot(second) / denominator) if denominator > 0 else None


def _mean_baseline(image: torch.Tensor, mean: list[float]) -> torch.Tensor:
    values = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
    return values.expand_as(image).clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--stats", default="outputs/data/train_cv_stats.json")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--save-maps", type=int, default=16)
    parser.add_argument("--sanity-samples", type=int, default=8)
    parser.add_argument("--sanity-steps", type=int, default=8)
    parser.add_argument(
        "--curve-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--maps-out", default=None)
    args = parser.parse_args()
    if sorted(args.curve_fractions) != args.curve_fractions:
        raise SystemExit("--curve-fractions must be sorted")
    if args.curve_fractions[0] != 0 or args.curve_fractions[-1] != 1:
        raise SystemExit("--curve-fractions must include endpoints 0 and 1")

    stats_path = Path(args.stats)
    stats_record = json.loads(stats_path.read_text())
    if stats_record.get("split") != "train-cv":
        raise SystemExit("attribution baselines must use train-cv statistics")
    statistics = stats_record["statistics"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, modality, model_type = load_checkpoint_model(args.ckpt, device=device)
    if modality != "fusion":
        raise SystemExit("RGB/T attribution requires a fusion checkpoint")
    model.eval()
    dataset = kaist_split_cv(args.root, "val-cv")

    rows = []
    saved_maps = []
    sanity_cases = []
    inspected = 0
    for sample_index in range(len(dataset)):
        if len(rows) >= args.limit:
            break
        sample = dataset[sample_index]
        scored = sample["classes"] == 1
        gt_boxes = sample["boxes"][scored]
        if gt_boxes.numel() == 0:
            continue
        inspected += 1
        rgb = (sample["rgb"].float() / 255.0).unsqueeze(0).to(device)
        thermal = (sample["thermal"].float() / 255.0).unsqueeze(0).to(device)
        rgb_baseline = _mean_baseline(rgb, statistics["rgb"]["mean"])
        thermal_baseline = _mean_baseline(thermal, statistics["thermal"]["mean"])
        with torch.no_grad():
            outputs = model(rgb, thermal)
            decoded = decode_outputs(outputs)[0]
            scores = candidate_scores(outputs)[0]
            selected = select_candidate(decoded, scores, gt_boxes, args.match_iou)
        if selected is None:
            continue
        gt_box = gt_boxes[selected["gt_index"]].to(device)
        rgb_attr, thermal_attr, input_score, baseline_score = integrated_gradients(
            model,
            rgb,
            thermal,
            rgb_baseline,
            thermal_baseline,
            selected["candidate_index"],
            args.steps,
        )
        rgb_map = attribution_map(rgb_attr).detach().cpu()
        thermal_map = attribution_map(thermal_attr).detach().cpu()
        rgb_cam, thermal_cam, cam_level = grad_cam(
            model, model_type, rgb, thermal, selected["candidate_index"]
        )
        rgb_cam = rgb_cam.cpu()
        thermal_cam = thermal_cam.cpu()
        rgb_metrics = localization_metrics(rgb_map, gt_box.cpu())
        thermal_metrics = localization_metrics(thermal_map, gt_box.cpu())
        total_mass = rgb_metrics["total_abs_attribution"] + thermal_metrics["total_abs_attribution"]
        curves = faithfulness_curves(
            model,
            rgb,
            thermal,
            rgb_baseline,
            thermal_baseline,
            rgb_map.to(device),
            thermal_map.to(device),
            selected["candidate_index"],
            gt_box,
            args.curve_fractions,
        )
        row = {
            "frame_id": sample["id"],
            "candidate": selected,
            "gt_box": gt_box.detach().cpu().tolist(),
            "target": "fixed pre-NMS objectness sigmoid * class sigmoid",
            "input_score": input_score,
            "joint_train_mean_baseline_score": baseline_score,
            "ig_completeness": {
                "score_difference": input_score - baseline_score,
                "attribution_sum": float(rgb_attr.sum() + thermal_attr.sum()),
                "absolute_error": abs(
                    (input_score - baseline_score) - float(rgb_attr.sum() + thermal_attr.sum())
                ),
            },
            "rgb": {
                **rgb_metrics,
                "modality_abs_mass_share": (
                    rgb_metrics["total_abs_attribution"] / max(total_mass, 1e-20)
                ),
            },
            "thermal": {
                **thermal_metrics,
                "modality_abs_mass_share": (
                    thermal_metrics["total_abs_attribution"] / max(total_mass, 1e-20)
                ),
            },
            "grad_cam": {
                "fusion_level": cam_level,
                "rgb": localization_metrics(rgb_cam, gt_box.cpu()),
                "thermal": localization_metrics(thermal_cam, gt_box.cpu()),
                "agreement_with_integrated_gradients": {
                    "rgb_map_correlation": centered_correlation(rgb_map, rgb_cam),
                    "thermal_map_correlation": centered_correlation(thermal_map, thermal_cam),
                },
            },
            "faithfulness": curves,
        }
        rows.append(row)
        if len(saved_maps) < args.save_maps:
            saved_maps.append(
                (
                    sample["id"],
                    rgb_map.numpy(),
                    thermal_map.numpy(),
                    rgb_cam.numpy(),
                    thermal_cam.numpy(),
                )
            )
        if len(sanity_cases) < args.sanity_samples:
            sanity_cases.append(
                {
                    "frame_id": sample["id"],
                    "rgb": rgb.detach(),
                    "thermal": thermal.detach(),
                    "rgb_baseline": rgb_baseline.detach(),
                    "thermal_baseline": thermal_baseline.detach(),
                    "candidate_index": selected["candidate_index"],
                    "rgb_map": rgb_map,
                    "thermal_map": thermal_map,
                }
            )
        print(f"{len(rows)}/{args.limit}: {sample['id']}")

    if not rows:
        raise SystemExit(f"no GT-matched candidates found in {inspected} positive frames")

    randomization = []
    for scope in ("head", "full_model"):
        randomized, randomized_modality, _ = load_checkpoint_model(args.ckpt, device=device)
        if randomized_modality != "fusion":
            raise RuntimeError(
                f"checkpoint changed modality while reloading: {randomized_modality!r}"
            )
        randomized.eval()
        randomize_parameters(
            randomized.head if scope == "head" else randomized,
            seed=20260806,
        )
        for case in sanity_cases:
            rgb_attr, thermal_attr, _, _ = integrated_gradients(
                randomized,
                case["rgb"],
                case["thermal"],
                case["rgb_baseline"],
                case["thermal_baseline"],
                case["candidate_index"],
                args.sanity_steps,
            )
            randomization.append(
                {
                    "scope": scope,
                    "frame_id": case["frame_id"],
                    "candidate_index": case["candidate_index"],
                    "rgb_map_correlation": centered_correlation(
                        case["rgb_map"], attribution_map(rgb_attr).cpu()
                    ),
                    "thermal_map_correlation": centered_correlation(
                        case["thermal_map"], attribution_map(thermal_attr).cpu()
                    ),
                }
            )

    checkpoint = Path(args.ckpt)
    output = Path(args.out)
    maps_output = Path(args.maps_out) if args.maps_out else output.with_suffix(".npz")
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_type": model_type,
        "split": "val-cv",
        "selection": (
            f"first {args.limit} frames, in frozen order, with a pre-NMS "
            f"candidate IoU >= {args.match_iou} to a scored GT"
        ),
        "integrated_gradients": {
            "reference": "train-cv per-channel mean",
            "path_intervals": args.steps,
            "target": "fixed candidate objectness sigmoid * class sigmoid",
        },
        "grad_cam": {
            "target": "same fixed candidate score",
            "layer": "deepest feature level that participates in fusion",
        },
        "scope_warning": (
            "attribution is descriptive/faithfulness evidence; causal modality "
            "reliance comes from fixed-checkpoint input interventions"
        ),
        "raw_per_candidate": rows,
        "parameter_randomization": {
            "method": "same fixed candidate after deterministic learned-head/full reset",
            "sanity_path_intervals": args.sanity_steps,
            "raw": randomization,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    maps_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        maps_output,
        frame_ids=np.asarray([row[0] for row in saved_maps]),
        rgb_saliency=np.stack([row[1] for row in saved_maps]),
        thermal_saliency=np.stack([row[2] for row in saved_maps]),
        rgb_grad_cam=np.stack([row[3] for row in saved_maps]),
        thermal_grad_cam=np.stack([row[4] for row in saved_maps]),
    )
    result["maps_npz"] = {
        "path": str(maps_output),
        "sha256": sha256_file(maps_output),
    }
    atomic_write_json(output, result)
    print(f"wrote {output} and {maps_output}")


if __name__ == "__main__":
    main()
