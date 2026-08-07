"""Raw-first representation and gate diagnostics for a fusion checkpoint.

The JSON contains per-frame measurements before aggregate summaries. The NPZ
contains every pooled branch vector used for CKA/effective-rank calculations.
Gate or attention values are diagnostics, not causal explanations; compare them
against ``evaluate_interventions.py`` before making a modality-use claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.data.kaist import kaist_split_cv
from prism.evaluate_model import load_checkpoint_model
from prism.provenance import atomic_save_npz, atomic_write_json, sha256_file


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float | None:
    """Centered linear CKA across samples."""
    if x.shape[0] < 2:
        return None
    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)
    numerator = (x.T @ y).square().sum()
    denominator = ((x.T @ x).square().sum() * (y.T @ y).square().sum()).sqrt()
    return float(numerator / denominator) if denominator > 0 else None


def effective_rank(x: torch.Tensor) -> dict:
    """Entropy effective rank of the centered sample covariance spectrum."""
    x = x.double() - x.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    spectrum = singular.square()
    if spectrum.sum() == 0:
        return {
            "effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "numerical_rank": 0,
            "condition_number_nonzero": None,
            "top_singular_values": singular[:16].tolist(),
        }
    probability = spectrum / spectrum.sum()
    probability = probability[probability > 0]
    rank = torch.exp(-(probability * probability.log()).sum())
    maximum = max(1, min(x.shape[0] - 1, x.shape[1]))
    tolerance = singular.max() * max(x.shape) * torch.finfo(singular.dtype).eps
    nonzero = singular[singular > tolerance]
    return {
        "effective_rank": float(rank),
        "normalized_effective_rank": float(rank / maximum),
        "numerical_rank": len(nonzero),
        "condition_number_nonzero": (
            float(nonzero.max() / nonzero.min()) if len(nonzero) else None
        ),
        "top_singular_values": singular[:16].tolist(),
    }


def _tensor_stats(tensor: torch.Tensor) -> list[dict]:
    flat = tensor.float().flatten(1)
    return [
        {
            "mean": float(row.mean()),
            "std": float(row.std(unbiased=False)),
            "near_zero_fraction": float((row < 0.05).float().mean()),
            "near_one_fraction": float((row > 0.95).float().mean()),
        }
        for row in flat
    ]


def operator_diagnostics(block) -> list[dict]:
    """One raw diagnostic row per sample from the most recent fusion call."""
    operator = getattr(block, "operator", block)
    batch_size = None
    fields: dict[str, list] = {}
    for name in ("last_alpha", "last_gate", "last_weights"):
        value = getattr(operator, name, None)
        if torch.is_tensor(value):
            stats = _tensor_stats(value)
            batch_size = len(stats)
            fields[name.removeprefix("last_")] = stats
    if hasattr(operator, "last_masks") and operator.last_masks is not None:
        channel, spatial = operator.last_masks
        ch_stats, sp_stats = _tensor_stats(channel), _tensor_stats(spatial)
        batch_size = len(ch_stats)
        fields["channel_mask"] = ch_stats
        fields["spatial_mask"] = sp_stats
    for name in ("last_offset_per_sample", "last_attention_entropy", "last_attention_peak"):
        value = getattr(operator, name, None)
        if torch.is_tensor(value):
            values = value.detach().cpu().flatten().tolist()
            batch_size = len(values)
            fields[name.removeprefix("last_")] = values
    if hasattr(operator, "alpha_value"):
        fields["global_alpha"] = [operator.alpha_value()]

    if batch_size is None:
        batch_size = 1
    rows = [{"operator": type(operator).__name__} for _ in range(batch_size)]
    for key, values in fields.items():
        if key == "global_alpha":
            for row in rows:
                row[key] = values[0]
        else:
            for row, value in zip(rows, values, strict=True):
                row[key] = value
    return rows


def operator_parameter_diagnostics(block) -> dict:
    """Weight norms and concat modality blocks for one fusion operator."""
    operator = getattr(block, "operator", block)
    result = {
        "operator": type(operator).__name__,
        "parameter_l2": {
            name: float(parameter.detach().float().norm())
            for name, parameter in operator.named_parameters()
        },
    }
    projection = getattr(operator, "proj", None)
    if isinstance(projection, torch.nn.Conv2d) and projection.in_channels == (
        2 * projection.out_channels
    ):
        weights = projection.weight.detach().float()
        channels = projection.out_channels
        rgb_norm = float(weights[:, :channels].norm())
        thermal_norm = float(weights[:, channels:].norm())
        result["concat_projection"] = {
            "rgb_block_l2": rgb_norm,
            "thermal_block_l2": thermal_norm,
            "rgb_to_thermal_norm_ratio": rgb_norm / max(thermal_norm, 1e-20),
            "singular_values": torch.linalg.svdvals(weights.flatten(1)).cpu().tolist(),
        }
    return result


def _sample_feature_stats(feature: torch.Tensor) -> dict:
    values = feature.detach().float()
    centered = values - values.mean()
    variance = centered.square().mean()
    kurtosis = float(centered.pow(4).mean() / variance.square()) if variance > 0 else None
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "rms": float(values.square().mean().sqrt()),
        "near_zero_fraction": float((values.abs() < 1e-6).float().mean()),
        "kurtosis": kurtosis,
    }


def _branches(model, model_type, rgb, thermal):
    if model_type == "multi-scale":
        rgb_features = dict(zip((3, 4, 5), model.backbone_rgb(rgb), strict=True))
        thermal_features = dict(zip((3, 4, 5), model.backbone_thermal(thermal), strict=True))
        levels = [int(level) for level in model.fusers]
        fusers = {level: model.fusers[str(level)] for level in levels}
    elif model_type == "one-shot":
        level = model.meet_level
        rgb_features = model.backbone_rgb.forward_upto(rgb, level)
        thermal_features = model.backbone_thermal.forward_upto(thermal, level)
        levels = [level]
        fusers = {level: model.fuser}
    else:
        raise ValueError(
            f"representation diagnostics require a two-stream graph, got {model_type!r}"
        )
    return rgb_features, thermal_features, levels, fusers


@torch.no_grad()
def collect(model, model_type, dataset, device, batch_size: int, limit: int):
    vectors: dict[str, list[torch.Tensor]] = {}
    raw_rows = []
    ids = []
    model.eval()
    for start in range(0, limit, batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, limit))]
        ids.extend(sample["id"] for sample in samples)
        rgb = torch.stack([sample["rgb"].float() / 255.0 for sample in samples]).to(device)
        thermal = torch.stack([sample["thermal"].float() / 255.0 for sample in samples]).to(device)
        r, t, levels, fusers = _branches(model, model_type, rgb, thermal)

        for level in levels:
            fused = fusers[level](r[level], t[level])
            rgb_gap = r[level].mean(dim=(2, 3)).cpu()
            thermal_gap = t[level].mean(dim=(2, 3)).cpu()
            fused_gap = fused.mean(dim=(2, 3)).cpu()
            vectors.setdefault(f"rgb_l{level}", []).append(rgb_gap)
            vectors.setdefault(f"thermal_l{level}", []).append(thermal_gap)
            vectors.setdefault(f"fused_l{level}", []).append(fused_gap)
            diagnostics = operator_diagnostics(fusers[level])
            if len(diagnostics) == 1 and len(samples) > 1:
                diagnostics *= len(samples)
            for index, sample in enumerate(samples):
                raw_rows.append(
                    {
                        "frame_id": sample["id"],
                        "level": level,
                        "rgb": _sample_feature_stats(r[level][index]),
                        "thermal": _sample_feature_stats(t[level][index]),
                        "fused": _sample_feature_stats(fused[index]),
                        "rgb_thermal_gap_cosine": float(
                            F.cosine_similarity(
                                rgb_gap[index : index + 1], thermal_gap[index : index + 1]
                            ).item()
                        ),
                        "operator": diagnostics[index],
                    }
                )

    packed = {name: torch.cat(value, dim=0) for name, value in vectors.items()}
    summary = {}
    for key in sorted(packed):
        summary[key] = effective_rank(packed[key])
    levels = sorted({int(key.rsplit("l", 1)[1]) for key in packed if key.startswith("rgb_")})
    for level in levels:
        summary[f"rgb_thermal_l{level}"] = {
            "linear_cka": linear_cka(packed[f"rgb_l{level}"], packed[f"thermal_l{level}"])
        }
        summary[f"fused_rgb_l{level}"] = {
            "linear_cka": linear_cka(packed[f"fused_l{level}"], packed[f"rgb_l{level}"])
        }
        summary[f"fused_thermal_l{level}"] = {
            "linear_cka": linear_cka(packed[f"fused_l{level}"], packed[f"thermal_l{level}"])
        }
        summary[f"operator_l{level}"] = operator_parameter_diagnostics(fusers[level])
    return ids, packed, raw_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, modality, model_type = load_checkpoint_model(args.ckpt, device=device)
    if modality != "fusion":
        raise SystemExit("representation diagnostics require a fusion checkpoint")
    dataset = kaist_split_cv(args.root, "val-cv")
    limit = min(args.limit, len(dataset))
    ids, vectors, raw_rows, summary = collect(
        model, model_type, dataset, device, args.batch_size, limit
    )

    checkpoint = Path(args.ckpt)
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": "val-cv",
        "frames": limit,
        "sampling": "first N frozen val-cv IDs; no random subsampling",
        "raw_per_frame_level": raw_rows,
        "derived_summary": summary,
        "interpretation_warning": (
            "gate/attention values and CKA are descriptive; validate modality-use "
            "claims with controlled interventions"
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw_out) if args.raw_out else output.with_suffix(".npz")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        raw_path,
        frame_ids=np.asarray(ids),
        **{name: tensor.numpy() for name, tensor in vectors.items()},
    )
    result["raw_npz"] = {
        "path": str(raw_path),
        "sha256": sha256_file(raw_path),
    }
    atomic_write_json(output, result)
    print(f"wrote {output} and {raw_path}")


if __name__ == "__main__":
    main()
