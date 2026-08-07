"""Profile parameters, dynamic MACs, latency, and peak memory for one checkpoint.

The output preserves every measured latency before reporting summaries. Dynamic
MAC accounting covers Conv2d, Linear, and both explicit cross-attention matrix
multiplications. A MAC is reported separately from the 2-FLOP multiply-add
convention to avoid the common factor-of-two ambiguity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prism.evaluate_model import load_checkpoint_model
from prism.models.fusion import CrossAttentionFuse
from prism.provenance import atomic_write_json, sha256_file


def count_dynamic_macs(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    total = 0
    handles = []

    def conv_hook(module, _inputs, output):
        nonlocal total
        output_elements = output.numel()
        kernel_macs = (
            module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        )
        total += output_elements * kernel_macs

    def linear_hook(module, _inputs, output):
        nonlocal total
        total += output.numel() * module.in_features

    def attention_hook(module, module_inputs, _output):
        nonlocal total
        query = module_inputs[0]
        batch, channels, height, width = query.shape
        key_tokens = height * width
        if key_tokens > module.max_kv_tokens:
            scale = (module.max_kv_tokens / key_tokens) ** 0.5
            key_height = max(1, int(height * scale))
            key_width = max(1, min(width, module.max_kv_tokens // key_height))
            key_tokens = key_height * key_width
        query_tokens = height * width
        # QK^T and attention@V, each B*Nq*Nk*C MACs.
        total += 2 * batch * query_tokens * key_tokens * channels

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, CrossAttentionFuse):
            handles.append(module.register_forward_hook(attention_hook))
    with torch.no_grad():
        model(*inputs)
    for handle in handles:
        handle.remove()
    return int(total)


def latency_samples(model, inputs, device: str, warmup: int, iterations: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        samples = []
        for _ in range(iterations):
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(*inputs)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                model(*inputs)
                samples.append(1000.0 * (time.perf_counter() - start_time))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, modality, model_type = load_checkpoint_model(args.ckpt, device=device)
    model.eval()
    image = torch.zeros(1, 3, args.height, args.width, device=device)
    inputs = (image, image.clone()) if modality == "fusion" else (image,)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    macs = count_dynamic_macs(model, inputs)
    samples = latency_samples(model, inputs, device, args.warmup, args.iterations)
    peak_memory = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    values = np.asarray(samples)

    checkpoint = Path(args.ckpt)
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_type": model_type,
        "modality": modality,
        "device": device,
        "input_shape": [1, 3, args.height, args.width],
        "parameters": parameters,
        "trainable_parameters": trainable,
        "macs": macs,
        "flops_2_per_mac": 2 * macs,
        "mac_accounting": "dynamic Conv2d + Linear + QK/AV attention matmuls",
        "latency_ms_raw": samples,
        "latency_ms_summary": {
            "count": len(samples),
            "mean": float(values.mean()),
            "std_population": float(values.std()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
        },
        "peak_allocated_bytes": peak_memory,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
