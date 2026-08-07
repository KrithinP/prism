"""From-scratch training loop for the architecture study (YOLOX on KAIST).

The frozen protocol (TODO.md Phase 3): SGD + linear warmup -> cosine decay, AMP
for the 12GB Blackwell card, five predeclared seeds, EXACTLY 40 epochs for
architecture runs. Two hard rules:

  1. THE DATA FIREWALL (TODO.md 4.4): --split only accepts train / train-cv.
     Validation and official-test data are unreachable from the training loader
     by construction; evaluation is handled by separate commands
     (prism.evaluate_model --official-test) that requires a frozen run manifest.
  2. Legacy `fuse_stages` fusion is quarantined (TODO.md 6.7): the fusion family
     must be early, one-shot, or multi-scale. LegacyTwoStreamYOLOX exists only
     to LOAD old checkpoints.

New-run fusion config:
  --fusion-family early [--fusion-operator is not used: input concat is fixed]
  --fusion-family one-shot --meet-level 4 --anchor thermal --fusion-operator concat
  --fusion-family multi-scale --fuse-subset 345 --anchor thermal --fusion-operator concat
Duplicated-input capacity controls (TODO.md 6.1): --fusion-inputs rgb|thermal feeds
both branches the same modality (RGB+RGB / T+T).
"""

from __future__ import annotations

import argparse
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prism.data.detection import (
    DetectionDataset,
    FusionDataset,
    detection_collate,
    fusion_collate,
)
from prism.data.kaist import kaist_split, kaist_split_cv
from prism.data.transforms import (
    Compose,
    PairedHFlip,
    PairedModalityDropout,
    RGBPhotometric,
    ThermalMisalignment,
)
from prism.eval.canonical import evaluate_reference
from prism.models.early_fusion import EarlyFusionYOLOX
from prism.models.loss import YOLOXLoss
from prism.models.multi_scale_fusion import MultiScaleFusionYOLOX
from prism.models.one_shot_fusion import OneShotFusionYOLOX
from prism.models.yolox import YOLOX
from prism.protocol import ARCH_EPOCHS, PROTOCOL_VERSION, RECIPE, SEEDS
from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    append_jsonl,
    artifact_record,
    atomic_torch_save,
    atomic_write_json,
    utc_now,
)


class Diverged(RuntimeError):
    """Training blew up at this learning rate. A finding, not a bug."""


# Divergence must be judged on a SMOOTHED loss, never a single batch.
# The loss is (5*iou + obj + cls) / num_fg, so a batch holding only one or two
# assignable pedestrians divides by a tiny number and legitimately reads in the
# hundreds with nothing wrong. A first attempt at this guard used a single-batch
# limit of 500 and killed every run at step 22 (2026-07-24).
# Real divergence looks different: sustained growth (obj 12 -> 989 over ~300 steps),
# which drags the running average up and holds it there.
DIVERGENCE_EMA_LIMIT = 300.0
DIVERGENCE_EMA_BETA = 0.98
DIVERGENCE_GRACE_STEPS = 200  # don't judge during warmup at all
HISTORY_EVERY_STEPS = RECIPE["history_every_steps"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """Seed every RNG used by paired augmentations in a dataloader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parameter_group_diagnostics(model) -> dict[str, dict[str, float]]:
    """L2 weight/gradient norms grouped by top-level module name."""
    accumulators: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        values = accumulators.setdefault(
            group, {"parameter_sq": 0.0, "gradient_sq": 0.0, "parameters": 0}
        )
        values["parameter_sq"] += float(parameter.detach().float().square().sum())
        values["parameters"] += parameter.numel()
        if parameter.grad is not None:
            values["gradient_sq"] += float(parameter.grad.detach().float().square().sum())
    return {
        group: {
            "parameter_l2": math.sqrt(values["parameter_sq"]),
            "gradient_l2_preclip": math.sqrt(values["gradient_sq"]),
            "gradient_to_weight": (
                math.sqrt(values["gradient_sq"]) / max(math.sqrt(values["parameter_sq"]), 1e-12)
            ),
            "parameters": int(values["parameters"]),
        }
        for group, values in sorted(accumulators.items())
    }


def build_optimizer(
    model,
    lr,
    momentum=RECIPE["momentum"],
    weight_decay=RECIPE["weight_decay"],
):
    """YOLOX-style: no weight decay on BN weights or biases (stabilises from-scratch)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):  # BN scale/shift and all biases
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def lr_factor(step, total_steps, warmup_steps):
    """Linear warmup 0->1, then cosine 1->0 across the rest of training."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def warmup_steps_for(total_steps: int) -> int:
    """Return the frozen warmup duration as a fraction of the full run."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    return max(1, round(total_steps * RECIPE["warmup_frac"]))


def require_training_split(split: str) -> None:
    """Reject held-out validation and official-test splits at the train boundary."""
    if split not in {"train", "train-cv"}:
        raise ValueError(f"split={split!r} is not a training split; use 'train' or 'train-cv'")


def require_fusion_inputs(fusion_inputs: str) -> None:
    if fusion_inputs not in {"paired", "rgb", "thermal"}:
        raise ValueError("fusion_inputs must be one of: 'paired', 'rgb', or 'thermal'")


def positive_indices(base):
    """Frame indices with >=1 scored pedestrian (scans XML only, no image decode)."""
    from prism.data.kaist import parse_annotation

    keep = []
    for i, fid in enumerate(base.ids):
        _, _, xml = base._paths(fid)
        if (parse_annotation(xml)["classes"] == 1).any():
            keep.append(i)
    return keep


def require_device(allow_cpu=False):
    """Refuse to train on CPU unless explicitly allowed.

    A silent CPU fallback is worse than a crash: a full run is ~100x slower and looks
    like the model is merely slow to converge. This bit us on 2026-07-24 when an NVIDIA
    driver upgrade left the loaded kernel module (580.159.03) out of sync with the
    userspace libs (580.173.02) -- nvidia-smi reported "Driver/library version mismatch",
    torch.cuda.is_available() went False, and a sweep started grinding on CPU. Fix was a
    reboot. Fail loudly instead.
    """
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        print("WARNING: training on CPU (--allow-cpu). This will be very slow.")
        return "cpu"
    raise RuntimeError(
        "CUDA is not available -- refusing to train on CPU.\n"
        "  Check:  nvidia-smi   (a 'Driver/library version mismatch' means the loaded\n"
        "          kernel module and the installed userspace driver disagree -- reboot)\n"
        '  Then:   python -c "import torch; print(torch.cuda.is_available())"\n'
        "  Override with --allow-cpu if you really mean it."
    )


def build_model(modality, fusion_family, base_ch, fuse_subset, meet_level, anchor, operator):
    """Returns (model, model_type, fusion_config_dict)."""
    if modality != "fusion":
        return YOLOX(num_classes=1, base_ch=base_ch), "unimodal", {}
    if fusion_family == "early":
        model = EarlyFusionYOLOX(num_classes=1, base_ch=base_ch)
        return model, "early", {"family": "early"}
    if fusion_family == "multi-scale":
        model = MultiScaleFusionYOLOX(
            num_classes=1,
            base_ch=base_ch,
            fuse_subset=fuse_subset,
            anchor=anchor,
            operator=operator,
        )
        cfg = {
            "family": "multi-scale",
            "fuse_subset": list(fuse_subset),
            "anchor": anchor,
            "operator": operator,
        }
        return model, "multi-scale", cfg
    if fusion_family == "one-shot":
        model = OneShotFusionYOLOX(
            num_classes=1, base_ch=base_ch, meet_level=meet_level, anchor=anchor, operator=operator
        )
        cfg = {
            "family": "one-shot",
            "meet_level": meet_level,
            "anchor": anchor,
            "operator": operator,
        }
        return model, "one-shot", cfg
    raise ValueError(f"unknown fusion family {fusion_family!r}")


@torch.no_grad()
def validate_mr(model, root, split, device, modality, input_policy):
    """Canonical MR^-2 (All) on a validation split, for the per-epoch health log."""
    from prism.evaluate_model import collect_frames

    was_training = model.training
    base = kaist_split_cv(root, split)
    try:
        frames = collect_frames(
            model, base, modality, device, input_policy=input_policy, day_ids=None
        )
        return evaluate_reference(frames)["all"]
    finally:
        model.train(was_training)


def train(
    root,
    modality="rgb",
    lr=RECIPE["lr"],
    epochs=ARCH_EPOCHS,
    batch_size=RECIPE["batch_size"],
    seed=0,
    limit=None,
    out="checkpoints",
    positives_only=False,
    split="train",
    allow_cpu=False,
    save_every=RECIPE["checkpoint_every_epochs"],
    base_ch=32,
    fusion_family="multi-scale",
    fuse_subset=(3, 4, 5),
    meet_level=4,
    anchor="thermal",
    operator="concat",
    fusion_inputs="paired",
    val_split="val-cv",
    val_every=RECIPE["validation_every_epochs"],
    run_id=None,
    attempt_id=None,
    run_args_hash=None,
    source_fingerprint=None,
    resume=None,
    history_path=None,
    modality_dropout_p=0.0,
    thermal_shift_px=0,
):
    require_training_split(split)
    require_fusion_inputs(fusion_inputs)
    if seed not in SEEDS:
        print(
            f"WARNING: seed {seed} is not in the predeclared set {SEEDS} -- "
            "this run cannot join a 5-seed mean unless it is a declared new block"
        )

    set_seed(seed)
    Path(out).mkdir(parents=True, exist_ok=True)  # fail fast on a bad path
    final_checkpoint = Path(out) / f"{modality}_seed{seed}.pt"
    if final_checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint: {final_checkpoint}")
    device = require_device(allow_cpu)

    if modality != "fusion" and (modality_dropout_p or thermal_shift_px):
        raise ValueError("fusion robustness augmentations require --modality fusion")
    transforms = [PairedHFlip(0.5), RGBPhotometric(0.2, 0.2)]
    if thermal_shift_px:
        transforms.append(ThermalMisalignment(thermal_shift_px))
    if modality_dropout_p:
        transforms.append(PairedModalityDropout(modality_dropout_p))
    aug = Compose(transforms)
    base = kaist_split_cv(root, split) if split == "train-cv" else kaist_split(root, split)
    is_fusion = modality == "fusion"
    if is_fusion:
        ds = FusionDataset(base, transforms=aug)
        collate = fusion_collate
    else:
        ds = DetectionDataset(base, modality=modality, transforms=aug)
        collate = detection_collate
    idx = positive_indices(base) if positives_only else list(range(len(ds)))
    if limit:
        idx = idx[:limit]
    ds = torch.utils.data.Subset(ds, idx)
    gen = torch.Generator().manual_seed(seed)  # deterministic shuffle per seed
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=RECIPE["num_workers"],
        collate_fn=collate,
        drop_last=True,
        generator=gen,
        worker_init_fn=seed_worker,
    )
    if len(loader) == 0:
        raise ValueError(
            f"training loader has zero batches: samples={len(ds)}, batch_size={batch_size}"
        )

    model, model_type, fusion_cfg = build_model(
        modality, fusion_family, base_ch, fuse_subset, meet_level, anchor, operator
    )
    model = model.to(device).train()
    lossfn = YOLOXLoss(num_classes=1)
    opt = build_optimizer(model, lr)
    amp_enabled = device == "cuda" and RECIPE["amp"]
    scaler = torch.amp.GradScaler(device, enabled=amp_enabled)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_steps = epochs * len(loader)
    warmup_steps = warmup_steps_for(total_steps)
    step = 0
    loss_ema = None
    val_log = []
    run_started = time.monotonic()
    history_file = Path(history_path) if history_path else Path(out) / "training_history.jsonl"
    start_record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "run_start",
        "timestamp_utc": utc_now(),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_args_hash": run_args_hash,
        "source_fingerprint": source_fingerprint,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "split": split,
        "dataset_samples": len(ds),
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": len(loader),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": lr,
        "device": device,
        "amp_enabled": amp_enabled,
        "save_every_epochs": save_every,
        "validation_split": val_split,
        "validation_every_epochs": val_every,
        "model_type": model_type,
        "base_channels": base_ch,
        "modality": modality,
        "fusion": fusion_cfg,
        "fusion_inputs": fusion_inputs,
        "training_interventions": {
            "modality_dropout_p": modality_dropout_p,
            "thermal_shift_px": thermal_shift_px,
        },
        "protocol": RECIPE,
    }
    if resume is None and history_file.exists():
        raise FileExistsError(f"refusing to overwrite training history: {history_file}")

    def log_history(row: dict, *, durable: bool = True) -> None:
        append_jsonl(
            history_file,
            {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "timestamp_utc": utc_now(),
                "run_id": run_id,
                "attempt_id": attempt_id,
                **row,
            },
            durable=durable,
        )

    if not history_file.exists():
        log_history(start_record)

    def checkpoint_state(epoch_number: int) -> dict:
        numpy_state = np.random.get_state()
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "modality": modality,
            "seed": seed,
            "epoch": epoch_number,
            "global_step": step,
            "loss_ema": loss_ema,
            "validation_history": val_log,
            "base_ch": base_ch,
            "model_type": model_type,
            "fusion": fusion_cfg,
            "fusion_inputs": fusion_inputs,
            "training_interventions": {
                "modality_dropout_p": modality_dropout_p,
                "thermal_shift_px": thermal_shift_px,
            },
            "run_id": run_id,
            "attempt_id": attempt_id,
            "run_args_hash": run_args_hash,
            "source_fingerprint": source_fingerprint,
            "protocol_version": PROTOCOL_VERSION,
            "protocol": RECIPE,
            "history_path": str(history_file.resolve()),
            "rng": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_state[0],
                    "state": numpy_state[1].tolist(),
                    "position": numpy_state[2],
                    "has_gauss": numpy_state[3],
                    "cached_gaussian": numpy_state[4],
                },
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "dataloader_generator": gen.get_state(),
            },
        }

    start_epoch = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        expected = {
            "modality": modality,
            "seed": seed,
            "model_type": model_type,
            "run_args_hash": run_args_hash,
            "source_fingerprint": source_fingerprint,
            "protocol_version": PROTOCOL_VERSION,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if value is not None and state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume checkpoint metadata mismatch: {mismatches}")
        start_epoch = int(state["epoch"])
        if not 0 < start_epoch < epochs:
            raise ValueError(f"resume epoch must be in [1,{epochs - 1}], got {start_epoch}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["global_step"])
        loss_ema = state.get("loss_ema")
        val_log = list(state.get("validation_history", []))
        rng = state["rng"]
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        gen.set_state(rng["dataloader_generator"].cpu())
        log_history(
            {
                "kind": "resume",
                "checkpoint": artifact_record(resume),
                "epoch": start_epoch,
                "step": step,
            }
        )
        print(f"resumed {resume} at epoch {start_epoch}, step {step}")

    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        epoch_sums = {
            "loss": 0.0,
            "iou": 0.0,
            "objectness": 0.0,
            "classification": 0.0,
            "num_foreground": 0,
        }
        epoch_steps = 0
        for batch in loader:
            if is_fusion:
                rgb, thermal, targets = batch
                rgb, thermal = rgb.to(device), thermal.to(device)
                model_in = (
                    (rgb, rgb)
                    if fusion_inputs == "rgb"
                    else (thermal, thermal)
                    if fusion_inputs == "thermal"
                    else (rgb, thermal)
                )
            else:
                imgs, targets = batch
                model_in = (imgs.to(device),)
            for g in opt.param_groups:
                g["lr"] = lr * lr_factor(step, total_steps, warmup_steps)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=amp_enabled):
                loss, parts = lossfn(model(*model_in), targets)

            # A diverged run is a RESULT ("this lr is unstable"), not a crash.
            lv = loss.item()
            if not torch.isfinite(loss):
                raise Diverged(f"non-finite loss at step {step} (lr={lr})")
            loss_ema = (
                lv
                if loss_ema is None
                else DIVERGENCE_EMA_BETA * loss_ema + (1 - DIVERGENCE_EMA_BETA) * lv
            )
            if step > DIVERGENCE_GRACE_STEPS and loss_ema > DIVERGENCE_EMA_LIMIT:
                raise Diverged(
                    f"loss EMA {loss_ema:.1f} exceeded {DIVERGENCE_EMA_LIMIT} "
                    f"at step {step} (lr={lr}) -- sustained blow-up"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            should_log = step == 0 or (step + 1) % HISTORY_EVERY_STEPS == 0
            groups = parameter_group_diagnostics(model) if should_log else None
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=RECIPE["max_grad_norm"]
            )
            scaler.step(opt)
            scaler.update()
            step += 1
            epoch_steps += 1
            epoch_sums["loss"] += lv
            epoch_sums["iou"] += float(parts["iou"])
            epoch_sums["objectness"] += float(parts["obj"])
            epoch_sums["classification"] += float(parts["cls"])
            epoch_sums["num_foreground"] += int(parts["num_fg"])
            if should_log:
                log_history(
                    {
                        "kind": "train_step",
                        "epoch": epoch + 1,
                        "step": step,
                        "lr": opt.param_groups[0]["lr"],
                        "loss": lv,
                        "loss_ema": loss_ema,
                        "iou": float(parts["iou"]),
                        "objectness": float(parts["obj"]),
                        "classification": float(parts["cls"]),
                        "num_foreground": int(parts["num_fg"]),
                        "gradient_l2_preclip": float(total_grad_norm),
                        "gradient_clip_max_norm": RECIPE["max_grad_norm"],
                        "amp_scale": scaler.get_scale(),
                        "parameter_groups": groups,
                    },
                    durable=False,
                )
            if step % HISTORY_EVERY_STEPS == 0:
                print(
                    f"e{epoch} s{step}/{total_steps} "
                    f"lr{opt.param_groups[0]['lr']:.4f} loss{loss.item():.3f} "
                    f"iou{parts['iou']:.3f} obj{parts['obj']:.3f} "
                    f"cls{parts['cls']:.3f} fg{parts['num_fg']}"
                )

        epoch_duration = time.monotonic() - epoch_started
        epoch_record = {
            "kind": "epoch_end",
            "epoch": epoch + 1,
            "step": step,
            "steps": epoch_steps,
            "samples": epoch_steps * batch_size,
            "duration_seconds": epoch_duration,
            "samples_per_second": epoch_steps * batch_size / epoch_duration,
            "lr": opt.param_groups[0]["lr"],
            "mean_loss": epoch_sums["loss"] / epoch_steps,
            "mean_iou": epoch_sums["iou"] / epoch_steps,
            "mean_objectness": epoch_sums["objectness"] / epoch_steps,
            "mean_classification": epoch_sums["classification"] / epoch_steps,
            "num_foreground": epoch_sums["num_foreground"],
            "loss_ema": loss_ema,
            "amp_scale": scaler.get_scale(),
        }
        if device == "cuda":
            epoch_record["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            epoch_record["cuda_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        log_history(epoch_record)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # validation MR (health log, TODO.md 5.3): canonical evaluator on the
        # frozen sequence-disjoint validation list; NOT used for checkpoint picks.
        if val_every and (epoch + 1) % val_every == 0:
            mr = validate_mr(model, root, val_split, device, modality, fusion_inputs)
            val_log.append({"epoch": epoch + 1, "val_mr_all": mr})
            log_history(
                {
                    "kind": "validation",
                    "epoch": epoch + 1,
                    "step": step,
                    "split": val_split,
                    "mr_all": mr,
                }
            )
            print(f"e{epoch + 1} val MR^-2 All = {mr * 100:.2f}%")

        if save_every and (epoch + 1) % save_every == 0 and (epoch + 1) != epochs:
            ck = Path(out) / f"{modality}_seed{seed}_e{epoch + 1}.pt"
            atomic_torch_save(ck, checkpoint_state(epoch + 1))
            log_history(
                {
                    "kind": "checkpoint",
                    "epoch": epoch + 1,
                    "step": step,
                    "final": False,
                    "artifact": artifact_record(ck),
                }
            )
            print(f"saved {ck}")

    ckpt = final_checkpoint
    atomic_torch_save(ckpt, checkpoint_state(epochs))
    final_artifact = artifact_record(ckpt)
    log_history(
        {
            "kind": "checkpoint",
            "epoch": epochs,
            "step": step,
            "final": True,
            "artifact": final_artifact,
        }
    )
    if val_log:
        atomic_write_json(Path(out) / "val_curve.json", {"validation_history": val_log})
    log_history(
        {
            "kind": "run_end",
            "status": "completed",
            "epoch": epochs,
            "step": step,
            "duration_seconds": time.monotonic() - run_started,
            "checkpoint": final_artifact,
            "validation_history": val_log,
        }
    )
    print(f"saved {ckpt}")
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--modality", default="rgb", choices=["rgb", "thermal", "fusion"])
    ap.add_argument("--lr", type=float, default=RECIPE["lr"])
    ap.add_argument(
        "--epochs",
        type=int,
        default=ARCH_EPOCHS,
        help="frozen at 40 for architecture runs (TODO.md 1)",
    )
    ap.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke: train on first N frames")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument(
        "--positives-only",
        action="store_true",
        help="restrict to frames containing >=1 pedestrian (for the smoke)",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "train-cv"],
        help="validation and official-test splits are firewalled from training",
    )
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument(
        "--save-every",
        type=int,
        default=RECIPE["checkpoint_every_epochs"],
    )
    ap.add_argument(
        "--base-ch",
        type=int,
        default=32,
        help="backbone width; use --base-ch 42 for the ~14M param control",
    )

    fusion = ap.add_argument_group("fusion architecture")
    fusion.add_argument(
        "--fusion-family",
        default="multi-scale",
        choices=["early", "multi-scale", "one-shot"],
        help="legacy fuse_stages is quarantined and cannot be trained",
    )
    fusion.add_argument(
        "--fuse-subset", default="345", help="multi-scale fusion levels, e.g. 345, 34, 4"
    )
    fusion.add_argument(
        "--fuse-stages",
        default=None,
        help="DEPRECATED alias of --fuse-subset (quarantined semantics)",
    )
    fusion.add_argument(
        "--meet-level",
        type=int,
        default=4,
        choices=[3, 4, 5],
        help="one-shot meeting level (dark3/dark4/dark5)",
    )
    fusion.add_argument(
        "--anchor",
        default="thermal",
        choices=["thermal", "rgb"],
        help="modality that feeds un-fused neck skips",
    )
    fusion.add_argument(
        "--fusion-operator",
        default="concat",
        choices=[
            "concat",
            "sum",
            "average",
            "scalar-mixture",
            "quality-gated",
            "channel-mixture",
            "gated-residual",
            "nested-residual",
            "gaff",
            "dmaf",
            "alignment",
            "cross-attention",
            "concat-attention-params",
            "concat-alignment-params",
        ],
    )
    fusion.add_argument(
        "--fusion-inputs",
        default="paired",
        choices=["paired", "rgb", "thermal"],
        help="rgb -> RGB+RGB and thermal -> T+T duplicated-input controls",
    )
    fusion.add_argument(
        "--modality-dropout-p",
        type=float,
        default=0.0,
        help="Block D only: probability of dropping exactly one modality",
    )
    fusion.add_argument(
        "--thermal-shift-px",
        type=int,
        default=0,
        help="Block D only: random thermal translation in input pixels",
    )

    ev = ap.add_argument_group("validation")
    ev.add_argument("--val-split", default="val-cv")
    ev.add_argument(
        "--val-every",
        type=int,
        default=RECIPE["validation_every_epochs"],
        help="evaluate canonical validation MR every N epochs (health log)",
    )
    ap.add_argument("--run-id", default=None, help="matrix-run identity recorded in the checkpoint")
    ap.add_argument("--attempt-id", default=None, help="unique runner attempt identity")
    ap.add_argument(
        "--run-args-hash", default=None, help="frozen matrix config hash recorded in the checkpoint"
    )
    ap.add_argument(
        "--source-fingerprint", default=None, help="content hash of executable study source"
    )
    ap.add_argument("--resume", default=None, help="exact end-of-epoch checkpoint to resume")
    ap.add_argument(
        "--history-path",
        default=None,
        help="attempt-specific append-only training history JSONL",
    )
    return ap


def main() -> None:
    a = build_arg_parser().parse_args()

    if a.limit is None and a.epochs != ARCH_EPOCHS:
        raise SystemExit(
            f"architecture runs are frozen at exactly {ARCH_EPOCHS} epochs; "
            "nonstandard epoch counts are allowed only with --limit smoke runs"
        )

    if a.fuse_stages is not None:
        print(
            "WARNING: --fuse-stages is the quarantined legacy flag; mapping it to "
            "--fuse-subset so old commands still run. New experiments must use the "
            "corrected graphs (one-shot / multi-scale with anchor fallback)."
        )
        a.fuse_subset = a.fuse_stages

    history_file = (
        Path(a.history_path) if a.history_path else Path(a.out) / "training_history.jsonl"
    )
    history_existed = history_file.exists()
    try:
        train(
            root=a.root,
            modality=a.modality,
            lr=a.lr,
            epochs=a.epochs,
            batch_size=a.batch_size,
            seed=a.seed,
            limit=a.limit,
            out=a.out,
            positives_only=a.positives_only,
            split=a.split,
            allow_cpu=a.allow_cpu,
            save_every=a.save_every,
            base_ch=a.base_ch,
            fusion_family=a.fusion_family,
            fuse_subset=tuple(int(c) for c in a.fuse_subset),
            meet_level=a.meet_level,
            anchor=a.anchor,
            operator=a.fusion_operator,
            fusion_inputs=a.fusion_inputs,
            val_split=a.val_split,
            val_every=a.val_every,
            run_id=a.run_id,
            attempt_id=a.attempt_id,
            run_args_hash=a.run_args_hash,
            source_fingerprint=a.source_fingerprint,
            resume=a.resume,
            history_path=a.history_path,
            modality_dropout_p=a.modality_dropout_p,
            thermal_shift_px=a.thermal_shift_px,
        )
    except BaseException as error:
        if not history_existed or a.resume is not None:
            append_jsonl(
                history_file,
                {
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                    "timestamp_utc": utc_now(),
                    "kind": "run_end",
                    "status": "failed",
                    "run_id": a.run_id,
                    "attempt_id": a.attempt_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
