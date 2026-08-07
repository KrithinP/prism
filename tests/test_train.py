"""The schedule must warm up then decay; the optimizer must exempt BN/bias from decay."""

import pytest
import torch
from torch import nn

from prism.models.yolox import YOLOX
from prism.protocol import RECIPE
from prism.train import (
    build_arg_parser,
    build_optimizer,
    lr_factor,
    parameter_group_diagnostics,
    require_fusion_inputs,
    require_training_split,
    validate_mr,
    warmup_steps_for,
)


def test_cli_defaults_match_frozen_protocol_without_undeclared_corruptions():
    args = build_arg_parser().parse_args(["--root", "data"])
    assert args.epochs == RECIPE["epochs"]
    assert args.lr == RECIPE["lr"]
    assert args.batch_size == RECIPE["batch_size"]
    assert args.save_every == RECIPE["checkpoint_every_epochs"]
    assert args.val_every == RECIPE["validation_every_epochs"]
    assert args.modality_dropout_p == 0.0
    assert args.thermal_shift_px == 0


def test_lr_schedule_warms_up_then_decays():
    total, warm = 1000, 100
    assert lr_factor(0, total, warm) == 0.0  # starts at zero
    assert abs(lr_factor(warm, total, warm) - 1.0) < 1e-6  # peaks at end of warmup
    assert lr_factor(total, total, warm) < 0.01  # cosine decays to ~0
    # monotonic climb through warmup, monotonic fall after
    assert lr_factor(50, total, warm) < lr_factor(90, total, warm)
    assert lr_factor(200, total, warm) > lr_factor(800, total, warm)


def test_warmup_steps_follow_frozen_fraction():
    assert warmup_steps_for(1_000) == 50
    assert warmup_steps_for(38_000) == 1_900


def test_warmup_steps_reject_nonpositive_runs():
    with pytest.raises(ValueError, match="positive"):
        warmup_steps_for(0)


@pytest.mark.parametrize("split", ["train", "train-cv"])
def test_training_split_accepts_only_training_partitions(split):
    require_training_split(split)


@pytest.mark.parametrize("split", ["val-cv", "test", "test-all"])
def test_training_split_rejects_held_out_partitions(split):
    with pytest.raises(ValueError, match="not a training split"):
        require_training_split(split)


@pytest.mark.parametrize("fusion_inputs", ["paired", "rgb", "thermal"])
def test_fusion_inputs_accepts_supported_modes(fusion_inputs):
    require_fusion_inputs(fusion_inputs)


def test_fusion_inputs_rejects_unknown_modes():
    with pytest.raises(ValueError, match="must be one of"):
        require_fusion_inputs("depth")


def test_optimizer_exempts_bn_and_bias_from_decay():
    model = YOLOX(num_classes=1)
    opt = build_optimizer(model, lr=0.01, weight_decay=5e-4)
    assert len(opt.param_groups) == 2
    decay_group = next(g for g in opt.param_groups if g["weight_decay"] > 0)
    nodecay_group = next(g for g in opt.param_groups if g["weight_decay"] == 0)
    assert len(decay_group["params"]) > 0 and len(nodecay_group["params"]) > 0
    # every no-decay param is 1-D (BN scale/shift or bias); every decay param is a kernel
    assert all(p.ndim <= 1 for p in nodecay_group["params"])
    assert all(p.ndim >= 2 for p in decay_group["params"])


def test_parameter_group_diagnostics_separates_top_level_modules():
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model(torch.ones(1, 3)).sum().backward()
    diagnostics = parameter_group_diagnostics(model)
    assert set(diagnostics) == {"0", "1"}
    assert all(row["gradient_l2_preclip"] > 0 for row in diagnostics.values())
    assert all(row["parameter_l2"] > 0 for row in diagnostics.values())


def test_validation_restores_training_mode(monkeypatch):
    model = nn.Linear(1, 1).train()

    def collect_frames(model, *_args, **_kwargs):
        model.eval()
        return [{"dummy": True}]

    monkeypatch.setattr("prism.train.kaist_split_cv", lambda *_args: object())
    monkeypatch.setattr("prism.evaluate_model.collect_frames", collect_frames)
    monkeypatch.setattr("prism.train.evaluate_reference", lambda _frames: {"all": 0.25})
    assert validate_mr(model, "root", "val-cv", "cpu", "rgb", "paired") == 0.25
    assert model.training
