"""Machine-enforced checks for the staged architecture-study protocol."""

from pathlib import Path

import torch
import yaml

from prism.protocol import ARCH_EPOCHS, PROTOCOL_VERSION, RECIPE, SEEDS, TEST_SPLITS
from scripts.run_experiments import find_resume_checkpoint, source_fingerprint
from scripts.select_winner import DEFAULT_GRAPH_ARMS


def test_frozen_core_protocol() -> None:
    assert ARCH_EPOCHS == 40
    assert PROTOCOL_VERSION == "2026-08-07-corrected-v4"
    assert SEEDS == (0, 1, 2, 3, 4)
    assert TEST_SPLITS == ("test", "test-day", "test-night")
    assert RECIPE["checkpoint_every_epochs"] == 10
    assert RECIPE["validation_every_epochs"] == 10
    assert RECIPE["history_every_steps"] == 20


def test_yaml_config_mirrors_executable_recipe() -> None:
    config = yaml.safe_load(Path("configs/architecture.yaml").read_text())
    optimization = config["optimization"]
    training = config["training"]
    evaluation = config["evaluation"]

    assert config["data"]["input_size"] == list(RECIPE["input_size"])
    assert config["data"]["batch_size"] == RECIPE["batch_size"]
    assert optimization["epochs"] == RECIPE["epochs"]
    assert optimization["base_lr"] == RECIPE["lr"]
    assert optimization["momentum"] == RECIPE["momentum"]
    assert optimization["weight_decay"] == RECIPE["weight_decay"]
    assert optimization["warmup_fraction"] == RECIPE["warmup_frac"]
    assert optimization["max_grad_norm"] == RECIPE["max_grad_norm"]
    assert optimization["amp"] == RECIPE["amp"]
    assert optimization["ema"] == RECIPE["ema"]
    assert training["num_workers"] == RECIPE["num_workers"]
    assert training["checkpoint_every_epochs"] == RECIPE["checkpoint_every_epochs"]
    assert training["validation_every_epochs"] == RECIPE["validation_every_epochs"]
    assert training["history_every_steps"] == RECIPE["history_every_steps"]
    assert training["seeds"] == list(SEEDS)
    assert evaluation["nms_iou"] == RECIPE["nms_iou"]
    assert evaluation["conf_floor"] == RECIPE["confidence_floor"]


def test_block_c_template_selection_requires_a_feature_graph() -> None:
    assert [f"B{i}" for i in range(2, 11)] == DEFAULT_GRAPH_ARMS
    assert "B1" not in DEFAULT_GRAPH_ARMS


def test_source_fingerprint_is_a_sha256() -> None:
    fingerprint = source_fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_resume_uses_latest_matching_periodic_checkpoint(tmp_path) -> None:
    for epoch in (10, 20):
        torch.save(
            {
                "run_args_hash": "args",
                "source_fingerprint": "source",
                "protocol_version": PROTOCOL_VERSION,
                "seed": 0,
                "epoch": epoch,
            },
            tmp_path / f"fusion_seed0_e{epoch}.pt",
        )
    found = find_resume_checkpoint(tmp_path, "fusion", 0, "args", "source", final_epoch=40)
    assert found.name == "fusion_seed0_e20.pt"
