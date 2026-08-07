"""The frozen architecture-study protocol (TODO.md section 1 and Phase 3).

These constants encode the locked experimental decisions. Every comparable arm
must agree on all of them except the declared intervention -- do not edit this
file to make a single run convenient. Any change is a new experimental block with
its own record in the run ledger.
"""

from __future__ import annotations

# The five predeclared seeds used for every core arm (TODO.md 5.2). Never add or
# replace seeds after a matrix starts; a seed is either in the predeclared list or
# it is a new experimental block.
SEEDS = (0, 1, 2, 3, 4)
PROTOCOL_VERSION = "2026-08-07-corrected-v4"

# Architecture-discovery runs train for EXACTLY 40 epochs (TODO.md 1). Initial
# exploratory runs used 16 epochs; those numbers are quarantined as exploratory.
# Initial experiments fix secondary choices such as epoch count so that major
# architectural decisions can be compared cleanly. Training duration and other
# optimization choices can be saturated after the fusion design is fixed.
ARCH_EPOCHS = 40

# The primary metric for architecture selection (TODO.md 8.6): Reasonable All
# log-average miss rate on VALIDATION. Day/Night MR, robustness, latency, and
# memory are secondary constrained metrics.
PRIMARY_METRIC = "reasonable_all_mr"

# The official improved-annotation test set is separate from selection
# (TODO.md 4.4): it is unreachable from the training path, and evaluating it
# requires an explicit --official-test flag plus a frozen run manifest.
TEST_SPLITS = ("test", "test-day", "test-night")

# Default frozen training recipe (mirrors configs/architecture.yaml).
RECIPE = {
    "lr": 0.01,
    "epochs": ARCH_EPOCHS,
    "batch_size": 8,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "warmup_frac": 0.05,
    "amp": True,
    "ema": False,
    "input_size": (640, 512),
    "augmentations": ("PairedHFlip(0.5)", "RGBPhotometric(0.2, 0.2)"),
    "num_workers": 4,
    "max_grad_norm": 10.0,
    "checkpoint_every_epochs": 10,
    "validation_every_epochs": 10,
    "history_every_steps": 20,
    "optimizer": "SGD nesterov",
    "label_assignment": "SimOTA",
    "loss_weights": {"iou": 5.0, "obj": 1.0, "cls": 1.0},
    "nms_iou": 0.65,
    "confidence_floor": 0.001,
}
