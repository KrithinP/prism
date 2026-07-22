
"""The schedule must warm up then decay; the optimizer must exempt BN/bias from decay."""
from prism.models.yolox import YOLOX
from prism.train import lr_factor, build_optimizer


def test_lr_schedule_warms_up_then_decays():
    total, warm = 1000, 100
    assert lr_factor(0, total, warm) == 0.0            # starts at zero
    assert abs(lr_factor(warm, total, warm) - 1.0) < 1e-6   # peaks at end of warmup
    assert lr_factor(total, total, warm) < 0.01        # cosine decays to ~0
    # monotonic climb through warmup, monotonic fall after
    assert lr_factor(50, total, warm) < lr_factor(90, total, warm)
    assert lr_factor(200, total, warm) > lr_factor(800, total, warm)


"""The schedule must warm up then decay; the optimizer must exempt BN/bias from decay."""
from prism.models.yolox import YOLOX
from prism.train import lr_factor, build_optimizer


def test_lr_schedule_warms_up_then_decays():
    total, warm = 1000, 100
    assert lr_factor(0, total, warm) == 0.0            # starts at zero
    assert abs(lr_factor(warm, total, warm) - 1.0) < 1e-6   # peaks at end of warmup
    assert lr_factor(total, total, warm) < 0.01        # cosine decays to ~0
    # monotonic climb through warmup, monotonic fall after
    assert lr_factor(50, total, warm) < lr_factor(90, total, warm)
    assert lr_factor(200, total, warm) > lr_factor(800, total, warm)


def test_optimizer_exempts_bn_and_bias_from_decay():
    model = YOLOX(num_classes=1)
    opt = build_optimizer(model, lr=0.01, weight_decay=5e-4)
    assert len(opt.param_groups) == 2
    decay_group = [g for g in opt.param_groups if g["weight_decay"] > 0][0]
    nodecay_group = [g for g in opt.param_groups if g["weight_decay"] == 0][0]
    assert len(decay_group["params"]) > 0 and len(nodecay_group["params"]) > 0
    # every no-decay param is 1-D (BN scale/shift or bias); every decay param is a kernel
    assert all(p.ndim <= 1 for p in nodecay_group["params"])
    assert all(p.ndim >= 2 for p in decay_group["params"])
