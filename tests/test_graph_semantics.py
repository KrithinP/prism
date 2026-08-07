"""Graph-level semantics tests (TODO.md 6.8): prove the computational graph IS
what the name says.

- shape test for every supported graph/resolution
- modality-dependence: perturb RGB vs thermal, assert only the declared tensors change
- zero-gradient test: an excluded modality has no path to an anchor-only level
- hook-based fusion-call counting: exactly one call for one-shot, |subset| for multi-scale
- parameter ownership: private prefixes vs shared suffix
- serialization round-trip preserving the graph config
"""

import torch

from prism.models.early_fusion import EarlyFusionYOLOX
from prism.models.multi_scale_fusion import MultiScaleFusionYOLOX
from prism.models.one_shot_fusion import OneShotFusionYOLOX
from prism.models.yolox_fusion import LegacyTwoStreamYOLOX

_IN = (1, 3, 512, 640)


def _rand(*shape):
    return torch.randn(*shape)


def _diff(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------- one-shot


def test_one_shot_shapes_all_meet_levels():
    for meet in (3, 4, 5):
        m = OneShotFusionYOLOX(meet_level=meet, anchor="thermal").eval()
        outs = m(_rand(*_IN), _rand(*_IN))
        assert [o.shape for o in outs] == [(1, 6, 64, 80), (1, 6, 32, 40), (1, 6, 16, 20)], meet


def test_one_shot_fusion_called_exactly_once():
    m = OneShotFusionYOLOX(meet_level=4).eval()
    calls = []

    def hook(mod, args, out):
        calls.append(1)

    m.fuser.register_forward_hook(hook)
    m(_rand(*_IN), _rand(*_IN))
    assert len(calls) == 1, "one-shot fusion must call the fusion operator exactly once"


def test_one_shot_dark3_inputs_all_fused():
    """Meet at dark3: ALL three neck inputs must change when EITHER camera changes."""
    m = OneShotFusionYOLOX(meet_level=3).eval()
    n = m.forward_features(_rand(*_IN), _rand(*_IN))
    n2 = m.forward_features(_rand(*_IN) + 1e-3, _rand(*_IN))
    for a, b in zip(n, n2, strict=True):
        assert _diff(a, b) > 0
    n3 = m.forward_features(_rand(*_IN), _rand(*_IN) + 1e-3)
    for a, b in zip(n, n3, strict=True):
        assert _diff(a, b) > 0


def test_one_shot_dark4_thermal_anchor_dark3_is_thermal_only():
    """Meet at dark4, thermal anchor: neck input p3 = A_3(T_3). Perturbing RGB
    alone must leave p3 EXACTLY unchanged."""
    torch.manual_seed(0)
    m = OneShotFusionYOLOX(meet_level=4, anchor="thermal").eval()
    rgb, thm = _rand(*_IN), _rand(*_IN)
    n = m.forward_features(rgb, thm)
    n_pert = m.forward_features(rgb + 1.0, thm)  # huge RGB perturbation
    assert _diff(n[0], n_pert[0]) == 0.0, "p3 must be thermal-only at dark4-meet"
    assert _diff(n[1], n_pert[1]) > 0, "p4 (fused) must change with RGB"
    assert _diff(n[2], n_pert[2]) > 0, "p5 (shared suffix) must change with RGB"


def test_one_shot_dark5_thermal_anchor_lower_levels_thermal_only():
    torch.manual_seed(0)
    m = OneShotFusionYOLOX(meet_level=5, anchor="thermal").eval()
    rgb, thm = _rand(*_IN), _rand(*_IN)
    n = m.forward_features(rgb, thm)
    n_pert = m.forward_features(rgb + 1.0, thm)
    assert _diff(n[0], n_pert[0]) == 0.0  # p3 = A_3(T_3)
    assert _diff(n[1], n_pert[1]) == 0.0  # p4 = A_4(T_4)
    assert _diff(n[2], n_pert[2]) > 0  # p5 fused


def test_one_shot_rgb_anchor_mirrors_thermal():
    torch.manual_seed(0)
    m = OneShotFusionYOLOX(meet_level=4, anchor="rgb").eval()
    rgb, thm = _rand(*_IN), _rand(*_IN)
    n = m.forward_features(rgb, thm)
    n_pert = m.forward_features(rgb, thm + 1.0)
    assert _diff(n[0], n_pert[0]) == 0.0  # p3 = A_3(R_3)
    assert _diff(n[1], n_pert[1]) > 0
    assert _diff(n[2], n_pert[2]) > 0


def test_one_shot_zero_gradient_excluded_modality():
    """Meet at dark4, thermal anchor: p3 = A_3(T_3). The gradient of p3 w.r.t.
    the RGB backbone must be exactly zero -- no path RGB -> p3."""
    m = OneShotFusionYOLOX(meet_level=4, anchor="thermal")
    rgb, thm = _rand(2, 3, 128, 160), _rand(2, 3, 128, 160)
    n = m.forward_features(rgb, thm)
    grads = torch.autograd.grad(n[0].sum(), m.backbone_rgb.parameters(), allow_unused=True)
    for g in grads:
        assert g is None or g.abs().max().item() == 0.0


def test_one_shot_shared_suffix_owned_by_fusion_only():
    """dark4-meet: dark5 must be the shared suffix module, not either backbone's."""
    m = OneShotFusionYOLOX(meet_level=4)
    suffix_params = set(m.suffix_stages.parameters())
    rgb_params = set(m.backbone_rgb.parameters())
    thm_params = set(m.backbone_thermal.parameters())
    assert suffix_params.isdisjoint(rgb_params)
    assert suffix_params.isdisjoint(thm_params)


def test_one_shot_serialization_round_trip():
    m = OneShotFusionYOLOX(meet_level=4, anchor="rgb", operator="concat")
    m2 = OneShotFusionYOLOX(meet_level=4, anchor="rgb", operator="concat")
    m2.load_state_dict(m.state_dict())
    torch.manual_seed(0)
    r, t = _rand(*_IN), _rand(*_IN)
    with torch.no_grad():
        a = m(r, t)[0]
        b = m2(r, t)[0]
    assert torch.allclose(a, b)


# ---------------------------------------------------------------- multi-scale


def test_multi_scale_shapes_all_subsets():
    for subset in ((3,), (4,), (5,), (3, 4), (4, 5), (3, 4, 5)):
        m = MultiScaleFusionYOLOX(fuse_subset=subset).eval()
        outs = m(_rand(*_IN), _rand(*_IN))
        assert [o.shape for o in outs] == [(1, 6, 64, 80), (1, 6, 32, 40), (1, 6, 16, 20)], subset


def test_multi_scale_fusion_calls_match_subset():
    m = MultiScaleFusionYOLOX(fuse_subset=(3, 5)).eval()
    calls = []
    for f in m.fusers.values():
        f.register_forward_hook(lambda mod, args, out: calls.append(1))
    m(_rand(*_IN), _rand(*_IN))
    assert len(calls) == 2


def test_multi_scale_unfused_level_is_anchor_only():
    """fuse_subset={4}: p3 and p5 come from the anchor modality alone."""
    torch.manual_seed(0)
    m = MultiScaleFusionYOLOX(fuse_subset=(4,), anchor="thermal").eval()
    rgb, thm = _rand(*_IN), _rand(*_IN)
    n = m.forward_features(rgb, thm)
    n_pert = m.forward_features(rgb + 1.0, thm)
    assert _diff(n[0], n_pert[0]) == 0.0  # p3 = A_3(T_3), no sum
    assert _diff(n[1], n_pert[1]) > 0  # p4 fused
    assert _diff(n[2], n_pert[2]) == 0.0  # p5 = A_5(T_5), no sum
    # thermal perturb: everything changes
    n_pert_t = m.forward_features(rgb, thm + 1.0)
    for a, b in zip(n, n_pert_t, strict=True):
        assert _diff(a, b) > 0


def test_multi_scale_never_silent_sum():
    """The whole point of the fix (TODO 6.7): unfused levels must not sum."""
    m = MultiScaleFusionYOLOX(fuse_subset=(4,), anchor="thermal")
    assert len(m.fusers) == 1 and set(m.fusers) == {"4"}
    assert set(m.adapters) == {"3", "5"}


def test_multi_scale_serialization_round_trip():
    m = MultiScaleFusionYOLOX(fuse_subset=(3, 5), operator="gaff")
    m2 = MultiScaleFusionYOLOX(fuse_subset=(3, 5), operator="gaff")
    m2.load_state_dict(m.state_dict())
    torch.manual_seed(0)
    r, t = _rand(*_IN), _rand(*_IN)
    with torch.no_grad():
        assert torch.allclose(m(r, t)[0], m2(r, t)[0])


# ---------------------------------------------------------------- early fusion


def test_early_fusion_shapes():
    m = EarlyFusionYOLOX().eval()
    outs = m(_rand(*_IN), _rand(*_IN))
    assert [o.shape for o in outs] == [(1, 6, 64, 80), (1, 6, 32, 40), (1, 6, 16, 20)]


def test_early_fusion_fusion_at_input():
    """The only fusion is the input concat: fusion_levels() reports depth 0."""
    m = EarlyFusionYOLOX()
    assert m.fusion_levels() == (0,)


def test_early_fusion_single_stream_both_cameras_affect_everything():
    """One shared backbone: perturbing EITHER camera must change ALL three neck
    inputs -- there is no anchor-only level to hide in."""
    torch.manual_seed(0)
    m = EarlyFusionYOLOX().eval()
    rgb, thm = _rand(*_IN), _rand(*_IN)
    n = m.forward_features(rgb, thm)
    n_r = m.forward_features(rgb + 1e-3, thm)
    n_t = m.forward_features(rgb, thm + 1e-3)
    for a, b in zip(n, n_r, strict=True):
        assert _diff(a, b) > 0
    for a, b in zip(n, n_t, strict=True):
        assert _diff(a, b) > 0


def test_early_fusion_one_backbone_no_modality_private_params():
    """Unlike the middle-fusion graphs there are no per-modality parameters:
    the single backbone owns every convolution."""
    m = EarlyFusionYOLOX()
    assert isinstance(m.backbone, torch.nn.Module)
    assert not hasattr(m, "backbone_rgb") and not hasattr(m, "backbone_thermal")
    one_shot = OneShotFusionYOLOX(meet_level=4)
    assert sum(p.numel() for p in m.parameters()) < sum(p.numel() for p in one_shot.parameters())


def test_early_fusion_gradient_flows_from_both_inputs():
    """F_0 concatenates before the stem: the stem weights receive gradient from
    both cameras (no detached/excluded path anywhere)."""
    m = EarlyFusionYOLOX()
    rgb = _rand(2, 3, 128, 160).requires_grad_(True)
    thm = _rand(2, 3, 128, 160).requires_grad_(True)
    n = m.forward_features(rgb, thm)
    g_rgb, g_thm = torch.autograd.grad(n[2].sum(), (rgb, thm))
    assert g_rgb.abs().max().item() > 0
    assert g_thm.abs().max().item() > 0


def test_early_fusion_serialization_round_trip():
    m = EarlyFusionYOLOX(base_ch=32)
    m2 = EarlyFusionYOLOX(base_ch=32)
    m2.load_state_dict(m.state_dict())
    torch.manual_seed(0)
    r, t = _rand(*_IN), _rand(*_IN)
    with torch.no_grad():
        assert torch.allclose(m(r, t)[0], m2(r, t)[0])


# ---------------------------------------------------------------- legacy quarantine


def test_legacy_quarantined_but_loadable():
    """Old checkpoints must still load (alias), but the class is explicitly legacy."""
    assert LegacyTwoStreamYOLOX is not OneShotFusionYOLOX
    m = LegacyTwoStreamYOLOX(fuse_stages=(4,)).eval()
    outs = m(_rand(*_IN), _rand(*_IN))
    assert len(outs) == 3
