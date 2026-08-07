"""Focused invariants for the corrected fusion operators and width handling."""

import torch

from prism.models.early_fusion import EarlyFusionYOLOX
from prism.models.fusion import (
    AlignmentAwareFuse,
    AlignmentParamMatchedConcatFuse,
    AttentionParamMatchedConcatFuse,
    CrossAttentionFuse,
    FusionOperatorBlock,
    GatedResidualFuse,
    NestedResidualFuse,
    QualityGatedFuse,
    build_fusion_operator,
)
from prism.models.multi_scale_fusion import MultiScaleFusionYOLOX
from prism.models.one_shot_fusion import OneShotFusionYOLOX


def test_nondefault_base_width_is_supported_by_every_graph():
    rgb = torch.randn(1, 3, 64, 96)
    thermal = torch.randn(1, 3, 64, 96)
    for model in (
        EarlyFusionYOLOX(base_ch=12),
        OneShotFusionYOLOX(base_ch=12, meet_level=4),
        MultiScaleFusionYOLOX(base_ch=12, fuse_subset=(3, 5)),
    ):
        outputs = model.eval()(rgb, thermal)
        assert [output.shape[1] for output in outputs] == [6, 6, 6]


def test_graph_tables_report_real_backbone_strides():
    model = OneShotFusionYOLOX(base_ch=12, meet_level=4)
    table = model.graph_table()
    assert "stride=8" in table
    assert "stride=16" in table
    assert "stride=32" in table
    assert "C=48" in table and "C=96" in table and "C=192" in table


def test_gated_residual_uses_graph_anchor():
    model = MultiScaleFusionYOLOX(
        base_ch=8, fuse_subset=(4,), anchor="thermal", operator="gated-residual"
    )
    fuser = model.fusers["4"]
    assert isinstance(fuser, FusionOperatorBlock)
    assert isinstance(fuser.operator, GatedResidualFuse)
    assert fuser.operator.anchor == "thermal"


def test_alignment_offsets_receive_gradient():
    module = AlignmentAwareFuse(8)
    rgb = torch.randn(2, 8, 8, 10, requires_grad=True)
    thermal = torch.randn(2, 8, 8, 10, requires_grad=True)
    module(rgb, thermal).square().mean().backward()
    gradient = module.offset_net[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_cross_attention_bounds_key_value_tokens_and_preserves_shape():
    module = CrossAttentionFuse(16, heads=4, max_kv_tokens=32)
    rgb = torch.randn(1, 16, 32, 40)
    thermal = torch.randn(1, 16, 32, 40)
    output = module(rgb, thermal)
    assert output.shape == rgb.shape


def test_anchor_sensitive_operators_receive_declared_anchor():
    for name, cls in (
        ("gated-residual", GatedResidualFuse),
        ("alignment", AlignmentAwareFuse),
        ("cross-attention", CrossAttentionFuse),
    ):
        block = build_fusion_operator(name, 16, anchor="rgb")
        assert isinstance(block.operator, cls)
        assert block.operator.anchor == "rgb"


def test_every_factory_operator_has_identical_post_normalization():
    for name in (
        "average",
        "concat",
        "scalar-mixture",
        "quality-gated",
        "channel-mixture",
        "gated-residual",
        "gaff",
        "dmaf",
        "alignment",
        "cross-attention",
        "concat-attention-params",
        "concat-alignment-params",
    ):
        block = build_fusion_operator(name, 16)
        assert isinstance(block, FusionOperatorBlock)
        assert block.norm.eps == 1e-3
        assert block.norm.momentum == 0.03


def test_quality_gate_starts_at_equal_trust_per_image():
    module = QualityGatedFuse(8)
    output = module(torch.zeros(2, 8, 4, 5), torch.ones(2, 8, 4, 5))
    assert torch.allclose(output, torch.full_like(output, 0.5))
    assert torch.allclose(module.last_alpha, torch.full((2, 1, 1, 1), 0.5))


def test_nested_residual_is_exact_anchor_at_initialization():
    rgb = torch.randn(2, 8, 4, 5)
    thermal = torch.randn(2, 8, 4, 5)
    module = build_fusion_operator("nested-residual", 8, anchor="rgb")
    assert isinstance(module, NestedResidualFuse)
    assert torch.equal(module(rgb, thermal), rgb)
    module = build_fusion_operator("nested-residual", 8, anchor="thermal")
    assert torch.equal(module(rgb, thermal), thermal)


def test_expensive_operators_have_exact_parameter_matched_concat_controls():
    channels = 16
    attention = CrossAttentionFuse(channels)
    attention_control = AttentionParamMatchedConcatFuse(channels)
    alignment = AlignmentAwareFuse(channels)
    alignment_control = AlignmentParamMatchedConcatFuse(channels)
    assert attention.params() == attention_control.params() == 4 * channels**2
    assert alignment.params() == alignment_control.params() == (20 * channels**2 + 4 * channels + 2)
