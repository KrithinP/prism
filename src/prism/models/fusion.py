"""Fusion operators: how two feature maps (one per camera) become one.

Every operator shares the signature ``fuse(rgb_feat, thermal_feat) -> fused_feat``
with the fused map the same shape as either input (TODO.md Phase 5.1 operator
ladder). Each operator documents its normalization, activation, initialization,
output width, residual path, and added capacity so operators are compared fairly
(Phase 5.2): same graph, same output width, matched normalization, capacity
reported.

For every learned operator, ``params_and_flops`` reports the added parameter and
FLOP count so the table can carry the capacity column.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from prism.models.blocks import ConvBNAct


class _ParamsFlopsMixin:
    """Added-capacity accounting: params and FLOPs per 640x512 input pair.

    FLOPs are counted on a per-level basis by the caller (which knows H,W); here we
    only expose parameter counts, which are shape-independent.
    """

    def params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def params_and_flops(self, h: int, w: int) -> tuple[int, int]:
        """(added params, added MACs at feature size h x w)."""
        flops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                flops += (
                    m.weight.shape[0]
                    * m.weight.shape[1]
                    * m.weight.shape[2]
                    * m.weight.shape[3]
                    * h
                    * w
                )
        return self.params(), flops


class ConcatFuse(_ParamsFlopsMixin, nn.Module):
    """Concatenate the two feature maps (2*C channels) and project back to C with a
    learned 1x1 conv. The primary learned linear-mixing anchor of the study.

    - normalization/activation: supplied by the common FusionOperatorBlock
    - initialization: default Kaiming convolution initialization
    - output width: C
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, rgb_feat, thermal_feat):
        return self.proj(torch.cat([rgb_feat, thermal_feat], dim=1))


class AttentionParamMatchedConcatFuse(_ParamsFlopsMixin, nn.Module):
    """Concat control with exactly the 4*C^2 weights of cross-attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.mix1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.mix2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, rgb_feat, thermal_feat):
        fused = self.proj(torch.cat([rgb_feat, thermal_feat], dim=1))
        return self.mix2(self.act(self.mix1(fused)))


class AlignmentParamMatchedConcatFuse(_ParamsFlopsMixin, nn.Module):
    """Spatial concat control with exactly the parameters of learned alignment."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.mix1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.mix2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.modality_scale = nn.Parameter(torch.ones(2))
        self.channel_scale = nn.Parameter(torch.ones(channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, rgb_feat, thermal_feat):
        inputs = torch.cat(
            [
                self.modality_scale[0] * rgb_feat,
                self.modality_scale[1] * thermal_feat,
            ],
            dim=1,
        )
        fused = self.proj(inputs)
        return self.mix2(self.act(self.mix1(fused))) * self.channel_scale.view(1, -1, 1, 1)


class SumFuse(nn.Module):
    """Zero-parameter element-wise sum. The honest floor operator."""

    def forward(self, rgb_feat, thermal_feat):
        return rgb_feat + thermal_feat

    def params_and_flops(self, h, w):
        return 0, 0


class AverageFuse(SumFuse):
    """Operator 1 of the ladder: equal-trust (r+t)/2, zero parameters."""

    def forward(self, rgb_feat, thermal_feat):
        return (rgb_feat + thermal_feat) / 2.0


class ScalarMixtureFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 3: one global scalar alpha blends the two cameras.

    alpha = sigmoid(a), a initialized to 0 so alpha starts at exactly 0.5
    (equal trust). Log its learned value -- a converged alpha of 0 or 1 means
    the model learned to discard one camera entirely.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.channels = channels

    def alpha_value(self):
        return torch.sigmoid(self.a).item()

    def forward(self, rgb_feat, thermal_feat):
        alpha = torch.sigmoid(self.a)
        return alpha * rgb_feat + (1 - alpha) * thermal_feat

    def params_and_flops(self, h, w):
        return self.params(), 0


class ChannelMixtureFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 4: per-channel mixing weights from a GAP of the concatenation.

    alpha_c = sigmoid(MLP(GAP([r;t]))) in (B, C, 1, 1); Z = alpha_c*r + (1-alpha_c)*t.
    Init bias of the last linear to 0 (equal trust), hidden weights to a small
    value so gates do not saturate at initialization (Phase 5.2).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.channels = channels

    def forward(self, rgb_feat, thermal_feat):
        g = torch.cat([rgb_feat, thermal_feat], dim=1)
        g = g.mean(dim=(2, 3))  # GAP -> (B, 2C)
        alpha = torch.sigmoid(self.mlp(g)).unsqueeze(-1).unsqueeze(-1)
        self.last_alpha = alpha.detach()
        return alpha * rgb_feat + (1 - alpha) * thermal_feat


class QualityGatedFuse(_ParamsFlopsMixin, nn.Module):
    """Per-image scalar trust gate estimated from both modality feature statistics.

    This is a controlled illumination/quality-conditioned arm, not a faithful
    reimplementation of MBNet's illumination-aware feature alignment. The gate
    sees global mean and standard deviation from each modality and decides one
    scalar alpha per image:

        Z = alpha(x_r, x_t) * r + (1-alpha(x_r, x_t)) * t.

    It separates *sample-adaptive modality trust* from the global scalar arm and
    the higher-capacity per-channel arm. Initialization is exactly equal trust.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(4 * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_alpha = None

    def forward(self, rgb_feat, thermal_feat):
        stats = torch.cat(
            [
                rgb_feat.mean(dim=(2, 3)),
                rgb_feat.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt(),
                thermal_feat.mean(dim=(2, 3)),
                thermal_feat.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt(),
            ],
            dim=1,
        )
        alpha = torch.sigmoid(self.mlp(stats)).view(-1, 1, 1, 1)
        self.last_alpha = alpha.detach()
        return alpha * rgb_feat + (1 - alpha) * thermal_feat


class GatedResidualFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 5: spatial/channel gate over one stream, residual into the other.

    g = sigmoid(G([r;t])); Z = r + g*t. The anchor argument controls which
    stream is the base (rgb-anchored = r + g*t, thermal-anchored = t + g*r).
    """

    def __init__(self, channels: int, anchor: str = "rgb"):
        super().__init__()
        self.anchor = anchor
        self.gate = nn.Sequential(
            ConvBNAct(2 * channels, channels, k=3),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.channels = channels

    def forward(self, rgb_feat, thermal_feat):
        g = torch.sigmoid(self.gate(torch.cat([rgb_feat, thermal_feat], dim=1)))
        self.last_gate = g.detach()
        if self.anchor == "rgb":
            return rgb_feat + g * thermal_feat
        return thermal_feat + g * rgb_feat


class NestedResidualFuse(_ParamsFlopsMixin, nn.Module):
    """Nested-at-initialization control: Z = base + alpha * complement.

    ``alpha`` is initialized to zero and is not passed through a sigmoid, so the
    fusion hypothesis class literally contains the anchor-only feature function.
    This arm intentionally bypasses the common post-normalization wrapper.
    Joint SGD may still move the anchor branch and does not guarantee preservation
    of an independently trained unimodal model. It is a first-principles control,
    not a capacity-matched member of the Block C operator ranking.
    """

    def __init__(self, channels: int, anchor: str = "thermal"):
        super().__init__()
        self.anchor = anchor
        self.alpha = nn.Parameter(torch.zeros(1))
        self.last_alpha = None

    def forward(self, rgb_feat, thermal_feat):
        base, complement = (
            (rgb_feat, thermal_feat) if self.anchor == "rgb" else (thermal_feat, rgb_feat)
        )
        self.last_alpha = self.alpha.detach().reshape(1, 1, 1, 1)
        return base + self.alpha * complement

    def params_and_flops(self, h, w):
        return self.params(), 0


class _ChannelAttention(nn.Module):
    """SE-style channel attention over C*2 channels (avg+max pooled)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels * 2 // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        return torch.sigmoid(self.mlp(avg + mx)).view(b, c, 1, 1)


class GAFFFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 6: GAFF/CBAM-style guided channel + spatial attention.

    Channel attention over the concatenated map, then a spatial gate from
    avg/max channel pooling, then a 1x1 projection back to C. Masks are
    returned for logging (never as causal explanations -- TODO Phase 9).

    - channel mask: (B, 2C, 1, 1)
    - spatial mask: (B, 1, H, W)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ch_att = _ChannelAttention(channels, reduction)
        self.sp_att = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.last_masks = None

    def forward(self, rgb_feat, thermal_feat):
        f = torch.cat([rgb_feat, thermal_feat], dim=1)
        ch_mask = self.ch_att(f)
        f = f * ch_mask
        avg = f.mean(dim=1, keepdim=True)
        mx = f.amax(dim=1, keepdim=True)
        spatial = self.sp_att(torch.cat([avg, mx], dim=1))
        f = f * spatial
        self.last_masks = (ch_mask, spatial)
        return self.proj(f)


class DMAFFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 7: MBNet/DMAF-style differential fusion.

    Channel reweighting estimated from the difference |r-t| (the modality
    imbalance signal), applied multiplicatively to the concatenated projection.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.att = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.att[-1].weight)
        nn.init.zeros_(self.att[-1].bias)
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, rgb_feat, thermal_feat):
        d = (rgb_feat - thermal_feat).abs().mean(dim=(2, 3))  # (B, C)
        w = torch.sigmoid(self.att(d)).unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1)
        self.last_weights = w.detach()
        f = torch.cat([rgb_feat, thermal_feat], dim=1) * w
        return self.proj(f)


class AlignmentAwareFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 8: predict a 2-D offset field and warp the complementary feature.

    Predict offsets for the thermal map relative to RGB, sample it with
    grid_sample, then concat-project. Regularization/offset logging:
    ``mean_offset()`` reports the mean |offset| in pixels -- if the data is
    aligned this should stay near zero.
    """

    def __init__(self, channels: int, anchor: str = "thermal"):
        super().__init__()
        self.anchor = anchor
        self.offset_net = nn.Sequential(
            ConvBNAct(2 * channels, channels, k=3),
            nn.Conv2d(channels, 2, 1),
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self._last_offset = None

    def mean_offset(self):
        return float(self._last_offset) if self._last_offset is not None else float("nan")

    def forward(self, rgb_feat, thermal_feat):
        base, complement = (
            (rgb_feat, thermal_feat) if self.anchor == "rgb" else (thermal_feat, rgb_feat)
        )
        b, _c, h, w = base.shape
        offset = self.offset_net(torch.cat([rgb_feat, thermal_feat], dim=1))  # (B,2,H,W)
        self._last_offset = offset.detach().abs().mean().item()
        self.last_offset_per_sample = offset.detach().abs().mean(dim=(1, 2, 3))
        # grid_sample: pixel coords = grid + offset (normalized to [-1,1])
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=rgb_feat.device),
            torch.linspace(-1, 1, w, device=rgb_feat.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).expand(b, -1, -1, 2).clone()
        # align_corners=True maps the first/last pixel centres to -1/+1, so one
        # pixel is exactly 2/(size-1) in normalized coordinates.
        # No detach: the offset field must receive gradient so the alignment
        # can actually learn (it was detached here once -- the operator could
        # never move its own offsets, TODO.md blocker #9).
        grid[..., 0] += offset[:, 0].tanh() * (2.0 / max(w - 1, 1))
        grid[..., 1] += offset[:, 1].tanh() * (2.0 / max(h - 1, 1))
        warped = F.grid_sample(complement, grid, mode="bilinear", align_corners=True)
        return self.proj(torch.cat([base, warped], dim=1))


class CrossAttentionFuse(_ParamsFlopsMixin, nn.Module):
    """Operator 9: cross-attention ceiling. Q from r, K/V from t.

    z = r + softmax(QK^T/sqrt(d))V. Queries retain the full feature resolution;
    keys/values are adaptively pooled to at most ``max_kv_tokens``. This makes
    the operator defined at dark3/4/5 without materializing a prohibitive
    (H*W)^2 matrix at dark3.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        max_kv_tokens: int = 256,
        anchor: str = "thermal",
    ):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels {channels} must be divisible by heads {heads}")
        self.channels = channels
        self.heads = heads
        self.max_kv_tokens = max_kv_tokens
        self.anchor = anchor
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = (channels // heads) ** -0.5

    def forward(self, rgb_feat, thermal_feat):
        query, kv = (rgb_feat, thermal_feat) if self.anchor == "rgb" else (thermal_feat, rgb_feat)
        b, c, h, w = query.shape
        if h * w > self.max_kv_tokens:
            # Preserve aspect ratio while bounding the attention matrix.
            scale = (self.max_kv_tokens / (h * w)) ** 0.5
            kh = max(1, int(h * scale))
            kw = max(1, min(w, self.max_kv_tokens // kh))
            kv = F.adaptive_avg_pool2d(kv, (kh, kw))
        kh, kw = kv.shape[-2:]
        q = self.q(query).view(b, self.heads, c // self.heads, h * w)
        k = self.k(kv).view(b, self.heads, c // self.heads, kh * kw)
        v = self.v(kv).view(b, self.heads, c // self.heads, kh * kw)
        attn = torch.softmax(q.transpose(2, 3) @ k * self.scale, dim=-1)
        normalizer = math.log(max(kh * kw, 2))
        self.last_attention_entropy = (
            -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=(1, 2))
            / normalizer
        ).detach()
        self.last_attention_peak = attn.amax(dim=-1).mean(dim=(1, 2)).detach()
        z = (attn @ v.transpose(2, 3)).transpose(2, 3).reshape(b, c, h, w)
        return query + self.out(z)


class FusionOperatorBlock(_ParamsFlopsMixin, nn.Module):
    """Common output contract for every operator in the comparison ladder.

    All operators produce C channels and then pass through the same YOLOX-style
    BatchNorm and SiLU. This removes a prior confound where projection-based
    operators received normalization/activation but additive operators did not.
    The underlying mechanism remains available as ``.operator`` for diagnostics.
    """

    def __init__(self, operator: nn.Module, channels: int):
        super().__init__()
        self.operator = operator
        self.norm = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, rgb_feat, thermal_feat):
        return self.act(self.norm(self.operator(rgb_feat, thermal_feat)))


class AdapterProjection(nn.Module):
    """Single-modality adapter A_l: projects a backbone level to the neck's width.

    In the current base-ch=32 architecture the backbone levels already match the
    neck's expected channels, so the adapter is the identity. The class exists so
    the graph is explicit and a width mismatch is caught loudly rather than by a
    shape error at runtime.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.identity = in_ch == out_ch
        self.proj = nn.Identity() if self.identity else ConvBNAct(in_ch, out_ch, k=1)

    def forward(self, x):
        return self.proj(x)


# name -> operator class, so experiments select the operator by string.
OPERATORS = {
    "concat": ConcatFuse,
    "concat-attention-params": AttentionParamMatchedConcatFuse,
    "concat-alignment-params": AlignmentParamMatchedConcatFuse,
    "sum": SumFuse,
    "average": AverageFuse,
    "scalar-mixture": ScalarMixtureFuse,
    "channel-mixture": ChannelMixtureFuse,
    "quality-gated": QualityGatedFuse,
    "gated-residual": GatedResidualFuse,
    "nested-residual": NestedResidualFuse,
    "gaff": GAFFFuse,
    "dmaf": DMAFFuse,
    "alignment": AlignmentAwareFuse,
    "cross-attention": CrossAttentionFuse,
}


def build_fusion_operator(name: str, channels: int, anchor: str = "thermal") -> nn.Module:
    """Construct an operator with the graph's declared anchor semantics.

    Most operators are symmetric or define their own RGB-query convention.
    Gated residual fusion is explicitly anchored, so it must receive the graph
    anchor rather than silently defaulting to RGB.
    """
    try:
        operator = OPERATORS[name]
    except KeyError as e:
        raise ValueError(f"unknown fusion operator {name!r}") from e
    if name in ("sum", "average"):
        instance = operator()
    elif name in ("gated-residual", "nested-residual", "alignment", "cross-attention"):
        instance = operator(channels, anchor=anchor)
    else:
        instance = operator(channels)
    if name == "nested-residual":
        return instance
    return FusionOperatorBlock(instance, channels)
