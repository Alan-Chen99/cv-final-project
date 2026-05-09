"""UNet with self-attention at bottleneck for velocity prediction.

Canonical implementation from spatial-4x-flow-matching experiments.
Architecture: encoder-decoder with skip connections, time conditioning,
and multi-head self-attention at 16x16 bottleneck resolution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def _num_groups(channels: int, max_groups: int = 32) -> int:
    """Find largest divisor of channels that is <= max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2D feature maps with residual connection."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)
        out = self.proj(out).view(B, C, H, W)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttentionUNet(nn.Module):
    """UNet with self-attention at bottleneck for flow matching velocity prediction.

    Default: 2 input channels (interpolated state + LR condition), 1 output channel (velocity).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        attn_heads: int = 4,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(ch, out_ch, time_emb_dim, dropout),
                        ResBlock(out_ch, out_ch, time_emb_dim, dropout),
                    ]
                )
            )
            self.down_samples.append(Downsample(out_ch))
            ch = out_ch

        # Bottleneck with self-attention
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch, num_heads=attn_heads)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(ch + out_ch, out_ch, time_emb_dim, dropout),
                        ResBlock(out_ch, out_ch, time_emb_dim, dropout),
                    ]
                )
            )
            self.up_samples.append(Upsample(ch))
            ch = out_ch

        self.final_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(t * 1000.0)
        h = self.init_conv(x)

        skips = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples, strict=True):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks, upsample in zip(self.up_blocks, self.up_samples, strict=True):
            h = upsample(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        return self.final_conv(F.silu(self.final_norm(h)))
