"""Diffusion Transformer (DiT) for velocity/noise prediction.

Vision Transformer with AdaLN-Zero conditioning, following Peebles & Xie (2023).
Patchifies 128x128 input into tokens, processes with transformer blocks,
then unpatchifies back to spatial output.
"""

import math

import torch
import torch.nn as nn


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


class TimestepEmbedder(nn.Module):
    """Maps scalar timestep to embedding vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t * 1000.0)


class PatchEmbed(nn.Module):
    """Convert 2D image into patch tokens via convolution."""

    def __init__(self, patch_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim)
        return x


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    AdaLN-Zero: time embedding produces (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
    where gamma/beta modulate LayerNorm and alpha gates the residual addition.
    All alpha params initialized to zero so each block starts as identity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        last_layer = self.adaLN_modulation[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + alpha1.unsqueeze(1) * h

        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + alpha2.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection to unpatchify."""

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        last_layer = self.adaLN_modulation[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb)
        gamma, beta = mod.chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h = self.linear(h)
        return h


class DiT(nn.Module):
    """Diffusion Transformer for velocity prediction.

    Input: [noisy_state, condition] concatenated (2 ch) at 128x128
    Output: predicted velocity (1 ch) at 128x128

    Args:
        img_size: Input spatial resolution
        patch_size: Patch size for tokenization
        in_channels: Total input channels (noisy + condition)
        out_channels: Output channels
        hidden_dim: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 2,
        out_channels: int = 1,
        hidden_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = TimestepEmbedder(hidden_dim)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, out_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to image: (B, N, ps^2*C) -> (B, C, H, W)."""
        ps = self.patch_size
        gs = self.grid_size
        c = self.out_channels
        x = x.reshape(-1, gs, gs, ps, ps, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(-1, c, gs * ps, gs * ps)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, condition], dim=1)
        x = self.patch_embed(x) + self.pos_embed
        t_emb = self.time_embed(t)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x)
        return x
