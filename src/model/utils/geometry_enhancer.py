import torch.nn as nn
import torch


class GeometryEnhancer(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, tokens, points_xy, image_hw):
        H, W = image_hw
        xy = points_xy.clone()
        xy[..., 0] /= W
        xy[..., 1] /= H

        rel = xy[:, :, None, :] - xy[:, None, :, :]
        dist = torch.norm(rel, dim=-1)  # [B, N, N]

        geo_feat = torch.stack([
            dist.mean(dim=2),
            dist.min(dim=2).values,
            dist.max(dim=2).values
        ], dim=-1)  # [B, N, 3]

        return tokens + self.proj(geo_feat)
