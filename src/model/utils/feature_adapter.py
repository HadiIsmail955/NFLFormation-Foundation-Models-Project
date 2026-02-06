import torch.nn as nn

class FeatureAdapter(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.adapter(x)
