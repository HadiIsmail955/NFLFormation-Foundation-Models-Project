import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class FormationHead(nn.Module):
    def __init__(self, in_channels: int, num_formations: int, dropout: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),  # [B,128,1,1]
            nn.Flatten(),             # [B,128]

            nn.Dropout(dropout),
            nn.Linear(128, num_formations),
        )

    def forward(self, x):
        return self.net(x)