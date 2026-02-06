import torch.nn as nn

class PresenceHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, Q):  # [B, K, D]
        return self.mlp(Q)  # [B, K, 1]