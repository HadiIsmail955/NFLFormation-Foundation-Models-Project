import torch.nn as nn

class RoleHead(nn.Module):
    def __init__(self, d_model=256, num_roles=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, num_roles)
        )

    def forward(self, Q):  # [B, K, D]
        return self.mlp(Q)  # [B, K, R]