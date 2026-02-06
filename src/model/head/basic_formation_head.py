import torch
import torch.nn as nn

class FormationHead(nn.Module):
    def __init__(self, d_model=256, num_formations=14):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, num_formations)
        )

    def forward(self, Q, present_logits=None):
        attn_logits = self.attn(Q)

        if present_logits is not None:
            present_mask = (present_logits.sigmoid() > 0.5).float()
            attn_logits = attn_logits + (present_mask - 1) * 1e4

        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, K, 1]

        Q_pool = (attn_weights * Q).sum(dim=1)  # [B, D]

        return self.classifier(Q_pool)  # [B, num_formations]