from torch import nn

class FormationHead(nn.Module):
    def __init__(self, d_model=256, n_classes=20):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, tokens, valid_mask=None):
        pad_mask = None
        if valid_mask is not None:
            pad_mask = ~valid_mask

        z = self.encoder(tokens, src_key_padding_mask=pad_mask)
        pooled = z.mean(dim=1) 
        return self.cls(pooled)
