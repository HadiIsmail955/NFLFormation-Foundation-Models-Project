import torch
import torch.nn as nn

class QueryInstanceDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1, num_queries=11):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory_tokens):
        B = memory_tokens.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        Q = self.decoder(tgt=queries, memory=memory_tokens)               # [B, K, D]
        return Q