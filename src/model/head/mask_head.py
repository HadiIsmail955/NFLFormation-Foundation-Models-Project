import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductMaskHead(nn.Module):
    def __init__(self, in_channels=256, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

    def forward(self, feat_map, queries, out_hw):
        Fmap = self.proj(feat_map)  
        B, D, Hs, Ws = Fmap.shape
        K = queries.shape[1]

        tokens = Fmap.flatten(2).transpose(1, 2)  

        mask_logits_small = torch.einsum("bkd,bnd->bkn", queries, tokens)

        mask_logits_small = mask_logits_small.view(B, K, Hs, Ws)

        mask_logits = F.interpolate(mask_logits_small, size=out_hw, mode="bilinear", align_corners=False)
        return mask_logits