import torch.nn as nn
from segment_anything import sam_model_registry

class SAMBackbone(nn.Module):
    def __init__(self, sam, unfreeze_last_blocks=0):
        super().__init__()
        self.encoder = sam.image_encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_last_blocks > 0:
            assert unfreeze_last_blocks <= len(self.encoder.blocks), (
                f"unfreeze_last_blocks={unfreeze_last_blocks} "
                f"exceeds number of encoder blocks={len(self.encoder.blocks)}"
            )

            for block in self.encoder.blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

                for m in block.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True

    def forward(self, x): # x: [B, 3, 1024, 1024]
        features = self.encoder(x)  
        return features # return: [B, 256, 64, 64]
