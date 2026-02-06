import torch.nn as nn
class SAMPromptEncoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        points=None,      # (B, N, 2)
        boxes=None,       # (B, N, 4)
        masks=None        # (B, N, H, W)
    ):
        return self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        