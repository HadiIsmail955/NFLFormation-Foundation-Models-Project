import torch.nn as nn
import torch.nn.functional as F

class SAMMaskDecoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder

        for p in self.mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, image_embeddings, sparse_emb, dense_emb, out_size,multimask_output=False):
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
        )

        return F.interpolate(
            low_res_masks,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )