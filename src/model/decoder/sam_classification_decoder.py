import torch
import torch.nn as nn


class SAMClassificationDecoder(nn.Module):
    def __init__(
        self,
        sam,
        k_layers: int = 3,
        unfreeze_last_k: int = 1,
    ):
        super().__init__()

        self.sam = sam
        self.k_layers = k_layers
        self.unfreeze_last_k = unfreeze_last_k

        self.blocks = nn.ModuleList(
            sam.mask_decoder.transformer.layers[:k_layers]
        )

        self._freeze_all()
        self._unfreeze_last_k()

    def _freeze_all(self):
        for p in self.blocks.parameters():
            p.requires_grad = False

    def _unfreeze_last_k(self):
        if self.unfreeze_last_k <= 0:
            return

        assert self.unfreeze_last_k <= len(self.blocks), (
            f"unfreeze_last_k={self.unfreeze_last_k} "
            f"exceeds number of blocks={len(self.blocks)}"
        )

        for blk in self.blocks[-self.unfreeze_last_k:]:
            for p in blk.parameters():
                p.requires_grad = True

    def forward(self, queries, keys, key_pe):
        for blk in self.blocks:
            query_pe = torch.zeros_like(queries)
            queries, keys = blk(
                queries=queries,
                keys=keys,
                query_pe=query_pe,
                key_pe=key_pe,
            )
        return queries
