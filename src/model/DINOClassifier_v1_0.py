import torch.nn as nn
from .backbone.dino_backbone import DINOBackbone
from .head.classification_head import ClassificationHead


class DINOClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        dino_type="vit_b",
        unfreeze_last_blocks=0,
        dropout=0.1,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = DINOBackbone(
            dino_type=dino_type,
            unfreeze_last_blocks=unfreeze_last_blocks,
            freeze=freeze_backbone,
        )

        self.classifier = ClassificationHead(
            in_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        feats = self.backbone(x)          # [B, C]
        logits = self.classifier(feats)   # [B, num_classes]
        return logits
