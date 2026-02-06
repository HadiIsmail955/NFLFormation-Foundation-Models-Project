import torch
import torch.nn as nn
from src.utils.mask_applier import apply_mask
from src.data_loader.transformations.ClassifierTransformer import ClassifierTransformer


class SAMDINOClassifier(nn.Module):
    def __init__(
        self,
        sam_model,
        dino_classifier,
        mask_mode="soft",
    ):
        super().__init__()

        self.sam = sam_model
        self.sam.eval()
        self.classifier = dino_classifier
        self.mask_mode = mask_mode
        self.classifier_transform = ClassifierTransformer()

    def forward(self, image):
        with torch.no_grad():
            sam_logits = self.sam(image)
            mask = torch.sigmoid(sam_logits)
        masked_image = apply_mask(image, mask, self.mask_mode)
        masked_image = self.classifier_transform(masked_image)
        return self.classifier(masked_image)
