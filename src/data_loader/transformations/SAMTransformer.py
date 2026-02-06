from torchvision import transforms
from torchvision.transforms import functional as TF


class SegTransform:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            ),
        ])

    def apply(self, image=None, mask=None, do_flip=False):
        if image is not None:
            if do_flip:
                image = TF.hflip(image)
            image = self.image_transform(image)

        if mask is not None:
            if do_flip:
                mask = TF.hflip(mask)
            mask = self._mask_to_tensor(mask)

        return image, mask

    def _mask_to_tensor(self, mask):
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor
