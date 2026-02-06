import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

class ClassifierTransformer:
    def __init__(self, backbone="dino3", img_size=1008):
        self.img_size = img_size
        
        if backbone.lower() == "dino3":
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
        elif backbone.lower() == "clip":
            self.mean = [0.48145466, 0.4578275, 0.40821073]
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:
            raise ValueError("Unsupported backbone")
        
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.image_transform(image)
        else:
            is_batched = image.dim() == 4
            if not is_batched:
                image = image.unsqueeze(0)  # [1,3,H,W]
            image = TF.resize(image, (self.img_size, self.img_size))
            image = TF.normalize(image, mean=self.mean, std=self.std)
            if not is_batched:
                image = image.squeeze(0)

        return image