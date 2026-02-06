import torch
import numpy as np
import cv2
import random
import torch.nn.functional as F

def apply_mask(image, mask, mode="soft"):

    if mode == "hard":
        return image * (mask > 0.5)

    elif mode == "soft":
        return image * (0.7 + 0.3 * mask)


    elif mode == "background_gray":
        gray = image.mean(dim=1, keepdim=True)
        return image * mask + gray * (1 - mask)

    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    
def compute_centers_from_masks(mask_logits, H, W, eps=1e-6):
    B, K, _, _ = mask_logits.shape
    probs = mask_logits.sigmoid()

    ys = torch.linspace(0, 1, H, device=mask_logits.device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=mask_logits.device).view(1, 1, 1, W)

    mass = probs.sum(dim=(2,3), keepdim=True) + eps
    x = (probs * xs).sum(dim=(2,3), keepdim=True) / mass
    y = (probs * ys).sum(dim=(2,3), keepdim=True) / mass

    centers = torch.cat([x, y], dim=-1).squeeze(2).squeeze(2)  # [B,K,2]
    return centers

def visualize_instances(image_np, masks, alpha=0.5, draw_centroid=True):
    vis = image_np.copy()

    for m in masks:
        mask = m["segmentation"]
        color = np.array([
            random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255),
        ])

        vis[mask] = (
            (1 - alpha) * vis[mask] + alpha * color
        ).astype(np.uint8)

        if draw_centroid:
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.circle(vis, (cx, cy), 4, (255,255,255), -1)

    return vis

def blur_outside_sharpen_inside_mask(image, mask, strength=1.5,blur_kernel=7):
    blur = F.avg_pool2d(
        image,
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2,
    )
    sharp = image + strength * (image - blur)
    out = sharp * mask + image * (1 - mask)
    return out