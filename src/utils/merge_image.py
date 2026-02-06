import cv2
import numpy as np
import torch

def save_overlay_with_metrics(
    image,
    gt_mask,
    pred_mask,
    metrics,
    save_path,
):

    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    gt = gt_mask.squeeze().cpu().numpy()
    pred = pred_mask.squeeze().cpu().numpy()

    overlay = img.copy()
    overlay[gt > 0.5] = overlay[gt > 0.5] * 0.7 + np.array([0, 255, 0]) * 0.3
    overlay[pred > 0.5] = overlay[pred > 0.5] * 0.7 + np.array([255, 0, 0]) * 0.3

    overlay = overlay.astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    color = (255, 255, 255)

    y0 = 22
    dy = 20

    lines = [
        f"{key}: {value:.4f}" for key, value in metrics.items()
    ]

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(
            overlay,
            line,
            (10, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, overlay[..., ::-1])

def save_classification_overlay(
    image,
    gt_label,
    pred_label,
    confidence,
    save_path,
    id_to_class,
):

    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()

        img = img - img.min()
        img = img / (img.max() + 1e-6)

        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
    else:
        raise TypeError("Image must be a torch.Tensor")

    img = img[..., ::-1]

    gt_name = id_to_class.get(gt_label, "unknown")
    pred_name = id_to_class.get(pred_label, "unknown")

    correct = (gt_label == pred_label)

    color = (0, 200, 0) if correct else (0, 0, 255)

    lines = [
        f"GT:   {gt_name}",
        f"PRED: {pred_name}",
        f"CONF: {confidence:.3f}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    x, y0 = 10, 28
    dy = 24

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(
            img,
            line,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, img)

# POSITION_COLORS = {
#     0: (0, 0, 0),
#     1: (255, 128, 0),   
#     2: (255, 0, 0),     
#     3: (0, 255, 0),     
#     4: (0, 200, 255),   
#     5: (255, 255, 0),   
#     6: (255, 0, 255),   
#     7: (0, 0, 255), 
#     8: (128, 128, 128),    
# }

# POSITION_NAMES = {
#     0: "BG",
#     1: "FB",
#     2: "QB",
#     3: "RB",
#     4: "SB",
#     5: "TE",
#     6: "WB",
#     7: "WR",
#     8: "OLINE",
# }

POSITION_COLORS = {
    0: (0, 0, 0), 
    1: (255, 0, 0),     
    2: (0, 255, 0),     
    3: (255, 255, 0),   
    4: (0, 0, 255), 
    5: (128, 128, 128),    
}
POSITION_NAMES = {
    0: "BG",
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "OLINE",
}


def draw_legend(img, x=10, y=20, dy=18):
    for p, name in POSITION_NAMES.items():
        color = POSITION_COLORS[p]

        cv2.rectangle(img, (x, y - 10), (x + 15, y), color, -1)
        cv2.putText(
            img,
            name,
            (x + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        y += dy

def overlay_multiclass_mask(image, cls_map, alpha=0.45):
    img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = img.copy()

    cls_map = cls_map.cpu().numpy()  

    for p, color in POSITION_COLORS.items():
        mask = (cls_map == p)
        if mask.sum() == 0:
            continue

        color_img = np.zeros_like(img)
        color_img[mask] = color

        overlay = cv2.addWeighted(overlay, 1.0, color_img, alpha, 0)

    return overlay
