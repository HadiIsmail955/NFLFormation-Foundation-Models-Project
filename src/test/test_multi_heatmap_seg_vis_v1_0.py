import os
import cv2
import csv
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import autocast

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMMultiSegmenter_v1_0 import SAMSegmenter
from src.utils.metrics import macro_metrics_from_logits
from src.data_loader.collate import presnap_collate_fn
from src.utils.merge_image import save_overlay_with_metrics
from src.utils.losses import make_loss
from src.utils.metrics import (
    dice_iou_from_logits,
    precision_recall_from_logits,
)


from src.utils.merge_image import POSITION_COLORS, draw_legend

# ID_TO_POSITION = {
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

ID_TO_POSITION = {
    0: "BG",
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "OLINE",
}

def extract_xy_position_from_outputs(seg_logits, heatmap_logits, threshold=0.6):

    seg_cls = torch.softmax(seg_logits, dim=0).argmax(dim=0).cpu().numpy()
    H, W = seg_cls.shape

    oline_cx, oline_cy = get_oline_centroid(seg_cls)

    heat = torch.sigmoid(heatmap_logits[0]).cpu().numpy()
    heat_bin = (heat > threshold).astype(np.uint8)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(heat_bin)

    results = []

    for cx, cy in centroids[1:]:
        x = int(cx)
        y = int(cy)

        patch = seg_cls[max(0,y-2):y+3, max(0,x-2):x+3]
        vals, counts = np.unique(patch, return_counts=True)
        pos_id = int(vals[counts.argmax()])
        pos_name = ID_TO_POSITION.get(pos_id, "unknown")

        lr = get_lr_alignment_from_oline(x, oline_cx, W)

        if oline_cx is not None:
            dx = x - oline_cx
            dy = y - oline_cy
            dist = float(np.sqrt(dx*dx + dy*dy))
        else:
            dx = dy = dist = 0.0

        results.append((
            x, y,
            pos_name,
            lr,
            round(dx,2),
            round(dy,2),
            round(dist,2)
        ))

    return results

def get_oline_centroid(seg_cls):
    # ys, xs = np.where(seg_cls == 8)
    ys, xs = np.where(seg_cls == 5)
    if len(xs) == 0:
        return None, None
    return xs.mean(), ys.mean()

def get_lr_alignment_from_oline(x, oline_cx, width, margin_ratio=0.06):

    if oline_cx is None:
        oline_cx = width / 2

    margin = width * margin_ratio

    if x < oline_cx - margin:
        return "LEFT"
    elif x > oline_cx + margin:
        return "RIGHT"
    else:
        return "CENTER"

def render_multilabel_mask(y, threshold=0.5):
    P, H, W = y.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for p, color in POSITION_COLORS.items():
        if p >= P:
            continue
        mask = y[p] > threshold
        if mask.sum() == 0:
            continue
        canvas[mask.cpu().numpy()] = color

    return canvas

@torch.no_grad()
def test_phase(cfg, logger):
    logger.logger.info("Initializing testing (mask-only)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    logger.logger.info(f"Using device: {device}")
    
    seg_tf = SegTransform()

    train_daraset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["train_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=cfg["flip_augmentation"],
        flip_prob=cfg["flip_prob"],
    )
    test_dataset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["test_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=cfg["flip_augmentation"],
        flip_prob=cfg["flip_prob"],
    )

    logger.logger.info(f"Train dataset loaded: {len(train_daraset)} samples")
    logger.logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_daraset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=presnap_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=presnap_collate_fn,
    )

    model = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=0,
        num_classes=cfg["num_positions"],
    ).to(device)

    ckpt_path = logger.get_best_checkpoint_path()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    logger.logger.info(f"Loaded checkpoint: {ckpt_path}")

    vis_dir = logger.get_viz_dir()
    os.makedirs(vis_dir, exist_ok=True)

    train_seg_vis_dir = os.path.join(vis_dir, "train", "segmentation")
    train_heatmap_vis_dir = os.path.join(vis_dir, "train", "heatmap")
    test_seg_vis_dir = os.path.join(vis_dir, "test", "segmentation")
    test_heatmap_vis_dir = os.path.join(vis_dir, "test", "heatmap")

    os.makedirs(train_seg_vis_dir, exist_ok=True)
    os.makedirs(train_heatmap_vis_dir, exist_ok=True)
    os.makedirs(test_seg_vis_dir, exist_ok=True)
    os.makedirs(test_heatmap_vis_dir, exist_ok=True)
    
    train_csv = open(os.path.join(vis_dir, "train/train_players.csv"), "w", newline="")
    test_csv  = open(os.path.join(vis_dir, "test/test_players.csv"), "w", newline="")

    train_writer = csv.writer(train_csv)
    test_writer  = csv.writer(test_csv)

    header = [
        "image_file",
        "image",
        "formation",
        "x","y",
        "position",
        "lr_align",
        "dx_from_oline",
        "dy_from_oline",
        "dist_from_oline"
    ]

    train_writer.writerow(header)
    test_writer.writerow(header)

    sample_idx = 0
    
    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in train_bar:
        x = batch["seg_image"].to(device, non_blocking=True)              # [B,3,H,W]
        y = batch["position_masks"].to(device, non_blocking=True).float()   # [B,P,H,W]
        # y = batch["position_heatmaps"].to(device, non_blocking=True).float()
        heatmap_y = batch["center_map"].to(device, non_blocking=True)
        formation = batch["formation_name"]      # [B]
        file_names = batch.get("file_name", None)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x,multimask_output=True)                                           # [B,P,H,W]
            heatmap_logits = model(x, multimask_output=False)

        gt_cls = y.argmax(dim=1)        # [B,H,W]

        for b in range(x.size(0)):
            H, W = gt_cls[b].shape
            pr_mask_img = render_multilabel_mask(torch.sigmoid(logits[b]))
            heatmap_probs = torch.sigmoid(heatmap_logits)
            draw_legend(pr_mask_img)
            combined = np.concatenate([pr_mask_img], axis=1)
            cv2.imwrite(
                os.path.join(train_seg_vis_dir, f"mask_{sample_idx:04d}.png"),
                combined[..., ::-1],  # RGB → BGR
            )

            heatmap_np = heatmap_probs[b].max(dim=0)[0].cpu().numpy()
            heatmap_img = (heatmap_np * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(train_heatmap_vis_dir, f"heatmap_raw_{sample_idx:04d}.png"),
                heatmap_img,
            )

            players = extract_xy_position_from_outputs(
                logits[b],
                heatmap_logits[b]
            )

            img_name = f"{sample_idx:04d}.png"
            image= file_names[b] if file_names is not None else img_name

            for row in players:
                train_writer.writerow([image, img_name, formation[b],*row])

            sample_idx += 1

    test_bar = tqdm(test_loader, desc="Testing", leave=False)

    for batch in test_bar:
        x = batch["seg_image"].to(device, non_blocking=True)              # [B,3,H,W]
        y = batch["position_masks"].to(device, non_blocking=True).float()   # [B,P,H,W]
        # y = batch["position_heatmaps"].to(device, non_blocking=True).float()
        heatmap_y = batch["center_map"].to(device, non_blocking=True)
        formation = batch["formation_name"]     # [B]
        file_names = batch.get("file_name", None)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x,multimask_output=True)                                           # [B,P,H,W]
            heatmap_logits = model(x, multimask_output=False)

        gt_cls = y.argmax(dim=1)        # [B,H,W]

        for b in range(x.size(0)):
            H, W = gt_cls[b].shape
            pr_mask_img = render_multilabel_mask(torch.sigmoid(logits[b]))
            heatmap_probs = torch.sigmoid(heatmap_logits)
            draw_legend(pr_mask_img)
            combined = np.concatenate([pr_mask_img], axis=1)
            cv2.imwrite(
                os.path.join(test_seg_vis_dir, f"mask_{sample_idx:04d}.png"),
                combined[..., ::-1],  # RGB → BGR
            )

            heatmap_np = heatmap_probs[b].max(dim=0)[0].cpu().numpy()
            heatmap_img = (heatmap_np * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(test_heatmap_vis_dir, f"heatmap_raw_{sample_idx:04d}.png"),
                heatmap_img,
            )

            players = extract_xy_position_from_outputs(
                logits[b],
                heatmap_logits[b]
            )

            img_name = f"{sample_idx:04d}.png"
            image= file_names[b] if file_names is not None else img_name

            for row in players:
                test_writer.writerow([image,img_name, formation[b],*row])

            sample_idx += 1

    train_csv.close()
    test_csv.close()
    logger.close()
