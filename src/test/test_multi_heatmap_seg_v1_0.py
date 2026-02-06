import os
import cv2
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

    dataset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["test_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=cfg["flip_augmentation"],
        flip_prob=cfg["flip_prob"],
    )

    logger.logger.info(f"Test dataset loaded: {len(dataset)} samples")

    test_loader = DataLoader(
        dataset,
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
    loss_fn = make_loss()

    total_dice = total_iou = total_prec = total_rec = 0.0
    heatmap_total_loss = heatmap_total_dice = heatmap_total_iou = heatmap_total_prec = heatmap_total_rec = 0.0
    
    n_batches = 0
    sample_idx = 0
    heatmap_sample_idx = 0

    test_bar = tqdm(test_loader, desc="Testing", leave=False)

    for batch in test_bar:
        x = batch["seg_image"].to(device, non_blocking=True)              # [B,3,H,W]
        y = batch["position_masks"].to(device, non_blocking=True).float()   # [B,P,H,W]
        # y = batch["position_heatmaps"].to(device, non_blocking=True).float()

        heatmap_y = batch["center_map"].to(device, non_blocking=True)

        count_masks = batch.get("num_masks", None)
        file_names = batch.get("file_name", None)

        for b in range(x.size(0)):
            print(f"count_masks: {count_masks[b]} | file_name: {file_names[b]}")

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x,multimask_output=True)                                           # [B,P,H,W]

        dice, iou, prec, rec = macro_metrics_from_logits(logits, y)

        total_dice += dice.item()
        total_iou  += iou.item()
        total_prec += prec.item()
        total_rec  += rec.item()
        n_batches += 1

        pred_cls = logits.argmax(dim=1)   # [B,H,W]
        gt_cls   = y.argmax(dim=1)        # [B,H,W]

        for b in range(x.size(0)):
            H, W = gt_cls[b].shape

            gt_mask_img = render_multilabel_mask(y[b])
            pr_mask_img = render_multilabel_mask(torch.sigmoid(logits[b]))

            draw_legend(gt_mask_img)
            draw_legend(pr_mask_img)

            cv2.putText(
                gt_mask_img,
                "GT",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                pr_mask_img,
                f"PRED | Dice={dice:.3f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            combined = np.concatenate([gt_mask_img, pr_mask_img], axis=1)

            cv2.imwrite(
                os.path.join(vis_dir, f"mask_{sample_idx:04d}.png"),
                combined[..., ::-1],  # RGB → BGR
            )
            print(f"count_masks: {count_masks[b]} for sample {sample_idx} | file_name: {file_names[b]}")

            sample_idx += 1

        with autocast(device_type="cuda", enabled=amp_enabled):
            heatmap_logits = model(x, multimask_output=False)
            heatmap_loss = loss_fn(heatmap_logits, heatmap_y)

        heatmap_dice_mean, heatmap_iou_mean = dice_iou_from_logits(heatmap_logits, heatmap_y)
        heatmap_prec_mean, heatmap_rec_mean = precision_recall_from_logits(heatmap_logits, heatmap_y)

        heatmap_dice_img, heatmap_iou_img = dice_iou_from_logits(
            heatmap_logits, heatmap_y, reduce=False
        )
        heatmap_prec_img, heatmap_rec_img = precision_recall_from_logits(
            heatmap_logits, heatmap_y, reduce=False
        )

        heatmap_total_loss += heatmap_loss.item()
        heatmap_total_dice += heatmap_dice_mean
        heatmap_total_iou += heatmap_iou_mean
        heatmap_total_prec += heatmap_prec_mean
        heatmap_total_rec += heatmap_rec_mean

        heatmap_probs = torch.sigmoid(heatmap_logits)
        heatmap_preds = (heatmap_probs > cfg.get("test_threshold", 0.5)).float()

        for heatmap_b in range(x.size(0)):
            heatmap_metrics = {
                "dice": heatmap_dice_img[heatmap_b].item(),
                "iou": heatmap_iou_img[heatmap_b].item(),
                "precision": heatmap_prec_img[heatmap_b].item(),
                "recall": heatmap_rec_img[heatmap_b].item(),
            }

            save_overlay_with_metrics(
                x[heatmap_b],
                heatmap_y[heatmap_b],
                heatmap_preds[heatmap_b],
                heatmap_metrics,
                os.path.join(vis_dir, f"sample_{heatmap_sample_idx:04d}.png"),
            )
            heatmap_sample_idx += 1

    metrics = {
        "test_dice": total_dice / n_batches,
        "test_iou": total_iou / n_batches,
        "test_precision": total_prec / n_batches,
        "test_recall": total_rec / n_batches,
        "heatmap_test_dice": heatmap_total_dice / n_batches,
        "heatmap_test_iou": heatmap_total_iou / n_batches,
        "heatmap_test_precision": heatmap_total_prec / n_batches,
        "heatmap_test_recall": heatmap_total_rec / n_batches,
    }

    for k, v in metrics.items():
        logger.logger.info(f"{k}: {v:.6f}")

    logger.logger.info(
        "Test results | "
        f"dice={metrics['test_dice']:.4f}, "
        f"iou={metrics['test_iou']:.4f}, "
        f"precision={metrics['test_precision']:.4f}, "
        f"recall={metrics['test_recall']:.4f}"
        f"heatmap_dice={metrics['heatmap_test_dice']:.4f}, "
        f"heatmap_iou={metrics['heatmap_test_iou']:.4f}, "
        f"heatmap_precision={metrics['heatmap_test_precision']:.4f}, "
        f"heatmap_recall={metrics['heatmap_test_recall']:.4f}"
    )

    logger.close()
