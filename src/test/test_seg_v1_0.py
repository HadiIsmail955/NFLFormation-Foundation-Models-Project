import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import autocast

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMMultiSegmenter_v1_0 import SAMSegmenter
from src.utils.merge_image import save_overlay_with_metrics
from src.utils.losses import make_loss
from src.utils.metrics import (
    dice_iou_from_logits,
    precision_recall_from_logits,
)
from src.data_loader.collate import presnap_collate_fn

@torch.no_grad()
def test_phase(cfg, logger):
    logger.logger.info("Initializing testing...")

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

    loss_fn = make_loss()

    vis_dir = logger.get_viz_dir()
    os.makedirs(vis_dir, exist_ok=True)

    total_loss = total_dice = total_iou = 0.0
    total_prec = total_rec = 0.0
    sample_idx = 0

    test_bar = tqdm(test_loader, desc="Testing", leave=False)

    for batch in test_bar:
        x = batch["seg_image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x,multimask_output=False)
            loss = loss_fn(logits, y)

        dice_mean, iou_mean = dice_iou_from_logits(logits, y)
        prec_mean, rec_mean = precision_recall_from_logits(logits, y)

        dice_img, iou_img = dice_iou_from_logits(logits, y, reduce=False)
        prec_img, rec_img = precision_recall_from_logits(logits, y, reduce=False)

        total_loss += loss.item()
        total_dice += dice_mean
        total_iou += iou_mean
        total_prec += prec_mean
        total_rec += rec_mean

        probs = torch.sigmoid(logits)
        preds = (probs > cfg.get("test_threshold", 0.5)).float()

        for b in range(x.size(0)):
            metrics = {
                "dice": dice_img[b].item(),
                "iou": iou_img[b].item(),
                "precision": prec_img[b].item(),
                "recall": rec_img[b].item(),
            }

            save_overlay_with_metrics(
                x[b],
                y[b],
                preds[b],
                metrics,
                os.path.join(vis_dir, f"sample_{sample_idx:04d}.png"),
            )
            sample_idx += 1

    n = len(test_loader)

    metrics = {
        "test_loss": total_loss / n,
        "test_dice": total_dice / n,
        "test_iou": total_iou / n,
        "test_precision": total_prec / n,
        "test_recall": total_rec / n,
    }

    for k, v in metrics.items():
        logger.logger.info(f"{k}: {v:.6f}")

    logger.logger.info(
        "Test results | "
        f"dice={metrics['test_dice']:.4f}, "
        f"iou={metrics['test_iou']:.4f}, "
        f"precision={metrics['test_precision']:.4f}, "
        f"recall={metrics['test_recall']:.4f}"
    )

    logger.close()