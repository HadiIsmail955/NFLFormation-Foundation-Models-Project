import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import autocast
from torch.amp import GradScaler

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMMultiSegmenter_v1_0 import SAMSegmenter

from src.utils.losses import make_loss, SegmentationClassLoss
from src.data_loader.collate import presnap_collate_fn


def log(msg, logger):
    logger.logger.info(msg)


def _macro_dice_iou_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    B, P, H, W = probs.shape
    probs_f = probs.view(B, P, -1)
    targ_f  = targets.view(B, P, -1)

    inter = (probs_f * targ_f).sum(dim=(0, 2))
    p_sum = probs_f.sum(dim=(0, 2))
    t_sum = targ_f.sum(dim=(0, 2))
    union = p_sum + t_sum - inter

    dice = (2 * inter + eps) / (p_sum + t_sum + eps)
    iou  = (inter + eps) / (union + eps)

    valid = (t_sum > 0)

    if valid.any():
        return dice[valid].mean(), iou[valid].mean()
    else:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)


def train_phase(cfg, logger):
    log("Initializing training...", logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    log(f"Using device: {device}", logger)

    seg_tf = SegTransform()

    dataset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["train_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=cfg["flip_augmentation"],
        flip_prob=cfg["flip_prob"],
    )

    log(f"Dataset loaded: {len(dataset)} samples", logger)

    val_len = int(len(dataset) * cfg["val_split"])
    train_len = len(dataset) - val_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    log(f"Split done -> train={train_len}, val={val_len}", logger)

    persistent = cfg["num_workers"] > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=persistent,
        collate_fn=presnap_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=persistent,
        collate_fn=presnap_collate_fn,
    )

    log(
        f"DataLoaders ready (batch_size={cfg['batch_size']}, workers={cfg['num_workers']})",
        logger,
    )

    log("Initializing model...", logger)

    model = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
        num_classes=cfg["num_positions"],
    ).to(device)

    if cfg.get("continue_from_ckpt", None) is not None:
        state = torch.load(cfg["continue_from_ckpt"], map_location=device)
        model.load_state_dict(state["model"])
        log(f"Continuing training from checkpoint: {cfg['continue_from_ckpt']}", logger)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model ready ({trainable_params:,} trainable parameters)", logger)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="max",
    #     factor=cfg["lr_decay_factor"],
    #     patience=math.ceil(cfg["patience"] // 2),
    #     threshold=cfg["threshold"],
    #     threshold_mode="rel",
    #     min_lr=cfg["min_lr"],
    #     verbose=True,
    # )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["lr_decay_factor"],
        patience=math.ceil(cfg["patience"] // 2),
        min_lr=cfg["min_lr"],
        verbose=True,
    )
    class_weights = torch.tensor([0.2, 2.5, 1.5, 1.5, 2.0, 1.0,], device=device)
    # class_weights = torch.tensor([
    #     0.05,   # background
    #     2.0,    # FB
    #     3.0,    # QB
    #     1.5,    # RB
    #     2.0,    # SB
    #     2.2,    # TE
    #     2.0,    # WB
    #     1.3,    # WR
    #     1.0     # OLINE
    # ], device=device)
    loss_fn = SegmentationClassLoss(class_weights)

    loss_fn_heatmap = make_loss()

    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    log("Starting training loop...", logger)

    best_dice = 0.0
    patience_counter = 0
    best_metric = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"\nEpoch {epoch}/{cfg['epochs']} started | lr: {current_lr:.6f}", logger)

        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Train]", leave=False)

        for batch in train_bar:
            x = batch["seg_image"].to(device, non_blocking=True)                # [B,3,H,W]
            y = batch["position_masks"].to(device, non_blocking=True).float()   # [B,P,H,W]
            # y = batch["position_heatmaps"].to(device, non_blocking=True).float()
            
            heatmap_y = batch["center_map"].to(device, non_blocking=True)

            count_masks = batch.get("num_masks", None)
            file_names = batch.get("file_name", None)
            for b in range(x.size(0)):
                print(f"count_masks: {count_masks[b]} | file_name: {file_names[b]}")

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(x,multimask_output=True)
                loss = loss_fn(logits, y)
                logits_heatmap = model(x,multimask_output=False)
                heatmap_loss = loss_fn_heatmap(logits_heatmap, heatmap_y)
                loss = loss + 0.5 * heatmap_loss    


            if not torch.isfinite(loss).all():
                log("Non-finite loss detected. Stopping training.", logger)
                logger.close()
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg["grad_clip"],
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        log("Running validation...", logger)

        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Val]", leave=False)

        with torch.no_grad():
            for batch in val_bar:
                x = batch["seg_image"].to(device, non_blocking=True)
                y = batch["position_masks"].to(device, non_blocking=True).float()
                # y = batch["position_heatmaps"].to(device, non_blocking=True).float()

                heatmap_y = batch["center_map"].to(device, non_blocking=True)

                count_masks = batch.get("num_masks", None)
                file_names = batch.get("file_name", None)
                
                for b in range(x.size(0)):
                    print(f"count_masks: {count_masks[b]} | file_name: {file_names[b]}")

                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(x, multimask_output=True)
                    loss = loss_fn(logits, y)
                    logits_heatmap = model(x,multimask_output=False)
                    heatmap_loss = loss_fn_heatmap(logits_heatmap, heatmap_y)
                    loss = loss + 0.5 * heatmap_loss 

                d, i = _macro_dice_iou_from_logits(logits, y)

                val_loss += loss.item()
                val_dice += d.item()
                val_iou += i.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        logger.log_epoch({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
        })

        log(
            f"Epoch {epoch} summary | train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, val_iou={val_iou:.4f}",
            logger,
        )

        # scheduler.step(val_dice)

        previous_lr = current_lr
        metric = val_loss + 0.5 * (1 - val_dice)
        scheduler.step(metric)
        current_lr = optimizer.param_groups[0]["lr"]

        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            logger.save_checkpoint(model, "best.pt", epoch=epoch, val_dice=val_dice)
        elif current_lr < previous_lr:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] and cfg.get("early_stopping", True):
            log("Early stopping triggered", logger)
            break

    log(f"Training finished. Best Dice: {best_dice:.4f}", logger)
    logger.close()
