import math

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import autocast
from torch.amp import GradScaler

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMFormationModel_v1_0 import SAMFormationModel

from src.data_loader.collate import presnap_collate_fn
# from src.utils.seed import set_seed

@torch.no_grad()
def compute_accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

def log(msg, logger):
    # print(msg)
    logger.logger.info(msg)


def train_phase(cfg, logger):
    # set_seed(cfg["seed"])
    log("Initializing training...", logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    log(f"Using device: {device}", logger)

    log("Loading dataset metadata...", logger)

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

    log("Splitting dataset into train / validation...", logger)

    val_len = int(len(dataset) * cfg["val_split"])
    train_len = len(dataset) - val_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    log(f"Split done -> train={train_len}, val={val_len}", logger)

    log("Creating DataLoaders...", logger)

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
        f"DataLoaders ready "
        f"(batch_size={cfg['batch_size']}, workers={cfg['num_workers']})",
        logger,
    )

    log("Initializing model...", logger)

    model = SAMFormationModel(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
        n_classes=cfg["n_classes"],
        k_decoder_layers=cfg["k_decoder_layers"],
        unfreeze_last_decoder_blocks=cfg.get("unfreeze_last_decoder_blocks",0),
    ).to(device)

    if cfg.get("continue_from_ckpt", None) is not None:
        state = torch.load(cfg.get("continue_from_ckpt", None), map_location=device)
        model.load_state_dict(state["model"])
        log(
            f"Continuing training from checkpoint: "
            f"{cfg.get('continue_from_ckpt', None) is not None}",
            logger,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model ready ({trainable_params:,} trainable parameters)", logger)

    # assert trainable_params < 6_000_000, "Encoder accidentally unfrozen!"

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # for accuracy
        factor=cfg["lr_decay_factor"],
        patience=math.ceil(cfg["patience"] // 2),
        threshold=cfg["threshold"],
        threshold_mode="rel",
        min_lr=cfg["min_lr"],
        verbose=True,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    scaler = GradScaler(
        device="cuda",
        enabled=amp_enabled,
    )

    log("Starting training loop...", logger)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"\nEpoch {epoch}/{cfg['epochs']} started | lr: {current_lr:.6f}", logger)

        model.train()
        train_loss = 0.0
        train_acc = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Train]",
            leave=False,
        )

        for batch in train_bar:
            x = batch["seg_image"].to(device, non_blocking=True)      
            points_xy = batch["centers"]
            points_label = batch["points_label"] 
            y = batch["formation_label"].to(device, non_blocking=True)
       

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(x, points_xy)
                loss = loss_fn(logits, y)

            if not torch.isfinite(loss):
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
            train_acc += compute_accuracy(logits.detach(), y)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        log("Running validation...", logger)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Val]", leave=False)

        with torch.no_grad():
            for batch in val_bar:
                x = batch["seg_image"].to(device, non_blocking=True)
                points_xy = batch["centers"]
                y = batch["formation_label"].to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(x, points_xy)
                    loss = loss_fn(logits, y)

                val_loss += loss.item()
                val_acc += compute_accuracy(logits, y)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        logger.log_epoch({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        log(
            f"Epoch {epoch} summary | "
            f"train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}",
            logger,
        )
        previous_lr = current_lr
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            logger.save_checkpoint(
                model,
                name="best.pt",
                epoch=epoch,
                val_acc=best_val_acc,
            )
            log("New best model saved", logger)
        elif current_lr < previous_lr:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] and cfg.get("early_stopping", True):
            log("Early stopping triggered", logger)
            break

    log(f"Training finished. Best val acc: {best_val_acc:.4f}", logger)
    logger.close()