import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
from torch.amp import GradScaler

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMSegmenter_v1_0 import SAMSegmenter
from src.model.DINOClassifier_v1_0 import DINOClassifier
from src.model.SAM_DINOClassifier_v1_0 import SAMDINOClassifier

from src.utils.metrics import accuracy_from_logits
from src.data_loader.collate import presnap_collate_fn
# from src.utils.seed import set_seed


def log(msg, logger):
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
        f"DataLoaders ready "
        f"(batch_size={cfg['batch_size']}, workers={cfg['num_workers']})",
        logger,
    )

    log("Initializing SAM model...", logger)

    sam_model = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["sam_ckpt_dir"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
    ).to(device)

    assert isinstance(cfg.get("seg_from_ckpt", None), str), \
        "seg_from_ckpt must be a valid checkpoint path"

    state = torch.load(cfg["seg_from_ckpt"], map_location=device)
    sam_model.load_state_dict(state.get("model", state), strict=True)

    log(f"Loaded SAM model from: {cfg['seg_from_ckpt']}", logger)

    sam_model.eval()
    for p in sam_model.parameters():
        p.requires_grad = False

    log("Initializing DINO classifier...", logger)

    dino_classifier = DINOClassifier(
        num_classes=cfg["num_classes"],
        dino_type=cfg["dino_type"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
        freeze_backbone=True,
    ).to(device)

    dino_classifier.backbone.encoder.eval()

    model = SAMDINOClassifier(
        sam_model=sam_model,
        dino_classifier=dino_classifier,
        mask_mode=cfg["mask_mode"],
    ).to(device)

    class_weights = torch.tensor([
        1/72, 1/71, 1/63, 1/54, 1/45, 1/44, 1/42,
        1/42, 1/39, 1/39, 1/36, 1/31, 1/27, 1/22
    ], device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          
        factor=cfg["lr_decay_factor"],          
        patience=math.ceil(cfg["patience"] // 3),          
        threshold=cfg["weight_decay"],
        min_lr=cfg["min_lr"],
        verbose=True,
    )

    scaler = GradScaler(enabled=amp_enabled)

    log("Starting training loop...", logger)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"\nEpoch {epoch}/{cfg['epochs']} | lr={current_lr:.6f}", logger)

        model.train()
        model.sam.eval()

        train_loss = 0.0
        train_acc = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            image = batch["seg_image"].to(device, non_blocking=True)
            label = batch["formation_label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                logits = model(image)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                cfg["grad_clip"],
            )

            scaler.step(optimizer)
            scaler.update()

            acc = accuracy_from_logits(logits, label)

            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        model.sam.eval()

        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                image = batch["seg_image"].to(device)
                label = batch["formation_label"].to(device)

                with autocast(enabled=amp_enabled):
                    logits = model(image)
                    loss = criterion(logits, label)

                acc = accuracy_from_logits(logits, label)

                val_loss += loss.item()
                val_acc += acc

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

        logger.logger.info(
            f"Epoch {epoch} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            logger.save_checkpoint(
                model,
                name="best.pt",
                epoch=epoch,
                val_acc=best_acc,
            )
            logger.logger.info("New best model saved")
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            logger.logger.info("Early stopping triggered")
            break

    logger.logger.info(
        f"Training finished. Best Val Accuracy: {best_acc:.4f}"
    )
    logger.close()
