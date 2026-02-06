import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.model.SAMSegmenter_v1_0 import SAMSegmenter
from src.model.DINOClassifier_v1_0 import DINOClassifier
from src.model.SAM_DINOClassifier_v1_0 import SAMDINOClassifier
from src.utils.merge_image import save_classification_overlay

FORMATION_MAP = {
    "shotgun": 0,
    "singleback": 1,
    "ace-left": 2,
    "ace-right": 3,
    "trips-left": 4,
    "trips-right": 5,
    "twins-right": 6,
    "bunch-left": 7,
    "bunch-right": 8,
    "i-formation": 9,
    "trey-left": 10,
    "trey-right": 11,
    "empty": 12,
    "double-tight": 13,
    "heavy": 14,
}

ID_TO_FORMATION = {v: k for k, v in FORMATION_MAP.items()}

@torch.no_grad()
def test_phase(cfg, logger):
    logger.logger.info("Initializing classification testing...")

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

    test_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )

    logger.logger.info(f"Test dataset loaded: {len(dataset)} samples")

    sam = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["sam_ckpt_dir"],
    ).to(device)

    for p in sam.parameters():
        p.requires_grad = False
    sam.eval()

    classifier = DINOClassifier(
        num_classes=cfg["num_classes"],
        dino_type=cfg["dino_type"],
    ).to(device)

    model = SAMDINOClassifier(
        sam_model=sam,
        dino_classifier=classifier,
        mask_mode=cfg["mask_mode"],
    ).to(device)

    ckpt_path = logger.get_best_checkpoint_path()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    model.eval()
    model.sam.eval()

    logger.logger.info(f"Loaded checkpoint: {ckpt_path}")

    criterion = nn.CrossEntropyLoss()

    vis_dir = logger.get_viz_dir()
    os.makedirs(vis_dir, exist_ok=True)

    total_loss = 0.0
    correct = 0
    total = 0
    sample_idx = 0

    for batch in tqdm(test_loader, desc="Testing", leave=False):
        images = batch["seg_image"].to(device, non_blocking=True)
        labels = batch["formation_label"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = probs.max(dim=1).values

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for b in range(images.size(0)):
            save_classification_overlay(
                image=images[b],
                gt_label=labels[b].item(),
                pred_label=preds[b].item(),
                confidence=confs[b].item(),
                save_path=os.path.join(vis_dir, f"sample_{sample_idx:04d}.png"),
                id_to_class=ID_TO_FORMATION,
            )
            sample_idx += 1

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    logger.logger.info(f"test_loss: {avg_loss:.6f}")
    logger.logger.info(f"test_accuracy: {accuracy:.4f}")
    logger.logger.info("Classification testing finished.")

    logger.close()
