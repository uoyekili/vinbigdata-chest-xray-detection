import argparse
import json
import os
import time

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src import config
from src.dataset import create_dataloaders
from src.metrics import compute_map
from src.model import build_model, get_model_info
from src.utils import setup_logging, get_logger, log_system_info, ensure_dir, format_number


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        valid_targets = [t for t in targets if len(t["boxes"]) > 0]
        if len(valid_targets) == 0:
            continue

        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            optimizer.step()

        loss_val = losses.item()
        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Validation."""
    model.eval()
    all_predictions = []
    all_targets = []

    pbar = tqdm(data_loader, desc="[Validation]")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            mask = pred_scores >= config.CONF_THRESHOLD
            
            all_predictions.append({
                "boxes": pred_boxes[mask],
                "scores": pred_scores[mask],
                "labels": pred_labels[mask],
            })
            all_targets.append({
                "boxes": target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"],
                "labels": target["labels"].numpy() if isinstance(target["labels"], torch.Tensor) else target["labels"],
            })

    return compute_map(all_predictions, all_targets, iou_threshold=0.4)


def train_model(model_name: str, fold: int):
    """Train a single model."""
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info(f"Training {model_name.upper()} - Fold {fold}")
    logger.info("=" * 60)

    train_loader, val_loader = create_dataloaders(fold=fold)
    model = build_model(model_name=model_name, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    info = get_model_info(model)
    logger.info(f"Trainable params: {format_number(info['trainable_params'])}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 10,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    scaler = GradScaler() if config.DEVICE.type == "cuda" else None
    ensure_dir(config.CHECKPOINT_DIR)
    
    checkpoint_path = config.get_checkpoint_path(model_name, fold)
    best_map = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_map": [], "lr": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch, scaler)
        current_lr = optimizer.param_groups[0]["lr"]

        val_map = evaluate(model, val_loader, config.DEVICE)

        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_map": best_map,
            }, checkpoint_path)
            logger.info(f"  [BEST] Saved best model (mAP={best_map:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"  [STOP] Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0
        history["train_loss"].append(avg_loss)
        history["val_map"].append(val_map)
        history["lr"].append(current_lr)

        logger.info(f"[{epoch:2d}/{config.NUM_EPOCHS}] Loss: {avg_loss:.4f} | mAP: {val_map:.4f} | Best: {best_map:.4f} | LR: {current_lr:.6f} | {elapsed:.1f}s")

    with open(f"{config.CHECKPOINT_DIR}/{model_name}_fold{fold}_history.json", "w") as f:
        json.dump(history, f)

    logger.info(f"[DONE] Training complete! Best mAP: {best_map:.4f}")
    return best_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                       choices=config.AVAILABLE_MODELS,
                       help="Model to train (None = all models)")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    # Initialize logging
    setup_logging(log_file=f"train_fold{args.fold}.log")
    logger = get_logger()
    log_system_info(logger)

    if args.model is None:
        results = {}
        for model_name in config.AVAILABLE_MODELS:
            results[model_name] = train_model(model_name, args.fold)
        
        logger.info("=" * 60)
        logger.info("Final Results:")
        for name, mAP in results.items():
            logger.info(f"  {name}: {mAP:.4f}")
    else:
        train_model(args.model, args.fold)
