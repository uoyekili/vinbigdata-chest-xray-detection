import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from src import config
from src.utils import get_logger


def get_train_transforms(image_size: int = config.IMAGE_SIZE):
    """Augmentation pipeline for training (medical images - minimal light transforms)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.2),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-5, 5),
                fill=0,
                p=0.4,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def get_val_transforms(image_size: int = config.IMAGE_SIZE):
    """Augmentation pipeline for validation/test (only resize & normalize)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1,
            min_visibility=0.1,
        ),
    )


class VinBigDataset(Dataset):
    """PyTorch Dataset for VinBigData Chest X-ray."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transforms=None,
        is_test: bool = False,
    ):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_test = is_test
        self.image_ids = df["image_id"].unique().tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        orig_h, orig_w = image.shape[:2]

        # Convert to RGB (required for pretrained models)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Load annotations
        records = self.df[self.df["image_id"] == image_id]

        if len(records) == 0 or self.is_test:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = records[["x_min", "y_min", "x_max", "y_max"]].values.astype(np.float32)
            # class_id: 0-13 -> label: 1-14 (0 is background)
            labels = records["class_id"].values.astype(np.int64) + 1

        # Clip boxes to image boundaries
        if len(boxes) > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 1, orig_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 1, orig_h)

            # Remove invalid boxes (width or height <= 0)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]

        # Apply augmentation
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed["labels"], dtype=np.int64) if transformed["labels"] else np.zeros((0,), dtype=np.int64)

        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        return image, target


class TestDataset(Dataset):
    """Dataset for test inference (no annotations)."""

    def __init__(
        self,
        image_ids: list,
        image_dir: str,
        image_size: int = config.IMAGE_SIZE,
    ):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        orig_h, orig_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        transformed = self.transform(image=image)
        return transformed["image"], image_id, orig_w, orig_h


def collate_fn(batch):
    """Custom collate function for detection models."""
    return tuple(zip(*batch))


def collate_fn_test(batch):
    """Collate function for test dataset."""
    images, image_ids, widths, heights = zip(*batch)
    return list(images), list(image_ids), list(widths), list(heights)


def create_dataloaders(
    processed_csv: str = config.FOLDS_TRAIN_CSV,
    fold: int = 0,
    num_folds: int = config.NUM_FOLDS,
    batch_size: int = config.BATCH_SIZE,
):
    """Create DataLoaders for training and validation."""
    df = pd.read_csv(processed_csv)
    unique_ids = df["image_id"].unique()

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    splits = list(kf.split(unique_ids))

    train_idx, val_idx = splits[fold]
    train_ids = unique_ids[train_idx]
    val_ids = unique_ids[val_idx]

    df_train = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["image_id"].isin(val_ids)].reset_index(drop=True)

    logger = get_logger()
    logger.info(f"Fold {fold}: Train = {len(train_ids)} images, Val = {len(val_ids)} images")

    train_dataset = VinBigDataset(
        df=df_train,
        image_dir=config.TRAIN_PNG_DIR,
        transforms=get_train_transforms(),
    )
    val_dataset = VinBigDataset(
        df=df_val,
        image_dir=config.TRAIN_PNG_DIR,
        transforms=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_test_dataloader(
    image_ids: list,
    image_dir: str = config.TEST_PNG_DIR,
    batch_size: int = 1,
):
    """Create DataLoader for test inference."""
    dataset = TestDataset(image_ids=image_ids, image_dir=image_dir)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn_test,
        pin_memory=True,
    )
    return loader
