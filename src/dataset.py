import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import config


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.2),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-5, 5),
                fill=0,
                p=0.4,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_area=1, min_visibility=0.1
        ),
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_area=1, min_visibility=0.1
        ),
    )


class VinBigDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = df["image_id"].unique().tolist()

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

        records = self.df[self.df["image_id"] == image_id]

        if len(records) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = records[["x_min", "y_min", "x_max", "y_max"]].values.astype(
                np.float32
            )
            labels = records["class_id"].values.astype(np.int64) + 1

        if len(boxes) > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 1, orig_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 1, orig_h)

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]

        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )
            image = transformed["image"]
            boxes = (
                np.array(transformed["bboxes"], dtype=np.float32)
                if transformed["bboxes"]
                else np.zeros((0, 4), dtype=np.float32)
            )
            labels = (
                np.array(transformed["labels"], dtype=np.int64)
                if transformed["labels"]
                else np.zeros((0,), dtype=np.int64)
            )

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels, "image_id": image_id}


def collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def create_dataloaders(train_csv, val_csv, image_dir):
    train_ds = VinBigDataset(pd.read_csv(train_csv), image_dir, get_train_transforms())
    val_ds = VinBigDataset(pd.read_csv(val_csv), image_dir, get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
