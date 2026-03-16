import os
import numpy as np
import pandas as pd

from src import config
from src.logger import get_logger


def split_dataset(preprocessed_csv, output_dir):
    """
    Split preprocessed dataset into train/val/test sets.
    """

    logger = get_logger()

    logger.info(f"Reading preprocessed CSV: {preprocessed_csv}")
    df = pd.read_csv(preprocessed_csv)

    logger.info(f"Loaded {len(df)} annotations from {df['image_id'].nunique()} images")

    unique_images = df["image_id"].unique()
    n = len(unique_images)

    indices = np.arange(n)
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(indices)

    train_n = int(n * config.TRAIN_SPLIT)
    val_n = int(n * config.VAL_SPLIT)

    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]

    train_imgs = unique_images[train_idx]
    val_imgs = unique_images[val_idx]
    test_imgs = unique_images[test_idx]

    df_train = df[df["image_id"].isin(train_imgs)].reset_index(drop=True)
    df_val = df[df["image_id"].isin(val_imgs)].reset_index(drop=True)
    df_test = df[df["image_id"].isin(test_imgs)].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    train_csv_path = os.path.join(output_dir, os.path.basename(config.TRAIN_CSV))
    val_csv_path = os.path.join(output_dir, os.path.basename(config.VAL_CSV))
    test_csv_path = os.path.join(output_dir, os.path.basename(config.TEST_CSV))

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    logger.info(f"Train: {len(df_train)} annotations ({len(train_imgs)} images)")
    logger.info(f"Val: {len(df_val)} annotations ({len(val_imgs)} images)")
    logger.info(f"Test: {len(df_test)} annotations ({len(test_imgs)} images)")
    logger.info(f"Split CSVs saved to {output_dir}")
