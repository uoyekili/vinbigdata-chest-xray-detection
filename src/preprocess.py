import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from ensemble_boxes import weighted_boxes_fusion
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src import config
from src.logger import get_logger


def read_dicom_header(dcm_path):
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return {"image_id": dcm_path.stem, "width": dcm.Columns, "height": dcm.Rows}
    except Exception:
        return None


def get_dicom_metadata_parallel(dicom_dir: str) -> pd.DataFrame:
    dicom_files = list(Path(dicom_dir).glob("*.dicom"))
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(read_dicom_header, dicom_files),
                total=len(dicom_files),
                desc="Extracting metadata",
            )
        )

    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


def process_single_dicom(args):
    """Process a single DICOM file (for parallel execution)."""
    dcm_path, output_dir, image_size = args
    out_path = os.path.join(output_dir, dcm_path.stem + ".png")

    # Skip if already exists
    if os.path.exists(out_path):
        return

    try:
        dcm = pydicom.dcmread(str(dcm_path))
        img = dcm.pixel_array.astype(np.float32)

        # Normalize to [0, 255]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
        img = img.astype(np.uint8)

        img = cv2.resize(img, (image_size, image_size))
        cv2.imwrite(out_path, img)
    except Exception as e:
        print(f"Error processing {dcm_path}: {e}")


def convert_dicom_to_png(dicom_dir: str, output_dir: str, image_size: int = 1024):
    """Convert DICOM images to PNG format using multiprocessing."""
    logger = get_logger()

    os.makedirs(output_dir, exist_ok=True)
    dicom_files = list(Path(dicom_dir).glob("*.dicom"))

    if not dicom_files:
        logger.error(f"No DICOM files found in {dicom_dir}")
        return 0

    logger.info(f"Converting {len(dicom_files)} DICOM files ...")

    args_list = [(dcm_path, output_dir, image_size) for dcm_path in dicom_files]

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_single_dicom, args_list),
                total=len(args_list),
                desc="Converting DICOMs",
            )
        )

    png_count = len(list(Path(output_dir).glob("*.png")))
    logger.info(f"✓ Converted {png_count} images to PNG in {output_dir}")


def merge_radiologist_annotations(
    df: pd.DataFrame, target_size: int = 1024
) -> pd.DataFrame:
    """Merge overlapping bounding boxes from multiple radiologists using WBF."""
    logger = get_logger()

    results = []

    # Filter out "No finding" (class_id = 14)
    df_original = len(df)
    df_with_box = df[df["class_id"] < 14].copy()
    removed = df_original - len(df_with_box)

    if removed > 0:
        logger.info(f"Removed {removed} 'No finding' annotations")

    for image_id, group in tqdm(
        df_with_box.groupby("image_id"), desc="Merging annotations"
    ):
        # Get image dimensions
        orig_w = group["width"].iloc[0]
        orig_h = group["height"].iloc[0]

        # Normalize coordinates to [0, 1]
        boxes = group[["x_min", "y_min", "x_max", "y_max"]].values.astype(float)
        boxes[:, [0, 2]] /= orig_w
        boxes[:, [1, 3]] /= orig_h
        boxes = np.clip(boxes, 0, 1)

        labels = group["class_id"].values.astype(int)
        scores = np.ones(len(boxes))

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            [boxes.tolist()],
            [scores.tolist()],
            [labels.tolist()],
            iou_thr=config.WBF_IOU_THR,
            skip_box_thr=config.WBF_SKIP_BOX_THR,
        )

        # Denormalize coordinates
        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            results.append(
                {
                    "image_id": image_id,
                    "class_id": int(label),
                    "x_min": box[0] * target_size,
                    "y_min": box[1] * target_size,
                    "x_max": box[2] * target_size,
                    "y_max": box[3] * target_size,
                    "confidence": score,
                    "width": target_size,
                    "height": target_size,
                }
            )

    df_merged = pd.DataFrame(results)
    logger.info(
        f"✓ Merged annotations: {len(df_with_box)} rows → {len(df_merged)} rows"
    )
    return df_merged


def split_local_test(df_merged: pd.DataFrame, test_ratio: float = 0.1):
    """Split 10% data for local testing."""
    logger = get_logger()
    logger.info(f"Splitting {test_ratio*100}% for local test...")

    # Split based on image_id to avoid leakage
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=config.RANDOM_SEED
    )
    train_idx, test_idx = next(splitter.split(df_merged, groups=df_merged["image_id"]))

    df_local_train = df_merged.iloc[train_idx]
    df_local_test = df_merged.iloc[test_idx]

    logger.info(
        f"Local Train set: {len(df_local_train)} annotations ({df_local_train['image_id'].nunique()} images)"
    )
    logger.info(
        f"Local Test set:  {len(df_local_test)} annotations ({df_local_test['image_id'].nunique()} images)"
    )

    return df_local_train, df_local_test


def preprocess_pipeline(
    csv_path: str,
    dicom_dir: str,
):
    """Run preprocessing: Convert DICOM → Merge annotations → Split Train/Test."""
    logger = get_logger()

    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    # Step 1: Convert DICOM to PNG
    logger.info(f"Step 1: Convert DICOM from {dicom_dir}")

    convert_dicom_to_png(dicom_dir, config.DATASET_DIR, config.IMAGE_SIZE)

    # Step 2: Read CSV and merge metadata
    logger.info(f"Step 2: Reading {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    logger.info(
        f"Loaded {len(df)} annotations from {len(df['image_id'].unique())} images"
    )

    # Step 3: Extract DICOM metadata
    logger.info("Step 3: Extracting DICOM metadata...")
    meta_df = get_dicom_metadata_parallel(dicom_dir)
    df = df.merge(meta_df, on="image_id", how="left")

    # Fill missing dimensions
    df["width"] = df["width"].fillna(config.IMAGE_SIZE)
    df["height"] = df["height"].fillna(config.IMAGE_SIZE)

    # Step 4: Merge radiologist annotations
    logger.info("Step 4: Merging annotations from multiple radiologists...")
    df_merged = merge_radiologist_annotations(df, config.IMAGE_SIZE)

    # Step 5: Save preprocessed data
    logger.info("Step 5: Saving preprocessed data...")
    df_merged.to_csv(config.PREPROCESSED_CSV, index=False)
    logger.info(f"Saved to {config.PREPROCESSED_CSV}")

    logger.info("=" * 70)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("=" * 70)

    return True
