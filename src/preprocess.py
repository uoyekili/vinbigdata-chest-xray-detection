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


def read_dicom_header(dcm_path):
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return {
            "image_id": dcm_path.stem,
            "width": dcm.Columns,
            "height": dcm.Rows
        }
    except Exception:
        return None


def get_dicom_metadata_parallel(dicom_dir: str) -> pd.DataFrame:
    dicom_files = list(Path(dicom_dir).glob("*.dicom"))
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(read_dicom_header, dicom_files), total=len(dicom_files), desc="Extracting metadata"))
    
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
    os.makedirs(output_dir, exist_ok=True)
    dicom_files = list(Path(dicom_dir).glob("*.dicom"))
    
    print(f"Converting {len(dicom_files)} DICOM files (Parallel)...")
    
    args_list = [(dcm_path, output_dir, image_size) for dcm_path in dicom_files]
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    # Max workers = number of CPU cores
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_dicom, args_list), total=len(args_list)))

    print(f"Saved images to {output_dir}")



def merge_radiologist_annotations(
    df: pd.DataFrame,
    target_size: int = 1024
) -> pd.DataFrame:
    """Merge overlapping bounding boxes from multiple radiologists using WBF."""
    results = []

    # Filter out "No finding" (class_id = 14)
    df_with_box = df[df["class_id"] != 14].copy()

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
                    "height": target_size
                }
            )

    return pd.DataFrame(results)


def split_local_test(df_merged: pd.DataFrame, test_ratio: float = 0.1):
    """Split 10% data for local testing (hold-out set)."""
    print(f"\nSplitting {test_ratio*100}% for local test (hold-out)...")
    
    # Split based on image_id to avoid leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_idx, test_idx = next(splitter.split(df_merged, groups=df_merged["image_id"]))
    
    df_local_train = df_merged.iloc[train_idx]
    df_local_test = df_merged.iloc[test_idx]
        
    print(f"Local Train/Val set: {len(df_local_train)} annotations ({df_local_train['image_id'].nunique()} images)")
    print(f"Local Test set:      {len(df_local_test)} annotations ({df_local_test['image_id'].nunique()} images)")
    
    return df_local_train, df_local_test



def preprocess_pipeline(
    csv_path: str,
    dicom_dir: str,
    output_dir: str,
):
    """Run preprocessing: Merge annotations -> Split Train/Local Test."""
    print("Extracting metadata from DICOMs...")
    meta_df = get_dicom_metadata_parallel(dicom_dir)
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Merge metadata
    df = df.merge(meta_df, on="image_id", how="left")
    
    # Check if we have width/height for all images
    if df["width"].isnull().any():
        print("Warning: Some images are missing metadata. Using defaults (1024x1024).")
        df["width"] = df["width"].fillna(1024)
        df["height"] = df["height"].fillna(1024)

    print("Merging annotations from multiple radiologists...")
    df_merged = merge_radiologist_annotations(df)
    
    # Split Local Test (10%)
    df_local_train, df_local_test = split_local_test(df_merged)

    # Save files
    local_train_path = os.path.join(output_dir, "local_train.csv")
    local_test_path = os.path.join(output_dir, "local_test.csv")
    
    df_local_train.to_csv(local_train_path, index=False)
    df_local_test.to_csv(local_test_path, index=False)
    
    print(f"Saved local train to {local_train_path}")
    print(f"Saved local test to {local_test_path}")


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Process Train Data
    if os.path.exists(config.RAW_TRAIN_DICOM_DIR):
        print("Processing Train DICOM...")
        convert_dicom_to_png(config.RAW_TRAIN_DICOM_DIR, config.TRAIN_PNG_DIR, image_size=1024)
    
    # Process Test Data
    if os.path.exists(config.RAW_TEST_DICOM_DIR):
        print("Processing Test DICOM...")
        convert_dicom_to_png(config.RAW_TEST_DICOM_DIR, config.TEST_PNG_DIR, image_size=1024)
    
    if os.path.exists(config.RAW_TRAIN_CSV):
        preprocess_pipeline(
            csv_path=config.RAW_TRAIN_CSV,
            dicom_dir=config.RAW_TRAIN_DICOM_DIR,
            output_dir=config.DATA_DIR,
        )
    
    print("Preprocessing completed!")
