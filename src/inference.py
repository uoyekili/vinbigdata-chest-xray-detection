"""Inference script - simple and straightforward."""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from src import config
from src.dataset import create_test_dataloader
from src.model import load_models
from src.ensemble import ensemble_multi_model, predict_single_model, filter_predictions, format_prediction_string
from src.utils import get_logger





def run_inference(use_ensemble: bool = True, fold: int = 0, output_path: str = None):
    """Run inference."""
    logger = get_logger()
    device = config.DEVICE
    
    # Find available models
    available_models = find_trained_models(fold)
    if not available_models:
        raise RuntimeError(f"No trained models found for fold {fold}")
    
    logger.info(f"Available models: {available_models}")
    
    # Get test images
    sample_df = pd.read_csv(config.SUBMISSION_CSV)
    image_ids = sample_df["image_id"].tolist()
    logger.info(f"Test images: {len(image_ids)}")
    
    loader = create_test_dataloader(image_ids, config.TEST_PNG_DIR, batch_size=1)
    predictions = {}
    
    if use_ensemble and len(available_models) > 1:
        logger.info(f"Running ENSEMBLE inference ({len(available_models)} models)...")
        models = load_models(available_models, fold, device)
        
        pbar = tqdm(loader, desc="Ensemble Inference")
        for images, ids, orig_ws, orig_hs in pbar:
            image = images[0]
            image_id = ids[0]
            pred = ensemble_multi_model(models, image, device)
            pred = filter_predictions(pred, config.CONF_THRESHOLD)
            predictions[image_id] = pred
    else:
        model_name = available_models[0]
        logger.info(f"Running SINGLE model inference ({model_name})...")
        model = load_models([model_name], fold, device)[0]
        
        pbar = tqdm(loader, desc=f"Inference ({model_name})")
        for images, ids, orig_ws, orig_hs in pbar:
            image = images[0]
            image_id = ids[0]
            pred = predict_single_model(model, image, device)
            pred = filter_predictions(pred, config.CONF_THRESHOLD)
            predictions[image_id] = pred
    
    # Create submission
    rows = []
    for _, row in sample_df.iterrows():
        image_id = row["image_id"]
        pred_string = format_prediction_string(predictions[image_id]) if image_id in predictions else "14 1.0 0 0 1 1"
        rows.append({"image_id": image_id, "PredictionString": pred_string})
    
    submission_df = pd.DataFrame(rows)
    
    if output_path is None:
        output_path = f"{config.OUTPUT_DIR}/submission.csv"
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    total_predictions = sum(len(p['boxes']) for p in predictions.values())
    images_with_findings = sum(1 for p in predictions.values() if len(p['boxes']) > 0)
    
    logger.info(f"✓ Submission saved: {output_path}")
    logger.info(f"  Total predictions: {total_predictions}")
    logger.info(f"  Images with findings: {images_with_findings}/{len(image_ids)}")


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Starting inference...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Use single model (not ensemble)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    run_inference(use_ensemble=not args.single, fold=args.fold, output_path=args.output)
    logger.info("Inference complete!")
