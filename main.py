import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from src import config
from src.logger import setup_logging, get_logger
from src.split import split_dataset
from src.utils import seed_everything
from src.preprocess import preprocess_pipeline
from src.training import train
from src.model import load_checkpoint
from src.dataset import VinBigDataset, get_val_transforms, collate_fn
from src.evaluation import evaluate
from src.ensemble import weighted_boxes_fusion
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Chest X-ray Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess DICOM files and merge annotations"
    )

    split_parser = subparsers.add_parser(
        "split", help="Split dataset into train/val/test"
    )

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--model",
        type=str,
        choices=config.AVAILABLE_MODELS,
        default="fasterrcnn",
        help="Model to train",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate ensemble on test set")
    eval_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=config.AVAILABLE_MODELS,
        default=config.AVAILABLE_MODELS,
        help="Models to evaluate (e.g., fasterrcnn retinanet_v2; default: all)",
    )

    args = parser.parse_args()

    seed_everything()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    setup_logging(log_file=os.path.join(config.OUTPUT_DIR, "logs", "run.log"))
    logger = get_logger()

    logger.info("=" * 70)
    logger.info(f"Command: {args.command}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    logger.info("=" * 70)

    if args.command == "preprocess":
        preprocess_pipeline(config.RAW_CSV, config.DICOM_DIR)
    elif args.command == "split":
        split_dataset(config.PREPROCESSED_CSV, config.DATA_DIR)
    elif args.command == "train":
        logger.info(f"Training {args.model.upper()}")
        train(args.model, config.TRAIN_CSV, config.VAL_CSV, config.PNG_DIR)
    elif args.command == "eval":
        logger.info(f"Evaluating models: {', '.join(args.models)}")
        run_eval(args.models)
    else:
        parser.print_help()


def run_eval(user_model_list):
    logger = get_logger()
    device = config.DEVICE

    if not os.path.exists(config.TEST_CSV):
        logger.error(f"Test CSV not found: {config.TEST_CSV}")
        logger.info("Run 'python main.py split' first")
        return

    # model_list uses same names as AVAILABLE_MODELS
    model_list = user_model_list

    logger.info(f"Loading test data from {config.TEST_CSV}")
    df_test = pd.read_csv(config.TEST_CSV)
    test_ds = VinBigDataset(df_test, config.PNG_DIR, get_val_transforms(), is_test=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Generate folder name
    if len(model_list) == 4 and set(model_list) == set(config.AVAILABLE_MODELS):
        folder_name = "all"
    else:
        folder_name = "_".join(sorted(model_list))

    # Create timestamped output directory structure
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    eval_base_dir = os.path.join(
        config.OUTPUT_DIR, "evaluation", folder_name, timestamp
    )

    config_dir = os.path.join(eval_base_dir, "config")
    metrics_dir = os.path.join(eval_base_dir, "metrics")
    cases_dir = os.path.join(eval_base_dir, "cases")
    logs_dir = os.path.join(eval_base_dir, "logs")

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(cases_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging to file
    eval_log_file = os.path.join(logs_dir, "evaluation.log")
    setup_logging(log_file=eval_log_file)
    logger = get_logger()

    logger.info("=" * 70)
    logger.info(f"Evaluation Start: {timestamp}")
    logger.info(f"Models: {', '.join(user_model_list)}")
    logger.info(f"Output: {eval_base_dir}")
    logger.info("=" * 70)

    # Save evaluation config
    eval_config = {
        "timestamp": timestamp,
        "models": user_model_list,
        "internal_models": model_list,
        "num_models": len(model_list),
        "conf_threshold": config.EVAL_CONF_THRESHOLD,
        "wbf_iou_thr": config.WBF_IOU_THR,
        "wbf_skip_box_thr": config.WBF_SKIP_BOX_THR,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "device": str(config.DEVICE),
    }
    config_path = os.path.join(config_dir, "eval_config.json")
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    logger.info(f"Saved eval_config.json")

    # Save model list
    model_list_data = {
        "num_models": len(model_list),
        "models": model_list,
    }
    model_list_path = os.path.join(config_dir, "model_list.json")
    with open(model_list_path, "w") as f:
        json.dump(model_list_data, f, indent=2)
    logger.info(f"Saved model_list.json")

    # Run ensemble evaluation
    logger.info(f"Ensemble with {len(model_list)} models: {', '.join(model_list)}")
    all_predictions, image_ids = ensemble_predict(model_list, test_loader, device)
    all_targets = get_targets(test_loader)

    # Evaluate with new structure
    evaluate(
        all_predictions,
        all_targets,
        eval_base_dir,
        cases_dir,
        metrics_dir,
        image_ids,
        df_test,
        config.PNG_DIR,
    )

    logger.info("=" * 70)
    logger.info(f"✓ Evaluation complete: {eval_base_dir}")
    logger.info("=" * 70)


def get_targets(data_loader):
    targets = []
    for _, batch_targets in data_loader:
        for target in batch_targets:
            targets.append(
                {
                    "boxes": (
                        target["boxes"].numpy()
                        if isinstance(target["boxes"], torch.Tensor)
                        else target["boxes"]
                    ),
                    "labels": (
                        target["labels"].numpy()
                        if isinstance(target["labels"], torch.Tensor)
                        else target["labels"]
                    ),
                }
            )
    return targets


def ensemble_predict(model_list, data_loader, device):
    logger = get_logger()

    # Load selected models for ensemble
    models = []
    for model_name in model_list:
        model_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}.pth")
        if os.path.exists(model_path):
            logger.info(f"Loading {model_name}")
            model = load_checkpoint(model_path, model_name, device)
            models.append((model_name, model))
        else:
            logger.warning(f"Model not found: {model_path}")

    if not models:
        logger.error("No models found!")
        return [], []

    logger.info(f"Ensemble: {len(models)} models with WBF")

    all_predictions = []
    all_image_ids = []
    with torch.no_grad():
        for images, batch_targets in tqdm(data_loader, desc="Ensemble Inference"):
            images = [img.to(device) for img in images]
            batch_size = len(images)

            batch_all_boxes = [[] for _ in range(batch_size)]
            batch_all_scores = [[] for _ in range(batch_size)]
            batch_all_labels = [[] for _ in range(batch_size)]

            for model_name, model in models:
                outputs = model(images)

                for i, output in enumerate(outputs):
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    mask = scores >= config.EVAL_CONF_THRESHOLD

                    batch_all_boxes[i].append(boxes[mask])
                    batch_all_scores[i].append(scores[mask])
                    batch_all_labels[i].append(labels[mask])

            for i, (boxes_list, scores_list, labels_list) in enumerate(
                zip(batch_all_boxes, batch_all_scores, batch_all_labels)
            ):
                # Get weights for current models from config
                weights = [config.WBF_WEIGHTS[model_name] for model_name, _ in models]

                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                    boxes_list,
                    scores_list,
                    labels_list,
                    iou_thr=config.WBF_IOU_THR,
                    skip_box_thr=config.WBF_SKIP_BOX_THR,
                    weights=weights,
                )

                all_predictions.append(
                    {
                        "boxes": fused_boxes,
                        "scores": fused_scores,
                        "labels": fused_labels.astype(np.int64),
                    }
                )

                img_id = batch_targets[i].get("image_id", len(all_image_ids))
                all_image_ids.append(img_id)

    return all_predictions, all_image_ids


if __name__ == "__main__":
    main()
