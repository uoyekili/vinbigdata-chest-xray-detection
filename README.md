# VinBigData Chest X-ray Detection

4-model ensemble for object detection on chest X-rays: Faster R-CNN + RetinaNet (FPN & FPN v2).

## Project Structure

```
vinbigdata-chest-xray-detection/
├── data/                          # Raw & processed data
│   ├── train.csv                  # Original DICOM annotations
│   ├── train_preprocessed.csv     # Merged PNG annotations
│   ├── train_split.csv            # Training split generated from the original dataset
│   ├── val_split.csv              # Validation split used during training
│   ├── test_split.csv             # Test split used only for evaluation
│   ├── train_dicom/               # Raw DICOM files
│   └── train_png/                 # Converted PNG images
│
├── src/
│   ├── preprocess.py              # DICOM→PNG, merge annotations
│   ├── split.py                   # Dataset train/val/test split
│   ├── dataset.py                 # VinBigDataset class
│   ├── model.py                   # Model builders (4 variants)
│   ├── training.py                # Training loop with AMP
│   ├── evaluation.py              # Evaluation, metrics, visualization
│   ├── ensemble.py                # WBF integration
│   ├── metrics.py                 # mAP & detection metrics
│   ├── config.py                  # Global configuration
│   └── utils.py                   # Utility functions
│
├── checkpoints/                   # Pre-trained model weights
│   ├── fasterrcnn.pth
│   └── retinanet.pth
│
├── output/                        # Generated outputs
│   ├── checkpoints/               # Trained models
│   ├── evaluation/                # Evaluation results (timestamped)
│   ├── analysis/                  # Analysis & visualizations
│   └── logs/                      # Training/evaluation logs
│
├── notebooks/
│   └── analysis.ipynb             # Data analysis notebook
│
├── main.py                        # Entry point (preprocess → train → eval)
├── requirements.txt               # Dependencies
└── README.md                      # Full documentation
```

## Output Structure

```
output/
├── checkpoints/              (trained models)
└── evaluation/
    └── {model_combination}/  (all, fasterrcnnv1, fasterrcnnv1_retinanetv2, etc.)
        └── {timestamp}/      (YYYY-MM-DD_HH-MM)
            ├── config/       (eval_config.json, model_list.json)
            ├── metrics/      (confusion_matrix.png, metrics_summary.json, etc.)
            ├── cases/        ({image_id}/ → visualization.png + metadata.json)
            └── logs/         (evaluation.log)
```

## Installation

```bash
pip install -r requirements.txt
```

Requires: GPU (CUDA)

## Usage

```bash
# 1. Preprocess: DICOM → PNG + merge annotations
python main.py preprocess

# 2. Split dataset (train/val/test)
python main.py split

# 3. Train models
python main.py train --model fasterrcnn
python main.py train --model fasterrcnn_v2
python main.py train --model retinanet
python main.py train --model retinanet_v2

# 4. Evaluate with ensemble
python main.py eval                                    # all 4 models
python main.py eval --models fasterrcnn                # single model
python main.py eval --models fasterrcnn retinanetv2    # 2 models
```

