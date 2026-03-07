# VinBigData Chest X-ray Detection

Simple & optimized detection ensemble for VinBigData competition.

## Features
- **Two models**: Faster R-CNN + RetinaNet
- **Ensemble**: Weighted Boxes Fusion (WBF)
- **Auto-analysis**: Per-class metrics + error analysis + prediction statistics
- **Clean code**: Minimal dependencies, easy to maintain

## Data Setup

1. **Download from [Kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)**
   - Kaggle CLI: `kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection`
   - Or download manually from competition page

2. **Extract to data folder:**
   ```
   data/
   ├── train_dicom/     # Extract train DICOM files here
   ├── test_dicom/      # Extract test DICOM files here
   ├── train.csv        # Copy from downloaded data
   ├── test.csv         # (optional)
   └── sample_submission.csv
   ```

3. **Run preprocessing** (converts DICOM → PNG, prepares annotations):
   ```bash
   python src/preprocess.py
   ```

## Quick Start

### 1. Train & Evaluate & Predict
```bash
# SIMPLE: Full pipeline (train → evaluate → submission)
./run.sh

# Or step-by-step:
python src/train.py                       # Train all models (fold 0 by default)
python src/evaluate.py                    # Evaluate on local test
python src/inference.py                   # Generate submission (auto goes to output/)

# All outputs saved automatically:
# - Logs: output/logs/run_*.log
# - Graphs: output/analysis/*.png
# - Results: output/analysis/image_results.csv
# - Submission: output/submission.csv
```


### 2. Advanced Options

**Specific fold (0-4):**
```bash
python src/train.py --fold 1
python src/evaluate.py --fold 1
```

**Train/evaluate single model (no ensemble):**
```bash
python src/train.py --model fasterrcnn
python src/evaluate.py --single
python src/inference.py --single
```

**Custom output path:**
```bash
python src/inference.py --output custom/path/submission.csv
```

## Outputs

- `checkpoints/` - Trained models
- `output/logs/` - Training & evaluation logs
  - `run_YYYYMMDD_HHMMSS.log` - Auto-timestamped log files
- `output/analysis/` - Evaluation results
  - `prediction_analysis.png` - 4-subplot analysis (distribution, scatter, confidence, class counts)
  - `class_performance.png` - Per-class AP chart with thresholds
  - `image_results.csv` - Detailed image-level metrics & errors
- `output/submission.csv` - Final submission

## Configuration

Edit `src/config.py`:
- `IMAGE_SIZE` - Input resolution (default: 1024)
- `BATCH_SIZE` - Training batch size (default: 4)
- `NUM_EPOCHS` - Training epochs (default: 30)
- `LEARNING_RATE` - Initial LR (default: 0.0001)
- `CONF_THRESHOLD` - Detection confidence threshold (default: 0.1)

## Models

| Model | Backbone | Status |
|-------|----------|--------|
| Faster R-CNN | ResNet50-FPN | ✓ Implemented |
| RetinaNet | ResNet50-FPN | ✓ Implemented |

## Key Files

```
src/
├── config.py       # Configuration
├── model.py        # Model definitions (Faster R-CNN, RetinaNet)
├── dataset.py      # Data loading & augmentation
├── train.py        # Training script
├── inference.py    # Inference & submission
├── evaluate.py     # Evaluation on local test
├── ensemble.py     # Ensemble utilities (WBF)
└── metrics.py      # mAP calculation
```

## Training Notes

- **Mixed Precision**: Enabled by default (faster + less memory)
- **Early Stopping**: Stops after 7 epochs without improvement
- **Gradient Clipping**: max_norm=5.0
- **Learning Rate Schedule**: OneCycleLR with warmup

## Ensemble Strategy

Uses **Weighted Boxes Fusion (WBF)**:
1. Get predictions from each model
2. Normalize boxes to [0, 1]
3. Apply WBF to merge overlapping boxes
4. Filter by confidence threshold

## Evaluation Metrics

- **mAP@0.4** - Main metric (default IoU threshold)
- **Per-class AP** - For each disease type
- **Prediction statistics** - Accuracy distribution
- **Error analysis** - Images with most missed/false predictions

## Tips

1. **Improve performance**:
   - Increase NUM_EPOCHS
   - Lower LEARNING_RATE
   - Increase BATCH_SIZE (if GPU memory allows)
   - Train on more folds and ensemble them

2. **Faster training**:
   - Reduce NUM_EPOCHS
   - Increase BATCH_SIZE
   - Lower IMAGE_SIZE (trade-off: accuracy)

3. **Better ensemble**:
   - Train multiple folds separately
   - Use different model architectures
   - Adjust WBF_IOU_THR (default: 0.5)

## Requirements

```
torch>=1.9
torchvision>=0.10
albumentations>=1.0
ensemble-boxes>=1.0
pandas
numpy
scikit-learn
matplotlib
pydicom
opencv-python
```

Install: `pip install -r requirements.txt`

**Note**: All outputs (logs, graphs, data) go to `./output/` folder. Make sure GPU is available for training.

## License

VinBigData Challenge - 2021

---

**Key Features**:
- **One command**: `./run.sh` does everything (train + evaluate + submission)
- **Auto logging**: All console output → `output/logs/run_*.log`
- **Auto analysis**: 4-subplot graphs + per-class metrics + error analysis
- **Simple code**: Easy to debug, modify, or extend
- **Optional args**: All have sensible defaults (fold=0, ensemble=True, etc.)
