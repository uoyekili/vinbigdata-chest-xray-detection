import torch
import os

# Device & Random Seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Model Configuration
CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]
NUM_CLASSES = len(CLASS_NAMES) + 1
AVAILABLE_MODELS = ["fasterrcnn", "fasterrcnn_v2", "retinanet", "retinanet_v2"]

# Training Hyperparameters
IMAGE_SIZE = 1024
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
NUM_WORKERS = 30
MAX_GRAD_NORM = 5.0
PATIENCE = 7

# Inference Thresholds
TRAIN_CONF_THRESHOLD = 0.05
EVAL_CONF_THRESHOLD = 0.3

# Data Paths
DATA_DIR = "data"
DICOM_DIR = os.path.join(DATA_DIR, "train_dicom")
RAW_CSV = os.path.join(DATA_DIR, "train.csv")
PNG_DIR = os.path.join(DATA_DIR, "train_png")
PREPROCESSED_CSV = os.path.join(DATA_DIR, "train_preprocessed.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_split.csv")
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Output Paths
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Ensemble Parameters
WBF_IOU_THR = 0.6
WBF_SKIP_BOX_THR = 0.0
WBF_WEIGHTS = {
    "fasterrcnn": 1.0,
    "fasterrcnn_v2": 1.0,
    "retinanet": 1.0,
    "retinanet_v2": 1.0,
}
