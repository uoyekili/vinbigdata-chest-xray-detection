import torch

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = "./data"
RAW_TRAIN_CSV = "./data/train.csv"
RAW_TRAIN_DICOM_DIR = "./data/train_dicom"
RAW_TEST_DICOM_DIR = "./data/test_dicom"
TRAIN_PNG_DIR = "./data/train_png"
TEST_PNG_DIR = "./data/test_png"
FOLDS_TRAIN_CSV = "./data/local_train.csv"
HOLDOUT_TEST_CSV = "./data/local_test.csv"
SUBMISSION_CSV = "./data/sample_submission.csv"
OUTPUT_DIR = "./output"
CHECKPOINT_DIR = "./checkpoints"

# =============================================================================
# CLASS & MODEL
# =============================================================================
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
NUM_CLASSES = len(CLASS_NAMES) + 1  # 15 (14 diseases + background)
AVAILABLE_MODELS = ["fasterrcnn", "retinanet"]

# =============================================================================
# TRAINING
# =============================================================================
IMAGE_SIZE = 1024
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
NUM_WORKERS = 4
MAX_GRAD_NORM = 5.0
PATIENCE = 7
NUM_FOLDS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# INFERENCE
# =============================================================================
WBF_IOU_THR = 0.4
WBF_SKIP_BOX_THR = 0.0001
CONF_THRESHOLD = 0.1
FINAL_CONF_THRESHOLD = 0.25  # Final filtering threshold for predictions


def get_checkpoint_path(model_name: str, fold: int) -> str:
    return f"{CHECKPOINT_DIR}/{model_name}_fold{fold}.pth"
