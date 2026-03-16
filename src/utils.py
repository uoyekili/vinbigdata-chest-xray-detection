import os
import random
import numpy as np
import torch

from src import config


def seed_everything(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_trained_models():
    available = []
    for model_name in config.AVAILABLE_MODELS:
        path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}.pth")
        if os.path.exists(path):
            available.append(model_name)
    return available
