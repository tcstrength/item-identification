import os
import json
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

logger.info(f"Load DotEnv: {load_dotenv()}")

def __get_torch_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
TORCH_DEVICE = __get_torch_device()
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_TEMP_DIR = PROJECT_ROOT.joinpath(os.environ.get("LABEL_STUDIO_TEMP_DIR", "local/temp/"))
LABEL_STUDIO_PROJECT_MAPPING = json.loads(os.environ.get("LABEL_STUDIO_PROJECT_MAPPING"))
LABEL_STUDIO_IMPORT_DATA_DIR = Path(PROJECT_ROOT).joinpath(os.environ.get("LABEL_STUDIO_IMPORT_DATA_DIR", "local/import/"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
