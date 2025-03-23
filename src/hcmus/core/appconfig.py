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

def __get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent

PROJECT_ROOT = __get_project_root()
TORCH_DEVICE = __get_torch_device()
LABEL_STUDIO_URL = os.environ["LABEL_STUDIO_URL"]
LABEL_STUDIO_API_KEY = os.environ["LABEL_STUDIO_API_KEY"]
LABEL_STUDIO_TEMP_DIR = Path(PROJECT_ROOT).joinpath(os.environ["LABEL_STUDIO_TEMP_DIR"])
LABEL_STUDIO_PROJECT_MAPPING = json.loads(os.environ["LABEL_STUDIO_PROJECT_MAPPING"])
IMPORT_DATA_DIR = Path(PROJECT_ROOT).joinpath(os.environ["IMPORT_DATA_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
