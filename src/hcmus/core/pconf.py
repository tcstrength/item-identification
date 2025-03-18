import os
from loguru import logger
from dotenv import load_dotenv

logger.info(f"Load DotEnv: {load_dotenv()}")
from pathlib import Path

def __get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent

PROJECT_ROOT = __get_project_root()
LABEL_STUDIO_URL=os.environ["LABEL_STUDIO_URL"]
LABEL_STUDIO_API_KEY=os.environ["LABEL_STUDIO_API_KEY"]
LABEL_STUDIO_PROJECT_ID=os.environ["LABEL_STUDIO_PROJECT_ID"]
LABEL_STUDIO_TEMP_DIR=Path(PROJECT_ROOT).joinpath(os.environ["LABEL_STUDIO_TEMP_DIR"])
IMPORT_DATA_DIR=Path(PROJECT_ROOT).joinpath(os.environ["IMPORT_DATA_DIR"])
