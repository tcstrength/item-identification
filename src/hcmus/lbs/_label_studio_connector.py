import os
import requests
import hashlib
from tqdm import tqdm
from typing import List, Dict
from loguru import logger
from hcmus.core import pconf
from hcmus.lbs import LabelStudioTask

class LabelStudioConnector:
    def __init__(self, url: str, api_key: str, project_id: int, verbose: bool = False):
        self._url = url.strip()
        self._project_id = project_id
        self._headers = {
            "Authorization": f"Token {api_key}"
        }
        self._verbose = verbose

    def _build_endpoint(self, component: str) -> str:
        endpoint_fmt = "{url}/api/projects/{project_id}/{component}"
        endpoint = endpoint_fmt.format(
            url=self._url,
            project_id=self._project_id,
            component=component
        )
        return endpoint

    def get_tasks(self, page_from: int, page_to: int, page_size: int = 100) -> List[LabelStudioTask]:
        tasks = []
        endpoint = self._build_endpoint("tasks")
        page = page_from
        bar = tqdm(range(page_from, page_to + 1), "Loading tasks")

        for page in bar:
            response = requests.get(
                endpoint,
                headers=self._headers,
                params={"page": page, "page_size": page_size},
            )

            if response.status_code != 200:
                logger.warning("Error fetching tasks:" + response.text)
                break

            page_tasks = response.json()
            page_tasks = [LabelStudioTask.model_validate(x) for x in page_tasks]
            if not page_tasks:
                break

            tasks.extend(page_tasks)

        return tasks

    def get_image(self, task: LabelStudioTask) -> str:
        image_path = task.data["image"].strip("/")
        image_url = f"{self._url}/{image_path}"
        file_ext = os.path.splitext(image_path)[1]
        file_name = hashlib.md5(image_path.encode()).hexdigest()
        save_path = pconf.LABEL_STUDIO_TEMP_DIR.joinpath(file_name)
        save_path = str(save_path) + file_ext

        if os.path.exists(save_path):
            if self._verbose:
                logger.info(f"Skip `{save_path}`.")
            return save_path

        os.makedirs(pconf.LABEL_STUDIO_TEMP_DIR, exist_ok=True)

        if self._verbose:
            logger.info(f"Save to `{save_path}`.")

        response = requests.get(image_url, headers=self._headers, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        return save_path

if __name__ == "__main__":
    from hcmus.core import pconf
    connector = LabelStudioConnector(
        url=pconf.LABEL_STUDIO_URL,
        api_key=pconf.LABEL_STUDIO_API_KEY,
        project_id=pconf.LABEL_STUDIO_PROJECT_ID
    )
    tasks = connector.get_tasks(1, 1, 10)

    tasks[0]
    connector.get_image(tasks[0])
