import os
import requests
import hashlib
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import List, Dict
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from hcmus.lbs import LabelStudioTask

class LabelStudioConnector:
    def __init__(self, url: str, api_key: str, project_id: int, temp_dir: str, verbose: bool = False):
        self._url = url.strip()
        self._project_id = project_id
        self._temp_dir = Path(temp_dir)
        self._headers = {
            "Authorization": f"Token {api_key}"
        }
        self._verbose = verbose

    def _build_endpoint(self, component: str) -> str:
        endpoint_fmt = "{url}/api/{component}"
        endpoint = endpoint_fmt.format(
            url=self._url,
            component=component
        )
        return endpoint

    def extract_labels(self, tasks: List[LabelStudioTask]) -> Dict[str, int]:
        labels = {}
        for task in tasks:
            if task.is_labeled == 0: continue

            for ann in task.annotations:
                for result in ann.result:
                    tmp = result.value.rectanglelabels
                    if len(tmp) > 1:
                        logger.warning(f"Unexpected labels: {tmp}")
                    elif len(tmp) == 0:
                        logger.warning(f"No label found: {task.id}")
                        continue

                    for l in tmp:
                        if l not in labels:
                            labels[l] = len(labels)
        return labels

    def transform_labels(
        self,
        dataset: List[Dict],
        label2idx: Dict[str, int],
        valid_labels: List[str],
        default_label: str = "object"
    ) -> List[Dict]:
        """Transform labels in dataset downloaded from self.download_dataset"""
        dataset = [x.copy() for x in dataset if x.get("target").get("labels")]
        idx2label = {v: k for k, v in label2idx.items()}
        for item in dataset:
            new_labels = []
            for idx in item.get("target").get("labels"):
                label_str = idx2label.get(idx, default_label)
                if label_str in valid_labels:
                    new_labels.append(idx2label[idx])
                else:
                    # Compatible with SKU110k
                    new_labels.append(default_label)
            item.get("target")["labels"] = new_labels
        return dataset

    def download_dataset(self, tasks: List[LabelStudioTask], labels: Dict[str, int] = None) -> List[Dict]:
        if labels is None:
            labels = self.extract_labels(tasks)
            logger.info(f"No labels input, auto extract {len(labels)} labels.")

        def download_one(task):
            path = self.get_image(task)
            tg_boxes = []
            tg_labels = []
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)
            width, height = image.size

            for ann in task.annotations:
                ann = task.annotations[0]
                for result in ann.result:
                    rect = result.value
                    label = rect.rectanglelabels[0]
                    label_idx = labels.get(label, -1)
                    x_min = width * (rect.x / 100)
                    y_min = height * (rect.y / 100)
                    x_max = x_min + (width * (rect.width / 100))
                    y_max = y_min + (height * (rect.height / 100))
                    tg_boxes.append([x_min, y_min, x_max, y_max])
                    tg_labels.append(label_idx)

            target = {
                "boxes": tg_boxes,
                "labels": tg_labels
            }

            return {
                "image": path,
                "task": task,
                "target": target
            }

        images = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(download_one, task): task for task in tasks}

            for future in tqdm(as_completed(futures), total=len(tasks), desc="Downloading images"):
                images.append(future.result())
        return images

    def get_total_tasks(self) -> int:
        endpoint = self._build_endpoint("projects")
        response = requests.get(
            endpoint,
            headers=self._headers,
            params={
                "ids": self._project_id,
                "include": "task_number"
            }
        )
        data = response.json()
        results = data.get("results")

        if len(results) == 0:
            logger.warning("No project found.")
            return 0

        return results[0].get("task_number")

    def get_tasks(
        self,
        page_from: int = 1,
        page_to: int = 100,
        page_size: int = 100
    ) -> List[LabelStudioTask]:
        tasks = []
        component = f"projects/{self._project_id}/tasks"
        endpoint = self._build_endpoint(component)
        page = page_from
        total_tasks = self.get_total_tasks()

        if (page_to - page_from + 1) * page_size > total_tasks:
            logger.warning(
                f"Page size is too large, only {total_tasks} tasks available."
            )
            page_to = total_tasks // page_size + 1
            logger.info(f"New `page_to` applied: {page_to}")

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

            # logger.info(response.content)

            page_tasks = response.json()
            page_tasks = [LabelStudioTask.model_validate(x) for x in page_tasks]
            # page_tasks = [x for x in page_tasks if x.is_labeled != 0]
            # print(page_tasks)w

            if not page_tasks:
                # logger.warning("Return empty.")
                break

            tasks.extend(page_tasks)

        return tasks

    def get_image(self, task: LabelStudioTask) -> str:
        image_path = task.data["image"].strip("/")
        image_url = f"{self._url}/{image_path}"
        file_ext = os.path.splitext(image_path)[1]
        file_name = hashlib.md5(image_path.encode()).hexdigest()
        save_path = self._temp_dir.joinpath(file_name)
        save_path = str(save_path) + file_ext

        if os.path.exists(save_path):
            if self._verbose:
                logger.info(f"Skip `{save_path}`.")
            return save_path

        os.makedirs(self._temp_dir, exist_ok=True)

        if self._verbose:
            logger.info(f"Save to `{save_path}`.")

        response = requests.get(image_url, headers=self._headers, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        return save_path

if __name__ == "__main__":
    from hcmus.core import appconfig
    connector = LabelStudioConnector(
        url=appconfig.LABEL_STUDIO_URL,
        api_key=appconfig.LABEL_STUDIO_API_KEY,
        project_id=appconfig.LABEL_STUDIO_PROJECT_ID
    )
    tasks = connector.get_tasks(1, 1, 10)

    tasks[0]
    connector.get_image(tasks[0])
