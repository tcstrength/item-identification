val_task_id_list = [
    4927, 4923, 4921, 4919, 4912,
    4910, 4908, 4906, 4904, 4902,
    4894, 4888, 4876, 4877, 4870,
    4868, 4861, 4860, 4856, 4852,
    4846, 4838, 4834, 4745, 4748,
    4751, 4753, 4755, 4757, 4766,
    4770, 4769, 4772, 4774, 4777,
    4782, 4785, 4788, 4790, 4795,
    4800, 4806, 4816, 4821, 4831
]

from hcmus.core import appconfig
import requests
import json

def get_task_data(task_id: int):
    headers = {"Authorization": f"Token {appconfig.LABEL_STUDIO_API_KEY}"}
    source_url = f"{appconfig.LABEL_STUDIO_URL}/api/tasks/{task_id}"
    response = requests.get(f"{source_url}/", headers=headers)
    task_data = response.json()
    return task_data


def import_task_to_project(task_data, project_id):
    headers = {"Authorization": f"Token {appconfig.LABEL_STUDIO_API_KEY}"}
    import_url = f"{appconfig.LABEL_STUDIO_URL}/api/projects/{project_id}/tasks"
    new_task = {
        "data": task_data["data"],
        "result": task_data["annotations"][0].get("result")
    }
    response = requests.post(import_url, json=new_task, headers=headers)
    data = response.json()
    task_id = data.get("id")
    print(f"Returned task ID: {task_id}")
    annotate_url = f"{appconfig.LABEL_STUDIO_URL}/api/tasks/{task_id}/annotations"
    response = requests.post(annotate_url, json=new_task, headers=headers)
    print(response.json())



# Get annotation from source project
target_project_id = 7
# source_task_id = val_task_id_list[0]
for source_task_id in val_task_id_list[1:]:
    task_data = get_task_data(source_task_id)
    import_task_to_project(task_data, target_project_id)


for task_id in val_task_id_list[2:]:
    print(task_id)
    headers = {"Authorization": f"Token {appconfig.LABEL_STUDIO_API_KEY}"}
    url = "http://jimica.ddns.net:8080/api/dm/actions?id=delete_tasks&tabID=15&project=11"
    payload = {
        "selectedItems": {
            "all": False,
            "included": [
                task_id
            ]
        },
        "project": "11"
    }
    response = requests.post(url, json=payload, headers=headers)
    print(response.status_code)

