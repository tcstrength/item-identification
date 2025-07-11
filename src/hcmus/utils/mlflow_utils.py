import os
import json
import mlflow
import torch

def log_json_artifact(obj, filename: str):
    with open(filename, "w") as f:
        json.dump(obj, f)
    mlflow.log_artifact(filename, "json")
    os.remove(filename)


def log_torch_artifact(obj, filename: str):
    torch.save(obj, filename)
    mlflow.log_artifact(filename, "torch")
    os.remove(filename)
