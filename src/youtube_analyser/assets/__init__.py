"""The models and the default people.toml shipped with the package.

Models are looked up in YTA_MODELS (a folder) when that is set, so a deployment can keep them
elsewhere (or add ones not shipped); otherwise in this package's models/ folder.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PEOPLE_TOML = HERE / "people.toml"


def models_dir():
    return Path(os.environ["YTA_MODELS"]) if os.environ.get("YTA_MODELS") else HERE / "models"


def model_path(name):
    """The path of a model file by name (e.g. "yolo11s-pose.onnx")."""
    return models_dir() / name
