from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
  data: np.ndarray
  target: np.ndarray
  fields: list[str]
  input_features: np.ndarray | None = None
  target_features: np.ndarray | None = None
  target_classes: np.ndarray | None = None


_DATA_DIR = Path(__file__).parent / "data"


def _load_dataset(name):
  path = _DATA_DIR / f"{name}.npz"

  with np.load(path, allow_pickle=False) as dataset:
    return Dataset(
      data=dataset["data"],
      input_features=dataset.get("input_features"),
      target=dataset["target"],
      target_features=dataset.get("target_features"),
      target_classes=dataset.get("target_classes"),
      fields=dataset.files,
    )
