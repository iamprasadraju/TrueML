from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    data: np.ndarray
    target: np.ndarray
    feature_names: np.ndarray | None = None
    target_names: np.ndarray | None = None


_DATA_DIR = Path(__file__).parent / "data"


def _load_dataset(name):
    path = _DATA_DIR / f"{name}.npz"

    with np.load(path, allow_pickle=False) as data:
        return Dataset(
            data=data["data"],
            target=data["target"],
            feature_names=data.get("feature_names"),
            target_names=data.get("target_names"),
        )
