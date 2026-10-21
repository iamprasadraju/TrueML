from ._base import Dataset, _load_dataset
from .loader import load_breast_cancer, load_diabetes, load_iris

__all__ = [
  "Dataset",
  "_load_dataset",
  "load_breast_cancer",
  "load_diabetes",
  "load_iris",
]
