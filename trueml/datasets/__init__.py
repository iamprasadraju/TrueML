from ._base import Dataset, _load_dataset


def load_iris():
    return _load_dataset("iris")


def load_breast_cancer():
    return _load_dataset("breast_cancer")


__all__ = [
    "Dataset",
    "load_breast_cancer",
    "load_iris",
]
