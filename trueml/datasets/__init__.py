from ._base import Dataset, _load_dataset


def load_iris():
    return _load_dataset("iris")


__all__ = [
    "Dataset",
    "load_iris",
]
