from ._base import _load_dataset


def load_iris(): return _load_dataset("iris")
def load_breast_cancer(): return _load_dataset("breast_cancer")
def load_diabetes(): return _load_dataset("diabetes")
