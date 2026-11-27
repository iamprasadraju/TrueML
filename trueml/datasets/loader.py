from ._base import _load_dataset


def load_iris(): return _load_dataset("iris")
def load_breast_cancer(): return _load_dataset("breast_cancer")

# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt
def load_diabetes(): return _load_dataset("diabetes")
