import numpy as np

class Tensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)
        self.shape = self.arr.shape
        self.ndim = self.arr.ndim
        self.dtype = self.arr.dtype
        
    def __add__(self, other):
        other = np.asarray(other)
        return Tensor(self.arr + other)

    def __matmul__(self, other):
        other = np.asarray(other)
        return Tensor(self.arr @ other)

    def dot(self, other):
        return Tensor(np.dot(self.arr, other))

    @property
    def T(self):
        return Tensor(self.arr.T)    

    def __getattr__(self, attr):
        return getattr(self.arr, attr)

    def __repr__(self):
        return (
            f"<Tensor {self.ndim}D "
            f"shape={self.shape} "
            f"dtype={self.dtype}>\n"
            f"{self.arr}"
        )
