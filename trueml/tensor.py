import numpy as np

from trueml._uops import UOp


class Tensor:
    def __init__(self, arr, uop=None):
        self.arr = np.asarray(arr)
        self.shape = self.arr.shape
        self.ndim = self.arr.ndim
        self.dtype = self.arr.dtype

        if uop is None:
            uop = UOp(op="LOAD", dtype=self.dtype, shape=self.shape, ndim=self.ndim)
        self.uop = uop

    @classmethod
    def _make(cls, arr, op, *srcs):
        arr = np.asarray(arr)
        uop = UOp(
            op=op,
            dtype=arr.dtype,
            shape=arr.shape,
            ndim=arr.ndim,
            src=tuple(s.uop for s in srcs),
        )
        return cls(arr, uop=uop)

    @property
    def metadata(self):
        return {
            "shape": self.shape,
            "ndim": self.ndim,
            "dtype": self.dtype,
        }

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return Tensor._make(self.arr + other.arr, "ADD", self, other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return Tensor._make(self.arr @ other.arr, "MATMUL", self, other)

    @property
    def T(self):
        return Tensor._make(self.arr.T, "TRANSPOSE", self)

    def __getattr__(self, attr):
        return getattr(self.arr, attr)

    def __repr__(self):
        return (
            f"<Tensor {self.ndim}D shape={self.shape} dtype={self.dtype}>\n{self.arr}"
        )
