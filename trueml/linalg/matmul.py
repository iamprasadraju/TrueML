import numpy as np


def matmul(matrixA, matrixB):
    """Naive pure-Python matrix multiplication.

    Computes the matrix product :math:`C = A \\times B` using three
    nested loops. This implementation is strictly for educational and
    debugging purposes to demonstrate how matrix multiplication works
    without hidden native C/Fortran layers.

    .. math::

        C_{ij} = \\sum_{k=1}^{\\text{cols}_A} A_{ik} B_{kj}

    Parameters
    ----------
    matrixA : numpy.ndarray of shape (m, k)
        Left operand matrix.
    matrixB : numpy.ndarray of shape (k, n)
        Right operand matrix.

    Returns
    -------
    numpy.ndarray of shape (m, n)
        The resulting product matrix.

    Raises
    ------
    ValueError
        If the number of columns in ``matrixA`` does not match the
        number of rows in ``matrixB``.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.linalg import matmul
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> matmul(A, B)
    array([[19., 22.],
           [43., 50.]])
    """
    rowsA = matrixA.shape[0]
    colsA = matrixA.shape[1]

    rowsB = matrixB.shape[0]
    colsB = matrixB.shape[1]

    if colsA == rowsB:
        matrixC = np.zeros((rowsA, colsB))
        for i in range(rowsA):
            for j in range(colsB):
                acc = 0
                for k in range(colsA):
                    acc += matrixA[i][k] * matrixB[k][j]
                matrixC[i][j] = acc
        return matrixC

    else:
        raise ValueError("no.of cols of A != no.of rows of B")


def npmatmul(matrixA, matrixB):
    """NumPy-native matrix multiplication.

    Delegates directly to NumPy's ``@`` operator, which internally
    hooks into highly optimized BLAS/LAPACK routines (e.g., OpenBLAS,
    MKL) written in C/Fortran.

    Parameters
    ----------
    matrixA : numpy.ndarray of shape (m, k)
        Left operand matrix.
    matrixB : numpy.ndarray of shape (k, n)
        Right operand matrix.

    Returns
    -------
    numpy.ndarray of shape (m, n)
        The resulting product matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.linalg import npmatmul
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> npmatmul(A, B)
    array([[19, 22],
           [43, 50]])
    """
    return matrixA @ matrixB
