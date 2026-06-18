import numpy as np

def matmul(matrixA, matrixB):
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
    return matrixA @ matrixB
