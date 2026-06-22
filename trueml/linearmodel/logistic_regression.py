class LogisticRegression:
    def __init__(self, n_features: int, lr: float = 0.01) -> None:
        pass
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def grad(self, X: np.ndarray, loss_gradient: np.ndarray) -> tuple[np.ndarray, float]:
        pass

    def backward(self, dw: np.ndarray, db: float) -> None:
        pass

        