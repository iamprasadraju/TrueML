import numpy as np


def lossfn_plot(func, optimum=(0, 0), cmap="viridis", figsize=(8, 6)):
    import matplotlib.pyplot as plt

    np.random.seed(40)

    # sample data
    x = np.random.randn(1000, 1)

    # default to (0, 0) minimum
    min_w, min_b = optimum
    # Add a little noise
    y = min_w * x + min_b + 0.5 * np.random.randn(1000, 1)

    w = np.linspace(-10, 10, 100)
    b = np.linspace(-10, 10, 100)
    W, B = np.meshgrid(w, b)

    Loss = np.zeros_like(W)

    # loss for 10000 models (100 * 100)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = x * W[i, j] + B[i, j]
            Loss[i, j] = func(y, y_pred)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(W, B, Loss, cmap=cmap)

    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Bias (b)")
    ax.set_zlabel("Loss")
    func_name = getattr(func, "__qualname__", type(func).__qualname__)
    ax.set_title(func_name)
    plt.show()
