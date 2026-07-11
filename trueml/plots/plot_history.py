import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def history(history):
    """Plot training metrics recorded during model training.

    Visualizes one or more metrics stored in a ``History`` object, with each
    metric displayed in its own subplot against training epochs.

    Args:
        history: A ``History`` object containing recorded metric values and
            corresponding epoch numbers.

    Returns:
        None. Displays the training history in a Jupyter notebook.

    Example:
        >>> model.fit(X, y, epochs=100)
        >>> history(model.history)
    """

    if not history.history:
        return

    clear_output(wait=True)

    fig, axes = plt.subplots(
        1, len(history.history), figsize=(5 * len(history.history), 4)
    )

    if len(history.history) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, history.history.items()):
        ax.plot(history.epochs, values, linewidth=2, label=name)

        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    display(fig)

    plt.close(fig)
