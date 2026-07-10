import matplotlib.pyplot as plt
from IPython.display import clear_output, display

def history(history):
    """Plot training metrics stored in History object.

    Args:
        history: History object containing training metrics.

    Returns:
        None (displays the plot in Jupyter notebook)

    Example:
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
