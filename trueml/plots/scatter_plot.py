import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


def scatter(
    x: ArrayLike,
    y: ArrayLike,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = "C0",
    alpha: float = 0.7,
    size: int = 40,
    marker: str = "o",
    grid: bool = False,
    figsize: tuple[int, int] | None = None,
):
    """Create a scatter plot.
    
    Args:
        x: The x-coordinates of the data points.
        y: The y-coordinates of the data points.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        color: The color of the points.
        alpha: The transparency of the points.
        size: The size of the points.
        marker: The marker style of the points.
        grid: Whether to show a grid.
        figsize: The size of the figure.
    
    Returns:
        None (displays the plot using matplotlib)

    Example:
        >>> scatter(x, y, title="Scatter Plot", xlabel="X", ylabel="Y")
        >>> scatter(x, y, color="red", alpha=0.5, size=50)
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 4))

    ax.scatter(
        x,
        y,
        s=size,
        c=color,
        marker=marker,
        alpha=alpha,
    )

    if title:
        ax.set_title(title or "")

    if xlabel:
        ax.set_xlabel(xlabel or "")

    if ylabel:
        ax.set_ylabel(ylabel or "")

    if grid:
        ax.grid(True, alpha=alpha)

    plt.tight_layout()
    plt.show()
