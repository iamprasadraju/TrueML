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
    """
    Create a scatter plot.

    A scatter plot visualizes the relationship between two numerical variables
    by representing each observation as a point. It is commonly used to identify
    correlations, trends, clusters, and outliers.

    Args:
        x: Values for the x-axis.
        y: Values for the y-axis.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        color: Color of the markers. Defaults to ``"C0"``.
        alpha: Transparency of the markers, ranging from ``0.0`` (fully
            transparent) to ``1.0`` (fully opaque). Defaults to ``0.7``.
        size: Marker size in points squared. Defaults to ``40``.
        marker: Marker style (for example, ``"o"``, ``"s"``, ``"^"``,
            or ``"x"``). Defaults to ``"o"``.
        grid: If ``True``, display grid lines. Defaults to ``False``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.

    Returns:
        None. Displays the scatter plot using Matplotlib.

    Example:
        >>> scatter(
        ...     x=heights,
        ...     y=weights,
        ...     title="Height vs Weight",
        ...     xlabel="Height (cm)",
        ...     ylabel="Weight (kg)",
        ...     color="royalblue",
        ...     alpha=0.6,
        ...     size=50,
        ...     marker="o",
        ...     grid=True,
        ...     figsize=(8, 5),
        ... )
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
