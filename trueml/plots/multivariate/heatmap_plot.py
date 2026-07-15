import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def heatmap(
    x: ArrayLike,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
):
    """Create a heatmap.

    A heatmap visualizes a two-dimensional matrix using colors, where each
    cell's color represents its numerical value. It is commonly used to
    display correlation matrices, confusion matrices, and other tabular
    numerical data.

    Args:
        x: Two-dimensional numerical data to visualize.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.
        cmap: Colormap used to represent values. Defaults to
            ``"viridis"``.
        colorbar: If ``True``, display a colorbar indicating the mapping
            between colors and values. Defaults to ``True``.
        xticklabels: Labels for the x-axis ticks. Defaults to ``None``.
        yticklabels: Labels for the y-axis ticks. Defaults to ``None``.

    Returns:
        None. Displays the heatmap using Matplotlib.

    Example:
        >>> corr = np.corrcoef(X.T)
        >>> heatmap(
        ...     corr,
        ...     title="Correlation Matrix",
        ...     xticklabels=feature_names,
        ...     yticklabels=feature_names,
        ...     cmap="coolwarm",
        ... )
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 5))

    im = ax.imshow(x, cmap=cmap, aspect="auto")

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    if colorbar:
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
