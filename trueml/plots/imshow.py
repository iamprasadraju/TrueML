import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


def imshow(
    x: ArrayLike,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    aspect: str = "auto",
):
    """Display a two-dimensional array as an image.

    Args:
        x: Two-dimensional data to display.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.
        cmap: Colormap used to map values to colors.
            Defaults to ``"viridis"``.
        colorbar: If ``True``, display a colorbar.
            Defaults to ``True``.
        aspect: Aspect ratio of the image. Defaults to ``"auto"``.

    Returns:
        None. Displays the image using Matplotlib.

    Example:
        >>> import numpy as np
        >>> data = np.random.rand(20, 20)
        >>> imshow(
        ...     data,
        ...     title="Random Matrix",
        ...     cmap="plasma",
        ... )
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 5))

    im = ax.imshow(
        x,
        cmap=cmap,
        aspect=aspect,
    )

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if colorbar:
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
