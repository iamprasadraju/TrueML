from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


def contour(
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] | None = None,
    levels: int = 20,
    cmap: str = "viridis",
    colorbar: bool = True,
):
    """Create a contour plot.

    A contour plot visualizes a three-dimensional surface on a two-dimensional
    plane using contour lines. Each contour line connects points having the
    same value.

    Args:
        X: Two-dimensional x-coordinate grid.
        Y: Two-dimensional y-coordinate grid.
        Z: Two-dimensional values evaluated on ``(X, Y)``.
        title: Title of the plot. Defaults to ``None``.
        xlabel: Label for the x-axis. Defaults to ``None``.
        ylabel: Label for the y-axis. Defaults to ``None``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``None``.
        levels: Number of contour levels. Defaults to ``20``.
        cmap: Colormap used for the contour lines.
            Defaults to ``"viridis"``.
        colorbar: If ``True``, display a colorbar.
            Defaults to ``True``.

    Returns:
        None. Displays the contour plot using Matplotlib.

    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = np.linspace(-5, 5, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X**2 + Y**2
        >>> contour(
        ...     X,
        ...     Y,
        ...     Z,
        ...     title="Contour Plot",
        ... )
    """

    fig, ax = plt.subplots(figsize=figsize or (6, 5))

    cs = ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
    )

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if colorbar:
        fig.colorbar(cs, ax=ax)

    plt.tight_layout()
    plt.show()