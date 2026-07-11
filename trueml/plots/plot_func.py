import numpy as np
import matplotlib.pyplot as plt


def function2d(
    func,
    x_range: tuple = (-10, 10),
    resolution: int = 100,
    figsize: tuple = (6, 4),
    title: str = None,
) -> None:
    """Plot a one-dimensional mathematical function.

    Evaluates a function over a specified range of x-values and displays the
    result as a 2D line plot.

    Args:
    func: A callable that accepts a NumPy array of x-values and returns
        the corresponding y-values.
    x_range: Lower and upper bounds of the x-axis as ``(min, max)``.
        Defaults to ``(-10, 10)``.
    resolution: Number of evenly spaced points used to evaluate the
        function. Higher values produce smoother curves. Defaults to
        ``100``.
    figsize: Figure size as ``(width, height)`` in inches.
        Defaults to ``(6, 4)``.
    title: Title of the plot. If ``None``, the function name is used.
        Defaults to ``None``.

    Returns:
        None. Displays the 2D function plot.

    Example:
        >>> def quadratic(x):
        ...     return x**2
        ...
        >>> function2d(
        ...     quadratic,
        ...     x_range=(-5, 5),
        ...     resolution=200,
        ...     title="Quadratic Function",
        ... )
    """
    x = np.linspace(*x_range, resolution)
    y = func(x)

    # get the function name
    plot_name = func.__qualname__.split(".")[0]
    fig, ax = plt.subplots()

    ax.plot(x, y)

    ax.set_title(title or plot_name)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend([plot_name])

    plt.show()


def function3d(
    func,
    x_range: tuple = (-10, 10),
    y_range: tuple = (-10, 10),
    resolution: int = 100,
    figsize: tuple = (6, 6),
    cmap: str = "viridis",
    title: str = None,
) -> None:
    """Plot a two-dimensional mathematical function as a 3D surface.

    Evaluates a function over a grid of x- and y-values and displays the
    resulting surface in three dimensions.

    Args:
        func: A callable that accepts two NumPy arrays ``(X, Y)`` and returns
            the corresponding z-values.
        x_range: Lower and upper bounds of the x-axis as ``(min, max)``.
            Defaults to ``(-10, 10)``.
        y_range: Lower and upper bounds of the y-axis as ``(min, max)``.
            Defaults to ``(-10, 10)``.
        resolution: Number of evenly spaced points along each axis used to
            evaluate the function. Higher values produce smoother surfaces.
            Defaults to ``100``.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``(6, 6)``.
        cmap: Colormap used to color the surface. Defaults to ``"viridis"``.
        title: Title of the plot. If ``None``, the function name is used.
            Defaults to ``None``.

    Returns:
        None. Displays the 3D surface plot.

    Example:
        >>> def paraboloid(x, y):
        ...     return x**2 + y**2
        ...
        >>> function3d(
        ...     paraboloid,
        ...     x_range=(-3, 3),
        ...     y_range=(-3, 3),
        ...     resolution=150,
        ...     cmap="plasma",
        ...     title="Paraboloid",
        ... )
    """
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)

    X, Y = np.meshgrid(x, y, copy=False)
    Z = func(X, Y)

    plot_name = func.__qualname__.split(".")[0]
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(projection="3d")
    # cmap: for color mapping of the surface (e.g., 'viridis', 'plasma', 'coolwarm', 'gist_earth')
    ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        shade=False,
        rcount=40,
        ccount=40,
    )

    if title is None:
        title = plot_name
    ax.set_title(title)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    plt.show()
