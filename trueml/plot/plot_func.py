import numpy as np
import matplotlib.pyplot as plt


def function2d(func, x_range: tuple = (-10, 10), resolution: int = 100, figsize: tuple = (6, 4), title: str = None) -> None:
    """Plot a 2D function.
    
    Args:
        func: The function to plot.
        x_range: The range of x values to plot.
        resolution: The number of points to plot.
        figsize: The size of the figure.
        title: The title of the plot.
    
    Returns:
        None
    
    Example:
        >>> def f(x):
        ...     return x**2
        >>> function2d(f)
    """
    x = np.linspace(*x_range, resolution)
    y = func(x)

    # get the function name
    plot_name = func.__qualname__.split(".")[0]
    fig, ax = plt.subplots()

    ax.plot(x, y)

    ax.set_title(title or plot_name)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend([plot_name])

    plt.show()


def function3d(func, x_range: tuple = (-10, 10), y_range: tuple = (-10, 10), resolution: int = 100, figsize: tuple =(6, 6), cmap: str ="viridis", title: str = None) -> None:
    """Plot a 3D function.
    
    Args:
        func: The function to plot.
        x_range: The range of x values to plot.
        y_range: The range of y values to plot.
        resolution: The number of points to plot.
        figsize: The size of the figure.
        cmap: The colormap to use for the plot.
        title: The title of the plot.
    
    Returns:
        None
    
    Example:
        >>> def f(x, y):
        ...     return x**2 + y**2
        >>> function3d(f)
    """
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)

    X, Y = np.meshgrid(x, y, copy=False)
    Z = func(X, Y)

    plot_name = func.__qualname__.split(".")[0]
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(projection='3d')
    # cmap: for color mapping of the surface (e.g., 'viridis', 'plasma', 'coolwarm', 'gist_earth')
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", linewidth=0, antialiased=False, shade=False, rcount=40, ccount=40)

    if title is None:
        title = plot_name
    ax.set_title(title)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()
