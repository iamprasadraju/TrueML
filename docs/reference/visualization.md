# Visualization

TrueML provides two visualization backends to help you analyze model behavior, loss surfaces, and training metrics:

1. **Matplotlib Backend (`trueml.viz`)**: Standard, static rendering suitable for scripts and simple plotting.
2. **Plotly Backend (`trueml.visualize`)**: Interactive, dynamic rendering suitable for Jupyter Notebooks and rich HTML outputs.

---

## Matplotlib Backend (`trueml.viz`)

**Module:** `trueml.viz`

### `plot2d`

```python
trueml.viz.plot2d(func, x_range=(-10, 10), resolution=100, figsize=(6, 4), title=None) -> None
```

Plots a 2D line graph of a single-variable function.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `callable` | — | The function to plot. Must accept and return a 1D NumPy array. |
| `x_range` | `tuple` | `(-10, 10)` | The `(min, max)` range of x-axis values. |
| `resolution` | `int` | `100` | The number of points to sample across the range. |
| `figsize` | `tuple` | `(6, 4)` | The figure dimensions. |
| `title` | `str` | `None` | The plot title. Defaults to the function's name. |

### `plot3d`

```python
trueml.viz.plot3d(func, x_range=(-10, 10), y_range=(-10, 10), resolution=100, figsize=(6, 6), cmap="gist_earth", title=None) -> None
```

Plots a 3D surface graph of a two-variable function.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `callable` | — | The function to plot. Must accept two 2D meshgrid arrays. |
| `x_range` | `tuple` | `(-10, 10)` | The range of x-axis values. |
| `y_range` | `tuple` | `(-10, 10)` | The range of y-axis values. |
| `resolution` | `int` | `100` | The number of points to sample across each axis. |
| `figsize` | `tuple` | `(6, 6)` | The figure dimensions. |
| `cmap` | `str` | `"gist_earth"` | The Matplotlib colormap applied to the surface. |

### `plot_metrics`

```python
trueml.viz.plot_metrics(epochs, *args, titles=None, labels=None, **kwargs)
```

Dynamically plots metrics across epochs during a training loop. It supports both script (`plt.ion()`) and Jupyter Notebook (`clear_output()`) environments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | `array-like` | — | The array of epoch numbers (the X-axis). |
| `*args` | `array-like` | — | Positional arrays representing metric values (Y-axis). |
| `titles` | `list` | `None` | List of titles for the subplots. |
| `labels` | `list` | `None` | List of legend labels for the plotted lines. |
| `**kwargs` | `array-like` | — | Keyword arguments representing grouped metrics. |

**Example:**
```python
from trueml.viz import plot_metrics

# Positional mode
plot_metrics(epochs, train_loss, train_acc, titles=['Loss', 'Accuracy'])

# Keyword mode (overlaying multiple lines per chart)
plot_metrics(epochs, Loss=[train_loss, val_loss], labels=[('Train', 'Val')])
```

---

## Plotly Backend (`trueml.visualize`)

**Module:** `trueml.visualize`

### `LivePlot`

```python
class trueml.visualize.LivePlot(title="Training Loss", labels=None, xlabel="Epoch", ylabel="Loss")
```

An object-oriented, Jupyter-native dynamic plotter that uses `plotly.graph_objects.FigureWidget` to update charts in real-time without screen flickering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | `"Training Loss"` | The chart title. |
| `labels` | `list` | `None` | List of legend labels. Dictates the number of tracked lines. |
| `xlabel` | `str` | `"Epoch"` | The X-axis title. |
| `ylabel` | `str` | `"Loss"` | The Y-axis title. |

**Methods:**
- `update(*values)`: Appends the provided scalar values to the lines and refreshes the widget. You can also call the instance directly: `plot(val1, val2)`.

**Example:**
```python
from trueml.visualize import LivePlot
plot = LivePlot("Performance", labels=["Train", "Val"])

for epoch in range(100):
    # ... training ...
    plot(train_loss, val_loss)
```

### `viz2d` & `viz3d`

```python
trueml.visualize.viz2d(func, x_range=(-5, 5), resolution=50, figsize=(500, 500))
trueml.visualize.viz3d(func, x_range=(-5, 5), y_range=(-5, 5), resolution=50, figsize=(500, 500))
```

Interactive, Plotly-powered equivalents to `plot2d` and `plot3d`. Note that `figsize` is provided in pixels (width, height) rather than inches.

---

## Backend Comparison

| Feature | Matplotlib (`trueml.viz`) | Plotly (`trueml.visualize`) |
|---------|----------------------------|-------------------------------|
| **Interactivity** | Static images | Hover, zoom, pan |
| **Environment** | Terminal scripts, Notebooks | Best in Jupyter Notebooks |
| **Live Updates** | Screen refresh (`clear_output`) | Native DOM widget updates |

## See Also
- [MSEloss](mse-loss.md) — Visualizing the loss surface of MSE using `viz3d`.
- [activations](activations.md) — Visualizing nonlinearities like sigmoid using `viz2d`.
