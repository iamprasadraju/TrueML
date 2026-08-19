# `trueml.viz` — Computation Graph Visualization

Trace tensor expressions into DAGs and render them as self-contained interactive HTML/SVG (browser or Jupyter). No bloat.

---

## Quick Start

```python
from trueml.tensor import Tensor
from trueml.viz import ComputationGraph
import numpy as np

# 1. build an expression
x = Tensor(np.random.randn(3, 4))
w = Tensor(np.random.randn(4, 2))
b = Tensor(np.ones((2,)))
y = x @ w + b

# 2. trace it
g = ComputationGraph()
g.build(y)

g.view(save="graph.html")         # self-contained HTML file
g.view(open_browser=True)         # opens in default browser
```

---

## API Reference

### `ComputationGraph`

#### `build(tensor)`

Walk the UOp graph rooted at `tensor.uop` via `UOp.toposort()` (iterative post-order DFS). Deduplicates by `id()` — shared inputs appear once. Safe for arbitrarily deep expressions.

Each tensor op owns one [`UOp`](../_uops.py) — a node holding `op`, `dtype`, `shape`, `ndim`, and its `src` UOps. Leaves are `LOAD` nodes and render as tensor boxes; real ops (`ADD`, `MATMUL`, `TRANSPOSE`, ...) render as circles.

```python
g = ComputationGraph()
g.build(output_tensor)

# incremental: call build() again to extend the same graph
g.build(another_tensor)
```

#### `view(save=None, open_browser=False, notebook=None)`

Self-contained HTML/SVG viewer. Toolbar: fit, zoom +/−, toggle captions, export SVG. Hover for tooltips, click to pin, drag to pan, scroll to zoom.

```python
g.view(save="graph.html")                  # save to file
g.view(open_browser=True)                  # open in browser
g.view()                                   # in Jupyter: inline interactive
g.view(save="g.html", open_browser=True)   # both
```

Returns `None`. Jupyter is auto-detected: the full interactive HTML is embedded via an iframe (base64 data URI), preserving all JS interactivity (tooltips, dragging, zooming). For a separate browser window use `open_browser=True`.

---

## Examples

### Linear Layer

```python
x = Tensor(np.random.randn(8, 3))
W = Tensor(np.random.randn(3, 5))
b = Tensor(np.zeros((8, 5)))
out = x @ W + b

g = ComputationGraph()
g.build(out)
g.view(save="linear.html", open_browser=True)
```

### Diamond DAG (Shared Inputs)

Shared tensors are deduplicated — `x` appears once with two outgoing edges.

```python
x = Tensor(np.random.randn(4, 4))
left = x + Tensor(np.ones((4, 4)))
right = x + Tensor(np.zeros((4, 4)))
out = left + right

g = ComputationGraph()
g.build(out)
g.view()
```

### Transpose + Matmul Chain

```python
A = Tensor(np.random.randn(3, 5))
B = Tensor(np.random.randn(3, 7))
out = A.T @ B   # (5,3) @ (3,7) -> (5,7)

g = ComputationGraph()
g.build(out)
g.view(save="transpose_matmul.html")
```

### Incremental Build

Call `build()` multiple times to grow the graph from different outputs.

```python
g = ComputationGraph()

x = Tensor(np.random.randn(2, 3))
w1 = Tensor(np.random.randn(3, 4))
w2 = Tensor(np.random.randn(3, 4))

out1 = x @ w1
out2 = x @ w2  # x is shared

g.build(out1)
g.build(out2)  # extends the graph, deduplicates x
g.view()
```

### Save for Docs / README

```python
g = ComputationGraph()
g.build(model_output)

# interactive HTML for GitHub Pages
g.view(save="docs/graph.html")
```

---

## Shape Propagation

Operation nodes automatically infer output shapes by forward-propagating through the DAG. Supported ops:

| Op | Rule |
|---|---|
| `MATMUL` | NumPy `@` semantics (batched) |
| `TRANSPOSE` | Reversed shape |
| `ADD`, `SUB`, `MUL`, `DIV`, `POW`, `RELU`, `SIGMOID`, `EXP`, `LOG`, `NEG` | `np.broadcast_shapes` |

Unknown ops inherit the first input's shape. Failed inference shows `shape=?`.

---

## Architecture

```
viz/
├── __init__.py    # exports ComputationGraph
└── cgraph.py      # graph builder, layout engine, HTML/SVG viewer
```

The graph nodes come from [`_uops.py`](../_uops.py) — a small `UOp` IR (`op`, `dtype`, `shape`, `ndim`, `src`) shared with the rest of trueml (autodiff will reuse the same IR). `ComputationGraph.build()` walks `tensor.uop.toposort()` and copies the UOps into a `networkx.DiGraph` for layout and rendering.

No external assets. CSS and JS are inlined string constants — the generated HTML is fully self-contained (works with `file://`, no server needed).

### Layout Algorithm

1. **Topological sort** → assign depth (longest path from root)
2. **Barycenter heuristic** → minimize edge crossings (10 iterations, bidirectional)
3. **Column layout** → left-to-right, nodes centered within layers

### HTML Viewer Features

- **Zoom**: scroll wheel (cursor-centered) or +/− buttons
- **Pan**: click and drag
- **Hover**: highlights connected edges, shows tooltip with metadata
- **Click**: pins tooltip to a node (click again or background to unpin)
- **Captions**: toggle metadata labels under op nodes
- **Export**: downloads the SVG as `graph.svg`