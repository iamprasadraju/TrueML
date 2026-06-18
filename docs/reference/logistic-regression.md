# LogisticRegression

**Module:** `trueml.linearmodel.LogisticRegression`

A binary logistic classifier of the form $p = \sigma(Xw + b)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function. Supports explicit gradient-descent training via the `backward` method.

---

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------|
| Forward | `forward` | `(X) → prob` | $p = \sigma(Xw + b)$ |
| Update | `backward` | `(dw, db) → None` | $w \gets w - \eta \cdot dw$ <br> $b \gets b - \eta \cdot db$ |

---

## Constructor

```python
LogisticRegression(n_features: int, lr: float = 0.01)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_features` | `int` | — | Dimensionality $d$ of the input space. |
| `lr` | `float` | `0.01` | Learning rate $\eta$. |

### Initial State

| Variable | Shape | Initial Value |
|----------|-------|---------------|
| `weights` | `(n_features,)` | $\mathcal{N}(0, 0.01^2)$ |
| `bias` | `scalar` | `0.0` |

---

## Method: `forward`

```python
model.forward(X: np.ndarray) -> np.ndarray
```

Computes the predicted probability $p = \sigma(Xw + b)$.

**Input:** $X \in \mathbb{R}^{n \times d}$ — design matrix.

**Output:** $p \in [0, 1]^{n}$ — predicted probabilities.

**Formula:**
$$
z_i = \sum_{j=1}^{d} X_{ij} w_j + b, \qquad
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

---

## Method: `backward`

```python
model.backward(dw: np.ndarray, db: float) -> None
```

Updates the model parameters via gradient descent.

**Input:** $dw \in \mathbb{R}^{d}$ — gradient with respect to $w$.

**Input:** $db \in \mathbb{R}$ — gradient with respect to $b$.

**Update rule:**
$$
\begin{aligned}
w &\gets w - \eta \cdot dw \\
b &\gets b - \eta \cdot db
\end{aligned}
$$

---

## Typical Loss Pairing

LogisticRegression is typically paired with a loss function that accepts probabilities and binary targets $y \in \{0, 1\}$. A common choice is the binary cross-entropy:

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \bigl[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \bigr]
$$

The gradient of this loss with respect to the logit $z$ is:

$$
\frac{\partial L}{\partial z} = \frac{1}{n} (p - y)
$$

which can be passed through the chain rule to obtain $dw$ and $db$:

$$
dw = \frac{1}{n} X^\mathsf{T} (p - y), \qquad db = \frac{1}{n} \sum (p - y)
$$
