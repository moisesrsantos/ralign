ralign — Rashomon Alignment
===========================

**ralign** is a Python package implementing *Rashomon Alignment* (RA), a framework for measuring functional similarity between classification models by comparing their decision boundaries rather than their predictive accuracy.

> Soares, C., van der Putten, P., Pfahringer, B., & Santos, M. (2026).
> **Rashomon Alignment**.
> *Faculty of Engineering, University of Porto / LIACC / Fraunhofer AICOS / BrightFactory / Leiden University / University of Waikato.*

---

## Motivation

Accuracy alone does not reveal whether two models err in the same regions of the instance space or define structurally similar decision boundaries. **Rashomon Alignment** addresses this gap by comparing model outputs directly. Two variants are defined:

| Variant | Symbol | Reference distribution | Description |
|---------|--------|----------------------|-------------|
| Distributional RA | **dRA** | Data-generating distribution P(X) | Agreement on a held-out test set; ecologically valid but sensitive to distribution shift |
| Geometric RA | **gRA** | Uniform distribution U(F) over the instance space | Agreement on uniformly sampled synthetic data; invariant to distribution shift, captures the full instance space |

Formally, for any finite evaluation set X = {x₁, …, xₙ}, the empirical agreement is:

```
θ_X(Mₐ, M_b) = (1/n) Σᵢ 1[Mₐ(xᵢ) = M_b(xᵢ)]  ∈ [0, 1]
```

- **dRA** estimates this expectation with xᵢ drawn from the test partition.
- **gRA** estimates it with xᵢ drawn uniformly from the bounding box of the training features.

A high dRA with low gRA indicates models that agree where data lies but disagree elsewhere — a form of apparent similarity that is fragile under distribution shift.

---

## Installation

```bash
pip install ralign
```

For development (includes test and build tools):

```bash
git clone https://github.com/mmrsantos/ralign.git
cd ralign
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.8, numpy, pandas, scikit-learn, matplotlib, umap-learn.

---

## Quick Start

```python
import numpy as np
from ralign import RashomonAlignment

# Predictions from two models on a held-out test set
pred_unpruned = np.array([0, 1, 1, 0, 1, 1, 0])
pred_pruned   = np.array([0, 1, 0, 0, 1, 1, 1])

# Distributional RA (agreement on test data)
dra = RashomonAlignment.distributional(pred_unpruned, pred_pruned)
print(f"dRA: {dra:.3f}")

# Geometric RA (agreement over the full instance space)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

X = pd.DataFrame({"f1": [1.2, 3.4, 2.1, 0.5, 4.0],
                  "f2": [0.3, 1.1, 2.2, 3.3, 0.8]})
y = [0, 1, 1, 0, 1]

unpruned = DecisionTreeClassifier(min_samples_split=2, random_state=42).fit(X, y)
pruned   = DecisionTreeClassifier(min_samples_split=5, random_state=42).fit(X, y)

result = RashomonAlignment.geometric(
    models=[unpruned, pruned],
    df=X,
    feature_cols=list(X.columns),
    n_samples=1000,
    random_state=42,
)
print(f"gRA: {result['alignment']:.3f}")
```

---

## Paper Experiment: Pruned vs. Unpruned Decision Trees

The paper benchmarks RA on 92 UCI datasets, comparing pruned and unpruned `DecisionTreeClassifier` models under 5-fold cross-validation. The experiment in [`examples/decision_tree_pruning_rashomon.py`](examples/decision_tree_pruning_rashomon.py) reproduces this setup exactly.

### Running the experiment

```bash
python examples/decision_tree_pruning_rashomon.py
```

This writes `examples/dt_pruning_rashomon_results.csv` with columns `dataset`, `accdiff` (pruned − unpruned accuracy), `dRA`, and `gRA`.

### Generating the figures

```bash
python examples/visualize_rashomon_results.py
```

Produces the following plots in `examples/img/`:

| File | Description |
|------|-------------|
| `hist_gRA.png` | Distribution of gRA across datasets |
| `hist_dRA.png` | Distribution of dRA across datasets |
| `hist_accdiff.png` | Distribution of accuracy difference |
| `scatter_gRA_accdiff.png` | gRA vs. accuracy difference (r ≈ 0.514) |
| `scatter_gRA_dRA.png` | gRA vs. dRA (r ≈ 0.745) |
| `quadrants_gRA_accdiff.png` | Quadrant analysis by medians of gRA and \|Δacc\| |

### Key findings from the paper

**E1 — gRA vs. accuracy difference (r = 0.514):**
Higher geometric alignment between pruned and unpruned trees is associated with smaller accuracy penalties, but the wide scatter shows the two measures are not redundant. Four regimes emerge when partitioning by their medians:

| Quadrant | Interpretation |
|----------|---------------|
| High \|Δacc\| + Low gRA (dominant) | Pruning changes the boundary globally and hurts accuracy — expected when boundaries are complex |
| Low \|Δacc\| + High gRA | Both trees agree globally and have equivalent accuracy — simple, well-captured boundaries |
| High \|Δacc\| + High gRA (critical) | Models agree over most of the space but differ exactly where the test set lies — accuracy hides the structural alignment |
| Low \|Δacc\| + Low gRA (rare) | Globally misaligned models unlikely to produce equivalent test accuracy by chance |

**E2 — gRA vs. dRA (r = 0.745):**
dRA is concentrated near 1.0 (models agree on observed data), while gRA is spread across [0, 1]. Several datasets show high dRA (> 0.8) with low gRA (< 0.3), cases where distributional alignment overestimates global structural similarity. No dataset shows high gRA with low dRA — consistent with the change-of-measure relationship: if two models agree across the whole space, they must also agree on any subsample.

### Experimental protocol (from the paper)

- **Datasets:** 92 UCI datasets covering healthcare, finance, biology, image and signal processing; 32–4 600 instances; 4–649 features; numerical, categorical, and mixed types.
- **Preprocessing:** numerical features imputed with column median and standardised; categorical features imputed with most-frequent value and one-hot encoded.
- **Models:** `DecisionTreeClassifier(random_state=42)`.
  - *Unpruned:* `min_samples_split=2`.
  - *Pruned:* `min_samples_split=10` + cost-complexity pruning with α selected per fold via an inner 80/20 split; the largest α in the pruning path is used (most aggressive pruning the data supports).
- **Evaluation:** 5-fold cross-validation (shuffled, `random_state=42`); dRA on the test partition; gRA on 1 000 synthetic instances drawn uniformly from the training bounding box.

---

## API Reference

All functionality is accessed via `RashomonAlignment`.

### `RashomonAlignment.distributional(a, b, method="jaccard")`

Compute agreement between two prediction arrays on the same set of instances.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a`, `b` | array-like | Prediction vectors (same length) |
| `method` | `"jaccard"` \| `"cohen_kappa"` | Agreement metric |

Returns `float` in [0, 1] (`"jaccard"`) or [−1, 1] (`"cohen_kappa"`).

---

### `RashomonAlignment.generate_uniform_samples(df, n_samples, numeric_cols, binary_cols, random_state)`

Generate a synthetic dataset by sampling each column uniformly within its observed range.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Reference data defining feature ranges |
| `n_samples` | `int` | `10000` | Number of synthetic instances |
| `numeric_cols` | list\|None | auto-detected | Columns sampled from U(min, max) |
| `binary_cols` | list\|None | auto-detected | Columns sampled from observed unique values |
| `random_state` | int\|None | `None` | RNG seed |

Returns `pd.DataFrame` with the same columns as `df`.

---

### `RashomonAlignment.geometric(models, df, feature_cols, n_samples, binary_cols, numeric_cols, random_state, alignment_method)`

Full gRA pipeline: generate synthetic data → apply models → compute agreement.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | list | — | Exactly two fitted estimators |
| `df` | `pd.DataFrame` | — | Training data (defines bounding box) |
| `feature_cols` | list | — | Feature columns to use |
| `n_samples` | `int` | `1000` | Synthetic instances |
| `alignment_method` | `str` | `"jaccard"` | Passed to `distributional` |

Returns `dict` with keys `generated`, `predictions`, `alignment`, `alignment_method`.

---

### `RashomonAlignment.plot_models(compare_result, feature_cols, model_names, figsize, random_state)`

Visualise model predictions and their agreement on the synthetic data using UMAP (falls back to t-SNE).

Returns a `matplotlib.figure.Figure` with three panels: model A predictions, model B predictions, and an agreement map (blue = agree, red = disagree).

---

## License

MIT
