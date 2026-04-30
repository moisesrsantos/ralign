"""Metrics for evaluating alignment between models in the Rashomon set."""
from typing import Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


class RashomonAlignment:
    """Compute alignment metrics between two prediction vectors."""

    @staticmethod
    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.mean(a == b))

    @staticmethod
    def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
        return float(cohen_kappa_score(a, b))

    @classmethod
    def distributional(cls, a: Sequence, b: Sequence, method: str = "jaccard") -> float:
        if method == "cohen_kappa":
            return cls.cohen_kappa(np.asarray(a), np.asarray(b))
        return cls.jaccard(np.asarray(a), np.asarray(b))

    @staticmethod
    def generate_uniform_samples(
        df: pd.DataFrame,
        n_samples: int = 10000,
        numeric_cols: Optional[Sequence[str]] = None,
        binary_cols: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic samples: numeric uniformly in observed range, binary sampled from observed uniques.

        Returns a DataFrame with same columns as requested.
        """

        rng = np.random.default_rng(random_state)
        X = df.copy()
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if binary_cols is None:
            binary_cols = [c for c in X.columns if X[c].dropna().nunique() == 2]

        # if a numeric column is binary (e.g., 0/1), treat it as binary
        numeric_cols = [c for c in numeric_cols if c not in set(binary_cols)]

        columns = {}
        for c in X.columns:
            if c in numeric_cols:
                col = X[c].dropna()
                if col.empty:
                    columns[c] = np.full(n_samples, np.nan)
                else:
                    lo, hi = float(col.min()), float(col.max())
                    if lo == hi:
                        columns[c] = np.full(n_samples, lo)
                    else:
                        columns[c] = rng.uniform(lo, hi, size=n_samples)
            elif c in binary_cols:
                vals = X[c].dropna().unique()
                if len(vals) == 0:
                    columns[c] = np.full(n_samples, np.nan)
                else:
                    columns[c] = rng.choice(vals, size=n_samples)
            else:
                vals = X[c].dropna().unique()
                if len(vals) == 0:
                    columns[c] = np.full(n_samples, np.nan)
                else:
                    columns[c] = rng.choice(vals, size=n_samples)

        out = pd.DataFrame(columns)
        return out

    @staticmethod
    def apply_models_to_generated(models: Sequence[Any], X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Apply models to samples. Try `predict`, then `predict_proba` (take argmax).

        Returns dict mapping index -> predictions (1D arrays).
        """
        results = {}
        for i, m in enumerate(models):
            try:
                preds = m.predict(X)
            except Exception:
                proba = m.predict_proba(X)
                preds = np.asarray(proba).argmax(axis=1)
            results[i] = np.asarray(preds)
        return results

    @classmethod
    def geometric(
        cls,
        models: Sequence[Any],
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        n_samples: int = 1000,
        binary_cols: Optional[Sequence[str]] = None,
        numeric_cols: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        alignment_method: str = "jaccard",
    ) -> Dict[str, Any]:
        """Generate samples, apply models, and compute alignment between model 0 and model 1.

        Returns a dict with keys: 'generated', 'predictions', 'alignment', 'alignment_method'.
        """
        gen = cls.generate_uniform_samples(df[feature_cols], n_samples, numeric_cols, binary_cols, random_state)
        preds = cls.apply_models_to_generated(models, gen)
        a, b = preds[0], preds[1]
        alignment = cls.distributional(a, b, method=alignment_method)
        return {"generated": gen, "predictions": preds, "alignment": alignment, "alignment_method": alignment_method}

    @staticmethod
    def plot_models(
        compare_result: Dict[str, Any],
        feature_cols: Sequence[str],
        model_names: Optional[Sequence[str]] = None,
        figsize=(12, 4),
        random_state: Optional[int] = None,
    ) -> plt.Figure:
        """Plot 2D embedding (umap preferred, TSNE fallback) showing model predictions and agreement.

        Returns matplotlib Figure.
        """
        gen = compare_result["generated"]
        preds = compare_result["predictions"]
        y0 = preds[0]
        y1 = preds[1]

        try:
            import umap as _umap

            emb = _umap.UMAP(random_state=random_state).fit_transform(gen[feature_cols])
            used = "UMAP"
        except Exception:
            from sklearn.manifold import TSNE

            emb = TSNE(n_components=2, random_state=random_state).fit_transform(gen[feature_cols])
            used = "TSNE"

        # deterministic color mapping
        classes = np.unique(np.concatenate([y0, y1]))
        cmap = plt.get_cmap("tab10")
        color_map = {c: cmap(i % 10) for i, c in enumerate(classes)}

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes = axes.ravel()
        
        # Create model names with embedding info
        if model_names is None:
            model_names = [f"Model 0", f"Model 1"]
        titles = [f"{model_names[0]} ({used})", f"{model_names[1]} ({used})", "Agreement"]
        
        for ax, data, title in zip(axes, (y0, y1, y0 == y1), titles):
            if title == "Agreement":
                colors = np.where(data, "#9fb4c7", "#d45d5d")
                ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=16, edgecolor="k", linewidth=0.2)
            else:
                cols = [color_map[v] for v in data]
                ax.scatter(emb[:, 0], emb[:, 1], c=cols, s=16, edgecolor="k", linewidth=0.2)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        # legend
        handles = [plt.Line2D([0], [0], marker="o", color="w", label=str(c), markerfacecolor=color_map[c], markersize=8) for c in classes]
        axes[1].legend(handles=handles, title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

