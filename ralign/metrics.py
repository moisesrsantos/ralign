"""Metrics for evaluating alignment between models in the Rashomon set."""
import warnings
from typing import Sequence, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import qmc
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

    @staticmethod
    def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
        """Wilson score confidence interval for a binomial proportion."""
        if n == 0:
            return 0.0, 1.0
        phat = successes / n
        denom = 1.0 + z ** 2 / n
        center = phat + z ** 2 / (2 * n)
        margin = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
        lo = (center - margin) / denom
        hi = (center + margin) / denom
        return max(0.0, lo), min(1.0, hi)

    @staticmethod
    def sobol_samples(
        df: pd.DataFrame,
        m: int,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate m low-discrepancy Sobol samples over the observed feature space.

        Numeric columns are mapped to the observed [min, max] range; categorical
        (including binary) columns are mapped to the observed unique categories.
        The base [0, 1) coordinates are drawn from a scrambled Sobol sequence for
        lower-variance coverage of the instance space than i.i.d. uniform draws.
        """
        X = df.copy()
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = [c for c in X.columns if X[c].dropna().nunique() == 2]
        numeric_cols = [c for c in numeric_cols if c not in set(categorical_cols)]
        categorical_cols = [c for c in X.columns if c in set(categorical_cols)] or [
            c for c in X.columns if c not in set(numeric_cols)
        ]

        d = len(numeric_cols) + len(categorical_cols)
        sampler = qmc.Sobol(d=max(d, 1), scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            u = sampler.random(m)

        columns = {}
        for j, c in enumerate(numeric_cols):
            col = X[c].dropna()
            if col.empty:
                columns[c] = np.full(m, np.nan)
                continue
            lo, hi = float(col.min()), float(col.max())
            if hi <= lo:
                hi = lo + 1e-9
            columns[c] = lo + u[:, j] * (hi - lo)

        offset = len(numeric_cols)
        for gi, c in enumerate(categorical_cols):
            vals = X[c].dropna().unique()
            if len(vals) == 0:
                columns[c] = np.full(m, np.nan)
                continue
            k = len(vals)
            col_u = u[:, offset + gi]
            cat_idx = np.minimum((col_u * k).astype(int), k - 1)
            columns[c] = np.asarray(vals)[cat_idx]

        return pd.DataFrame(columns)[list(df.columns)]

    @classmethod
    def _qmc_agreement(
        cls,
        models: Sequence[Any],
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        m: int,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], float, float, float]:
        """Draw m Sobol samples, apply model 0 and model 1, and score their agreement.

        Shared by `geometric` and `qmc_gra` so both compute gRA the same way -
        via Sobol QMC sampling plus a Wilson confidence interval on the
        agreement rate - at whatever sample size the caller needs.

        Returns (generated, predictions, agreement_rate, ci_low, ci_high).
        """
        gen = cls.sobol_samples(df[list(feature_cols)], m, numeric_cols, categorical_cols, seed=seed)
        preds = cls.apply_models_to_generated(models, gen)
        agree = preds[0] == preds[1]
        successes = int(np.sum(agree))
        p = successes / m
        ci_low, ci_high = cls.wilson_ci(successes, m)
        return gen, preds, p, ci_low, ci_high

    @classmethod
    def geometric(
        cls,
        models: Sequence[Any],
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        n_samples: int = 1000,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        alignment_method: str = "jaccard",
    ) -> Dict[str, Any]:
        """Geometric Rashomon Alignment (gRA): agreement between model 0 and model 1
        over synthetic Sobol samples spanning the observed instance space.

        Returns a dict with keys: 'generated', 'predictions', 'alignment',
        'alignment_method', 'ci_low', 'ci_high' (Wilson CI on the agreement rate).
        """
        gen, preds, p, ci_low, ci_high = cls._qmc_agreement(
            models, df, feature_cols, n_samples, numeric_cols, categorical_cols, seed
        )
        alignment = p if alignment_method == "jaccard" else cls.distributional(
            preds[0], preds[1], method=alignment_method
        )
        return {
            "generated": gen,
            "predictions": preds,
            "alignment": alignment,
            "alignment_method": alignment_method,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    @classmethod
    def qmc_gra(
        cls,
        models: Sequence[Any],
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        m_low: int = 1000,
        m_high: int = 8000,
        seed: Optional[int] = None,
        flag_threshold: float = 0.03,
    ) -> Dict[str, Any]:
        """Estimate gRA at two sample sizes to check convergence of the Sobol QMC estimate.

        Model-agnostic: model 0 and model 1 are compared only through their
        `.predict` outputs on synthetic Sobol samples (see `geometric` /
        `_qmc_agreement`), with no inspection of model internals. `flag_diff`
        signals whether the `m_low` and `m_high` estimates disagree by more
        than `flag_threshold`, i.e. the estimate has not yet converged.
        """
        _, _, p_low, _, _ = cls._qmc_agreement(models, df, feature_cols, m_low, numeric_cols, categorical_cols, seed)
        seed_high = None if seed is None else seed + 1
        _, _, p_high, ci_low, ci_high = cls._qmc_agreement(
            models, df, feature_cols, m_high, numeric_cols, categorical_cols, seed_high
        )

        return {
            "gra": p_high,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "method": "qmc",
            "m_low": m_low,
            "m_high": m_high,
            "gra_m_low": p_low,
            "gra_m_high": p_high,
            "flag_diff": abs(p_low - p_high) > flag_threshold,
        }

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

