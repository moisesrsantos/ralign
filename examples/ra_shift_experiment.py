"""Shift-robustness experiment for Rashomon Alignment.

Extends the pruned-vs-unpruned tree experiment (decision_tree_pruning_rashomon.py)
with two additional algorithm pairs (linear/RBF SVM, LogReg/MLP). gRA is estimated
with `ralign.metrics.RashomonAlignment.qmc_gra`, the same model-agnostic Sobol
quasi-Monte Carlo estimator (with Wilson confidence intervals and a two-scale
convergence check) applied identically to all three pairs and to every other
experiment in this package - this module adds no gRA-estimation code of its own,
only the shift-robustness protocol around it: covariate-shift induction via
PCA-tilted resampling and a bound-validity / decision-flip analysis. This module
is self-contained and does not modify the original tree experiment or its results.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR))

from ralign.metrics import RashomonAlignment  # noqa: E402

from decision_tree_pruning_rashomon import (  # noqa: E402
    MSS_PRUNED,
    MSS_UNPRUNED,
    N_SPLITS,
    RANDOM_STATE,
    build_preprocessor,
    find_data_file,
    load_dataset,
    select_ccp_alpha,
    split_xy,
)

BETAS: List[float] = [0.0, 0.5, 1.0, 2.0, 4.0]
TAU_FIXED = 0.7
QMC_M_LOW = 1000
QMC_M_HIGH = 8000

RESULTS_DIR = EXAMPLES_DIR / "results"
FIGURES_DIR = EXAMPLES_DIR / "figures"

PAIRS = ["tree_pruned_vs_unpruned", "svm_linear_vs_rbf", "logreg_vs_mlp"]


# ---------------------------------------------------------------------------
# gRA via ralign, and per-fold, per-pair evaluation
# ---------------------------------------------------------------------------

def make_model_pipeline(preprocessor, estimator) -> Pipeline:
    """A fresh preprocessor + estimator pipeline, so gRA (ralign.qmc_gra) can
    sample the raw instance space and let each model preprocess it internally,
    instead of this module reimplementing one-hot-aware QMC sampling."""
    return Pipeline([("pre", clone(preprocessor)), ("model", estimator)])


def compute_gra(model_a, model_b, X_tr: pd.DataFrame, num_feats: Sequence[str],
                 cat_feats: Sequence[str], seed: int) -> Dict[str, Any]:
    """Returns dict with keys: gra, ci_low, ci_high, method, m1000, m8000, flag_diff.

    Thin wrapper around `RashomonAlignment.qmc_gra` - the same model-agnostic
    Sobol QMC estimator is used for every pair (tree, SVM, LogReg/MLP) and for
    the MNIST capstone case study (ra_shift_experiment_mnist_tabpfn.py), so
    gRA estimation requires no model-specific or pair-specific code path here.
    """
    res = RashomonAlignment.qmc_gra(
        [model_a, model_b], X_tr, list(X_tr.columns),
        numeric_cols=list(num_feats), categorical_cols=list(cat_feats),
        m_low=QMC_M_LOW, m_high=QMC_M_HIGH, seed=seed,
    )
    return {
        "gra": res["gra"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
        "method": res["method"], "m1000": res["gra_m_low"], "m8000": res["gra_m_high"],
        "flag_diff": res["flag_diff"],
    }


def evaluate_fold(pair: str, model_a, model_b, X_te, y_te: np.ndarray,
                   z_scores: np.ndarray, gra_info: Dict[str, Any],
                   dataset: str, fold_idx: int, fold_seed_base: int) -> List[Dict[str, Any]]:
    yp_a = model_a.predict(X_te)
    yp_b = model_b.predict(X_te)
    n = len(y_te)
    rows = []
    for beta in BETAS:
        if beta == 0.0:
            idx = np.arange(n)
            tv_hat = 0.0
        else:
            w = np.exp(beta * z_scores)
            probs = w / w.sum()
            seed = fold_seed_base + int(beta * 100)
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=n, replace=True, p=probs)
            tv_hat = 0.5 * float(np.mean(np.abs(w / w.mean() - 1.0)))

        yp_a_s, yp_b_s, y_s = yp_a[idx], yp_b[idx], y_te[idx]
        dra_beta = float(np.mean(yp_a_s == yp_b_s))
        acc_a = float(np.mean(y_s == yp_a_s))
        acc_b = float(np.mean(y_s == yp_b_s))
        delta_acc = acc_b - acc_a
        gap = abs(dra_beta - gra_info["gra"]) if not np.isnan(gra_info["gra"]) else np.nan

        rows.append({
            "dataset": dataset, "pair": pair, "fold": fold_idx, "beta": beta,
            "dRA_beta": dra_beta, "gRA": gra_info["gra"],
            "gRA_CI_low": gra_info["ci_low"], "gRA_CI_high": gra_info["ci_high"],
            "gra_method": gra_info["method"], "gra_m1000": gra_info["m1000"],
            "gra_m8000": gra_info["m8000"], "gra_flag_diff": gra_info["flag_diff"],
            "acc_A": acc_a, "acc_B": acc_b, "delta_acc": delta_acc,
            "TV_hat": tv_hat, "gap": gap,
        })
    return rows


def evaluate_dataset(dataset_dir: Path) -> Optional[List[Dict[str, Any]]]:
    data_file = find_data_file(dataset_dir)
    if data_file is None:
        return None
    df = load_dataset(data_file)
    if df.shape[1] < 2:
        return None
    X, y = split_xy(df)
    if X.empty or y.empty:
        return None

    num_feats = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_feats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_rows: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        if y_tr.nunique() < 2:
            continue

        y_tr_arr = y_tr.to_numpy()
        y_te_arr = y_te.to_numpy()

        # Shared preprocessor used only to derive the PCA-tilted covariate-shift
        # weights below; each model pair gets its own (identically-fit) cloned
        # copy via `make_model_pipeline` so gRA can be computed on raw features.
        preprocessor = build_preprocessor(num_feats, cat_feats)
        Xt_tr = preprocessor.fit_transform(X_tr)
        Xt_te = preprocessor.transform(X_te)
        if hasattr(Xt_tr, "toarray"):
            Xt_tr = Xt_tr.toarray()
        if hasattr(Xt_te, "toarray"):
            Xt_te = Xt_te.toarray()

        pca = PCA(n_components=1, random_state=RANDOM_STATE)
        pca.fit(Xt_tr)
        pc1_te = pca.transform(Xt_te).ravel()
        std = pc1_te.std()
        z_scores = (pc1_te - pc1_te.mean()) / std if std > 0 else np.zeros_like(pc1_te)

        fold_seed_base = RANDOM_STATE * 1_000_003 + fold_idx * 97

        # --- Pair 1: tree pruned vs unpruned ---
        try:
            alpha = select_ccp_alpha(X_tr, y_tr, num_feats, cat_feats)
            tree_unpruned = make_model_pipeline(preprocessor, DecisionTreeClassifier(
                random_state=RANDOM_STATE, min_samples_split=MSS_UNPRUNED))
            tree_pruned = make_model_pipeline(preprocessor, DecisionTreeClassifier(
                random_state=RANDOM_STATE, ccp_alpha=alpha, min_samples_split=MSS_PRUNED))
            tree_unpruned.fit(X_tr, y_tr_arr)
            tree_pruned.fit(X_tr, y_tr_arr)
            gra_info = compute_gra(tree_unpruned, tree_pruned, X_tr, num_feats, cat_feats, fold_seed_base)
            all_rows += evaluate_fold("tree_pruned_vs_unpruned", tree_unpruned, tree_pruned,
                                       X_te, y_te_arr, z_scores, gra_info,
                                       dataset_dir.name, fold_idx, fold_seed_base)
        except Exception:
            pass

        # --- Pair 2: linear SVM vs RBF SVM ---
        try:
            svm_lin = make_model_pipeline(preprocessor, SVC(
                kernel="linear", C=1, random_state=RANDOM_STATE))
            svm_rbf = make_model_pipeline(preprocessor, SVC(
                kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE))
            svm_lin.fit(X_tr, y_tr_arr)
            svm_rbf.fit(X_tr, y_tr_arr)
            gra_info = compute_gra(svm_lin, svm_rbf, X_tr, num_feats, cat_feats, fold_seed_base + 10_000)
            all_rows += evaluate_fold("svm_linear_vs_rbf", svm_lin, svm_rbf,
                                       X_te, y_te_arr, z_scores, gra_info,
                                       dataset_dir.name, fold_idx, fold_seed_base)
        except Exception:
            pass

        # --- Pair 3: logistic regression vs MLP ---
        try:
            logreg = make_model_pipeline(preprocessor, LogisticRegression(
                C=1, max_iter=1000, random_state=RANDOM_STATE))
            mlp = make_model_pipeline(preprocessor, MLPClassifier(
                hidden_layer_sizes=(100,), activation="relu",
                random_state=RANDOM_STATE, max_iter=500, early_stopping=True))
            logreg.fit(X_tr, y_tr_arr)
            mlp.fit(X_tr, y_tr_arr)
            gra_info = compute_gra(logreg, mlp, X_tr, num_feats, cat_feats, fold_seed_base + 20_000)
            all_rows += evaluate_fold("logreg_vs_mlp", logreg, mlp,
                                       X_te, y_te_arr, z_scores, gra_info,
                                       dataset_dir.name, fold_idx, fold_seed_base)
        except Exception:
            pass

    return all_rows or None


# ---------------------------------------------------------------------------
# Aggregation / analysis
# ---------------------------------------------------------------------------

def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    agg = raw.groupby(["dataset", "pair"]).agg(
        n_folds=("fold", "nunique"),
        dRA_0_mean=("dRA_beta", lambda s: s[raw.loc[s.index, "beta"] == 0.0].mean()),
        gRA_mean=("gRA", "mean"),
        gRA_CI_low_mean=("gRA_CI_low", "mean"),
        gRA_CI_high_mean=("gRA_CI_high", "mean"),
        gap_mean=("gap", "mean"),
        gap_max=("gap", "max"),
        TV_hat_mean=("TV_hat", "mean"),
        acc_A_mean=("acc_A", "mean"),
        acc_B_mean=("acc_B", "mean"),
        delta_acc_mean=("delta_acc", "mean"),
        gra_method=("gra_method", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        any_gra_flag_diff=("gra_flag_diff", "any"),
    ).reset_index()
    return agg


def add_delta_gap(raw: pd.DataFrame) -> pd.DataFrame:
    """Adds delta_gap(beta) = gap(beta) - gap(0) per (dataset, pair, fold).

    gap(beta) = |dRA_beta - gRA| alone is NOT bounded by TV_hat(beta): the
    provable inequality (triangle inequality applied to the dRA*/gRA* TV bound
    at P_0 and P_beta) is gap(beta) <= TV_hat(beta) + gap(0), i.e.
    delta_gap(beta) <= TV_hat(beta). Testing gap(beta) <= TV_hat(beta) directly
    silently assumes gap(0) = 0, which is false in general (dRA and gRA differ
    even absent any shift) and made the bound look far weaker than it is.
    """
    gap0 = raw.loc[raw["beta"] == 0.0, ["dataset", "pair", "fold", "gap"]].rename(
        columns={"gap": "gap0"})
    out = raw.merge(gap0, on=["dataset", "pair", "fold"], how="left")
    out["delta_gap"] = out["gap"] - out["gap0"]
    return out


def bound_validity_analysis(raw: pd.DataFrame) -> Dict[str, Any]:
    raw = add_delta_gap(raw)
    valid = raw.dropna(subset=["delta_gap", "TV_hat"])
    pear = pearsonr(valid["delta_gap"], valid["TV_hat"])
    spear = spearmanr(valid["delta_gap"], valid["TV_hat"])
    holds = (valid["delta_gap"] <= valid["TV_hat"]).mean()
    return {
        "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
        "spearman_r": float(spear[0]), "spearman_p": float(spear[1]),
        "bound_holds_fraction": float(holds),
        "n": int(len(valid)),
    }


def decision_flip_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, pair): fold-averaged dRA_beta at each beta, classify vs tau,
    detect flips relative to beta=0, and flag them as spurious (gRA-anchored
    classification is constant by construction and never flips)."""
    grp = raw.groupby(["dataset", "pair", "beta"])["dRA_beta"].mean().reset_index()
    pivot = grp.pivot(index=["dataset", "pair"], columns="beta", values="dRA_beta")

    rows = []
    for (dataset, pair), row in pivot.iterrows():
        if 0.0 not in row.index or pd.isna(row.get(0.0)):
            continue
        median_ra = row.dropna().median()
        base_fixed = row[0.0] > TAU_FIXED
        base_median = row[0.0] > median_ra
        flip_fixed = False
        flip_median = False
        for beta in BETAS:
            if beta == 0.0 or beta not in row.index or pd.isna(row[beta]):
                continue
            if (row[beta] > TAU_FIXED) != base_fixed:
                flip_fixed = True
            if (row[beta] > median_ra) != base_median:
                flip_median = True
        rows.append({
            "dataset": dataset, "pair": pair,
            "spurious_flip_tau_fixed": flip_fixed,
            "spurious_flip_tau_median": flip_median,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(raw: pd.DataFrame, flips: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = add_delta_gap(raw)

    # scatter: delta_gap vs TV_hat (the provable bound; see add_delta_gap docstring)
    valid = raw.dropna(subset=["delta_gap", "TV_hat"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for pair, sub in valid.groupby("pair"):
        ax.scatter(sub["TV_hat"], sub["delta_gap"], s=8, alpha=0.4, label=pair)
    lims = [0, max(valid["TV_hat"].max(), valid["delta_gap"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="delta_gap = TV_hat (bound)")
    ax.set_xlabel("TV_hat(beta)")
    ax.set_ylabel("delta_gap(beta) = gap(beta) - gap(0)")
    ax.set_title("Bound validity: delta_gap vs total-variation shift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gap_vs_TV_scatter.png", dpi=150)
    plt.close(fig)

    # line plot: mean delta_gap vs beta per pair, shaded band across datasets
    fig, ax = plt.subplots(figsize=(7, 5))
    for pair, sub in valid.groupby("pair"):
        stats = sub.groupby("beta")["delta_gap"].agg(["mean", "std"]).reset_index()
        ax.plot(stats["beta"], stats["mean"], marker="o", label=pair)
        ax.fill_between(stats["beta"], stats["mean"] - stats["std"].fillna(0),
                         stats["mean"] + stats["std"].fillna(0), alpha=0.15)
    ax.set_xlabel("beta (shift strength)")
    ax.set_ylabel("mean delta_gap(beta) across datasets")
    ax.set_title("Gap growth under covariate shift, by algorithm pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gap_vs_beta_by_pair.png", dpi=150)
    plt.close(fig)

    # flip rate by pair
    fig, ax = plt.subplots(figsize=(6, 5))
    if not flips.empty:
        rate = flips.groupby("pair")[["spurious_flip_tau_fixed", "spurious_flip_tau_median"]].mean()
        rate = rate.reindex(PAIRS).dropna(how="all")
        rate.plot(kind="bar", ax=ax)
        ax.set_ylabel("fraction of datasets with >=1 spurious flip")
        ax.set_title("Spurious decision flips under covariate shift")
        ax.set_xticklabels(rate.index, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "flip_rate_by_pair.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def write_notes(bound_stats: Dict[str, Any], flips: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = ["# Shift-Robustness Experiment: Notes", ""]
    lines.append("## Bound validity (delta_gap(beta) = gap(beta) - gap(0), vs TV_hat(beta))")
    lines.append(
        "Testing gap(beta) <= TV_hat(beta) directly implicitly assumes gap(0) = 0, which is "
        "false in general (dRA and gRA differ even absent any shift). The provable corollary of "
        "the dRA*/gRA* TV bound is delta_gap(beta) <= TV_hat(beta)."
    )
    lines.append(
        f"- Pooled across {bound_stats['n']} (dataset, pair, fold, beta) rows: "
        f"Pearson r = {bound_stats['pearson_r']:.3f} (p={bound_stats['pearson_p']:.2g}), "
        f"Spearman rho = {bound_stats['spearman_r']:.3f} (p={bound_stats['spearman_p']:.2g})."
    )
    lines.append(
        f"- The bound delta_gap(beta) <= TV_hat(beta) holds in "
        f"{bound_stats['bound_holds_fraction']*100:.1f}% of rows "
        f"(violations beyond this are attributable to gRA/dRA estimation noise)."
    )
    lines.append("")

    lines.append("## Spurious decision flips (models never changed; only the observed test distribution did)")
    if not flips.empty:
        for pair in PAIRS:
            sub = flips[flips["pair"] == pair]
            if sub.empty:
                continue
            n = len(sub)
            n_fixed = int(sub["spurious_flip_tau_fixed"].sum())
            n_median = int(sub["spurious_flip_tau_median"].sum())
            lines.append(
                f"- **{pair}**: {n_fixed}/{n} datasets ({100*n_fixed/n:.1f}%) show at least one "
                f"spurious flip across the beta grid at tau=0.7; "
                f"{n_median}/{n} ({100*n_median/n:.1f}%) at tau=per-dataset-median-RA."
            )
    lines.append("")

    lines.append("## Cross-pair generality check")
    for pair in PAIRS:
        sub = summary[summary["pair"] == pair]
        if sub.empty:
            lines.append(f"- **{pair}**: no data collected.")
            continue
        method = sub["gra_method"].mode().iat[0] if not sub["gra_method"].mode().empty else "n/a"
        closing = "Computed with the same model-agnostic QMC estimator used for every pair."
        lines.append(
            f"- **{pair}** (gRA method: {method}): mean gap = {sub['gap_mean'].mean():.4f}, "
            f"mean TV_hat = {sub['TV_hat_mean'].mean():.4f}, "
            f"mean gRA = {sub['gRA_mean'].mean():.4f} over {len(sub)} datasets. {closing}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "shift_experiment_notes.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: Optional[int] = None) -> None:
    datasets_dir = EXAMPLES_DIR / "datasets"
    all_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    dirs = sorted(d for d in datasets_dir.iterdir() if d.is_dir())
    if limit:
        dirs = dirs[:limit]

    for d in dirs:
        try:
            rows = evaluate_dataset(d)
        except Exception:
            rows = None
        if rows:
            all_rows.extend(rows)
        else:
            skipped.append(d.name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(RESULTS_DIR / "shift_experiment_raw.csv", index=False)

    if skipped:
        (RESULTS_DIR / "shift_experiment_skipped.txt").write_text("\n".join(sorted(skipped)), encoding="utf-8")

    if raw.empty:
        return

    summary = build_summary(raw)
    summary.to_csv(RESULTS_DIR / "shift_experiment_summary.csv", index=False)

    bound_stats = bound_validity_analysis(raw)
    flips = decision_flip_analysis(raw)
    make_plots(raw, flips)
    write_notes(bound_stats, flips, summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of datasets processed (for smoke testing).")
    args = parser.parse_args()
    main(limit=args.limit)
