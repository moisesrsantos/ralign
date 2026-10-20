"""Capstone case study: shift-robustness on a single dataset (MNIST), with a
tabular foundation model in the loop.

This is deliberately NOT part of the 92-dataset sweep in ra_shift_experiment.py.
It reuses that module's core, model-agnostic gRA/TV-bound machinery (`compute_gra`,
backed by `ralign.metrics.RashomonAlignment.qmc_gra` - Sobol QMC sampling with
Wilson CIs - plus PCA-tilted covariate-shift induction, bound-validity and
decision-flip analysis) but applies it to a single, higher-dimensional, more
"real-world" dataset (MNIST digits) and swaps the LogReg-vs-MLP pair for a
TabPFN-vs-MLP pair, to illustrate that RA generalises beyond both classical
tabular benchmarks and classical (non-foundation-model) classifiers.

Practical constraints that shape the design, documented here rather than left
implicit:
  - TabPFN (github.com/PriorLabs/tabpfn) refuses to run on CPU with more than
    1,000 training rows per fold unless explicitly overridden, since attention
    over a large context is slow without a GPU. We therefore subsample MNIST
    to 120 stratified instances per digit (1,200 total), which keeps every
    fold's training partition under that limit with no override needed.
  - Raw MNIST is 784 pixels; both for computational tractability and so that
    "uniform sampling over the instance space" (Sect. gRA) samples points that
    are at least a plausible approximation of the manifold of real digit
    images rather than an intractably vast, almost entirely non-digit-like
    784-dimensional pixel cube, we reduce to the top 150 PCA components (fit
    per fold on the training partition only, like every other per-fold
    transform in this pipeline) before applying any of the three pairs. This
    retains ~90% of pixel variance (measured on a held-out MNIST sample: 50
    components retain 65%, 100 retain 85%, 150 retain 90%), the standard
    rule-of-thumb target for PCA dimensionality reduction; TabPFN's own
    accuracy plateaus well before 100 components, so 150 is chosen for a
    faithful instance space rather than for classifier accuracy.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR))

from ra_shift_experiment import (  # noqa: E402
    BETAS,
    add_delta_gap,
    bound_validity_analysis,
    compute_gra,
    decision_flip_analysis,
    evaluate_fold,
)
from decision_tree_pruning_rashomon import MSS_PRUNED, MSS_UNPRUNED, N_SPLITS, RANDOM_STATE  # noqa: E402

N_PER_CLASS = 120        # 1,200 total instances, stratified across 10 digits
N_PCA = 150               # instance space F = top-150 PCA components (~90% variance)
DATASET_NAME = "mnist"
PAIRS = ["tree_pruned_vs_unpruned", "svm_linear_vs_rbf", "tabpfn_vs_mlp"]

RESULTS_DIR = EXAMPLES_DIR / "results"
FIGURES_DIR = EXAMPLES_DIR / "figures"


def load_mnist_subsample() -> tuple[np.ndarray, np.ndarray]:
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, cache=True,
        data_home=str(EXAMPLES_DIR / "datasets" / ".openml_cache"),
    )
    X, y = mnist.data, mnist.target.astype(int)
    rng = np.random.default_rng(RANDOM_STATE)
    idx_parts = []
    for c in range(10):
        class_idx = np.where(y == c)[0]
        idx_parts.append(rng.choice(class_idx, size=N_PER_CLASS, replace=False))
    idx = np.concatenate(idx_parts)
    return X[idx], y[idx]


def select_ccp_alpha(Xt_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> float:
    """Same protocol as decision_tree_pruning_rashomon.select_ccp_alpha (inner
    80/20 split, take the most aggressive alpha_K), operating directly on the
    already-preprocessed array instead of a raw dataframe."""
    if len(np.unique(y_tr)) < 2:
        return 0.0
    try:
        X_tr2, _, y_tr2, _ = train_test_split(
            Xt_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    except ValueError:
        X_tr2, _, y_tr2, _ = train_test_split(
            Xt_tr, y_tr, test_size=0.2, random_state=seed)
    tree = DecisionTreeClassifier(random_state=seed, min_samples_split=MSS_PRUNED)
    tree.fit(X_tr2, y_tr2)
    alphas = np.unique(tree.cost_complexity_pruning_path(X_tr2, y_tr2).ccp_alphas)
    if alphas.size == 0:
        return 0.0
    return float(alphas[-1])


PCA_FEATURE_COLS = [f"pc{i}" for i in range(N_PCA)]


def main() -> None:
    X, y = load_mnist_subsample()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_rows: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X)):
        print(f"=== fold {fold_idx} ===", flush=True)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
        pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE).fit(X_tr_s)
        Xt_tr, Xt_te = pca.transform(X_tr_s), pca.transform(X_te_s)
        Xt_tr_df = pd.DataFrame(Xt_tr, columns=PCA_FEATURE_COLS)

        pca_shift = PCA(n_components=1, random_state=RANDOM_STATE).fit(Xt_tr)
        pc1_te = pca_shift.transform(Xt_te).ravel()
        std = pc1_te.std()
        z_scores = (pc1_te - pc1_te.mean()) / std if std > 0 else np.zeros_like(pc1_te)

        fold_seed_base = RANDOM_STATE * 1_000_003 + fold_idx * 97

        # --- Pair 1: tree pruned vs unpruned ---
        alpha = select_ccp_alpha(Xt_tr, y_tr, RANDOM_STATE)
        tree_unpruned = DecisionTreeClassifier(
            random_state=RANDOM_STATE, min_samples_split=MSS_UNPRUNED).fit(Xt_tr, y_tr)
        tree_pruned = DecisionTreeClassifier(
            random_state=RANDOM_STATE, ccp_alpha=alpha, min_samples_split=MSS_PRUNED).fit(Xt_tr, y_tr)
        gra_info = compute_gra(tree_unpruned, tree_pruned, Xt_tr_df, PCA_FEATURE_COLS, [], fold_seed_base)
        all_rows += evaluate_fold("tree_pruned_vs_unpruned", tree_unpruned, tree_pruned,
                                   Xt_te, y_te, z_scores, gra_info, DATASET_NAME, fold_idx, fold_seed_base)
        print("  tree pair done", flush=True)

        # --- Pair 2: linear SVM vs RBF SVM ---
        svm_lin = SVC(kernel="linear", C=1, random_state=RANDOM_STATE).fit(Xt_tr, y_tr)
        svm_rbf = SVC(kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE).fit(Xt_tr, y_tr)
        gra_info = compute_gra(svm_lin, svm_rbf, Xt_tr_df, PCA_FEATURE_COLS, [], fold_seed_base + 10_000)
        all_rows += evaluate_fold("svm_linear_vs_rbf", svm_lin, svm_rbf,
                                   Xt_te, y_te, z_scores, gra_info, DATASET_NAME, fold_idx, fold_seed_base)
        print("  svm pair done", flush=True)

        # --- Pair 3: TabPFN vs MLP ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tabpfn = TabPFNClassifier(device="cpu", random_state=RANDOM_STATE)
            tabpfn.fit(Xt_tr, y_tr)
        mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                             random_state=RANDOM_STATE, max_iter=500, early_stopping=True)
        mlp.fit(Xt_tr, y_tr)
        gra_info = compute_gra(tabpfn, mlp, Xt_tr_df, PCA_FEATURE_COLS, [], fold_seed_base + 20_000)
        all_rows += evaluate_fold("tabpfn_vs_mlp", tabpfn, mlp,
                                   Xt_te, y_te, z_scores, gra_info, DATASET_NAME, fold_idx, fold_seed_base)
        print("  tabpfn/mlp pair done", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(RESULTS_DIR / "mnist_tabpfn_raw.csv", index=False)

    bound_stats = bound_validity_analysis(raw)
    flips = decision_flip_analysis(raw)
    flips.to_csv(RESULTS_DIR / "mnist_tabpfn_flips.csv", index=False)

    per_pair = raw.groupby("pair").agg(
        gRA_mean=("gRA", "mean"),
        dRA0_mean=("dRA_beta", lambda s: s[raw.loc[s.index, "beta"] == 0.0].mean()),
    ).reindex(PAIRS)

    make_plots(raw)
    write_notes(bound_stats, flips, per_pair, raw)
    print("done")


def make_plots(raw: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = {"tree_pruned_vs_unpruned": "Tree (pruned vs unpruned)",
              "svm_linear_vs_rbf": "SVM (linear vs RBF)",
              "tabpfn_vs_mlp": "TabPFN vs MLP"}

    raw = add_delta_gap(raw)
    valid = raw.dropna(subset=["delta_gap", "TV_hat"])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for pair, g in valid.groupby("pair"):
        ax.scatter(g["TV_hat"], g["delta_gap"], s=40, alpha=0.75, label=labels.get(pair, pair))
    lim = max(valid["TV_hat"].max(), valid["delta_gap"].max()) * 1.05 if len(valid) else 1.0
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="bound: delta_gap = TV_hat")
    ax.set_xlabel("TV_hat(beta)")
    ax.set_ylabel("delta_gap(beta) = gap(beta) - gap(0)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mnist_gap_vs_TV.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for pair, sub in valid.groupby("pair"):
        stats = sub.groupby("beta")["dRA_beta"].mean().reset_index()
        ax.plot(stats["beta"], stats["dRA_beta"], marker="o", label=labels.get(pair, pair))
        gra_val = sub["gRA"].mean()
        ax.axhline(gra_val, linestyle=":", alpha=0.5,
                   color=ax.get_lines()[-1].get_color())
    ax.set_xlabel("beta (shift strength)")
    ax.set_ylabel("dRA(beta)  (dotted lines: gRA, shift-invariant)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mnist_dRA_vs_beta.png", dpi=150)
    plt.close(fig)


def write_notes(bound_stats: Dict[str, Any], flips: pd.DataFrame,
                 per_pair: pd.DataFrame, raw: pd.DataFrame) -> None:
    lines = ["# MNIST capstone case study: notes", ""]
    lines.append("## Per-pair gRA / dRA(beta=0)")
    for pair in PAIRS:
        if pair not in per_pair.index or pd.isna(per_pair.loc[pair, "gRA_mean"]):
            continue
        lines.append(f"- **{pair}**: gRA = {per_pair.loc[pair, 'gRA_mean']:.3f}, "
                      f"dRA(beta=0) = {per_pair.loc[pair, 'dRA0_mean']:.3f}")
    lines.append("")
    lines.append("## Bound validity (pooled across pairs and folds, this dataset only)")
    lines.append(f"- n={bound_stats['n']}, Pearson r={bound_stats['pearson_r']:.3f} "
                 f"(p={bound_stats['pearson_p']:.2g}), bound holds in "
                 f"{bound_stats['bound_holds_fraction']*100:.1f}% of rows.")
    lines.append("")
    lines.append("## Decision flips (tau=0.7), this single dataset")
    for _, row in flips.iterrows():
        lines.append(f"- **{row['pair']}**: flip at tau=0.7 -> {row['spurious_flip_tau_fixed']}; "
                     f"flip at tau=median -> {row['spurious_flip_tau_median']}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mnist_tabpfn_notes.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
