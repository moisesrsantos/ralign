from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ralign.metrics import RashomonAlignment  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
MSS_UNPRUNED = 2
MSS_PRUNED = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_data_file(dataset_dir: Path):
    for ext in (".data", ".csv", ".txt", ".dat"):
        files = sorted(dataset_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None, sep=None, engine="python",
        na_values=["?", "NA", "NaN", "nan", ""], skipinitialspace=True,
    )
    # Remove colunas totalmente nulas
    df = df.dropna(axis=1, how="all")
    # Remove linhas totalmente nulas
    df = df.dropna(axis=0, how="all")
    return df


def split_xy(df: pd.DataFrame, min_class_count: int = 3):
    # Sempre considerar a última coluna como target (classificação)
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    # Target sempre categórico (LabelEncoder via factorize)
    y = y.astype(str)
    # Remover classes com poucas amostras para evitar erros no KFold/stratify
    counts = y.value_counts()
    valid_classes = counts[counts >= min_class_count].index
    keep = y.isin(valid_classes)
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    y = pd.Series(pd.factorize(y)[0], index=y.index)
    return X, y


def build_preprocessor(num_feats, cat_feats) -> ColumnTransformer:
    tx = []
    if num_feats:
        tx.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_feats))
    if cat_feats:
        tx.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_feats))
    return ColumnTransformer(tx, remainder="drop")


def make_pipeline(preprocessor, ccp_alpha=0.0, min_samples_split=MSS_UNPRUNED):
    return Pipeline([
        ("pre", clone(preprocessor)),
        ("model", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            ccp_alpha=ccp_alpha,
            min_samples_split=min_samples_split,
        )),
    ])


def select_ccp_alpha(X_train, y_train, num_feats, cat_feats) -> float:
    if y_train.nunique() < 2:
        return 0.0

    # Tentar stratify; se falhar (classes raras no fold), usar sem stratify
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            random_state=RANDOM_STATE, stratify=y_train,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            random_state=RANDOM_STATE,
        )

    pre = build_preprocessor(num_feats, cat_feats)
    X_tr_p, X_val_p = pre.fit_transform(X_tr), pre.transform(X_val)

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_split=MSS_PRUNED)
    tree.fit(X_tr_p, y_tr)

    alphas = np.unique(tree.cost_complexity_pruning_path(X_tr_p, y_tr).ccp_alphas)
    if alphas.size == 0:
        return 0.0
    # Usar o maior valor possível de pruning
    return float(alphas[-1])


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_dir: Path):
    data_file = find_data_file(dataset_dir)
    if data_file is None:
        return None

    df = load_dataset(data_file)
    if df.shape[1] < 2:
        return None

    X, y = split_xy(df)
    if X.empty or y.empty:
        return None


    # Detectar colunas numéricas e categóricas corretamente
    num_feats = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_feats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    preprocessor = build_preprocessor(num_feats, cat_feats)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accdiffs, dra_vals, gra_vals = [], [], []

    for tr_idx, te_idx in cv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        alpha = select_ccp_alpha(X_tr, y_tr, num_feats, cat_feats)

        unpruned = make_pipeline(preprocessor)
        pruned = make_pipeline(preprocessor, ccp_alpha=alpha, min_samples_split=MSS_PRUNED)

        unpruned.fit(X_tr, y_tr)
        pruned.fit(X_tr, y_tr)

        yp_u, yp_p = unpruned.predict(X_te), pruned.predict(X_te)

        accdiffs.append(accuracy_score(y_te, yp_p) - accuracy_score(y_te, yp_u))
        dra_vals.append(RashomonAlignment.distributional(yp_u, yp_p))
        gra_vals.append(RashomonAlignment.geometric(
            [unpruned, pruned], X_tr,
            feature_cols=X_tr.columns, n_samples=1000,
            random_state=RANDOM_STATE,
        )["alignment"])

    return {
        "dataset": dataset_dir.name,
        "accdiff": float(np.mean(accdiffs)),
        "dRA": float(np.mean(dra_vals)),
        "gRA": float(np.mean(gra_vals)),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    datasets_dir = Path(__file__).resolve().parent / "datasets"
    results, skipped = [], []

    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            r = evaluate_dataset(d)
        except Exception:
            r = None
        (results if r else skipped).append(r or d.name)

    out = Path(__file__).resolve().parent
    pd.DataFrame(results).sort_values("dataset").to_csv(out / "dt_pruning_rashomon_results.csv", index=False)
    if skipped:
        (out / "dt_pruning_rashomon_skipped.txt").write_text("\n".join(sorted(skipped)), encoding="utf-8")


if __name__ == "__main__":
    main()
