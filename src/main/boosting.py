import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score


def select_top5_features_fast(
    df: pd.DataFrame,
    target_col: str = "target",
    max_sample: int = 100000,
    time_budget: float = 25.0,
) -> tuple[list[str], pd.Series, float]:
    """
    Быстрый отбор 5 лучших признаков через CatBoost CV.
    Возвращает: (top_5_names, feature_importance_series, cv_auc)
    """
    start = time.time()

    n = min(len(df), max_sample)
    idx = np.random.RandomState(42).choice(len(df), n, replace=False)
    df_s = df.iloc[idx].copy()

    drop_cols = {target_col, "row_id", "user_id", "product_id", "id", "index"}
    X = df_s.drop(columns=[c for c in drop_cols if c in df_s.columns], errors="ignore")
    y = df_s[target_col]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("__UNKNOWN__").astype(str)
        else:
            X[col] = (
                pd.to_numeric(X[col], errors="coerce")
                .fillna(-999)
                .replace([np.inf, -np.inf], -999)
            )

    cat_indices = [i for i, c in enumerate(X.columns) if X[c].dtype == "object"]

    params = {
        "iterations": 150,
        "depth": 4,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "silent": True,
        "random_seed": 42,
        "auto_class_weights": "Balanced",
        "thread_count": -1,
    }

    CATBOOST_PARAMS = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": 1,
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
    }

    params = CATBOOST_PARAMS

    cv_auc = cross_val_score(
        CatBoostClassifier(**params), X, y, cv=3, scoring="roc_auc", n_jobs=1
    ).mean()

    model = CatBoostClassifier(**params)
    model.fit(X, y, cat_features=cat_indices if cat_indices else None, verbose=False)

    fi = pd.Series(model.get_feature_importance(), index=X.columns).sort_values(
        ascending=False
    )
    top5 = fi.index[:5].tolist()

    elapsed = time.time() - start

    print(f"\n Fast CV completed in {elapsed:.1f}s")
    print(f"    CV AUC: {cv_auc:.4f}")
    print(f"    Top 5: {top5}")

    return top5, fi, cv_auc