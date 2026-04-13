import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.main.config import MAX_CATBOOST_EVAL_RETRIES


@dataclass
class CatBoostEvalResult:
    top5: list[str]
    feature_importance: pd.Series
    cv_auc: float
    target_correlation: dict[str, float]
    train_roc_auc: float
    class_imbalance_ratio: float
    class_counts: dict[str, int]
    confusion_matrix: np.ndarray
    confusion_matrix_labels: list
    top_errors_text: str
    final_fit_time_sec: float
    total_eval_time_sec: float


def _is_categorical_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_categorical_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s):
        return True
    if pd.api.types.is_string_dtype(s):
        return True
    return False


def _prepare_xy(
    df: pd.DataFrame, target_col: str, drop_cols: set[str]
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[target_col]
    cat_indices: list[int] = []
    X = X.copy()
    for j, col in enumerate(X.columns):
        raw = df[col]
        if _is_categorical_series(raw):
            X[col] = raw.fillna("__UNKNOWN__").astype(str)
            cat_indices.append(j)
        else:
            X[col] = (
                pd.to_numeric(raw, errors="coerce")
                .fillna(-999)
                .replace([np.inf, -np.inf], -999)
            )
    return X, y, cat_indices


def _numeric_target_correlations(X: pd.DataFrame, y: pd.Series, cat_indices: set[int]):
    out: dict[str, float] = {}
    yf = pd.to_numeric(y, errors="coerce")
    for i, col in enumerate(X.columns):
        if i in cat_indices:
            continue
        xc = pd.to_numeric(X[col], errors="coerce")
        m = xc.notna() & yf.notna()
        if m.sum() < 2:
            continue
        c = xc[m].corr(yf[m])
        if pd.notna(c):
            out[col] = float(c)
    return out


def _class_imbalance_stats(y: pd.Series) -> tuple[float, dict[str, int]]:
    vc = y.value_counts()
    counts = {str(k): int(v) for k, v in vc.items()}
    if len(vc) < 2:
        return 1.0, counts
    ratio = float(vc.min() / vc.max())
    return ratio, counts


def _train_roc_auc(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    proba = model.predict_proba(X)
    y_arr = np.asarray(y)
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_arr, proba[:, 1]))
    return float(
        roc_auc_score(y_arr, proba, multi_class="ovr", average="macro")
    )


def _top_errors_summary(
    y: pd.Series,
    proba_full: np.ndarray,
    classes: np.ndarray,
    n: int = 5,
) -> str:
    """Топ ошибок по «уверенности» модели (бинарная классификация, порог 0.5)."""
    classes = np.asarray(classes)
    y_arr = np.asarray(y)
    if (
        len(classes) != 2
        or proba_full.shape[1] != 2
        or proba_full.shape[0] != len(y_arr)
    ):
        return (
            "Top errors: детализация FP/FN по вероятностям — только для бинарной классификации; "
            f"классов: {len(classes)}."
        )

    proba_pos = proba_full[:, 1]
    y_bin = (y_arr == classes[1]).astype(int)
    pred = (proba_pos >= 0.5).astype(int)

    fp_mask = (y_bin == 0) & (pred == 1)
    fn_mask = (y_bin == 1) & (pred == 0)
    parts: list[str] = []

    if fp_mask.any():
        idx = np.flatnonzero(fp_mask)
        order = idx[np.argsort(-proba_pos[idx])][:n]
        parts.append(
            "False positive (истина меньший класс, предсказан положительный): "
            + ", ".join(f"строка_в_выборке#{i} p={proba_pos[i]:.3f}" for i in order)
        )
    if fn_mask.any():
        idx = np.flatnonzero(fn_mask)
        order = idx[np.argsort(proba_pos[idx])][:n]
        parts.append(
            "False negative (истина положительный класс, предсказан отрицательный): "
            + ", ".join(f"строка_в_выборке#{i} p={proba_pos[i]:.3f}" for i in order)
        )

    if not parts:
        return "Ошибок при пороге 0.5 на этой выборке нет."
    return " | ".join(parts)


def select_top5_features_fast(
    df: pd.DataFrame,
    target_col: str = "target",
    max_sample: int = 50_000,
    time_budget: float = 25.0,
) -> CatBoostEvalResult:
    """
    Отбор топ-5 признаков через CatBoost.
    ROC-AUC на валидации — один быстрый holdout (80/20, shuffle, без стратификации).
    Затем финальная модель на всей подвыборке для важности признаков.
    При превышении max_sample — случайная подвыборка строк (random_state=42).
    """
    start = time.perf_counter()

    drop_cols = {target_col, "row_id", "user_id", "product_id", "id", "index"}
    n = min(len(df), max_sample)
    if len(df) > n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), size=n, replace=False)
        df_s = df.iloc[idx].copy()
    else:
        df_s = df.copy()

    X, y, cat_indices = _prepare_xy(df_s, target_col, drop_cols)
    cat_set = set(cat_indices)

    imb_ratio, class_counts = _class_imbalance_stats(y)

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

    n_classes = int(y.nunique())

    max_attempts = MAX_CATBOOST_EVAL_RETRIES
    cv_auc = 0.0
    fi = pd.Series(dtype=float)
    top5: list[str] = []
    target_corr: dict[str, float] = {}
    train_auc = 0.0
    cm = np.zeros((2, 2), dtype=int)
    cm_labels: list = []
    top_err = ""
    final_fit_sec = 0.0

    for attempt in range(1, max_attempts + 1):
        try:
            if n_classes >= 2 and len(X) >= 10:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    shuffle=True,
                    random_state=42,
                )
                clf = CatBoostClassifier(**CATBOOST_PARAMS)
                clf.fit(
                    X_tr,
                    y_tr,
                    cat_features=cat_indices if cat_indices else None,
                    verbose=False,
                )
                proba = clf.predict_proba(X_val)
                if y_val.nunique() < 2:
                    cv_auc = 0.0
                elif proba.shape[1] == 2:
                    cv_auc = float(roc_auc_score(y_val, proba[:, 1]))
                else:
                    cv_auc = float(
                        roc_auc_score(
                            y_val,
                            proba,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
            else:
                cv_auc = 0.0

            model = CatBoostClassifier(**CATBOOST_PARAMS)
            t0 = time.perf_counter()
            model.fit(
                X,
                y,
                cat_features=cat_indices if cat_indices else None,
                verbose=False,
            )
            final_fit_sec = time.perf_counter() - t0

            fi = pd.Series(
                model.get_feature_importance(), index=X.columns
            ).sort_values(ascending=False)
            top5 = fi.index[:5].tolist()
            target_corr = _numeric_target_correlations(X, y, cat_set)
            train_auc = _train_roc_auc(model, X, y)

            y_pred = model.predict(X).ravel()
            cl_order = np.asarray(getattr(model, "classes_", np.unique(np.asarray(y))))
            cm = confusion_matrix(np.asarray(y), y_pred, labels=cl_order)
            cm_labels = [str(x) for x in cl_order]

            proba_full = model.predict_proba(X)
            if proba_full.shape[1] == 2:
                top_err = _top_errors_summary(y, proba_full, cl_order)
            else:
                top_err = (
                    "Top errors: для multiclass используйте confusion matrix; "
                    "детализация FP/FN по вероятностям не выводится."
                )

            break
        except Exception as e:
            print(
                f"\n select_top5_features_fast: попытка {attempt}/{max_attempts}: {e}"
            )
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 16))

    total_sec = time.perf_counter() - start
    print(f"    Holdout ROC-AUC (20% val): {cv_auc:.4f}")
    print(f"    Top 5: {top5}")

    return CatBoostEvalResult(
        top5=top5,
        feature_importance=fi,
        cv_auc=cv_auc,
        target_correlation=target_corr,
        train_roc_auc=train_auc,
        class_imbalance_ratio=imb_ratio,
        class_counts=class_counts,
        confusion_matrix=cm,
        confusion_matrix_labels=cm_labels,
        top_errors_text=top_err,
        final_fit_time_sec=final_fit_sec,
        total_eval_time_sec=total_sec,
    )
