import os
from typing import Union

import numpy as np
import pandas as pd

from src.main.utils.traceback_extractor import extract_exec_error

CodeArg = Union[str, list[str]]


def _normalize_feature_codes(code: CodeArg) -> list[str]:
    if isinstance(code, str):
        return [code]
    if not code:
        raise ValueError("Список кодов генерации признаков пуст")
    return list(code)


def _apply_feature_codes(df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    out = df
    for i, c in enumerate(codes):
        ns = {"pd": pd, "np": np}
        exec(c, ns)
        if "generate_features" not in ns or not callable(ns["generate_features"]):
            raise ValueError(
                f"В фрагменте feature-кода #{i + 1} отсутствует функция generate_features(df)"
            )
        out, _ = ns["generate_features"](out.copy())
    return out


def export_final_output(
    code: CodeArg,
    selected_features: list[str],
    train_df_features: pd.DataFrame,
    test_df_features: pd.DataFrame,
    original_train: pd.DataFrame,
    original_test: pd.DataFrame,
    id_col: str,
    target_col: str,
):
    """
    Применяет цепочку кодов generate_features к train/test после merge,
    оставляет только нужные колонки и сохраняет output/train.csv, output/test.csv.

    Формат:
        train.csv = [id_col, target_col] + selected_features
        test.csv  = [id_col] + selected_features
    """
    codes = _normalize_feature_codes(code)

    if not isinstance(selected_features, list) or not all(
        isinstance(c, str) for c in selected_features
    ):
        raise TypeError("selected_features должен быть list[str]")

    if len(selected_features) == 0:
        raise ValueError("Список selected_features пуст")

    if len(selected_features) > 5:
        raise ValueError(
            f"selected_features содержит {len(selected_features)} признаков. Нужно не больше 5."
        )

    if id_col not in original_train.columns:
        raise ValueError(f"id_col='{id_col}' отсутствует в original_train")

    if id_col not in original_test.columns:
        raise ValueError(f"id_col='{id_col}' отсутствует в original_test")

    if target_col not in original_train.columns:
        raise ValueError(f"target_col='{target_col}' отсутствует в original_train")

    try:
        train_full = _apply_feature_codes(train_df_features.copy(), codes)
        test_full = _apply_feature_codes(test_df_features.copy(), codes)
    except Exception as e:
        err_snippet = codes[-1]
        line_no, error_line = extract_exec_error(err_snippet, e)
        err_text = (
            f"Ошибка при выполнении feature-кода: {type(e).__name__}: {e}. "
            f"Строка {line_no}: {repr(error_line)}"
        )
        raise RuntimeError(err_text) from e

    missing_train = [col for col in selected_features if col not in train_full.columns]
    missing_test = [col for col in selected_features if col not in test_full.columns]

    if missing_train:
        raise ValueError(
            f"В train после generate_features отсутствуют выбранные признаки: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"В test после generate_features отсутствуют выбранные признаки: {missing_test}"
        )

    train_export_cols = [id_col, target_col] + selected_features
    test_export_cols = [id_col] + selected_features

    final_train = pd.concat(
        [
            original_train[[id_col, target_col]].reset_index(drop=True),
            train_full[selected_features].reset_index(drop=True),
        ],
        axis=1,
    )

    final_test = pd.concat(
        [
            original_test[[id_col]].reset_index(drop=True),
            test_full[selected_features].reset_index(drop=True),
        ],
        axis=1,
    )

    if len(final_train.columns) != len(set(final_train.columns)):
        raise ValueError(f"В final_train есть дубли колонок: {list(final_train.columns)}")

    if len(final_test.columns) != len(set(final_test.columns)):
        raise ValueError(f"В final_test есть дубли колонок: {list(final_test.columns)}")

    final_train = final_train[train_export_cols]
    final_test = final_test[test_export_cols]

    for col in selected_features:
        if col in final_train.columns:
            if str(final_train[col].dtype) in ("object", "category"):
                final_train[col] = final_train[col].astype(str)
        if col in final_test.columns:
            if str(final_test[col].dtype) in ("object", "category"):
                final_test[col] = final_test[col].astype(str)

    os.makedirs("output", exist_ok=True)
    final_train.to_csv("output/train.csv", index=False)
    final_test.to_csv("output/test.csv", index=False)

    print(f"Готово. Train: {final_train.shape} | Test: {final_test.shape}")
    print(f"Train columns: {list(final_train.columns)}")
    print(f"Test columns: {list(final_test.columns)}")
