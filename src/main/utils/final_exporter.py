import pandas as pd
import numpy as np
import os

from src.main.utils.response_parsers import extract_code
from src.main.utils.traceback_extractor import extract_exec_error
import os
import pandas as pd
import numpy as np

from src.main.utils.traceback_extractor import extract_exec_error


def export_final_output(
    code: str,
    selected_features: list[str],
    train_df_features: pd.DataFrame,
    test_df_features: pd.DataFrame,
    original_train: pd.DataFrame,
    original_test: pd.DataFrame,
    id_col: str,
    target_col: str,
):
    """
    Применяет финальный код генерации признаков к train/test после merge,
    оставляет только нужные колонки и сохраняет output/train.csv, output/test.csv.

    Формат:
        train.csv = [id_col, target_col] + selected_features
        test.csv  = [id_col] + selected_features
    """
    if not isinstance(selected_features, list) or not all(isinstance(c, str) for c in selected_features):
        raise TypeError("selected_features должен быть list[str]")

    if len(selected_features) == 0:
        raise ValueError("Список selected_features пуст")

    if len(selected_features) > 5:
        raise ValueError(
            f"selected_features содержит {len(selected_features)} признаков. Нужно не больше 5."
        )

    # проверяем обязательные колонки в оригинальных данных
    if id_col not in original_train.columns:
        raise ValueError(f"id_col='{id_col}' отсутствует в original_train")

    if id_col not in original_test.columns:
        raise ValueError(f"id_col='{id_col}' отсутствует в original_test")

    if target_col not in original_train.columns:
        raise ValueError(f"target_col='{target_col}' отсутствует в original_train")

    try:
        ns = {"pd": pd, "np": np}
        exec(code, ns)

        if "generate_features" not in ns or not callable(ns["generate_features"]):
            raise ValueError("В коде отсутствует функция generate_features(df)")

        train_full, _ = ns["generate_features"](train_df_features.copy())
        test_full, _ = ns["generate_features"](test_df_features.copy())

    except Exception as e:
        line_no, error_line = extract_exec_error(code, e)
        err_text = (
            f"Ошибка при выполнении feature-кода: {type(e).__name__}: {e}. "
            f"Строка {line_no}: {repr(error_line)}"
        )
        raise RuntimeError(err_text) from e

    # проверяем, что все выбранные признаки реально существуют
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

    # собираем итоговые датафреймы строго по формату задания
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

    # защита от дублей колонок
    if len(final_train.columns) != len(set(final_train.columns)):
        raise ValueError(f"В final_train есть дубли колонок: {list(final_train.columns)}")

    if len(final_test.columns) != len(set(final_test.columns)):
        raise ValueError(f"В final_test есть дубли колонок: {list(final_test.columns)}")

    # проверка порядка колонок
    final_train = final_train[train_export_cols]
    final_test = final_test[test_export_cols]

    # object/category -> str только для selected_features
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