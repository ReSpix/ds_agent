import pandas as pd
import numpy as np
import os


def export_final_output(
    final_feature_code: str,
    top5_final: list[str],
    merged_train: pd.DataFrame,
    merged_test: pd.DataFrame,
    original_train: pd.DataFrame,
    original_test: pd.DataFrame,
):
    # 1. Применяем сгенерированный код к данным после Merge
    ns = {"pd": pd, "np": np}
    exec(final_feature_code, ns)
    train_full, _ = ns["generate_features"](merged_train.copy())
    test_full, _ = ns["generate_features"](merged_test.copy())

    # 2. Собираем колонки: ВСЕ оригинальные + ТОП-5 новых (дубли исключаются автоматически)
    final_train_cols = list(original_train.columns) + [
        c for c in top5_final if c not in original_train.columns
    ]
    final_test_cols = list(original_test.columns) + [
        c for c in top5_final if c not in original_test.columns
    ]

    # 3. Формируем финальные датафреймы
    final_train = train_full[final_train_cols]
    final_test = test_full[final_test_cols]

    # 4. Категорики → str (чтобы scoring.py сам подхватил их в cat_features)
    for col in top5_final:
        if col in final_train.columns and final_train[col].dtype == "object":
            final_train[col] = final_train[col].astype(str)
            final_test[col] = final_test[col].astype(str)

    print(f" Готово. Train: {final_train.shape} | Test: {final_test.shape}")
    print(f" Итоговые колонки: {list(final_train.columns)}")

    # 5. Сохранение
    os.makedirs("output", exist_ok=True)
    final_train.to_csv("output/train.csv", index=False)
    final_test.to_csv("output/test.csv", index=False)
