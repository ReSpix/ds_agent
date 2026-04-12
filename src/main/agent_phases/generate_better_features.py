from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

from src.main.prompts.text import ADVICED_FEATURE_PROMPT
from src.main.utils.response_parsers import extract_code


def validate_generated_features(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    new_cols: list[str],
    existing_cols: list[str] | None = None,
    min_features: int = 8,
) -> list[str]:
    if not isinstance(df_after, pd.DataFrame):
        raise TypeError("generate_features должна возвращать pd.DataFrame первым значением")

    if not isinstance(new_cols, list):
        raise TypeError("generate_features должна возвращать list[str] вторым значением")

    if not all(isinstance(col, str) for col in new_cols):
        raise TypeError("Все элементы new_cols должны быть строками")

    cleaned_cols = []
    seen = set()
    for col in new_cols:
        if col not in seen:
            cleaned_cols.append(col)
            seen.add(col)

    if len(cleaned_cols) < min_features:
        raise ValueError(f"Слишком мало фич: {len(cleaned_cols)}. Нужно минимум {min_features}")

    missing = [col for col in cleaned_cols if col not in df_after.columns]
    if missing:
        raise ValueError(f"Функция заявила новые признаки, которых нет в DataFrame: {missing}")

    old_cols = set(df_before.columns)
    bad_reused = [col for col in cleaned_cols if col in old_cols]
    if bad_reused:
        raise ValueError(
            f"Среди new_cols есть уже существовавшие колонки: {bad_reused}"
        )

    if existing_cols:
        existing_cols_set = set(existing_cols)
        duplicated_existing = [col for col in cleaned_cols if col in existing_cols_set]
        if duplicated_existing:
            raise ValueError(
                f"Новые признаки дублируют ранее созданные признаки: {duplicated_existing}"
            )

    actually_new = [col for col in df_after.columns if col not in df_before.columns]
    if not actually_new:
        raise ValueError("Функция не добавила ни одной новой колонки в DataFrame")

    not_actually_new = [col for col in cleaned_cols if col not in actually_new]
    if not_actually_new:
        raise ValueError(
            f"Колонки из new_cols не являются реально новыми: {not_actually_new}"
        )

    bad_constant = []
    bad_all_nan = []

    for col in cleaned_cols:
        series = df_after[col]
        if series.isna().all():
            bad_all_nan.append(col)
        elif series.nunique(dropna=False) <= 1:
            bad_constant.append(col)

    if bad_all_nan:
        raise ValueError(f"Есть признаки, состоящие только из NaN: {bad_all_nan}")

    if bad_constant:
        raise ValueError(f"Есть константные признаки: {bad_constant}")

    return cleaned_cols


def run_phase2_with_advice(
    task_desc: str,
    df: pd.DataFrame,
    profile: str,
    llm: GigaChat,
    advice: str = "",
    existing_cols: list[str] | None = None,
    existing_code: str = "",
    min_features: int = 8,
    max_attempts: int = 5,
):
    existing_cols = existing_cols or []

    prompt = ADVICED_FEATURE_PROMPT.format(
        task_desc=task_desc,
        profile=profile,
    )

    extra_context_parts = []

    if advice.strip():
        extra_context_parts.append(
            "Рекомендации аналитика, которые нужно учесть:\n"
            f"{advice.strip()}"
        )

    if existing_cols:
        extra_context_parts.append(
            "Ранее уже были созданы признаки. "
            "НЕЛЬЗЯ дублировать или переопределять их.\n"
            f"Список уже созданных признаков:\n{existing_cols}"
        )

    if existing_code.strip():
        extra_context_parts.append(
            "Ниже код предыдущей генерации признаков. "
            "Используй его как контекст, но НЕ копируй старые признаки повторно.\n"
            f"{existing_code}"
        )

    prompt += (
        "\n\nДополнительные ограничения:\n"
        "1. Верни ТОЛЬКО Python-код.\n"
        "2. Код должен содержать функцию generate_features(df).\n"
        "3. Функция должна вернуть (df_result, new_cols).\n"
        "4. new_cols должен содержать только действительно НОВЫЕ признаки.\n"
        "5. Нельзя менять или удалять существующие колонки.\n"
        "6. Нельзя повторять признаки из existing_cols.\n"
        "7. Не используй inplace=True для операций над отдельными колонками.\n"
        "8. Вместо df[col].fillna(..., inplace=True) пиши df[col] = df[col].fillna(...).\n"
    )

    if extra_context_parts:
        prompt += "\n\n" + "\n\n".join(extra_context_parts)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Начни. Верни только код полной функции generate_features(df)."),
    ]

    last_error = None

    for attempt in range(1, max_attempts + 1):
        print(f"\nГенерация (итерация улучшения): попытка {attempt}/{max_attempts}")

        try:
            resp = llm.invoke(messages)
            raw_text = str(resp.content) if hasattr(resp, "content") else str(resp)
            code = extract_code(raw_text)

            ns: dict[str, Any] = {"pd": pd, "np": np}
            exec(code, ns)

            if "generate_features" not in ns:
                raise ValueError("В ответе нет функции generate_features")

            fn = ns["generate_features"]
            if not callable(fn):
                raise TypeError("generate_features существует, но не является функцией")

            df_input = df.copy(deep=True)
            result = fn(df_input)

            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    "generate_features должна возвращать ровно два значения: (df_result, new_cols)"
                )

            df_res, new_cols = result

            new_cols = validate_generated_features(
                df_before=df,
                df_after=df_res,
                new_cols=new_cols,
                existing_cols=existing_cols,
                min_features=min_features,
            )

            print(f"Успех! Сгенерировано {len(new_cols)} новых фич.")
            return code, df_res, new_cols

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(last_error)

            if attempt == max_attempts:
                raise RuntimeError(
                    f"Генерация провалилась после {max_attempts} попыток. "
                    f"Последняя ошибка: {last_error}"
                ) from e

            messages.append(
                HumanMessage(
                    content=(
                        f"[ОШИБКА]\n{last_error}\n\n"
                        "Исправь код полностью. Верни ПОЛНУЮ функцию generate_features(df) целиком.\n"
                        "Не объясняй ничего текстом. Верни только код."
                    )
                )
            )

    raise RuntimeError("Генерация провалилась")