import pandas as pd
import numpy as np
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.main.prompts.text import ADVICED_FEATURE_PROMPT
from src.main.utils.response_parsers import extract_code


def run_phase2_with_advice(
    task_desc: str, df: pd.DataFrame, profile: str, llm: GigaChat, advice: str = ""
):
    prompt = ADVICED_FEATURE_PROMPT.format(task_desc=task_desc, profile=profile)

    if advice.strip():
        prompt += f"\n\n⚡ [УЧТИ РЕКОМЕНДАЦИИ АНАЛИТИКА]:\n{advice}\nПримени их, строго соблюдая ⛔ ограничения."

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Начни. Верни ТОЛЬКО код."),
    ]

    for attempt in range(1, 6):
        print(f"\n Генерация (итерация 2): попытка {attempt}/5")
        try:
            resp = llm.invoke(messages)
            code = extract_code(
                str(resp.content) if hasattr(resp, "content") else str(resp)
            )

            ns = {"pd": pd, "np": np}
            exec(code, ns)
            df_res, new_cols = ns["generate_features"](df.copy())
           
            assert len(new_cols) >= 8, f"Слишком мало фич: {len(new_cols)}"
            print(f" Успех! Сгенерировано {len(new_cols)} фич.")
            return code, df_res, new_cols

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f" {err_msg}")
            if attempt == 5:
                raise RuntimeError("Генерация провалилась")
            messages.append(
                HumanMessage(
                    content=f"[ОШИБКА]: {err_msg}\nИсправь ТОЛЬКО проблемную строку. Верни полную функцию."
                )
            )
    return None, None, []
