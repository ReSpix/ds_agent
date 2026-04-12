import pandas as pd
import numpy as np
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.main.prompts.text import FEATURE_PROMPT
from src.main.utils.response_parsers import extract_code
from src.main.utils.traceback_extractor import extract_exec_error


def run_feature_phase(
    task_desc: str,
    data_profile: str,
    df: pd.DataFrame,
    llm: GigaChat,
    target_col: str = "target",
):
    prompt = FEATURE_PROMPT.format(
        task_desc=task_desc, profile=data_profile, target_col=target_col
    )

    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        print(f"\n Генерация фич: попытка {attempt}/{max_attempts}")
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            print(f" Ошибка LLM: {e}")
            if attempt >= max_attempts:
                raise
            continue
        code = extract_code(str(resp.content) if hasattr(resp, "content") else str(resp))

        try:
            ns = {"pd": pd, "np": np}
            exec(code, ns)
            func = ns["generate_features"]

            df_res, new_cols = func(df.copy())
         
            print(f" Успех! Новые колонки: {new_cols}")
            return df_res, code, new_cols

        except Exception as e:
            print(f" Ошибка: {e}")
            line_no, line_text = extract_exec_error(code, e)
            prompt += f"\n\n[ОШИБКА]: {e} в строке {line_no}: {line_text}\nИсправь код и верни заново."

    raise RuntimeError(
        f"Не удалось сгенерировать фичи за {max_attempts} попыток"
    )
