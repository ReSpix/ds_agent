import pandas as pd
import numpy as np
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pathlib import Path

from src.main.utils.response_parsers import extract_code
from src.main.utils.traceback_extractor import extract_exec_error
from src.main.config import MAX_MERGE_CODE_ATTEMPTS
from src.main.prompts.text import MERGE_PROMPT


def merge_phase(
    task_desc: str,
    data_dir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    llm: GigaChat,
):
    schema = []
    for f in sorted(Path(data_dir).glob("*.csv")):
        if f.name in ("train.csv", "test.csv"):
            continue
        df = pd.read_csv(f, on_bad_lines='skip')
        schema.append(
            f"📄 {f.name}\nКолонки: {list(df.columns)}\nПример:\n{df.head(2).to_string()}\n"
        )

    messages = [
        SystemMessage(content=MERGE_PROMPT),
        HumanMessage(
            content=f"Задача: {task_desc}\n\nСхема данных:\n{''.join(schema)}"
        ),
    ]

    max_attempts = MAX_MERGE_CODE_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        print(f"\nПопытка merge {attempt} (лимит {max_attempts})")
        try:
            response = llm.invoke(messages)
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            if attempt >= max_attempts:
                raise
            continue
        code = extract_code(
            str(response.content) if hasattr(response, "content") else str(response)
        )

        print(code)

        messages.append(AIMessage(content=code))

        try:
            ns = {"pd": pd, "np": np, "Path": Path}
            exec(code, ns)
            func = ns["merge_data"]

            merged_train, merged_test = func(train_df.copy(), test_df.copy(), data_dir)
            print(f" Успех! Train: {merged_train.shape}, Test: {merged_test.shape}")
            return (
                merged_train,
                merged_test,
                code,
            )  

        except Exception as e:
            line_no, error_line = extract_exec_error(code, e)
            err_text = (
                f"❌ Ошибка '{type(e).__name__}: {e}' в строке {line_no}:"
                + repr(error_line)
            )
            print(err_text)
            messages.append(
                HumanMessage(
                    content=f"{err_text}\nИсправь код и верни заново. Ни в коем случае не допускай ту же ошибку еще раз. Будь внимательнее к задаче. ОБЯЗАТЕЛЬНО вначале напиши комментарий почему ты допустил ошибку и как будешь ее исправлять."
                )
            )

    raise RuntimeError(
        f"Не удалось получить рабочий merge-код за {max_attempts} попыток"
    )
