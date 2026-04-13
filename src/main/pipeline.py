import inspect
import json
import os

import numpy as np
import pandas as pd

from src.main.agent_phases.feature_advice import generate_advice
from src.main.agent_phases.generate_better_features import run_phase2_with_advice
from src.main.agent_phases.generate_features_primary import run_feature_phase
from src.main.agent_phases.merge_dataset import merge_phase
from src.main.boosting import select_top5_features_fast
from src.main.llm_builder import build_gigachat
from src.main.utils.atrifact_saver import ArtifactSaver
from src.main.utils.data_profiler import build_compact_profile
from src.main.utils.final_exporter import export_final_output
from src.main.utils.response_parsers import extract_json
from src.main.config import MAX_IMPROVE_ROUNDS, TARGET_ROC_AUC


def deduplicate_cols(cols: list[str]) -> list[str]:
    seen = set()
    result = []
    for col in cols:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def get_base_candidate_cols(
    merged_df: pd.DataFrame,
    original_df: pd.DataFrame,
    target_col: str,
) -> list[str]:
    original_cols = set(original_df.columns)
    return [c for c in merged_df.columns if c not in original_cols and c != target_col]


def apply_feature_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    ns = {"pd": pd, "np": np}
    exec(code, ns)
    if "generate_features" not in ns or not callable(ns["generate_features"]):
        raise ValueError("В коде отсутствует функция generate_features(df)")
    df_res, _ = ns["generate_features"](df.copy())
    if not isinstance(df_res, pd.DataFrame):
        raise TypeError("generate_features должна возвращать DataFrame первым значением")
    return df_res


def parse_eval_result(result):
    """
    Универсальный разбор результата select_top5_features_fast.
    Поддерживает и старый tuple, и новый объект CatBoostEvalResult.
    """
    if isinstance(result, tuple) and len(result) == 3:
        top5, fi, score = result
        return top5, fi, score

    # Новый объект
    attr_candidates = {
        "top5": ["top_features", "selected_features", "top5", "best_features"],
        "fi": ["feature_importance", "feature_importances", "importance", "importances"],
        "score": ["cv_auc", "score", "holdout_auc", "roc_auc", "auc"],
    }

    top5 = None
    fi = None
    score = None

    for name in attr_candidates["top5"]:
        if hasattr(result, name):
            top5 = getattr(result, name)
            break

    for name in attr_candidates["fi"]:
        if hasattr(result, name):
            fi = getattr(result, name)
            break

    for name in attr_candidates["score"]:
        if hasattr(result, name):
            score = getattr(result, name)
            break

    if top5 is None or fi is None or score is None:
        raise TypeError(
            "Не удалось разобрать результат select_top5_features_fast. "
            f"Тип: {type(result)}, содержимое: {result}"
        )

    return top5, fi, score


def call_phase2_with_compat(
    task_desc: str,
    df: pd.DataFrame,
    profile: str,
    llm,
    advice: str,
    existing_cols: list[str],
    existing_code: str,
):
    """
    Совместимость со старой и новой сигнатурой run_phase2_with_advice.
    """
    sig = inspect.signature(run_phase2_with_advice)
    params = sig.parameters

    kwargs = {
        "task_desc": task_desc,
        "df": df,
        "profile": profile,
        "llm": llm,
        "advice": advice,
    }

    if "existing_cols" in params:
        kwargs["existing_cols"] = existing_cols

    if "existing_code" in params:
        kwargs["existing_code"] = existing_code

    return run_phase2_with_advice(**kwargs)

def merge_final_features(original_df, features_df, id_col, top_features):
    # Оставляем только ID и новые топ-5 признаков
    new_data = features_df[[id_col] + list(top_features)].drop_duplicates(subset=[id_col])
    # Делаем left join к оригиналу — это гарантирует сохранность всех исходных колонок
    return original_df.merge(new_data, on=id_col, how='left')

def main():
    llm = build_gigachat()
    py_code_saver = ArtifactSaver("artifacts", "py")

    with open("data/readme.txt", "r", encoding="utf-8") as file:
        task_description = file.read()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print(
        f"Параметры остановки: TARGET_ROC_AUC={TARGET_ROC_AUC}, "
        f"MAX_IMPROVE_ROUNDS={MAX_IMPROVE_ROUNDS}"
    )

    # ===== 1. MERGE =====
    m_train, m_test, merge_code = merge_phase(
        task_desc=task_description,
        data_dir="data",
        train_df=train,
        test_df=test,
        llm=llm,
    )
    py_code_saver.save("merge", merge_code)

    # ===== 2. ID / TARGET =====
    target_prompt = f"""
Изучи описание данных:
{task_description}

Определи главную колонку-идентификатор (ID) и целевую переменную (Target).
Верни ТОЛЬКО JSON:
{{"id_col": "...", "target_col": "..."}}
"""

    response = extract_json(llm.invoke(target_prompt).content)
    print(response)

    keys = json.loads(response)
    id_col = keys.get("id_col", "id")
    target_col = keys.get("target_col", "target")

    # ===== 3. PROFILE =====
    profile = build_compact_profile(m_train, target_col)

    # ===== 4. BASE CANDIDATES =====
    base_candidate_cols = get_base_candidate_cols(
        merged_df=m_train,
        original_df=train,
        target_col=target_col,
    )
    print(f"Базовых candidate-признаков после merge: {len(base_candidate_cols)}")

    # ===== 5. PHASE 1 =====
    df_phase1, phase1_code, phase1_new_cols = run_feature_phase(
        task_desc=task_description,
        data_profile=profile,
        df=m_train,
        llm=llm,
        target_col=target_col,
    )
    py_code_saver.save("feature_engineering_phase1", phase1_code)

    candidate_cols_phase1 = deduplicate_cols(base_candidate_cols + phase1_new_cols)

    eval_df_phase1 = df_phase1[candidate_cols_phase1 + [target_col]].copy()
    phase1_result = select_top5_features_fast(
        eval_df_phase1,
        target_col=target_col,
    )
    top5_phase1, fi_phase1, score_phase1 = parse_eval_result(phase1_result)

    print("\n[PHASE 1]")
    print("Top-5:", top5_phase1)
    print("Score:", score_phase1)

    # ===== 6. BEST STATE INIT =====
    best_score = score_phase1
    best_top5 = list(top5_phase1)
    best_fi = fi_phase1
    best_df = df_phase1.copy()
    best_candidate_cols = list(candidate_cols_phase1)
    best_generated_cols = list(phase1_new_cols)
    best_code_chain = [phase1_code]

    current_df = df_phase1.copy()
    current_generated_cols = list(phase1_new_cols)
    current_code_chain = [phase1_code]

    # ===== 7. ITERATIVE IMPROVEMENT =====
    round_idx = 0
    while best_score < TARGET_ROC_AUC and round_idx < MAX_IMPROVE_ROUNDS:
        round_idx += 1
        print(
            f"\n=== Итерация улучшения {round_idx} "
            f"(лучший holdout ROC-AUC={best_score:.4f}, цель {TARGET_ROC_AUC}) ==="
        )

        try:
            advice = generate_advice(
                task_desc=task_description,
                df_profile=profile,
                cv_auc=best_score,
                feature_importance=dict(best_fi) if not isinstance(best_fi, dict) else best_fi,
                feature_code="\n\n".join(current_code_chain),
                llm=llm,
            )

            new_code, new_df, new_cols = call_phase2_with_compat(
                task_desc=task_description,
                df=current_df,
                profile=profile,
                llm=llm,
                advice=advice,
                existing_cols=deduplicate_cols(base_candidate_cols + current_generated_cols),
                existing_code="\n\n".join(current_code_chain),
            )

            py_code_saver.save(f"feature_engineering_round_{round_idx}", new_code)

            current_df = new_df.copy()
            current_generated_cols = deduplicate_cols(current_generated_cols + new_cols)
            current_code_chain = current_code_chain + [new_code]

            current_candidate_cols = deduplicate_cols(
                base_candidate_cols + current_generated_cols
            )

            eval_df_current = current_df[current_candidate_cols + [target_col]].copy()
            current_result = select_top5_features_fast(
                eval_df_current,
                target_col=target_col,
            )
            top5_current, fi_current, score_current = parse_eval_result(current_result)

            print(f"    Holdout ROC-AUC (20% val): {score_current:.4f}")
            print(f"    Top 5: {top5_current}")

            if score_current > best_score:
                print("    Есть улучшение, сохраняем лучший state.")
                best_score = score_current
                best_top5 = list(top5_current)
                best_fi = fi_current
                best_df = current_df.copy()
                best_candidate_cols = list(current_candidate_cols)
                best_generated_cols = list(current_generated_cols)
                best_code_chain = list(current_code_chain)

        except Exception as e:
            print(f"Итерация улучшения: ошибка, переходим к следующей попытке: {e}")
            continue

    if best_score < TARGET_ROC_AUC:
        raise ValueError(
            f"Лучший holdout ROC-AUC {best_score:.4f} < цели {TARGET_ROC_AUC}. "
            f"Итераций улучшения: {round_idx}; лимит MAX_IMPROVE_ROUNDS={MAX_IMPROVE_ROUNDS}."
        )
    else:
        print(
            f"\nЦель достигнута: holdout ROC-AUC >= {TARGET_ROC_AUC} (лучший {best_score:.4f})."
        )

    print("\n[FINAL BEST]")
    print("Best score:", best_score)
    print("Best top-5:", best_top5)

    # ===== 8. APPLY BEST CODE CHAIN TO TEST =====
    final_test_features = m_test.copy()
    for idx, code in enumerate(best_code_chain, start=1):
        print(f"Применяем feature-chain к test: шаг {idx}/{len(best_code_chain)}")
        final_test_features = apply_feature_code(final_test_features, code)

    # ===== 9. EXPORT =====
    export_final_output(
        code=best_code_chain[-1],
        selected_features=best_top5,
        train_df_features=best_df, 
        test_df_features=final_test_features, 
        original_train=train,
        original_test=test,
        id_col=id_col,
        target_col=target_col,
    )

    # 2. Формируем идеальную структуру: Исходные колонки + 5 лучших фичей
    final_train_to_export = merge_final_features(
        train, best_df, id_col, best_top5
    )
    final_test_to_export = merge_final_features(
        test, final_test_features, id_col, best_top5
    )

    # 3. Жестко перезаписываем CSV-файлы поверх тех, что создал export_final_output

    os.makedirs("output", exist_ok=True)
    final_train_to_export.to_csv("output/train.csv", index=False)
    final_test_to_export.to_csv("output/test.csv", index=False)


if __name__ == "__main__":
    main()