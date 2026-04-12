import json

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


def deduplicate_cols(cols: list[str]) -> list[str]:
    seen = set()
    result = []
    for col in cols:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def main():
    llm = build_gigachat()
    py_code_saver = ArtifactSaver("artifacts", "py")

    with open("data/readme.txt", "r", encoding="utf-8") as file:
        task_description = file.read()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # сколько дополнительных итераций после первой генерации
    max_extra_rounds = 6
    patience = 2

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

    # ===== 4. PHASE 1 =====
    df_current, current_code, current_new_cols = run_feature_phase(
        task_desc=task_description,
        data_profile=profile,
        df=m_train,
        llm=llm,
        target_col=target_col,
    )
    py_code_saver.save("feature_engineering_phase1", current_code)

    # все найденные кандидаты
    all_candidate_cols = deduplicate_cols(current_new_cols)

    # лучший известный набор
    best_df = df_current
    best_code = current_code

    # оцениваем первую итерацию
    best_eval_df = best_df[all_candidate_cols + [target_col]].copy()
    best_top5, best_fi, best_cv_auc = select_top5_features_fast(
        best_eval_df,
        target_col=target_col,
    )

    print("\n[PHASE 1]")
    print("Top-5:", best_top5)
    print("CV AUC:", best_cv_auc)

    no_improve_rounds = 0
    previous_round_code = current_code
    previous_round_df = df_current
    previous_round_cols = list(current_new_cols)

    # ===== 5. EXTRA ROUNDS =====
    for round_idx in range(2, max_extra_rounds + 2):
        print(f"\n[ROUND {round_idx}]")

        try:
            # advice строим по лучшему текущему состоянию
            advice = generate_advice(
                task_desc=task_description,
                df_profile=profile,
                cv_auc=best_cv_auc,
                feature_importance=dict(best_fi),
                feature_code=best_code,
                llm=llm,
            )

            # новая генерация идет поверх текущего лучшего df
            new_code, new_df, new_cols = run_phase2_with_advice(
                task_desc=task_description,
                df=best_df,
                profile=profile,
                llm=llm,
                advice=advice,
                existing_cols=all_candidate_cols,
                existing_code=best_code,
            )

            py_code_saver.save(f"feature_engineering_phase{round_idx}", new_code)

            # отдельно посмотрим качество только новых фич текущего раунда
            round_only_df = new_df[new_cols + [target_col]].copy()
            round_top5, round_fi, round_cv_auc = select_top5_features_fast(
                round_only_df,
                target_col=target_col,
            )

            print("Top-5 текущего раунда:", round_top5)
            print("CV AUC текущего раунда:", round_cv_auc)

            # объединяем кандидаты со всеми предыдущими
            merged_candidate_cols = deduplicate_cols(all_candidate_cols + new_cols)

            # общий финальный отбор среди всех найденных фич
            merged_eval_df = new_df[merged_candidate_cols + [target_col]].copy()
            merged_top5, merged_fi, merged_cv_auc = select_top5_features_fast(
                merged_eval_df,
                target_col=target_col,
            )

            print("Top-5 после объединения:", merged_top5)
            print("CV AUC после объединения:", merged_cv_auc)

            improved = merged_cv_auc > best_cv_auc

            if improved:
                print("Есть улучшение, сохраняем новый лучший результат.")
                best_df = new_df
                best_code = new_code
                best_top5 = merged_top5
                best_fi = merged_fi
                best_cv_auc = merged_cv_auc
                all_candidate_cols = merged_candidate_cols
                no_improve_rounds = 0
            else:
                print("Улучшения нет, раунд не принимаем.")
                no_improve_rounds += 1

            previous_round_code = new_code
            previous_round_df = new_df
            previous_round_cols = list(new_cols)

            if no_improve_rounds >= patience:
                print(
                    f"Нет улучшений {no_improve_rounds} раунда подряд. "
                    "Останавливаем поиск."
                )
                break

        except Exception as e:
            print(f"[WARN] Раунд {round_idx} провалился: {e}")
            no_improve_rounds += 1

            if no_improve_rounds >= patience:
                print(
                    f"Слишком много неудачных/бесполезных раундов подряд ({no_improve_rounds}). "
                    "Останавливаем поиск."
                )
                break

    print("\n[FINAL]")
    print("Best top-5:", best_top5)
    print("Best CV AUC:", best_cv_auc)

    if best_code == current_code:
        final_test_features_base = m_test
    else:
        ns = {"pd": pd, "np": np}
        exec(current_code, ns)
        test_after_phase1, _ = ns["generate_features"](m_test.copy())
        final_test_features_base = test_after_phase1

    # ===== 7. EXPORT =====
    export_final_output(
        code=best_code,
        selected_features=best_top5,
        train_df_features=best_df,
        test_df_features=final_test_features_base,
        original_train=train,
        original_test=test,
        id_col=id_col,
        target_col=target_col,
    )


if __name__ == "__main__":
    main()