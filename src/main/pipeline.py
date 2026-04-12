import json

import pandas as pd

from src.main.agent_phases.feature_advice import generate_advice
from src.main.agent_phases.generate_better_features import run_phase2_with_advice
from src.main.boosting import select_top5_features_fast
from src.main.llm_builder import build_gigachat
from src.main.agent_phases.merge_dataset import merge_phase
from src.main.agent_phases.generate_features_primary import run_feature_phase

from src.main.utils.atrifact_saver import ArtifactSaver
from src.main.utils.data_profiler import build_compact_profile
from src.main.utils.final_exporter import export_final_output
from src.main.utils.response_parsers import extract_json
from src.main.training_feedback import build_training_feedback_text

TARGET_CV_AUC = 0.8
MAX_IMPROVE_ROUNDS = 35


def main():
    llm = build_gigachat()
    py_code_saver = ArtifactSaver("artifacts", "py")

    with open("data/readme.txt", "r", encoding="utf-8") as file:
        task_description = "\n".join(file.readlines())

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    m_train, m_test, merge_code = merge_phase(
        task_desc=task_description,
        data_dir="data",
        train_df=train,
        test_df=test,
        llm=llm,
    )
    py_code_saver.save("merge", merge_code)

    target_promt = f""" 
        Изучи описание данных: {task_description} 
        Определи главную колонку-идентификатор (ID) и целевую переменную (Target).  
        Верни ТОЛЬКО JSON: {{"id_col": "...", "target_col": "..."}} 
    """

    response = extract_json(llm.invoke(target_promt).content)
    print(response)
    keys = json.loads(response)
    id_col = keys.get("id_col", "id_col")
    target_col = keys.get("target_col", "target")

    profile = build_compact_profile(m_train, target_col)
    df_with_features, feature_code, _new_cols = run_feature_phase(
        task_desc=task_description,
        data_profile=profile,
        df=m_train,
        llm=llm,
        target_col=target_col,
    )
    py_code_saver.save("feature_engineering_phase1", feature_code)

    working_df = df_with_features.copy()
    feature_code_list: list[str] = [feature_code]

    eval_res = select_top5_features_fast(working_df, target_col=target_col)

    improve_round = 0
    while eval_res.cv_auc < TARGET_CV_AUC and improve_round < MAX_IMPROVE_ROUNDS:
        improve_round += 1
        print(
            f"\n=== Итерация улучшения {improve_round}/{MAX_IMPROVE_ROUNDS} "
            f"(CV ROC-AUC={eval_res.cv_auc:.4f}, цель {TARGET_CV_AUC}) ==="
        )

        feedback = build_training_feedback_text(eval_res)

        try:
            advice = generate_advice(
                task_desc=task_description,
                df_profile=profile,
                cv_auc=eval_res.cv_auc,
                feature_importance=dict(eval_res.feature_importance),
                feature_code=feature_code_list[-1],
                llm=llm,
                training_feedback=feedback,
            )
            code2, df2, _cols2 = run_phase2_with_advice(
                task_desc=task_description,
                df=working_df,
                profile=profile,
                llm=llm,
                advice=advice,
                training_feedback=feedback,
            )
        except Exception as e:
            print(f"Итерация улучшения остановлена: {e}")
            break

        if df2 is None:
            print("Фаза 2 не вернула датафрейм, останавливаем цикл.")
            break

        feature_code_list.append(code2)
        working_df = df2
        eval_res = select_top5_features_fast(working_df, target_col=target_col)

    if eval_res.cv_auc < TARGET_CV_AUC:
        print(
            f"\nПредупреждение: CV ROC-AUC {eval_res.cv_auc:.4f} < {TARGET_CV_AUC} "
            f"после {improve_round} итераций улучшения (лимит {MAX_IMPROVE_ROUNDS})."
        )
    else:
        print(f"\nДостигнута цель: CV ROC-AUC >= {TARGET_CV_AUC} ({eval_res.cv_auc:.4f}).")

    py_code_saver.save(
        "feature_engineering_all_rounds",
        "\n\n# --- next round ---\n\n".join(feature_code_list),
    )

    export_final_output(
        feature_code_list,
        eval_res.top5,
        m_train,
        m_test,
        train,
        test,
        id_col=id_col,
        target_col=target_col,
    )


if __name__ == "__main__":
    main()
