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
    id_col = keys.get('id_col', 'id_col')
    target_col = keys.get('target_col', 'target')
    
    profile = build_compact_profile(m_train, target_col)
    df_with_features, feature_code, new_cols = run_feature_phase(
        task_desc=task_description,
        data_profile=profile,
        df=m_train,
        llm=llm,
        target_col=target_col,
    )
    py_code_saver.save("feature_engineering", feature_code)

    output_df = df_with_features[[*new_cols, target_col]]
    top5, fi, cv_auc = select_top5_features_fast(output_df, target_col=target_col)

    advice = generate_advice(
        task_desc=task_description,
        df_profile=profile,
        cv_auc=cv_auc,
        feature_importance=dict(fi),
        feature_code=feature_code,
        llm=llm,
    )

    code2, df2, cols2 = run_phase2_with_advice(
        task_desc=task_description, df=m_train, profile=profile, llm=llm, advice=advice
    )

    top5_final, fi_final, cv_final = select_top5_features_fast(
        df2 if df2 is not None else output_df, target_col=target_col
    )

    export_final_output(
        code2 if code2 is not None else feature_code,
        top5_final if top5_final is not None else top5,
        m_train,
        m_test,
        train,
        test,
    )


if __name__ == "__main__":
    main()
