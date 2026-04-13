"""Текстовая сводка метрик обучения для следующей итерации LLM."""

from __future__ import annotations

import numpy as np

from src.main.boosting import CatBoostEvalResult


def build_training_feedback_text(result: CatBoostEvalResult, max_corr_lines: int = 15) -> str:
    fi_sorted = sorted(
        dict(result.feature_importance).items(), key=lambda x: -x[1]
    )
    top_fi = fi_sorted[:8]
    weak_fi = fi_sorted[-5:] if len(fi_sorted) > 5 else []

    corr_sorted = sorted(
        result.target_correlation.items(), key=lambda x: -abs(x[1])
    )[:max_corr_lines]

    cm = result.confusion_matrix
    labels = result.confusion_matrix_labels
    cm_block = (
        "Confusion matrix [строки=истинный класс, столбцы=предсказание]:\n"
        f"Классы (порядок): {labels}\n"
        + np.array2string(cm, separator=", ")
    )

    counts_line = ", ".join(f"{k}={v}" for k, v in sorted(result.class_counts.items()))

    lines = [
        f"ROC-AUC: holdout 20% (shuffle, random_state=42, без стратификации)={result.cv_auc:.4f}; "
        f"на всей текущей обучающей подвыборке (in-sample)={result.train_roc_auc:.4f}",
        f"Топ-5 признаков по CatBoost Feature importance (имена): {', '.join(result.top5)}",
        "Feature importance (имя -> важность, топ): "
        + "; ".join(f"{k}={v:.2f}" for k, v in top_fi),
        "Слабые по важности: "
        + "; ".join(f"{k}={v:.2f}" for k, v in weak_fi),
        f"Class imbalance: ratio (minority/majority)={result.class_imbalance_ratio:.4f}; "
        f"counts по классам: {counts_line}",
        cm_block,
        f"Top errors (бинарная задача, порог 0.5; индексы — позиция строки в текущей подвыборке): "
        f"{result.top_errors_text}",
        f"Training time: финальное обучение CatBoost на подвыборке={result.final_fit_time_sec:.2f}s; "
        f"полное время оценки (holdout-модель + финальное обучение + метрики)={result.total_eval_time_sec:.2f}s",
        "Корреляция признака с таргетом (Pearson, только числовые колонки после подготовки), "
        "по убыванию |corr|: "
        + (
            "; ".join(f"{k}={v:+.3f}" for k, v in corr_sorted)
            if corr_sorted
            else "(нет числовых колонок для корреляции)"
        ),
    ]
    return "\n".join(lines)
