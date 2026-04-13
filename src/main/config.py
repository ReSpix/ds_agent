"""
Порог качества и лимиты итераций.

По умолчанию лимиты очень большие: остановка в основном по TARGET_ROC_AUC.
Переопределение через переменные окружения (см. имена в _env_*).
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# Целевой ROC-AUC (holdout в select_top5_features_fast); цикл улучшения крутится, пока лучший score ниже.
TARGET_ROC_AUC = _env_float("TARGET_ROC_AUC", 0.75)

# Внешний цикл улучшения признаков (после phase 1)
MAX_IMPROVE_ROUNDS = _env_int("MAX_IMPROVE_ROUNDS", 10**9)

# Повторы внутри фаз (LLM / merge / CatBoost)
MAX_MERGE_CODE_ATTEMPTS = _env_int("MAX_MERGE_CODE_ATTEMPTS", 10**6)
MAX_FEATURE_PHASE_ATTEMPTS = _env_int("MAX_FEATURE_PHASE_ATTEMPTS", 10**6)
MAX_PHASE2_ATTEMPTS = _env_int("MAX_PHASE2_ATTEMPTS", 10**6)
MAX_ADVICE_LLM_ATTEMPTS = _env_int("MAX_ADVICE_LLM_ATTEMPTS", 10**6)
MAX_CATBOOST_EVAL_RETRIES = _env_int("MAX_CATBOOST_EVAL_RETRIES", 10**6)
