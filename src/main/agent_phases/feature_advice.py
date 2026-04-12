from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage


def generate_advice(
    task_desc: str,
    df_profile: str,
    cv_auc: float,
    feature_importance: dict,
    feature_code: str,
    llm: GigaChat,
) -> str:
    """Генерирует текстовые рекомендации на основе CV CatBoost."""
    fi_sorted = sorted(feature_importance.items(), key=lambda x: -x[1])
    top_str = ", ".join([f"{k} ({v:.1f}%)" for k, v in fi_sorted[:3]])
    weak_str = ", ".join([f"{k} ({v:.1f}%)" for k, v in fi_sorted[-3:]])

    prompt = f"""Ты — Senior ML Analyst. Проанализируй результаты генерации признаков для CatBoost.
ЗАДАЧА: {task_desc}
ПРОФИЛЬ ДАННЫХ: {df_profile}
БЫЛИ СОЗДАНЫ НОВЫЕ ФИЧИ: {feature_code}
РЕЗУЛЬТАТЫ: CV AUC = {cv_auc:.4f} | Топ: {top_str} | Слабые: {weak_str}

ПРАВИЛА:
1. Объясни, какой бизнес-сигнал ловят топ-фичи.
2. Укажи 2-3 пропущенных паттерна (статусы, магические числа, отношения счётчиков, пороги).
3. Дай РОВНО 3 конкретных совета для создания новых фич. Формулируй как команды: "создай флаг...", "возьми log1p...", "раздели..." итд.
4. Советы ОБЯЗАНЫ быть совместимы с деревьями: БЕЗ нормализации, БЕЗ factorize(), БЕЗ деления на mean/std.
5. Верни ТОЛЬКО текст рекомендаций. Без markdown, без вступлений, без списков. Максимум 200 слов."""

    resp = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content="Проанализируй и верни рекомендации."),
        ]
    )
    return str(resp.content) if hasattr(resp, "content") else str(resp)
