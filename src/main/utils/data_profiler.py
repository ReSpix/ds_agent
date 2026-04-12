import pandas as pd


def build_compact_profile(df: pd.DataFrame, target_col: str = "target") -> str:
    lines = [f"📊 SHAPE: {df.shape[0]} строк, {df.shape[1]} колонок"]

    if target_col in df.columns:
        dist = df[target_col].value_counts(normalize=True).to_dict()
        lines.append(f"🎯 TARGET: {dist}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Убираем явные ID и сам таргет из списка
    skip = {"row_id", "id", target_col, "index"}
    num_cols = [c for c in num_cols if c not in skip]
    cat_cols = [c for c in cat_cols if c not in skip]

    lines.append(f"\n🔢 NUMERIC ({len(num_cols)}):")
    for c in num_cols[:15]:  # лимит для экономии токенов
        s = df[c].describe()
        lines.append(
            f"  {c}: μ={s['mean']:.2f} σ={s['std']:.2f} min={s['min']:.2f} max={s['max']:.2f} NaN={df[c].isna().mean():.1%}"
        )

    lines.append(f"\n📝 CATEGORICAL ({len(cat_cols)}):")
    for c in cat_cols[:10]:
        top = df[c].value_counts().head(2).to_dict()
        lines.append(
            f"  {c}: uniq={df[c].nunique()} top={top} NaN={df[c].isna().mean():.1%}"
        )

    return "\n".join(lines)
