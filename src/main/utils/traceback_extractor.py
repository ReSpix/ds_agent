def extract_exec_error(code: str, exception: Exception) -> tuple[int | None, str]:
    """Находит точную строку ошибки внутри кода, выполненного через exec()."""
    tb = exception.__traceback__
    target_lineno = None

    # Идём до конца traceback, запоминаем последний фрейм из '<string>'
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<string>":
            target_lineno = tb.tb_lineno
        tb = tb.tb_next

    if target_lineno is None:
        return None, f"{type(exception).__name__}: {exception}"

    lines = code.splitlines()
    if not (1 <= target_lineno <= len(lines)):
        return None, ""
        # return f"{type(exception).__name__}: {exception} (line {target_lineno} out of bounds)"

    # Контекст: ±2 строки вокруг ошибки
    start = max(0, target_lineno - 3)
    end = min(len(lines), target_lineno + 2)

    context = []
    for i in range(start, end):
        marker = ">>>" if i == target_lineno - 1 else "   "
        context.append(f"{marker} {i+1:3d}: {lines[i]}")

    exc_name = type(exception).__name__
    return target_lineno, "\n".join(context)
