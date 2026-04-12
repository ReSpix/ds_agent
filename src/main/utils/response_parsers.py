import re


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
