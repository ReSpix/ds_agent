import os
from langchain_gigachat import GigaChat

GIGACHAT_CONGIF = {
    "model": "GigaChat-2-Max",
    "verify_ssl_certs": False,
    "profanity_check": False,
    "credentials": os.getenv("GIGACHAT_CREDENTIALS"),
    "scope": os.getenv("GIGACHAT_SCOPE"),
    "temperature": 0.3,
    "max_tokens": 4096,
}

def build_gigachat():
    return GigaChat(**GIGACHAT_CONGIF)