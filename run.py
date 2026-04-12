"""Simple entrypoint for running the baseline example."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv

import src.main.pipeline as pipeline

def main() -> None:
    load_dotenv()
    pipeline.main()

if __name__ == "__main__":
    main()
