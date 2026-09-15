"""`python -m scripts.check_env` — sai 0 se da pra trabalhar, 1 se nao da.

Cada linha de erro traz o comando que resolve o problema.
"""

from __future__ import annotations

import importlib.util
import sys

REQUISITOS = ("langchain", "langgraph", "faiss", "sentence_transformers", "rouge_score", "sacrebleu")


def checar() -> int:
    problemas = []

    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        problemas.append(f"Python {sys.version_info.major}.{sys.version_info.minor} fora da faixa >=3.11,<3.13.")

    for pacote in REQUISITOS:
        if importlib.util.find_spec(pacote) is None:
            problemas.append(f"Pacote '{pacote}' ausente — rode: pip install -e \".[dev]\"")

    if not problemas:
        print("Ambiente OK.")
        return 0

    print("Ambiente incompleto:")
    for p in problemas:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(checar())
