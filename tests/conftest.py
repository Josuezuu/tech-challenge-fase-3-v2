from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _sem_chave_openai_real(monkeypatch):
    """Nenhum teste deve conseguir gastar dinheiro na API paga por acidente."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
