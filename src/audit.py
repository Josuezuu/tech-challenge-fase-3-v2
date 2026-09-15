"""Log auditavel por execucao — sem PII, reconstituivel pelo trace_id."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src import config


@dataclass
class RegistroAuditoria:
    trace_id: str
    pergunta: str
    intencao: str
    fontes: tuple[str, ...]
    guardrails_disparados: tuple[str, ...]
    llm_provider: str
    latencia_s: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def novo_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def registrar(registro: RegistroAuditoria, caminho: Path = config.AUDITORIA_LOG) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(registro), ensure_ascii=False) + "\n")


def buscar_por_trace_id(trace_id: str, caminho: Path = config.AUDITORIA_LOG) -> dict | None:
    if not caminho.exists():
        return None
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            registro = json.loads(linha)
            if registro.get("trace_id") == trace_id:
                return registro
    return None
