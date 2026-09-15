from __future__ import annotations

from pathlib import Path

import pytest

from src import db, graph, llm, safety


class IndiceFalso:
    """Substitui o FAISS real nos testes — sem embeddings, sem GPU, sem rede."""

    def __init__(self, documentos):
        self._documentos = documentos

    def buscar_mmr(self, pergunta, k=4):
        return self._documentos


def _documento(chunk_id, codigo_fonte, texto):
    from langchain_core.documents import Document

    return Document(page_content=texto, metadata={"chunk_id": chunk_id, "codigo_fonte": codigo_fonte})


@pytest.fixture()
def indice_com_sepse():
    return IndiceFalso(
        [_documento("HSA-PROT-001 § Conduta [0.0]", "HSA-PROT-001",
                     "sepse adulto conduta abrir pulseira laranja coletar pacote laboratorial")]
    )


@pytest.fixture()
def conexao_db(tmp_path: Path):
    conn = db.conectar(tmp_path / "hospital.db")
    db.criar_schema(conn)
    db.seed(conn, n_pacientes=5)
    yield conn
    conn.close()


@pytest.fixture()
def grafo_compilado(indice_com_sepse, conexao_db):
    return graph.montar_grafo(indice_com_sepse, conexao_db, llm.ProviderFake())


@pytest.mark.parametrize(
    "pergunta, paciente_codigo, esperado",
    [
        ("gere o modelo de receita", None, "documento"),
        ("qual a conduta para sepse?", None, "protocolo"),
        ("qual a conduta para sepse?", "PACIENTE_001", "triagem_paciente"),
        ("qual o cardapio de hoje?", None, "fora_escopo"),
    ],
)
def test_classificar_intencao_e_deterministico(pergunta, paciente_codigo, esperado):
    assert graph.classificar_intencao(pergunta, paciente_codigo) == esperado


def test_guardrail_pergunta_distingue_prescricao_de_pergunta_clinica():
    bloqueada = graph.no_guardrail_pergunta(
        {"pergunta_sanitizada": "qual a dose exata de dipirona?", "guardrails_disparados": []}
    )
    assert bloqueada["intencao"] == "recusado_prescricao"
    assert bloqueada["resposta"] == safety.RECUSA_PRESCRICAO

    liberada = graph.no_guardrail_pergunta(
        {"pergunta_sanitizada": "qual a conduta para sepse?", "guardrails_disparados": []}
    )
    assert "resposta" not in liberada


def test_grafo_recusa_pergunta_fora_de_escopo(grafo_compilado):
    estado_final = grafo_compilado.invoke({"pergunta": "qual o cardapio de hoje?", "paciente_codigo": None})
    assert estado_final["intencao"] == "fora_escopo"
    assert safety.RECUSA_ESCOPO in estado_final["resposta"]


def test_grafo_bloqueia_prescricao_e_preserva_a_mensagem_especifica(grafo_compilado):
    """Regressao: o no de recusar nao pode sobrescrever a recusa de prescricao com a de fora_escopo."""
    estado_final = grafo_compilado.invoke({"pergunta": "qual a dose exata de dipirona?", "paciente_codigo": None})
    assert "prescricao_na_pergunta" in estado_final["guardrails_disparados"]
    assert safety.RECUSA_PRESCRICAO in estado_final["resposta"]


def test_grafo_aplica_disclaimer_em_toda_resposta(grafo_compilado):
    estado_final = grafo_compilado.invoke({"pergunta": "qual o cardapio de hoje?", "paciente_codigo": None})
    assert safety.tem_disclaimer(estado_final["resposta"])


def test_grafo_recupera_protocolo_com_fonte_e_registra_trace_id(grafo_compilado):
    estado_final = grafo_compilado.invoke(
        {"pergunta": "qual a conduta inicial para sepse adulto?", "paciente_codigo": None}
    )
    assert estado_final["intencao"] == "protocolo"
    assert estado_final["fontes"]
    assert estado_final.get("trace_id")
